from __future__ import annotations

import importlib.util
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fedicl_mqa.core.config import Config
from fedicl_mqa.core.schema import MCQExample
from fedicl_mqa.modeling.loader import ModelBundle
from fedicl_mqa.training import loop
from fedicl_mqa.training.checkpointing import CheckpointManager, TrainerState


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "right"

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]

    def __call__(self, text, **kwargs):
        return {"input_ids": [4 + len(text) % 7, 5]}


def _config() -> Config:
    config = Config()
    config.hardware.device = "cpu"
    config.training.train_micro_batch_size = 2
    config.training.gradient_accumulation_steps = 2
    config.training.dataloader_num_workers = 0
    config.training.pin_memory = False
    config.training.save_every_steps = 1
    return config


def _examples() -> list[MCQExample]:
    return [
        MCQExample(
            example_id=f"q{index}",
            question=f"Question {index}?",
            options=("a", "bb", "ccc", "dddd"),
            label=index % 4,
            split="train",
        )
        for index in range(5)
    ]


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
class TrainingLoopTests(unittest.TestCase):
    def _model(self):
        import torch

        class Marker:
            pass

        markers = []

        class TrackGraph(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                ctx.marker = Marker()
                markers.append(weakref.ref(ctx.marker))
                return value.clone()

            @staticmethod
            def backward(ctx, grad):
                return grad

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.25))
                self.config = SimpleNamespace(use_cache=False)

            def forward(self, input_ids, **kwargs):
                if any(marker() is not None for marker in markers):
                    raise AssertionError("previous batch's autograd graph is still alive")
                loss = ((self.weight * input_ids.float() - 1) ** 2).mean()
                return SimpleNamespace(loss=TrackGraph.apply(loss))

        return Model(), markers

    def test_graph_is_released_and_partial_accumulation_group_is_preserved(self):
        import torch
        from torch.utils.data import DataLoader

        model, markers = self._model()
        config = _config()
        # Independent reference update, including the final one-batch group.
        expected = torch.nn.Parameter(model.weight.detach().clone())
        optimizer = torch.optim.AdamW(
            [expected], lr=config.training.learning_rate, weight_decay=0.0
        )
        loader = DataLoader(
            loop.AnswerOnlyDataset(_examples(), _Tokenizer(), config.model.max_seq_length),
            batch_size=2,
            shuffle=True,
            generator=torch.Generator().manual_seed(42),
            collate_fn=loop.AnswerCollator(_Tokenizer()),
        )
        for index, batch in enumerate(loader):
            size = 2 if index < 2 else 1
            loss = ((expected * batch["input_ids"].float() - 1) ** 2).mean() / size
            loss.backward()
            if index in (1, 2):
                torch.nn.utils.clip_grad_norm_([expected], config.training.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        state, _ = loop.train(
            ModelBundle(model, _Tokenizer(), torch.device("cpu")),
            _examples(),
            config,
            seed=42,
            epochs=1,
            kind="centralized",
        )
        torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)
        self.assertTrue(all(marker() is None for marker in markers))
        self.assertEqual((state.global_step, state.target_exposures, state.epoch), (2, 5, 1))

    def test_failed_batch_reports_context_and_does_not_commit_a_checkpoint(self):
        import torch

        model, _ = self._model()
        state = TrainerState(kind="centralized", seed=42)
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(
                temporary, config_hash="c", model_id="m", model_revision="r"
            )
            with (
                patch.object(model, "forward", side_effect=RuntimeError("CUDA test failure")),
                self.assertLogs(loop.logger, level="ERROR") as messages,
                self.assertRaisesRegex(RuntimeError, "CUDA test failure"),
            ):
                loop.train(
                    ModelBundle(model, _Tokenizer(), torch.device("cpu")),
                    _examples(),
                    _config(),
                    seed=42,
                    epochs=1,
                    kind="centralized",
                    checkpoint_manager=manager,
                    initial_state=state,
                )
            self.assertIsNone(manager.latest())
            self.assertEqual((state.global_step, state.target_exposures), (0, 0))
            self.assertIn("during forward at epoch 1/1 batch 1/3", messages.output[0])
            self.assertIn("input shape (2, 8)", messages.output[0])

    def test_nonfinite_loss_cannot_update_weights_or_save_a_checkpoint(self):
        import torch

        model, _ = self._model()
        initial_weight = model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(
                temporary, config_hash="c", model_id="m", model_revision="r"
            )
            with (
                patch.object(
                    model, "forward", return_value=SimpleNamespace(loss=model.weight * float("nan"))
                ),
                self.assertLogs(loop.logger, level="ERROR"),
                self.assertRaisesRegex(RuntimeError, "non-finite training loss"),
            ):
                loop.train(
                    ModelBundle(model, _Tokenizer(), torch.device("cpu")),
                    _examples(),
                    _config(),
                    seed=42,
                    epochs=1,
                    kind="centralized",
                    checkpoint_manager=manager,
                )
            self.assertIsNone(manager.latest())
            torch.testing.assert_close(model.weight, initial_weight, rtol=0, atol=0)

    def test_windows_cuda_sync_uses_the_selected_device_and_propagates_errors(self):
        import torch

        device = torch.device("cuda:1")
        error = RuntimeError("queued CUDA failure")
        with (
            patch.object(loop.sys, "platform", "win32"),
            patch.object(torch.cuda, "synchronize", side_effect=error) as sync,
            self.assertRaises(RuntimeError) as caught,
        ):
            loop._synchronize_before_release(device)
        self.assertIs(caught.exception, error)
        sync.assert_called_once_with(device)

    def test_nonfinite_gradients_cannot_update_weights(self):
        import torch

        model, _ = self._model()
        initial_weight = model.weight.detach().clone()
        model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        state = TrainerState(kind="centralized", seed=42)
        with (
            self.assertLogs(loop.logger, level="ERROR") as messages,
            self.assertRaisesRegex(RuntimeError, "non-finite"),
        ):
            loop.train(
                ModelBundle(model, _Tokenizer(), torch.device("cpu")),
                _examples(),
                _config(),
                seed=42,
                epochs=1,
                kind="centralized",
                initial_state=state,
            )
        torch.testing.assert_close(model.weight, initial_weight, rtol=0, atol=0)
        self.assertEqual(state.global_step, 0)
        self.assertIn("during optimizer step", messages.output[0])

    def test_cpu_and_non_windows_training_do_not_synchronize_cuda(self):
        import torch

        with patch.object(torch.cuda, "synchronize") as sync:
            with patch.object(loop.sys, "platform", "win32"):
                loop._synchronize_before_release(torch.device("cpu"))
            with patch.object(loop.sys, "platform", "linux"):
                loop._synchronize_before_release(torch.device("cuda"))
        sync.assert_not_called()


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("torch", "peft", "transformers")),
    "requires PyTorch, PEFT and Transformers",
)
class ResumeIntegrationTests(unittest.TestCase):
    def _bundle(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(123)
        base = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
            )
        )
        model = get_peft_model(
            base,
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.2,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
        return ModelBundle(model, _Tokenizer(), torch.device("cpu"))

    def test_mid_epoch_resume_matches_uninterrupted_adapter_optimizer_and_counters(self):
        import torch
        from peft import get_peft_model_state_dict

        config = _config()
        with tempfile.TemporaryDirectory() as temporary:

            def manager(name):
                return CheckpointManager(
                    Path(temporary) / name,
                    config_hash=config.hash,
                    model_id=config.model.id,
                    model_revision=config.model.revision,
                )

            full_manager, resumed_manager = manager("full"), manager("resumed")
            full = self._bundle()
            full_state, _ = loop.train(
                full,
                _examples(),
                config,
                seed=42,
                epochs=2,
                kind="centralized",
                checkpoint_manager=full_manager,
            )
            interrupted = self._bundle()
            save = resumed_manager.save

            def stop_after_commit(*args, **kwargs):
                save(*args, **kwargs)
                raise RuntimeError("simulated interruption after checkpoint commit")

            with (
                patch.object(resumed_manager, "save", side_effect=stop_after_commit),
                self.assertRaisesRegex(RuntimeError, "simulated interruption"),
            ):
                loop.train(
                    interrupted,
                    _examples(),
                    config,
                    seed=42,
                    epochs=2,
                    kind="centralized",
                    checkpoint_manager=resumed_manager,
                )
            restored = self._bundle()
            torch.manual_seed(999)  # The checkpoint must restore RNG for LoRA dropout.
            resumed_state, _ = loop.train(
                restored,
                _examples(),
                config,
                seed=42,
                epochs=2,
                kind="centralized",
                checkpoint_manager=resumed_manager,
                resume="auto",
            )
            torch.testing.assert_close(
                get_peft_model_state_dict(restored.model),
                get_peft_model_state_dict(full.model),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                torch.load(resumed_manager.latest() / "optimizer.pt", weights_only=True),
                torch.load(full_manager.latest() / "optimizer.pt", weights_only=True),
                rtol=0,
                atol=0,
            )
            for key in (
                "epoch",
                "batch_in_epoch",
                "global_step",
                "optimizer_updates",
                "target_exposures",
            ):
                self.assertEqual(getattr(resumed_state, key), getattr(full_state, key))
            self.assertEqual(resumed_state.target_exposures, 10)


if __name__ == "__main__":
    unittest.main()
