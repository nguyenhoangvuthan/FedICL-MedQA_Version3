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
    def test_cuda_forward_limit_does_not_change_cpu_batch_size(self):
        import torch

        self.assertEqual(loop._forward_batch_size(torch.device("cuda:1"), 8), 1)
        self.assertEqual(loop._forward_batch_size(torch.device("cpu"), 8), 8)

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

    def test_failure_after_first_chunk_cannot_step_or_checkpoint_partial_gradients(self):
        import torch

        model, _ = self._model()
        initial_weight = model.weight.detach().clone()
        state = TrainerState(kind="federated-client-1", seed=42)
        calls = 0

        def fail_second_backward(gradient):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("CUDA out of memory")
            return gradient

        model.weight.register_hook(fail_second_backward)

        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(
                temporary, config_hash="c", model_id="m", model_revision="r"
            )
            with (
                patch.object(loop, "_forward_batch_size", return_value=1),
                self.assertLogs(loop.logger, level="ERROR") as messages,
                self.assertRaisesRegex(RuntimeError, "CUDA out of memory"),
            ):
                loop.train(
                    ModelBundle(model, _Tokenizer(), torch.device("cpu")),
                    _examples(),
                    _config(),
                    seed=42,
                    epochs=1,
                    kind=state.kind,
                    checkpoint_manager=manager,
                    initial_state=state,
                )
            self.assertIsNone(manager.latest())
            self.assertIn("during backward", messages.output[0])
            self.assertEqual((state.global_step, state.target_exposures), (0, 0))
            torch.testing.assert_close(model.weight, initial_weight, rtol=0, atol=0)

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

    def test_split_loss_and_gradients_match_qwen_with_unequal_answer_lengths(self):
        import torch

        model = self._bundle().model
        model.eval()  # Disable dropout to compare the mathematical objective.
        batch = loop.AnswerCollator(_Tokenizer())(
            [
                {"input_ids": [1] * 9 + [4], "attention_mask": [1] * 10,
                 "labels": [-100] * 9 + [4]},
                {"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1] * 5,
                 "labels": [-100, -100, 3, 4, 5]},
                {"input_ids": [1, 2, 3], "attention_mask": [1] * 3,
                 "labels": [-100, 2, 3]},
            ]
        )
        reference = model(**batch).loss
        reference.backward()
        gradients = {
            name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
        }
        expected_loss = reference.item()
        del reference
        for limit in (1, 2):  # Also exercise an uneven final chunk.
            with self.subTest(limit=limit):
                model.zero_grad(set_to_none=True)
                actual_loss = 0.0
                chunks = list(loop._training_chunks(batch, limit))
                self.assertEqual(sum(chunk["input_ids"].shape[0] for chunk, _ in chunks), 3)
                self.assertEqual(chunks[-1][0]["input_ids"].shape[1], 8)
                for chunk, weight in chunks:
                    loss = model(**chunk).loss * weight
                    actual_loss += loss.item()
                    loss.backward()
                    del loss
                self.assertAlmostEqual(actual_loss, expected_loss, places=6)
                for name, p in model.named_parameters():
                    if name in gradients:
                        torch.testing.assert_close(p.grad, gradients[name], rtol=1e-5, atol=1e-7)

    def test_split_mid_epoch_resume_matches_uninterrupted_training(self):
        with patch.object(loop, "_forward_batch_size", return_value=1):
            self.test_mid_epoch_resume_matches_uninterrupted_adapter_optimizer_and_counters()

    def test_resume_unsplit_checkpoint_with_memory_guard_and_unchanged_config(self):
        import torch

        from fedicl_mqa.core.io import file_sha256

        config = _config()
        original_hash = config.hash
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(
                temporary, config_hash=config.hash,
                model_id=config.model.id, model_revision=config.model.revision,
            )
            save = manager.save

            def stop_after_commit(*args, **kwargs):
                # Emulate a historical checkpoint without the new runtime metadata.
                kwargs["extra"].pop("forward_micro_batch_size", None)
                save(*args, **kwargs)
                raise RuntimeError("simulated interruption")

            with (
                patch.object(manager, "save", side_effect=stop_after_commit),
                self.assertRaisesRegex(RuntimeError, "simulated interruption"),
            ):
                loop.train(
                    self._bundle(), _examples(), config, seed=42, epochs=2,
                    kind="centralized", checkpoint_manager=manager,
                )
            original = manager.latest()
            hashes = {p: file_sha256(p) for p in original.rglob("*") if p.is_file()}
            with patch.object(loop, "_forward_batch_size", return_value=1):
                state, telemetry = loop.train(
                    self._bundle(), _examples(), config, seed=42, epochs=2,
                    kind="centralized", checkpoint_manager=manager, resume="auto",
                )
            self.assertEqual(config.hash, original_hash)
            self.assertEqual((state.global_step, state.target_exposures, state.epoch), (4, 10, 2))
            self.assertEqual(telemetry["forward_micro_batch_size"], 1)
            self.assertEqual(telemetry["effective_batch_size"], 4)
            self.assertEqual(hashes, {p: file_sha256(p) for p in hashes})
            optimizer_state = torch.load(manager.latest() / "optimizer.pt", weights_only=True)
            self.assertEqual(optimizer_state["state"][0]["step"], 4)

    def test_resume_past_a_corrupt_future_checkpoint_matches_uninterrupted_training(self):
        import torch
        from peft import get_peft_model_state_dict

        config = _config()
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(
                temporary,
                config_hash=config.hash,
                model_id=config.model.id,
                model_revision=config.model.revision,
            )
            full = self._bundle()
            full_state, _ = loop.train(
                full,
                _examples(),
                config,
                seed=42,
                epochs=1,
                kind="local-client-1",
                checkpoint_manager=manager,
            )
            # Keep step 1 valid, but make every later checkpoint unusable. Auto-resume
            # must replay the final batch and save both colliding names successfully.
            for checkpoint in manager.root.glob("checkpoint-*"):
                if checkpoint.name != "checkpoint-step-00000001":
                    (checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(b"damaged")
            resumed = self._bundle()
            resumed_state, _ = loop.train(
                resumed,
                _examples(),
                config,
                seed=42,
                epochs=1,
                kind="local-client-1",
                checkpoint_manager=manager,
                resume="auto",
            )
            torch.testing.assert_close(
                get_peft_model_state_dict(resumed.model),
                get_peft_model_state_dict(full.model),
                rtol=0,
                atol=0,
            )
            self.assertEqual(resumed_state.global_step, full_state.global_step)
            self.assertEqual(resumed_state.target_exposures, full_state.target_exposures)
            self.assertEqual(resumed_state.epoch, full_state.epoch)
            manager.verify(manager.latest())
            archives = list((manager.root / ".invalid-checkpoints").iterdir())
            self.assertEqual(len(archives), 2)
            for archived in archives:
                self.assertEqual(
                    (archived / "adapter" / "adapter_model.safetensors").read_bytes(), b"damaged"
                )

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
