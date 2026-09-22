from __future__ import annotations

import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fedicl_mqa.core.io import read_json
from fedicl_mqa.modeling.loader import ModelBundle
from fedicl_mqa.training import validation
from fedicl_mqa.training.checkpointing import CheckpointManager, TrainerState
from tests.training.test_checkpointing import _FakeModel, _FakeTorch
from tests.training.test_loop import _examples, _Tokenizer


def held_out():
    return [
        replace(q, example_id=f"v-{q.example_id}", split="validation", question=f"Held out {i}")
        for i, q in enumerate(_examples())
    ]


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.manager = CheckpointManager(
            self.tmp.name, config_hash="config", model_id="model", model_revision="revision"
        )
        self.bundle = SimpleNamespace(model=_FakeModel())

    def save(self, epoch, *, kind="centralized", seed=42, round_index=0):
        with patch("fedicl_mqa.training.checkpointing._torch", return_value=_FakeTorch):
            return self.manager.save(
                f"checkpoint-epoch-{epoch:04d}",
                model=self.bundle.model,
                optimizer=None,
                trainer_state=TrainerState(
                    kind=kind, seed=seed, epoch=epoch, global_step=epoch, round_index=round_index
                ),
            )

    def select(self, **kwargs):
        return validation.select_existing_checkpoints(
            self.bundle,
            self.manager,
            held_out(),
            max_length=512,
            expected_kind=kwargs.get("kind", "centralized"),
            expected_seed=42,
        )

    def test_explicit_epoch_loads_three_instead_of_last_or_best(self):
        for epoch in (1, 3, 8):
            self.save(epoch)
        self.manager.mark_best("checkpoint-epoch-0001")
        before = {p: p.read_bytes() for p in self.manager.root.rglob("*") if p.is_file()}
        name, metadata = validation.epoch_checkpoint(
            self.manager, 3, expected_kind="centralized", expected_seed=42
        )
        self.assertEqual(name, "checkpoint-epoch-0003")
        self.assertEqual(metadata["epoch"], 3)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_explicit_epoch_fails_without_checkpoint_and_rejects_wrong_seed(self):
        self.save(8)
        with self.assertRaisesRegex(FileNotFoundError, "end-of-epoch 3"):
            validation.epoch_checkpoint(
                self.manager, 3, expected_kind="centralized", expected_seed=42
            )
        self.save(3, seed=43)
        with self.assertRaisesRegex(ValueError, "identity"):
            validation.epoch_checkpoint(
                self.manager, 3, expected_kind="centralized", expected_seed=42
            )

    def test_explicit_epoch_resolves_global_round_three(self):
        with patch("fedicl_mqa.training.checkpointing._torch", return_value=_FakeTorch):
            self.manager.save(
                "checkpoint-round-0003",
                model=self.bundle.model,
                optimizer=None,
                trainer_state=TrainerState(kind="federated", seed=42, epoch=3, round_index=3),
            )
        name, metadata = validation.epoch_checkpoint(
            self.manager, 3, expected_kind="federated", expected_seed=42
        )
        self.assertEqual(name, "checkpoint-round-0003")
        self.assertEqual(metadata["round"], 3)

    def test_explicit_epoch_does_not_accept_mid_epoch_step_checkpoint(self):
        with patch("fedicl_mqa.training.checkpointing._torch", return_value=_FakeTorch):
            self.manager.save(
                "checkpoint-step-00001000",
                model=self.bundle.model,
                optimizer=None,
                trainer_state=TrainerState(
                    kind="centralized", seed=42, epoch=3, global_step=1000, batch_in_epoch=25
                ),
            )
        with self.assertRaises(FileNotFoundError):
            validation.epoch_checkpoint(
                self.manager, 3, expected_kind="centralized", expected_seed=42
            )

    def test_selects_earlier_winner_preserves_checkpoints_and_resume_pointer(self):
        for epoch in (1, 2, 8):
            self.save(epoch)
        self.manager.mark_best("checkpoint-epoch-0008")
        before = {p: p.read_bytes() for p in Path(self.tmp.name).rglob("*") if p.is_file()}
        with (
            patch.object(self.manager, "load") as load,
            patch.object(validation, "validation_loss", side_effect=[1.0, 0.4, 0.9]),
        ):
            report = self.select()
        self.assertEqual(report["best_checkpoint"], "checkpoint-epoch-0002")
        self.assertEqual(load.call_count, 3)
        for call in load.call_args_list:
            self.assertFalse(call.kwargs["restore_rng"])
            self.assertNotIn("optimizer", call.kwargs)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertEqual(validation.validation_winner(self.manager)[0], "checkpoint-epoch-0002")
        with (
            patch.object(self.manager, "load") as load,
            patch.object(validation, "validation_loss") as loss,
        ):
            self.assertEqual(self.select()["best_checkpoint"], "checkpoint-epoch-0002")
            load.assert_not_called()
            loss.assert_not_called()

    def test_interrupted_sweep_resumes_scoring_and_rejects_partial_winner(self):
        for epoch in (1, 2, 8):
            self.save(epoch)
        with (
            patch.object(self.manager, "load"),
            patch.object(
                validation, "validation_loss", side_effect=[0.4, RuntimeError("interrupted")]
            ),
            self.assertRaisesRegex(RuntimeError, "interrupted"),
        ):
            self.select()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validation.validation_winner(self.manager)
        with (
            patch.object(self.manager, "load") as load,
            patch.object(validation, "validation_loss", side_effect=[0.5, 0.9]),
        ):
            self.assertEqual(self.select()["best_checkpoint"], "checkpoint-epoch-0001")
            self.assertEqual(load.call_count, 2)

    def test_tie_selects_earlier_checkpoint_and_skips_round_zero(self):
        self.save(0, kind="federated")
        self.save(1, kind="federated", round_index=1)
        self.save(8, kind="federated", round_index=8)
        with (
            patch.object(self.manager, "load") as load,
            patch.object(validation, "validation_loss", return_value=0.5),
        ):
            self.assertEqual(
                self.select(kind="federated")["best_checkpoint"], "checkpoint-epoch-0001"
            )
            self.assertEqual(load.call_count, 2)

    def test_rejects_different_seed_and_empty_directory(self):
        with self.assertRaisesRegex(ValueError, "no valid trained"):
            self.select()
        self.save(1, seed=43)
        with self.assertRaisesRegex(ValueError, "identity"):
            self.select()

    def test_corrupt_checkpoint_is_skipped_and_nonfinite_loss_fails(self):
        path = self.save(1)
        (path / "hashes.json").unlink()
        self.save(8)
        with (
            patch.object(self.manager, "load"),
            patch.object(validation, "validation_loss", return_value=float("nan")),
            self.assertRaisesRegex(ValueError, "non-finite"),
        ):
            self.select()
        self.assertFalse((self.manager.root / "best_validation_checkpoint.txt").exists())

    def test_changed_validation_cohort_and_tampered_winner_are_rejected(self):
        path = self.save(1)
        with (
            patch.object(self.manager, "load"),
            patch.object(validation, "validation_loss", return_value=0.5),
        ):
            self.select()
        with self.assertRaisesRegex(ValueError, "cohort"):
            validation.select_existing_checkpoints(
                self.bundle,
                self.manager,
                held_out()[:1],
                max_length=512,
                expected_kind="centralized",
                expected_seed=42,
            )
        (path / "adapter" / "adapter_model.safetensors").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            validation.validation_winner(self.manager)

    def test_validation_rejects_test_data_and_train_overlap(self):
        with self.assertRaisesRegex(ValueError, "held-out"):
            validation.validation_identity(_examples())
        with self.assertRaisesRegex(ValueError, "overlaps"):
            validation.validate_fit_separation(
                _examples(), [replace(_examples()[0], split="validation", example_id="another-id")]
            )


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("torch", "peft", "transformers")),
    "requires PyTorch, PEFT and Transformers",
)
class LossTests(unittest.TestCase):
    def test_token_weighting_inference_only_and_mode_rng_restoration(self):
        import torch

        class Tokenizer(_Tokenizer):
            def __call__(self, text, **kwargs):
                return {"input_ids": [4] * (1 if text.endswith("A") else 3)}

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.config = SimpleNamespace(use_cache=True)
                self.counts = []

            def forward(self, labels, **kwargs):
                assert not torch.is_grad_enabled()
                assert not self.training
                tokens = int((labels[:, 1:] != -100).sum())
                self.counts.append(tokens)
                return SimpleNamespace(loss=self.weight * tokens)

        model = Model()
        tokenizer = Tokenizer()
        tokenizer.padding_side = "left"
        bundle = ModelBundle(model, tokenizer, torch.device("cpu"))
        rng = torch.get_rng_state().clone()
        value = validation.validation_loss(bundle, held_out(), 512)
        self.assertGreater(len(set(model.counts)), 1)
        self.assertAlmostEqual(value, sum(n * n for n in model.counts) / sum(model.counts))
        torch.testing.assert_close(torch.get_rng_state(), rng)
        self.assertTrue(model.training)
        self.assertTrue(model.config.use_cache)
        self.assertEqual(tokenizer.padding_side, "left")
        self.assertIsNone(model.weight.grad)
        with (
            patch.object(model, "forward", side_effect=RuntimeError("oom")),
            self.assertRaisesRegex(RuntimeError, "oom"),
        ):
            validation.validation_loss(bundle, held_out(), 512)
        self.assertTrue(model.training)
        self.assertEqual(tokenizer.padding_side, "left")

    def test_real_lora_load_and_validation_do_not_update_saved_parameters(self):
        from tests.training.test_loop import ResumeIntegrationTests

        bundle = ResumeIntegrationTests()._bundle()
        with tempfile.TemporaryDirectory() as root:
            manager = CheckpointManager(root, config_hash="c", model_id="m", model_revision="r")
            checkpoint = manager.save(
                "checkpoint-epoch-0001",
                model=bundle.model,
                optimizer=None,
                trainer_state=TrainerState(kind="centralized", seed=42, epoch=1, global_step=1),
            )
            before = {p: p.read_bytes() for p in checkpoint.rglob("*") if p.is_file()}
            with patch("fedicl_mqa.training.loop.train", side_effect=AssertionError("no training")):
                report = validation.select_existing_checkpoints(
                    bundle,
                    manager,
                    held_out()[:2],
                    max_length=512,
                    expected_kind="centralized",
                    expected_seed=42,
                )
            self.assertGreater(report["best_validation_loss"], 0)
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)
            self.assertTrue(all(p.grad is None for p in bundle.model.parameters()))
            self.assertTrue(read_json(manager.root / "validation_selection.json")["complete"])
