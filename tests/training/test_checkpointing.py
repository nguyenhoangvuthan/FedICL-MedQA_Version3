from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fedicl_mqa.training.checkpointing import CheckpointManager, TrainerState


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeTorch:
    cuda = _FakeCuda()

    @staticmethod
    def get_rng_state() -> list[int]:
        return [1, 2, 3]

    @staticmethod
    def save(payload: object, path: Path) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")


class _FakeModel:
    @staticmethod
    def save_pretrained(path: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization
        path.mkdir(parents=True)
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")
        (path / "adapter_model.safetensors").write_bytes(b"adapter")


class CheckpointTests(unittest.TestCase):
    def test_atomic_checkpoint_pointer_and_hash_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(
                temporary,
                config_hash="config",
                model_id="model",
                model_revision="revision",
                keep=2,
            )
            with patch("fedicl_mqa.training.checkpointing._torch", return_value=_FakeTorch):
                checkpoint = manager.save(
                    "checkpoint-step-0001",
                    model=_FakeModel(),
                    optimizer=None,
                    trainer_state=TrainerState(kind="local", seed=42, global_step=1),
                    is_best=True,
                )
            self.assertEqual(manager.latest(), checkpoint)
            self.assertEqual(manager.resolve("best"), checkpoint)
            manager.verify(checkpoint)
            temporary_checkpoints = [
                path for path in checkpoint.parent.iterdir() if path.name.startswith(".checkpoint-")
            ]
            self.assertFalse(temporary_checkpoints)
            (checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                manager.verify(checkpoint)

    def test_unexpected_checkpoint_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(
                temporary,
                config_hash="config",
                model_id="model",
                model_revision="revision",
            )
            with patch("fedicl_mqa.training.checkpointing._torch", return_value=_FakeTorch):
                checkpoint = manager.save(
                    "checkpoint-step-0001",
                    model=_FakeModel(),
                    optimizer=None,
                    trainer_state=TrainerState(kind="local", seed=42),
                )
            (checkpoint / "unexpected.bin").write_bytes(b"unexpected")
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                manager.verify(checkpoint)

    def test_latest_recovers_a_stale_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(
                temporary,
                config_hash="config",
                model_id="model",
                model_revision="revision",
            )
            with patch("fedicl_mqa.training.checkpointing._torch", return_value=_FakeTorch):
                first = manager.save(
                    "checkpoint-step-0001",
                    model=_FakeModel(),
                    optimizer=None,
                    trainer_state=TrainerState(kind="local", seed=42, global_step=1),
                )
                second = manager.save(
                    "checkpoint-step-0002",
                    model=_FakeModel(),
                    optimizer=None,
                    trainer_state=TrainerState(kind="local", seed=42, global_step=2),
                )
            (Path(temporary) / "last_checkpoint.txt").write_text(
                f"{first.name}\n", encoding="utf-8"
            )
            self.assertEqual(manager.latest(), second)
            self.assertEqual(manager.resolve("last"), second)


if __name__ == "__main__":
    unittest.main()
