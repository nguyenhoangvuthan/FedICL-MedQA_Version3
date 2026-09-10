from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from fedicl_mqa.training.checkpointing import CheckpointManager, TrainerState


@contextmanager
def _working_directory(path: str | Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


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


class RelativeRootTests(unittest.TestCase):
    """output_dir in the shipped configs is relative, so root often is too."""

    def _manager(self, root: str | Path) -> CheckpointManager:
        return CheckpointManager(root, config_hash="config", model_id="model", model_revision="sha")

    def test_resolve_accepts_what_latest_returns(self) -> None:
        """latest() hands load() a rooted path; resolve() must not root it a second time."""
        with tempfile.TemporaryDirectory() as temporary, _working_directory(temporary):
            manager = self._manager(Path("outputs") / "a5000" / "local" / "client-0")
            checkpoint = manager.root / "checkpoint-epoch-0001-step-00000235"
            checkpoint.mkdir(parents=True)
            self.assertEqual(manager.resolve(checkpoint), checkpoint.resolve())

    def test_a_bare_name_is_still_read_as_root_relative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(Path(temporary) / "run")
            checkpoint = manager.root / "checkpoint-0001"
            checkpoint.mkdir(parents=True)
            self.assertEqual(manager.resolve("checkpoint-0001"), checkpoint.resolve())

    def test_root_is_absolute_so_paths_cannot_depend_on_the_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, _working_directory(temporary):
            self.assertTrue(self._manager("outputs/a5000/local").root.is_absolute())


class WorkingDirectoryCleanupTests(unittest.TestCase):
    def test_relative_root_tests_leave_directory_before_removal(self) -> None:
        """Enforce the Windows directory-lock constraint on every test platform."""
        original = tempfile.TemporaryDirectory

        class CheckedTemporaryDirectory(original):
            def __exit__(self, *args):
                root = Path(self.name).resolve()
                cwd = Path.cwd().resolve()
                if cwd == root or root in cwd.parents:
                    raise AssertionError("temporary directory is still the current directory")
                return super().__exit__(*args)

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(RelativeRootTests)
        result = unittest.TestResult()
        with patch.object(tempfile, "TemporaryDirectory", CheckedTemporaryDirectory):
            suite.run(result)
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)

    def test_working_directory_is_restored_after_an_exception(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with _working_directory(temporary):
                    raise RuntimeError("test failure")
            self.assertEqual(Path.cwd(), previous)


class MissingCheckpointMessageTests(unittest.TestCase):
    """A missing checkpoint printed only its path, which explained nothing."""

    def _manager(self, root: str | Path) -> CheckpointManager:
        return CheckpointManager(root, config_hash="config", model_id="model", model_revision="sha")

    def test_message_names_the_path_and_the_stale_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(temporary)
            with self.assertRaises(FileNotFoundError) as caught:
                manager.resolve("checkpoint-epoch-0001-step-00000235")
            message = str(caught.exception)
            self.assertIn("checkpoint-epoch-0001-step-00000235", message)
            self.assertIn("last_checkpoint.txt", message)
            self.assertNotEqual(message, str(manager.root / "checkpoint-epoch-0001-step-00000235"))

    def test_a_reference_outside_the_root_is_still_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = self._manager(Path(temporary) / "run")
            outside = Path(temporary) / "elsewhere"
            outside.mkdir()
            with self.assertRaises(ValueError):
                manager.resolve(outside)


if __name__ == "__main__":
    unittest.main()
