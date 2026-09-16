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
    def test_invalid_collision_is_archived_and_replaced_with_a_verified_checkpoint(self) -> None:
        for damage in ("missing-manifest", "bad-hash", "missing-state"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temporary:
                manager = CheckpointManager(
                    temporary, config_hash="config", model_id="model", model_revision="revision"
                )
                name = "checkpoint-step-00001000"
                with patch("fedicl_mqa.training.checkpointing._torch", return_value=_FakeTorch):
                    checkpoint = manager.save(
                        name,
                        model=_FakeModel(),
                        optimizer=None,
                        trainer_state=TrainerState(kind="local", seed=42, global_step=1000),
                    )
                    if damage == "missing-manifest":
                        (checkpoint / "hashes.json").unlink()
                    elif damage == "bad-hash":
                        (checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(b"bad")
                    else:
                        (checkpoint / "state.json").unlink()
                    original = {
                        str(p.relative_to(checkpoint)): p.read_bytes()
                        for p in checkpoint.rglob("*")
                        if p.is_file()
                    }
                    self.assertIsNone(manager.latest())
                    manager.save(
                        name,
                        model=_FakeModel(),
                        optimizer=None,
                        trainer_state=TrainerState(kind="local", seed=42, global_step=1000),
                    )
                manager.verify(checkpoint)
                self.assertEqual(manager.latest(), checkpoint)
                archives = list((manager.root / ".invalid-checkpoints").iterdir())
                self.assertEqual(len(archives), 1)
                self.assertEqual(
                    original,
                    {
                        str(p.relative_to(archives[0])): p.read_bytes()
                        for p in archives[0].rglob("*")
                        if p.is_file()
                    },
                )

    def test_valid_collision_is_preserved_even_for_an_incompatible_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(
                temporary, config_hash="config", model_id="model", model_revision="revision"
            )
            with patch("fedicl_mqa.training.checkpointing._torch", return_value=_FakeTorch):
                checkpoint = manager.save(
                    "checkpoint-step-00001000",
                    model=_FakeModel(),
                    optimizer=None,
                    trainer_state=TrainerState(kind="local", seed=42, global_step=1000),
                )
                original = (checkpoint / "state.json").read_bytes()
                for config_hash in ("config", "different-config"):
                    other = CheckpointManager(
                        temporary,
                        config_hash=config_hash,
                        model_id="model",
                        model_revision="revision",
                    )
                    with self.assertRaisesRegex(FileExistsError, "valid checkpoint.*--resume auto"):
                        other.save(
                            checkpoint.name,
                            model=_FakeModel(),
                            optimizer=None,
                            trainer_state=TrainerState(kind="local", seed=42, global_step=1000),
                        )
                self.assertEqual((checkpoint / "state.json").read_bytes(), original)
                manager.verify(checkpoint)
                self.assertFalse((manager.root / ".invalid-checkpoints").exists())

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


class WindowsRenameRetryTests(unittest.TestCase):
    """A transient handle on a freshly written file (antivirus, indexer) makes the
    directory rename fail with PermissionError on Windows; the checkpoint itself is
    complete and hashed, so the rename is retried rather than the checkpoint dropped."""

    def _save(self, manager: CheckpointManager) -> Path:
        with patch("fedicl_mqa.training.checkpointing._torch", return_value=_FakeTorch):
            return manager.save(
                "checkpoint-step-00002750",
                model=_FakeModel(),
                optimizer=None,
                trainer_state=TrainerState(kind="centralized", seed=42),
            )

    def test_transient_permission_error_is_retried_and_the_checkpoint_survives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(
                temporary, config_hash="config", model_id="model", model_revision="revision"
            )
            real_replace = os.replace
            failures = {"left": 3}
            waits: list[float] = []

            def flaky_replace(src, dst):
                # Only the checkpoint directory rename is denied; atomic file writes
                # inside it go through untouched, as on a real machine.
                if Path(src).is_dir() and failures["left"]:
                    failures["left"] -= 1
                    raise PermissionError(5, "Access is denied")
                real_replace(src, dst)

            with (
                patch("fedicl_mqa.training.checkpointing.os.replace", side_effect=flaky_replace),
                patch("fedicl_mqa.training.checkpointing.time.sleep", side_effect=waits.append),
            ):
                checkpoint = self._save(manager)
            self.assertTrue((checkpoint / "hashes.json").exists())
            self.assertEqual(len(waits), 3)
            self.assertFalse(list(Path(temporary).glob(".checkpoint-*")))
            manager.verify(checkpoint)

    def test_persistent_permission_error_names_the_likely_cause(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = CheckpointManager(
                temporary, config_hash="config", model_id="model", model_revision="revision"
            )
            real_replace = os.replace

            def denied_directory_replace(src, dst):
                if Path(src).is_dir():
                    raise PermissionError(5, "Access is denied")
                real_replace(src, dst)

            with (
                patch(
                    "fedicl_mqa.training.checkpointing.os.replace",
                    side_effect=denied_directory_replace,
                ),
                patch("fedicl_mqa.training.checkpointing.time.sleep"),
                self.assertRaisesRegex(PermissionError, "antivirus"),
            ):
                self._save(manager)
            self.assertFalse(list(Path(temporary).glob(".checkpoint-*")))


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
