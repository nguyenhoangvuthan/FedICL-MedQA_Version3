from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fedicl_mqa.core.config import Config
from fedicl_mqa.core.io import file_sha256, read_json, write_json
from fedicl_mqa.data.preparation import _verify_partition_files
from fedicl_mqa.training.checkpointing import CheckpointManager

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "migrate_context_budget.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("migrate_context_budget", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MigrationTestCase(unittest.TestCase):
    """Builds the parts of a sealed run that record the configuration hash."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "outputs" / "a5000"
        self.addCleanup(self._tmp.cleanup)
        self.script = _load_script()

        self.config = Config()
        self.config.experiment.output_dir = str(self.root)
        # Reproduce a run sealed before the budget was raised.
        self.config.model.max_seq_length = 2048
        self.old_hash = self.config.hash
        self.root.mkdir(parents=True)
        self.sealed = self.root / "sealed_config.json"
        write_json(self.sealed, self.config.to_dict())

        self.data_root = self.root / "data" / "medqa"
        client = self.data_root / "client_0"
        client.mkdir(parents=True)
        (client / "fit.jsonl").write_text('{"example_id": "a"}\n', encoding="utf-8")
        manifest = self.data_root / "partition_manifest.json"
        write_json(manifest, {"config_hash": self.old_hash, "assignments": {}})
        write_json(
            self.data_root / "file_hashes.json",
            {
                "partition_manifest.json": file_sha256(manifest),
                "client_0/fit.jsonl": file_sha256(client / "fit.jsonl"),
            },
        )

        self.checkpoint = (
            self.root / "training" / "medqa" / "local" / "seed-42" / "client-0" / "checkpoint-0001"
        )
        (self.checkpoint / "adapter").mkdir(parents=True)
        (self.checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(b"weights")
        write_json(
            self.checkpoint / "state.json",
            {
                "format_version": CheckpointManager.FORMAT_VERSION,
                "config_hash": self.old_hash,
                "model_id": self.config.model.id,
                "model_revision": self.config.model.revision,
                "trainer_state": {"epoch": 1, "global_step": 235, "round_index": 0},
                "metrics": {},
                "extra": {},
            },
        )
        self._rehash_checkpoint()

    def _rehash_checkpoint(self) -> None:
        write_json(
            self.checkpoint / "hashes.json",
            {
                str(path.relative_to(self.checkpoint)): file_sha256(path)
                for path in sorted(self.checkpoint.rglob("*"))
                if path.is_file() and path.name != "hashes.json"
            },
        )

    def _manager(self, config_hash: str) -> CheckpointManager:
        return CheckpointManager(
            self.checkpoint.parent,
            config_hash=config_hash,
            model_id=self.config.model.id,
            model_revision=self.config.model.revision,
        )

    def run_script(self, *argv: str) -> None:
        with mock.patch("sys.argv", ["migrate_context_budget.py", *argv]):
            self.script.main()

    def new_hash(self) -> str:
        after = Config.from_mapping(read_json(self.sealed))
        return after.hash


class GuardTests(MigrationTestCase):
    def test_a_field_outside_the_allowlist_is_refused(self) -> None:
        """Only fields that cannot change a training input may be re-stamped."""
        with self.assertRaises(SystemExit) as caught:
            self.run_script(
                "--config", str(self.sealed), "--set", "lora.rank=32", "--apply"
            )
        self.assertIn("not migratable", str(caught.exception))
        self.assertEqual(read_json(self.sealed)["lora"]["rank"], 16)

    def test_dry_run_writes_nothing(self) -> None:
        self.run_script("--config", str(self.sealed), "--set", "model.max_seq_length=4096")
        self.assertEqual(read_json(self.sealed)["model"]["max_seq_length"], 2048)
        self.assertEqual(
            read_json(self.data_root / "partition_manifest.json")["config_hash"], self.old_hash
        )

    def test_a_no_op_change_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            self.run_script(
                "--config", str(self.sealed), "--set", "model.max_seq_length=2048", "--apply"
            )


class ApplyTests(MigrationTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.run_script(
            "--config", str(self.sealed), "--set", "model.max_seq_length=4096", "--apply"
        )

    def test_the_sealed_config_carries_the_new_value(self) -> None:
        self.assertEqual(read_json(self.sealed)["model"]["max_seq_length"], 4096)
        self.assertNotEqual(self.new_hash(), self.old_hash)

    def test_the_partition_still_verifies(self) -> None:
        """file_hashes.json covers partition_manifest.json, so it must be recomputed."""
        _verify_partition_files(self.data_root)
        self.assertEqual(
            read_json(self.data_root / "partition_manifest.json")["config_hash"], self.new_hash()
        )

    def test_the_checkpoint_still_verifies(self) -> None:
        """hashes.json covers state.json, so it must be recomputed alongside it."""
        self._manager(self.new_hash()).verify(self.checkpoint)
        self.assertEqual(
            read_json(self.checkpoint / "state.json")["config_hash"], self.new_hash()
        )

    def test_the_checkpoint_is_selectable_under_the_new_hash(self) -> None:
        self.assertEqual(self._manager(self.new_hash()).latest(), self.checkpoint)

    def test_the_checkpoint_is_rejected_under_the_old_hash(self) -> None:
        """The mechanism still works; only this run's stamp moved."""
        self.assertIsNone(self._manager(self.old_hash).latest())

    def test_a_record_documents_the_change(self) -> None:
        records = list((self.root / "migrations").glob("*.json"))
        self.assertEqual(len(records), 1)
        record = read_json(records[0])
        self.assertEqual(record["old_config_hash"], self.old_hash)
        self.assertEqual(record["new_config_hash"], self.new_hash())
        self.assertEqual(record["changes"], ["model.max_seq_length=4096"])
        self.assertIn("fail-closed", record["rationale"])


if __name__ == "__main__":
    unittest.main()
