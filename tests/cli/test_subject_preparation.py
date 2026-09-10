from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fedicl_mqa.cli.commands.data import command_prepare_data
from fedicl_mqa.core.config import Config, ControlSettings
from fedicl_mqa.core.io import file_sha256, read_json, write_json


class SubjectPreparationTests(unittest.TestCase):
    def test_exclusion_audit_is_persisted_and_covered_by_partition_hash(self):
        def load(name, **kwargs):
            split = kwargs["split"]
            self.assertIn(split, {"train", "validation"})
            return [
                dict(
                    id=f"{split}-{i}",
                    question=f"unique{'validation' if split == 'train' and i == 1 else split}{i}",
                    opa="a",
                    opb="b",
                    opc="c",
                    opd="d",
                    cop=i % 4,
                    subject_name="Unknown" if i == 0 else ("Anatomy" if i % 2 else "Pathology"),
                )
                for i in range(401)
            ]

        with tempfile.TemporaryDirectory() as temporary:
            config = Config()
            config.experiment.output_dir = temporary
            config.data.dataset = "medmcqa"
            config.data.num_clients = 2
            config.controls = ControlSettings(min_validation_per_subject=1)
            with (
                patch("fedicl_mqa.cli.commands.data.seal_config", return_value=config),
                patch.dict(sys.modules, {"datasets": SimpleNamespace(load_dataset=load)}),
            ):
                command_prepare_data(argparse.Namespace(config="unused"))
            root = Path(temporary) / "data" / "medmcqa"
            audit = read_json(root / "subject_audit.json")
            manifest = read_json(root / "partition_manifest.json")
            self.assertEqual(
                manifest["protocol"]["subject_audit"],
                {k: v for k, v in audit.items() if k != "config_hash"},
            )
            self.assertEqual(
                audit["source_subject_filter"]["source_splits"]["train"]["excluded_ids"],
                ["train-0"],
            )
            hashes = read_json(root / "file_hashes.json")
            self.assertEqual(
                hashes["partition_manifest.json"], file_sha256(root / "partition_manifest.json")
            )
            self.assertFalse(
                {"train-0", "validation-0", "train-1"}
                & {a["example_id"] for a in manifest["assignments"]}
            )
            excluded = {
                r["example_id"]
                for group in audit["decontamination"]["exclusions"].values()
                for r in group["records"]
            }
            self.assertIn("train-1", excluded)

    def test_completed_old_partition_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Config()
            config.experiment.output_dir = temporary
            config.data.dataset = "medmcqa"
            config.controls = ControlSettings()
            audit_path = Path(temporary) / "data" / "medmcqa" / "subject_audit.json"
            write_json(audit_path, {"config_hash": config.hash})
            before = audit_path.read_bytes()
            with (
                patch("fedicl_mqa.cli.commands.data.seal_config", return_value=config),
                patch("fedicl_mqa.cli.commands.data.load_native_dataset") as loader,
                self.assertRaisesRegex(ValueError, "new output_dir"),
            ):
                command_prepare_data(argparse.Namespace(config="unused"))
            loader.assert_not_called()
            self.assertEqual(before, audit_path.read_bytes())
