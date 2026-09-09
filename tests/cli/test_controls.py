from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

from fedicl_mqa.cli import paths, pipeline
from fedicl_mqa.cli.commands import evaluation, training
from fedicl_mqa.core.config import Config, ControlSettings
from fedicl_mqa.core.io import file_sha256, object_hash, read_json, write_json
from fedicl_mqa.core.schema import MCQExample
from fedicl_mqa.evaluation.arms import active_arms
from fedicl_mqa.training import workflows
from fedicl_mqa.training.checkpointing import TrainerState


def controlled(root):
    c = Config()
    c.data.dataset = "medmcqa"
    c.controls = ControlSettings()
    c.experiment.output_dir = str(root)
    return c


class ConfigControlTests(unittest.TestCase):
    def test_old_configuration_serialization_and_hash_are_unchanged(self):
        config = Config()
        legacy = asdict(config)
        legacy.pop("controls")
        self.assertEqual(config.to_dict(), legacy)
        self.assertEqual(config.hash, object_hash(legacy))
        self.assertEqual(len(active_arms(config)), 8)
        config.controls = ControlSettings()
        with self.assertRaisesRegex(ValueError, "native MedMCQA"):
            config.validate()
        config.data.dataset = "medmcqa"
        config.validate()
        self.assertEqual(len(active_arms(config)), 13)
        self.assertNotEqual(config.hash, object_hash(legacy))
        self.assertEqual(Config.from_mapping(config.to_dict()).hash, config.hash)

    def test_controlled_pipeline_obeys_new_dependencies(self):
        config = controlled("/tmp/unused-controls-test")
        names = [s.name for s in pipeline.build_steps(config, split="test", force=False)]
        self.assertEqual(names.count("build-priors"), 1)
        self.assertLess(names.index("select-round"), names.index("build-priors"))
        self.assertLess(names.index("build-priors"), names.index("train-local-matched"))
        self.assertLess(names.index("train-local-matched"), names.index("evaluate-all"))

    def test_failed_retrieval_preflight_is_not_skipped_on_resume(self):
        with tempfile.TemporaryDirectory() as root:
            config = controlled(root)
            target = paths.data_root(config) / "retrieval_audit.json"
            write_json(
                target,
                {"config_hash": config.hash, "capacity_failures": 0, "over_budget_queries": 1},
            )
            steps = {s.name: s for s in pipeline.build_steps(config, split="test", force=False)}
            self.assertFalse(steps["audit-retrieval"].is_done())
            audit = read_json(target)
            audit["over_budget_queries"] = 0
            write_json(target, audit)
            self.assertTrue(steps["audit-retrieval"].is_done())


class ControlCommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = controlled(self.tmp.name)
        write_json(
            paths.checkpoint_root(self.config, "federated") / "selected_round.json", {"round": 6}
        )
        self.parts = {}
        self.rows = []
        for c in range(5):
            self.parts[c] = {r: [] for r in ("fit", "support", "validation", "test")}
            for s in ("Anatomy", "Pathology"):
                for i in range(10):
                    q = MCQExample(
                        f"{c}-{s}-{i}", "Q", ("a", "b", "c", "d"), 0, "validation", subject=s
                    )
                    self.parts[c]["validation"].append(q)
                    self.rows.append(
                        {
                            "prediction": {
                                "example_id": q.example_id,
                                "client_id": c,
                                "subject": s,
                                "gold": 0,
                                "seed": 42,
                                "predicted": int(i < (c + 1 if s == "Anatomy" else 8 - c)),
                            }
                        }
                    )
                self.parts[c]["support"].append(
                    MCQExample(f"s-{c}-{s}", "Q", ("a", "b", "c", "d"), 0, "train", subject=s)
                )
            self.parts[c]["fit"] = self.parts[c]["support"] * 3
        self.source = (
            paths.evaluation_dir(self.config, "F0", seed=42, split="validation", round_index=6)
            / "predictions.jsonl"
        )
        self.source.parent.mkdir(parents=True)
        self.source.write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        write_json(
            self.source.with_name("summary.json"),
            {
                "config_hash": self.config.hash,
                "arm": "F0",
                "split": "validation",
                "seed": 42,
                "run_metadata": {"selected_round": 6},
                "predictions_sha256": file_sha256(self.source),
            },
        )

    def build_prior(self):
        with (
            patch.object(training, "seal_config", return_value=self.config),
            patch.object(training, "load_partition", return_value=self.parts),
        ):
            training.command_build_priors(argparse.Namespace(config="unused", seed=42, round=6))
        return paths.priors_path(self.config, seed=42, round_index=6)

    def test_prior_artifact_binds_validation_and_rejects_source_tampering_before_gpu(self):
        prior = self.build_prior()
        audit = read_json(prior.with_suffix(".audit.json"))
        self.assertEqual(audit["source_split"], "validation")
        self.assertEqual(audit["prior_sha256"], file_sha256(prior))
        self.assertEqual(audit["evidence"]["0"]["Anatomy"]["n"], 40)
        self.source.write_text(self.source.read_text() + "\n")
        with (
            patch.object(evaluation, "load_partition", return_value=self.parts),
            patch.object(evaluation, "_load_arm_checkpoint") as gpu,
        ):
            with self.assertRaisesRegex(ValueError, "provenance hash mismatch"):
                evaluation._run_single_evaluation(
                    self.config, "F2", seed=42, split="test", round_index=None
                )
            gpu.assert_not_called()

    def test_prior_builder_rejects_test_ids_even_with_valid_summary_hash(self):
        payload = self.source.read_text().replace("0-Anatomy-0", "test-item")
        self.source.write_text(payload)
        summary = read_json(self.source.with_name("summary.json"))
        summary["predictions_sha256"] = file_sha256(self.source)
        write_json(self.source.with_name("summary.json"), summary)
        with self.assertRaisesRegex(ValueError, "complete validation cohort"):
            self.build_prior()

    def test_resumed_evaluation_checks_prior_source_again(self):
        prior = self.build_prior()
        target = (
            paths.evaluation_dir(self.config, "F2", seed=42, split="test", round_index=None)
            / "predictions.jsonl"
        )
        target.parent.mkdir(parents=True)
        target.write_text("{}\n")
        summary = {
            "config_hash": self.config.hash,
            "arm": "F2",
            "seed": 42,
            "split": "test",
            "predictions_sha256": file_sha256(target),
            "run_metadata": {"selected_round": 6, "prior_sha256": file_sha256(prior)},
        }
        evaluation.validate_summary_identity(
            self.config, summary, arm="F2", seed=42, split="test", round_index=None
        )
        self.source.write_text(self.source.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "provenance hash mismatch"):
            evaluation.validate_summary_identity(
                self.config, summary, arm="F2", seed=42, split="test", round_index=None
            )

    def test_matched_loader_uses_selected_round_and_rejects_partial_checkpoint(self):
        state = TrainerState(kind="local-matched-client-0", seed=42000, epoch=1, target_exposures=6)
        bundle = SimpleNamespace(model=object())
        with (
            patch.object(evaluation, "load_lora_bundle", return_value=bundle),
            patch.object(evaluation, "adapter_state", return_value={}),
            patch.object(evaluation, "load_partition", return_value=self.parts),
            patch.object(evaluation, "CheckpointManager") as manager,
        ):
            manager.return_value.load.return_value = SimpleNamespace(trainer_state=state)
            _, before = evaluation._load_arm_checkpoint(self.config, "LM0", 42, None)
            with self.assertRaisesRegex(ValueError, "incomplete training budget"):
                before(0, bundle.model)
            state.epoch = 6
            state.target_exposures = 36
            before(0, bundle.model)
            root = manager.call_args.args[0]
            self.assertIn("local-matched/round-6/seed-42/client-0", root.as_posix())

    def test_matched_training_keeps_local_epoch_config_and_uses_six_epochs(self):
        calls = []

        def train(bundle, examples, config, **kwargs):
            calls.append(kwargs)
            return None, {"total_target_exposures": len(examples) * kwargs["epochs"]}

        with (
            patch.object(
                workflows, "load_lora_bundle", return_value=SimpleNamespace(model=object())
            ),
            patch.object(workflows, "adapter_state", return_value={}),
            patch.object(workflows, "set_adapter_state"),
            patch.object(workflows, "configure_runtime"),
            patch.object(workflows, "train", side_effect=train),
        ):
            workflows.train_local_clients(
                self.config,
                {c: r["fit"] for c, r in self.parts.items()},
                seed=42,
                output_root=paths.matched_local_root(self.config),
                fl_rounds=6,
            )
        self.assertEqual(self.config.training.local_epochs, 1)
        self.assertEqual([c["epochs"] for c in calls], [6] * 5)
        self.assertTrue(all(c["kind"].startswith("local-matched") for c in calls))

    def test_summary_from_another_selected_round_is_rejected(self):
        summary = read_json(self.source.with_name("summary.json"))
        summary["run_metadata"]["selected_round"] = 4
        with self.assertRaisesRegex(ValueError, "stale or incompatible"):
            evaluation.validate_summary_identity(
                self.config, summary, arm="F0", seed=42, split="validation", round_index=6
            )
