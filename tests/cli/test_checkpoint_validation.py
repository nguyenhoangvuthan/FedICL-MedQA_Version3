from __future__ import annotations

import argparse
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fedicl_mqa.cli import parser, paths
from fedicl_mqa.cli.commands import checkpoint_validation, evaluation
from fedicl_mqa.core.config import Config, ControlSettings
from tests.training.test_loop import _examples


class CheckpointValidationCommands(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = Config()
        self.config.experiment.output_dir = self.tmp.name
        self.parts = {
            c: {
                "fit": [replace(_examples()[0], example_id=f"train{c}", question=f"Train {c}")],
                "support": [],
                "validation": [
                    replace(
                        _examples()[0],
                        example_id=f"val{c}",
                        question=f"Validation {c}",
                        split="validation",
                    )
                ],
                "test": [],
            }
            for c in range(5)
        }

    def test_all_modes_select_existing_adapters_without_invoking_training(self):
        for c in range(5):
            (paths.checkpoint_root(self.config, "local") / "seed-42" / f"client-{c}").mkdir(
                parents=True
            )
        (paths.checkpoint_root(self.config, "federated") / "seed-42" / "global").mkdir(parents=True)
        (paths.checkpoint_root(self.config, "centralized") / "seed-42").mkdir(parents=True)
        args = parser.build_parser().parse_args(
            ["validate-checkpoints", "--config", "c.yaml", "--mode", "all", "--seed", "42"]
        )
        with (
            patch.object(checkpoint_validation, "seal_config", return_value=self.config),
            patch.object(checkpoint_validation, "load_partition", return_value=self.parts),
            patch.object(checkpoint_validation, "load_lora_bundle") as load_model,
            patch.object(
                checkpoint_validation,
                "select_existing_checkpoints",
                return_value={
                    "best_checkpoint": "checkpoint-epoch-0002",
                    "best_validation_loss": 0.2,
                },
            ) as select,
            patch("fedicl_mqa.training.loop.train", side_effect=AssertionError("must not train")),
        ):
            args.func(args)
        self.assertEqual(load_model.call_count, 3)
        self.assertEqual(select.call_count, 7)
        self.assertEqual(
            [c.kwargs["expected_kind"] for c in select.call_args_list],
            [*(f"local-client-{c}" for c in range(5)), "federated", "centralized"],
        )
        self.assertEqual(
            [c.kwargs["expected_seed"] for c in select.call_args_list],
            [42000, 42001, 42002, 42003, 42004, 42, 42],
        )
        self.assertEqual([len(c.args[2]) for c in select.call_args_list], [1, 1, 1, 1, 1, 5, 5])

    def test_missing_server_checkpoints_fail_before_loading_model(self):
        args = argparse.Namespace(
            config="unused", mode="centralized", seed=42, all_seeds=False, fl_round=None
        )
        with (
            patch.object(checkpoint_validation, "seal_config", return_value=self.config),
            patch.object(checkpoint_validation, "load_partition", return_value=self.parts),
            patch.object(checkpoint_validation, "load_lora_bundle") as load,
            self.assertRaisesRegex(FileNotFoundError, "checkpoint directory"),
        ):
            checkpoint_validation.command_validate_checkpoints(args)
        load.assert_not_called()

    def test_best_evaluation_loads_selected_adapter_for_all_three_modes(self):
        self.config.data.dataset = "medmcqa"
        self.config.controls = ControlSettings()
        for arm in ("L0", "F0", "C0"):
            with self.subTest(arm=arm):
                manager = MagicMock()
                metadata = {}
                with (
                    patch.object(
                        evaluation, "load_lora_bundle", return_value=SimpleNamespace(model=object())
                    ),
                    patch.object(evaluation, "adapter_state", return_value={}),
                    patch.object(evaluation, "CheckpointManager", return_value=manager),
                    patch.object(evaluation, "_training_partition", return_value=self.parts),
                    patch.object(
                        evaluation,
                        "validation_winner",
                        return_value=("checkpoint-epoch-0002", {"validation_loss": 0.3}),
                    ),
                    patch.object(
                        evaluation,
                        "selected_round",
                        side_effect=AssertionError("must not load fixed round"),
                    ),
                ):
                    bundle, before = evaluation._load_arm_checkpoint(
                        self.config,
                        arm,
                        42,
                        None,
                        best_validation=True,
                        selection_metadata=metadata,
                    )
                    if before:
                        before(0, bundle.model)
                manager.load.assert_called_once_with(
                    "checkpoint-epoch-0002", model=bundle.model, restore_rng=False
                )
                self.assertTrue(metadata)

    def test_best_evaluation_writes_separate_results_and_selection_metadata(self):
        with (
            patch.object(evaluation, "load_partition", return_value=self.parts),
            patch.object(evaluation, "_load_arm_checkpoint", return_value=(object(), None)),
            patch.object(evaluation, "evaluate_arm", return_value={}) as evaluate,
        ):
            evaluation._run_single_evaluation(
                self.config,
                "F0",
                seed=42,
                split="test",
                round_index=None,
                best_validation=True,
            )
        kwargs = evaluate.call_args.kwargs
        self.assertEqual(Path(kwargs["output_dir"]).name, "best-validation")
        self.assertEqual(kwargs["run_metadata"]["checkpoint_selection"], "best-validation")
        self.assertNotEqual(
            kwargs["output_dir"],
            paths.evaluation_dir(self.config, "F0", seed=42, split="test", round_index=None),
        )

    def test_prior_and_round_overrides_are_rejected_for_new_selection(self):
        for arm, round_index in [("B0", None), ("F2", None), ("F0", 8)]:
            with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, "without round"):
                evaluation._run_single_evaluation(
                    self.config,
                    arm,
                    seed=42,
                    split="test",
                    round_index=round_index,
                    best_validation=True,
                )

    def test_parser_accepts_best_checkpoint_and_defaults_to_original_protocol(self):
        args = ["evaluate", "--config", "c.json", "--arm", "C0", "--seed", "42"]
        self.assertEqual(parser.build_parser().parse_args(args).checkpoint, "protocol")
        self.assertEqual(
            parser.build_parser().parse_args([*args, "--checkpoint", "best-validation"]).checkpoint,
            "best-validation",
        )

    def test_epoch_three_evaluation_for_all_three_modes(self):
        self.config.data.dataset = "medmcqa"
        self.config.controls = ControlSettings()
        for arm in ("L0", "F0", "C0"):
            manager = MagicMock()
            metadata = {}
            with (
                self.subTest(arm=arm),
                patch.object(
                    evaluation, "load_lora_bundle", return_value=SimpleNamespace(model=object())
                ),
                patch.object(evaluation, "adapter_state", return_value={}),
                patch.object(evaluation, "CheckpointManager", return_value=manager),
                patch.object(evaluation, "_training_partition", return_value=self.parts),
                patch.object(
                    evaluation,
                    "epoch_checkpoint",
                    return_value=("checkpoint-epoch-0003", {"epoch": 3}),
                ) as select,
                patch.object(
                    evaluation, "selected_round", side_effect=AssertionError("wrong round")
                ),
                patch.object(
                    evaluation, "validation_winner", side_effect=AssertionError("wrong best")
                ),
            ):
                bundle, before = evaluation._load_arm_checkpoint(
                    self.config, arm, 42, None, epoch=3, selection_metadata=metadata
                )
                if before:
                    before(0, bundle.model)
                self.assertEqual(select.call_args.args[1], 3)
            manager.load.assert_called_once_with(
                "checkpoint-epoch-0003", model=bundle.model, restore_rng=False
            )
            self.assertTrue(metadata)

    def test_epoch_cli_writes_separate_epoch_three_results(self):
        args = parser.build_parser().parse_args(
            ["evaluate", "--config", "c.json", "--arm", "C0", "--seed", "42", "--epoch", "3"]
        )
        with (
            patch.object(evaluation, "seal_config", return_value=self.config),
            patch.object(evaluation, "load_partition", return_value=self.parts),
            patch.object(evaluation, "_load_arm_checkpoint", return_value=(object(), None)) as load,
            patch.object(
                evaluation, "evaluate_arm", return_value={"pipeline_accuracy": 0.6}
            ) as run,
        ):
            args.func(args)
        self.assertEqual(load.call_args.kwargs["epoch"], 3)
        self.assertEqual(run.call_args.kwargs["output_dir"].name, "epoch-3")
        self.assertEqual(run.call_args.kwargs["run_metadata"]["checkpoint_selection"], "epoch-3")

    def test_epoch_rejects_best_and_round_override(self):
        with self.assertRaises(SystemExit):
            parser.build_parser().parse_args(
                [
                    "evaluate",
                    "--config",
                    "c",
                    "--arm",
                    "C0",
                    "--seed",
                    "42",
                    "--epoch",
                    "3",
                    "--checkpoint",
                    "best-validation",
                ]
            )
        for epoch, round_index in [(0, None), (3, 4)]:
            with self.assertRaises(ValueError):
                evaluation._run_single_evaluation(
                    self.config, "F0", seed=42, split="test", round_index=round_index, epoch=epoch
                )
