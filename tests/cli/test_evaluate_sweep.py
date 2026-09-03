from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from fedicl_mqa import cli
from fedicl_mqa.core.config import Config
from fedicl_mqa.core.io import write_json


def _config(output_dir: Path) -> Config:
    config = Config()
    config.experiment.output_dir = str(output_dir)
    return config


class EffectiveSeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.config = _config(Path(self._tmp.name))
        self.addCleanup(self._tmp.cleanup)

    def test_base_arms_run_once_without_a_seed(self) -> None:
        """B0/B1 ignore trained checkpoints, so every seed would repeat the same work."""
        for arm in ("B0", "B1"):
            self.assertEqual(cli._effective_seeds(self.config, arm), [None])

    def test_checkpoint_arms_run_once_per_configured_seed(self) -> None:
        for arm in ("L0", "L1", "F0", "F1", "F2", "C0"):
            self.assertEqual(cli._effective_seeds(self.config, arm), [42, 43, 44])

    def test_seed_list_follows_the_config(self) -> None:
        self.config.experiment.training_seeds = [7, 8]
        self.assertEqual(cli._effective_seeds(self.config, "F0"), [7, 8])


class OutputLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.config = _config(self.root)
        self.addCleanup(self._tmp.cleanup)

    def test_seeded_layout_matches_the_existing_evaluate_command(self) -> None:
        path = cli._evaluation_output_dir(
            self.config, "F0", seed=42, split="test", round_index=6
        )
        expected = self.root / "evaluations" / "medqa" / "F0" / "seed-42" / "test" / "round-6"
        self.assertEqual(path, expected)

    def test_base_arm_layout_uses_deterministic_and_selected(self) -> None:
        path = cli._evaluation_output_dir(
            self.config, "B1", seed=None, split="test", round_index=None
        )
        expected = (
            self.root / "evaluations" / "medqa" / "B1" / "deterministic" / "test" / "selected"
        )
        self.assertEqual(path, expected)


class SweepTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.config = _config(self.root)
        self.addCleanup(self._tmp.cleanup)

    def _seed_summary(self, arm: str, seed: int | None, accuracy: float) -> None:
        output = cli._evaluation_output_dir(
            self.config, arm, seed=seed, split="test", round_index=None
        )
        write_json(output / "summary.json", {"pipeline_accuracy": accuracy})

    def test_runs_every_seed_when_nothing_exists(self) -> None:
        calls: list[int | None] = []

        def fake_run(config: Config, arm: str, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs["seed"])
            return {"pipeline_accuracy": 0.5}

        with mock.patch.object(cli, "_run_single_evaluation", fake_run):
            accuracies = cli._sweep_arm(
                self.config, "F0", split="test", round_index=None, force=False
            )
        self.assertEqual(calls, [42, 43, 44])
        self.assertEqual(accuracies, [0.5, 0.5, 0.5])

    def test_completed_runs_are_skipped_and_their_accuracy_reused(self) -> None:
        self._seed_summary("F0", 42, 0.71)

        def fake_run(config: Config, arm: str, **kwargs: Any) -> dict[str, Any]:
            self.assertNotEqual(kwargs["seed"], 42, "seed 42 was already complete")
            return {"pipeline_accuracy": 0.5}

        with mock.patch.object(cli, "_run_single_evaluation", fake_run):
            accuracies = cli._sweep_arm(
                self.config, "F0", split="test", round_index=None, force=False
            )
        self.assertEqual(accuracies, [0.71, 0.5, 0.5])

    def test_force_reruns_completed_evaluations(self) -> None:
        self._seed_summary("F0", 42, 0.71)
        calls: list[int | None] = []

        def fake_run(config: Config, arm: str, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs["seed"])
            return {"pipeline_accuracy": 0.5}

        with mock.patch.object(cli, "_run_single_evaluation", fake_run):
            accuracies = cli._sweep_arm(
                self.config, "F0", split="test", round_index=None, force=True
            )
        self.assertEqual(calls, [42, 43, 44])
        self.assertEqual(accuracies, [0.5, 0.5, 0.5])

    def test_base_arm_sweeps_exactly_one_run(self) -> None:
        calls: list[int | None] = []

        def fake_run(config: Config, arm: str, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs["seed"])
            return {"pipeline_accuracy": 0.4}

        with mock.patch.object(cli, "_run_single_evaluation", fake_run):
            accuracies = cli._sweep_arm(
                self.config, "B1", split="test", round_index=None, force=False
            )
        self.assertEqual(calls, [None])
        self.assertEqual(accuracies, [0.4])

    def test_round_is_ignored_for_non_federated_arms(self) -> None:
        """evaluate-all carries one --round; only F0/F1/F2 may act on it."""
        seen: list[int | None] = []

        def fake_run(config: Config, arm: str, **kwargs: Any) -> dict[str, Any]:
            seen.append(kwargs["round_index"])
            return {"pipeline_accuracy": 0.3}

        with mock.patch.object(cli, "_run_single_evaluation", fake_run):
            cli._sweep_arm(self.config, "C0", split="test", round_index=6, force=False)
            cli._sweep_arm(self.config, "F0", split="test", round_index=6, force=False)
        self.assertEqual(seen[:3], [None, None, None])
        self.assertEqual(seen[3:], [6, 6, 6])


class ParserTests(unittest.TestCase):
    def test_evaluate_arm_defaults_to_the_test_split(self) -> None:
        args = cli.build_parser().parse_args(
            ["evaluate-arm", "--config", "c.yaml", "--arm", "B1"]
        )
        self.assertEqual(args.split, "test")
        self.assertEqual(args.arm, "B1")
        self.assertIsNone(args.gpu)
        self.assertFalse(args.force)

    def test_evaluate_all_needs_only_a_config(self) -> None:
        args = cli.build_parser().parse_args(["evaluate-all", "--config", "c.yaml"])
        self.assertEqual(args.split, "test")
        self.assertEqual(args.func, cli.command_evaluate_all)

    def test_gpu_flag_is_accepted_after_every_subcommand(self) -> None:
        for argv in (
            ["train", "--config", "c.yaml", "--mode", "local", "--seed", "42", "--gpu", "1"],
            ["evaluate-all", "--config", "c.yaml", "--gpu", "0"],
            ["doctor", "--config", "c.yaml", "--gpu", "1"],
        ):
            self.assertIn(cli.build_parser().parse_args(argv).gpu, (0, 1))

    def test_gpu_flag_rejects_devices_outside_the_pair(self) -> None:
        with self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["evaluate-all", "--config", "c.yaml", "--gpu", "2"])


class GpuSelectionTests(unittest.TestCase):
    def test_main_restricts_the_process_to_the_chosen_gpu(self) -> None:
        recorded: dict[str, str | None] = {}

        def fake_command(args: Any) -> None:
            recorded["visible"] = os.environ.get("CUDA_VISIBLE_DEVICES")

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(cli, "command_evaluate_all", fake_command),
            mock.patch.object(cli, "apply_hf_token", lambda *a, **k: "none"),
        ):
            cli.main(["evaluate-all", "--config", "c.yaml", "--gpu", "1"])
        self.assertEqual(recorded["visible"], "1")

    def test_main_leaves_the_variable_alone_without_the_flag(self) -> None:
        recorded: dict[str, str | None] = {}

        def fake_command(args: Any) -> None:
            recorded["visible"] = os.environ.get("CUDA_VISIBLE_DEVICES")

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(cli, "command_evaluate_all", fake_command),
            mock.patch.object(cli, "apply_hf_token", lambda *a, **k: "none"),
        ):
            cli.main(["evaluate-all", "--config", "c.yaml"])
        self.assertIsNone(recorded["visible"])


if __name__ == "__main__":
    unittest.main()


class CudaDiagnosisTests(unittest.TestCase):
    """doctor must name the actual cause, not just report that CUDA is missing."""

    def test_cpu_only_build_is_named_explicitly(self) -> None:
        message = cli._cuda_failure_reason(torch_version="2.14.0+cpu", cuda_build=None)
        self.assertIn("CPU-only", message)
        self.assertIn("2.14.0+cpu", message)
        self.assertIn("download.pytorch.org", message)

    def test_cuda_build_without_a_visible_device_reports_the_driver_side(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            message = cli._cuda_failure_reason(torch_version="2.14.0+cu128", cuda_build="12.8")
        self.assertNotIn("CPU-only", message)
        self.assertIn("driver", message.lower())

    def test_message_surfaces_a_restrictive_visible_device_list(self) -> None:
        """--gpu 3 on a two-GPU box must not look like a broken driver."""
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}, clear=True):
            message = cli._cuda_failure_reason(torch_version="2.14.0+cu128", cuda_build="12.8")
        self.assertIn("CUDA_VISIBLE_DEVICES", message)
        self.assertIn("3", message)


class MissingConfigTests(unittest.TestCase):
    """A sealed config that was never written must say which command writes it."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_missing_sealed_config_names_prepare_data(self) -> None:
        missing = self.root / "outputs" / "a5000" / "sealed_config.json"
        with self.assertRaises(FileNotFoundError) as caught:
            cli._require_existing_config(missing)
        message = str(caught.exception)
        self.assertIn("prepare-data", message)
        self.assertIn("configs/a5000.yaml", message)

    def test_suggestion_follows_the_output_directory_name(self) -> None:
        missing = self.root / "outputs" / "a5000-smoke" / "sealed_config.json"
        with self.assertRaises(FileNotFoundError) as caught:
            cli._require_existing_config(missing)
        self.assertIn("configs/a5000-smoke.yaml", str(caught.exception))

    def test_missing_yaml_config_does_not_suggest_prepare_data(self) -> None:
        with self.assertRaises(FileNotFoundError) as caught:
            cli._require_existing_config(self.root / "configs" / "nope.yaml")
        self.assertNotIn("prepare-data", str(caught.exception))

    def test_existing_file_is_returned_unchanged(self) -> None:
        present = self.root / "a5000.yaml"
        present.write_text("experiment: {}\n", encoding="utf-8")
        self.assertEqual(cli._require_existing_config(present), present)
