from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from fedicl_mqa.cli import paths
from fedicl_mqa.cli.commands import evaluation
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
            self.assertEqual(evaluation._effective_seeds(self.config, arm), [None])

    def test_checkpoint_arms_run_once_per_configured_seed(self) -> None:
        for arm in ("L0", "L1", "F0", "F1", "F2", "C0"):
            self.assertEqual(evaluation._effective_seeds(self.config, arm), [42, 43, 44])

    def test_seed_list_follows_the_config(self) -> None:
        self.config.experiment.training_seeds = [7, 8]
        self.assertEqual(evaluation._effective_seeds(self.config, "F0"), [7, 8])


class SweepTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.config = _config(self.root)
        self.addCleanup(self._tmp.cleanup)

    def _seed_summary(self, arm: str, seed: int | None, accuracy: float) -> None:
        output = paths.evaluation_dir(
            self.config, arm, seed=seed, split="test", round_index=None
        )
        write_json(output / "summary.json", {"pipeline_accuracy": accuracy})

    def test_runs_every_seed_when_nothing_exists(self) -> None:
        calls: list[int | None] = []

        def fake_run(config: Config, arm: str, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs["seed"])
            return {"pipeline_accuracy": 0.5}

        with mock.patch.object(evaluation, "_run_single_evaluation", fake_run):
            accuracies = evaluation._sweep_arm(
                self.config, "F0", split="test", round_index=None, force=False
            )
        self.assertEqual(calls, [42, 43, 44])
        self.assertEqual(accuracies, [0.5, 0.5, 0.5])

    def test_completed_runs_are_skipped_and_their_accuracy_reused(self) -> None:
        self._seed_summary("F0", 42, 0.71)

        def fake_run(config: Config, arm: str, **kwargs: Any) -> dict[str, Any]:
            self.assertNotEqual(kwargs["seed"], 42, "seed 42 was already complete")
            return {"pipeline_accuracy": 0.5}

        with mock.patch.object(evaluation, "_run_single_evaluation", fake_run):
            accuracies = evaluation._sweep_arm(
                self.config, "F0", split="test", round_index=None, force=False
            )
        self.assertEqual(accuracies, [0.71, 0.5, 0.5])

    def test_force_reruns_completed_evaluations(self) -> None:
        self._seed_summary("F0", 42, 0.71)
        calls: list[int | None] = []

        def fake_run(config: Config, arm: str, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs["seed"])
            return {"pipeline_accuracy": 0.5}

        with mock.patch.object(evaluation, "_run_single_evaluation", fake_run):
            accuracies = evaluation._sweep_arm(
                self.config, "F0", split="test", round_index=None, force=True
            )
        self.assertEqual(calls, [42, 43, 44])
        self.assertEqual(accuracies, [0.5, 0.5, 0.5])

    def test_base_arm_sweeps_exactly_one_run(self) -> None:
        calls: list[int | None] = []

        def fake_run(config: Config, arm: str, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs["seed"])
            return {"pipeline_accuracy": 0.4}

        with mock.patch.object(evaluation, "_run_single_evaluation", fake_run):
            accuracies = evaluation._sweep_arm(
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

        with mock.patch.object(evaluation, "_run_single_evaluation", fake_run):
            evaluation._sweep_arm(self.config, "C0", split="test", round_index=6, force=False)
            evaluation._sweep_arm(self.config, "F0", split="test", round_index=6, force=False)
        self.assertEqual(seen[:3], [None, None, None])
        self.assertEqual(seen[3:], [6, 6, 6])


if __name__ == "__main__":
    unittest.main()
