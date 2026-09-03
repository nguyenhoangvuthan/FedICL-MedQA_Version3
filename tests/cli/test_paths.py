from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fedicl_mqa.cli import paths
from fedicl_mqa.core.config import Config


class MissingConfigTests(unittest.TestCase):
    """A sealed config that was never written must say which command writes it."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_missing_sealed_config_names_prepare_data(self) -> None:
        missing = self.root / "outputs" / "a5000" / "sealed_config.json"
        with self.assertRaises(FileNotFoundError) as caught:
            paths.require_existing_config(missing)
        message = str(caught.exception)
        self.assertIn("prepare-data", message)
        self.assertIn("configs/a5000.yaml", message)

    def test_suggestion_follows_the_output_directory_name(self) -> None:
        missing = self.root / "outputs" / "a5000-smoke" / "sealed_config.json"
        with self.assertRaises(FileNotFoundError) as caught:
            paths.require_existing_config(missing)
        self.assertIn("configs/a5000-smoke.yaml", str(caught.exception))

    def test_missing_yaml_config_does_not_suggest_prepare_data(self) -> None:
        with self.assertRaises(FileNotFoundError) as caught:
            paths.require_existing_config(self.root / "configs" / "nope.yaml")
        self.assertNotIn("prepare-data", str(caught.exception))

    def test_existing_file_is_returned_unchanged(self) -> None:
        present = self.root / "a5000.yaml"
        present.write_text("experiment: {}\n", encoding="utf-8")
        self.assertEqual(paths.require_existing_config(present), present)


class LayoutTests(unittest.TestCase):
    """The output tree is built here and nowhere else, so pin its shape."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.config = Config()
        self.config.experiment.output_dir = str(self.root)
        self.addCleanup(self._tmp.cleanup)

    def test_results_are_grouped_under_the_arm(self) -> None:
        self.assertEqual(
            paths.evaluation_dir(self.config, "F0", seed=42, split="test", round_index=6),
            self.root / "arms" / "medqa" / "F0" / "seed-42" / "test" / "round-6",
        )

    def test_base_arm_results_are_not_seed_specific(self) -> None:
        self.assertEqual(
            paths.evaluation_dir(self.config, "B1", seed=None, split="test", round_index=None),
            self.root / "arms" / "medqa" / "B1" / "deterministic" / "test" / "selected",
        )

    def test_checkpoints_are_keyed_by_family_not_by_arm(self) -> None:
        """L0/L1 and F0/F1/F2 share one adapter; copying it per arm would let them drift."""
        self.assertEqual(
            paths.checkpoint_root(self.config, "federated"),
            self.root / "training" / "medqa" / "federated",
        )

    def test_prior_lives_under_the_arm_that_consumes_it(self) -> None:
        self.assertEqual(
            paths.priors_path(self.config, seed=42, round_index=6),
            self.root / "arms" / "medqa" / "F2" / "priors" / "seed-42" / "round-6.json",
        )

    def test_summary_marks_a_finished_run(self) -> None:
        self.assertEqual(
            paths.summary_path(self.config, "C0", seed=43, split="test", round_index=None).name,
            "summary.json",
        )

    def test_per_arm_log_and_shared_comparison_table(self) -> None:
        self.assertEqual(
            paths.arm_log_path(self.config, "F2"),
            self.root / "arms" / "medqa" / "F2" / "run_log.json",
        )
        self.assertEqual(
            paths.comparison_path(self.config, "json"), self.root / "arms_comparison.json"
        )
        self.assertEqual(
            paths.pipeline_state_path(self.config), self.root / "pipeline_state.yaml"
        )


if __name__ == "__main__":
    unittest.main()
