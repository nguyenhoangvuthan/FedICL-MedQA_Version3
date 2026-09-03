from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fedicl_mqa.cli import paths, results
from fedicl_mqa.core.config import Config
from fedicl_mqa.core.io import read_json, write_json


class ResultsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.config = Config()
        self.config.experiment.output_dir = str(self.root)
        self.addCleanup(self._tmp.cleanup)

    def write_summary(
        self, arm: str, seed: int | None, accuracy: float, *, split: str = "test", **extra: float
    ) -> None:
        payload = {
            "arm": arm,
            "seed": seed,
            "split": split,
            "pipeline_accuracy": accuracy,
            "position_macro_f1": accuracy - 0.01,
            "conditional_likelihood_accuracy": accuracy + 0.02,
            "ece": 0.05,
            "exact_match_coverage": 0.99,
            "semantic_fallback_rate": 0.01,
            **extra,
        }
        write_json(
            paths.summary_path(self.config, arm, seed=seed, split=split, round_index=None),
            payload,
        )


class ArmLogTests(ResultsTestCase):
    def test_first_record_creates_the_log(self) -> None:
        results.record_arm_run(
            self.config, "F0", seed=42, split="test", status="completed", accuracy=0.61
        )
        payload = read_json(paths.arm_log_path(self.config, "F0"))
        self.assertEqual(payload["arm"], "F0")
        self.assertEqual(len(payload["records"]), 1)
        self.assertEqual(payload["records"][0]["status"], "completed")
        self.assertEqual(payload["records"][0]["seed"], 42)

    def test_records_accumulate_across_runs(self) -> None:
        for seed in (42, 43, 44):
            results.record_arm_run(
                self.config, "F0", seed=seed, split="test", status="completed", accuracy=0.5
            )
        payload = read_json(paths.arm_log_path(self.config, "F0"))
        self.assertEqual([r["seed"] for r in payload["records"]], [42, 43, 44])

    def test_failure_records_the_error_and_no_accuracy(self) -> None:
        results.record_arm_run(
            self.config, "F2", seed=42, split="test", status="failed", error="no prior"
        )
        record = read_json(paths.arm_log_path(self.config, "F2"))["records"][0]
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error"], "no prior")
        self.assertIsNone(record["accuracy"])

    def test_record_carries_the_config_hash_for_traceability(self) -> None:
        results.record_arm_run(
            self.config, "B1", seed=None, split="test", status="skipped", accuracy=0.4
        )
        payload = read_json(paths.arm_log_path(self.config, "B1"))
        self.assertEqual(payload["config_hash"], self.config.hash)


class ComparisonTableTests(ResultsTestCase):
    def test_table_is_built_from_summaries_on_disk(self) -> None:
        self.write_summary("B1", None, 0.40)
        for seed, value in zip((42, 43, 44), (0.50, 0.52, 0.54), strict=True):
            self.write_summary("F0", seed, value)
        table = results.update_comparison(self.config, split="test")
        self.assertEqual(sorted(table["arms"]), ["B1", "F0"])
        self.assertAlmostEqual(table["arms"]["F0"]["mean"]["pipeline_accuracy"], 0.52)
        self.assertEqual(table["arms"]["F0"]["runs"], 3)
        self.assertEqual(table["arms"]["B1"]["runs"], 1)

    def test_single_run_has_zero_spread(self) -> None:
        self.write_summary("B0", None, 0.31)
        table = results.update_comparison(self.config, split="test")
        self.assertEqual(table["arms"]["B0"]["std"]["pipeline_accuracy"], 0.0)

    def test_arms_without_results_are_absent(self) -> None:
        self.write_summary("B1", None, 0.40)
        table = results.update_comparison(self.config, split="test")
        self.assertNotIn("F2", table["arms"])

    def test_only_the_requested_split_is_included(self) -> None:
        self.write_summary("F0", 42, 0.50, split="validation")
        table = results.update_comparison(self.config, split="test")
        self.assertEqual(table["arms"], {})

    def test_both_files_are_written(self) -> None:
        self.write_summary("B1", None, 0.40)
        results.update_comparison(self.config, split="test")
        self.assertTrue(paths.comparison_path(self.config, "json").exists())
        markdown = paths.comparison_path(self.config, "md").read_text(encoding="utf-8")
        self.assertIn("B1", markdown)
        self.assertIn("pipeline_accuracy", markdown.replace(" ", "_").lower())

    def test_table_is_rewritten_as_more_arms_finish(self) -> None:
        """The table must stay usable while a long sweep is still running."""
        self.write_summary("B1", None, 0.40)
        first = results.update_comparison(self.config, split="test")
        self.assertEqual(list(first["arms"]), ["B1"])
        self.write_summary("C0", 42, 0.60)
        second = results.update_comparison(self.config, split="test")
        self.assertEqual(sorted(second["arms"]), ["B1", "C0"])

    def test_rendered_table_orders_arms_by_accuracy(self) -> None:
        self.write_summary("B1", None, 0.40)
        self.write_summary("C0", 42, 0.60)
        self.write_summary("B0", None, 0.30)
        rendered = results.render_comparison(results.update_comparison(self.config, split="test"))
        rows = [line for line in rendered.splitlines() if line.startswith("| ")]
        armed = [row.split("|")[1].strip() for row in rows[2:]]
        self.assertEqual(armed, ["C0", "B1", "B0"])


if __name__ == "__main__":
    unittest.main()
