from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from fedicl_mqa.cli import paths, pipeline
from fedicl_mqa.core.config import Config


def _step(name: str, *, done: bool = False, calls: list[str] | None = None,
          fails: bool = False) -> pipeline.Step:
    def run() -> None:
        if calls is not None:
            calls.append(name)
        if fails:
            raise RuntimeError(f"{name} exploded")

    return pipeline.Step(name=name, run=run, is_done=lambda: done)


class StepDefinitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config()
        self.config.experiment.output_dir = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def test_ten_steps_in_dependency_order(self) -> None:
        names = [step.name for step in pipeline.build_steps(self.config, split="test", force=False)]
        self.assertEqual(
            names,
            [
                "prepare-data",
                "audit-retrieval",
                "train-local",
                "train-federated",
                "evaluate-f0-validation",
                "select-round",
                "train-centralized",
                "build-priors",
                "evaluate-all",
                "report",
            ],
        )

    def test_data_steps_report_done_once_their_artifact_exists(self) -> None:
        steps = {s.name: s for s in pipeline.build_steps(self.config, split="test", force=False)}
        self.assertFalse(steps["audit-retrieval"].is_done())
        audit = paths.data_root(self.config) / "retrieval_audit.json"
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text("{}", encoding="utf-8")
        self.assertTrue(steps["audit-retrieval"].is_done())

    def test_force_never_reports_a_step_as_done(self) -> None:
        audit = paths.data_root(self.config) / "retrieval_audit.json"
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text("{}", encoding="utf-8")
        steps = {s.name: s for s in pipeline.build_steps(self.config, split="test", force=True)}
        self.assertFalse(steps["audit-retrieval"].is_done())


class ExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config()
        self.config.experiment.output_dir = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def state(self) -> dict:
        return yaml.safe_load(
            paths.pipeline_state_path(self.config).read_text(encoding="utf-8")
        )

    def test_runs_every_step_in_order(self) -> None:
        calls: list[str] = []
        steps = [_step("one", calls=calls), _step("two", calls=calls)]
        pipeline.execute(self.config, steps)
        self.assertEqual(calls, ["one", "two"])
        self.assertEqual([s["status"] for s in self.state()["steps"]], ["done", "done"])

    def test_a_completed_step_is_skipped_without_running(self) -> None:
        calls: list[str] = []
        steps = [_step("one", done=True, calls=calls), _step("two", calls=calls)]
        pipeline.execute(self.config, steps)
        self.assertEqual(calls, ["two"])
        self.assertEqual([s["status"] for s in self.state()["steps"]], ["skipped", "done"])

    def test_failure_stops_the_run_and_records_why(self) -> None:
        calls: list[str] = []
        steps = [_step("one", calls=calls), _step("two", fails=True), _step("three", calls=calls)]
        with self.assertRaises(RuntimeError):
            pipeline.execute(self.config, steps)
        self.assertEqual(calls, ["one"], "steps after the failure must not run")
        statuses = [s["status"] for s in self.state()["steps"]]
        self.assertEqual(statuses, ["done", "failed", "pending"])
        self.assertIn("two exploded", self.state()["steps"][1]["error"])

    def test_state_is_written_before_the_first_step_runs(self) -> None:
        """A long first step must still leave something to watch."""
        observed: list[str] = []

        def peek() -> None:
            observed.extend(s["status"] for s in self.state()["steps"])

        pipeline.execute(self.config, [pipeline.Step("one", peek, lambda: False)])
        self.assertEqual(observed, ["running"])

    def test_state_records_position_and_config_hash(self) -> None:
        pipeline.execute(self.config, [_step("one")])
        state = self.state()
        self.assertEqual(state["config_hash"], self.config.hash)
        self.assertEqual(state["dataset"], self.config.data.dataset)
        self.assertEqual(state["steps"][0]["index"], 1)
        self.assertIsNotNone(state["steps"][0]["finished_at"])
        self.assertIsNotNone(state["steps"][0]["duration_seconds"])

    def test_rerunning_after_a_failure_resumes(self) -> None:
        calls: list[str] = []
        with self.assertRaises(RuntimeError):
            pipeline.execute(
                self.config, [_step("one", calls=calls), _step("two", fails=True)]
            )
        calls.clear()
        pipeline.execute(
            self.config, [_step("one", done=True, calls=calls), _step("two", calls=calls)]
        )
        self.assertEqual(calls, ["two"])
        self.assertEqual([s["status"] for s in self.state()["steps"]], ["skipped", "done"])


if __name__ == "__main__":
    unittest.main()
