from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import yaml

from fedicl_mqa.cli import paths, pipeline
from fedicl_mqa.cli.commands import evaluation as evaluation_commands
from fedicl_mqa.core.config import Config, ControlSettings, ICLTrainingSettings


def _step(
    name: str, *, done: bool = False, calls: list[str] | None = None, fails: bool = False
) -> pipeline.Step:
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


class ArmSubsetStepTests(unittest.TestCase):
    """`pipeline --arms` keeps only the steps the requested arms depend on."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config()
        self.config.experiment.output_dir = self._tmp.name
        self.config.data.dataset = "medmcqa"
        self.config.controls = ControlSettings()
        self.config.icl_training = ICLTrainingSettings()
        self.addCleanup(self._tmp.cleanup)

    def names(self, arms: list[str]) -> list[str]:
        return [
            s.name for s in pipeline.build_steps(self.config, split="test", force=False, arms=arms)
        ]

    def test_federated_arms_skip_local_centralized_and_priors(self) -> None:
        self.assertEqual(
            self.names(["F0", "F1", "FT0", "FT1"]),
            [
                "prepare-data",
                "audit-retrieval",
                "audit-training-icl",
                "train-federated",
                "evaluate-f0-validation",
                "select-round",
                "train-federated-icl",
                "evaluate-arms",
                "report",
            ],
        )

    def test_prior_arms_pull_in_build_priors(self) -> None:
        names = self.names(["F1", "FP"])
        self.assertIn("build-priors", names)
        self.assertNotIn("train-federated-icl", names)
        self.assertLess(names.index("select-round"), names.index("build-priors"))

    def test_matched_local_needs_the_selected_round_but_not_federated_icl(self) -> None:
        names = self.names(["LM0"])
        self.assertIn("train-federated", names)
        self.assertIn("select-round", names)
        self.assertIn("train-local-matched", names)
        self.assertNotIn("train-federated-icl", names)
        self.assertNotIn("train-local", names)

    def test_inactive_arm_is_rejected(self) -> None:
        self.config.icl_training = None
        with self.assertRaisesRegex(ValueError, "not enabled"):
            self.names(["F0", "FT0"])

    def test_evaluate_arms_step_evaluates_each_requested_arm(self) -> None:
        seen: list[str] = []
        steps = {
            s.name: s
            for s in pipeline.build_steps(
                self.config, split="test", force=False, arms=["FT1", "F0"]
            )
        }
        with mock.patch.object(
            evaluation_commands, "command_evaluate_arm", lambda ns: seen.append(ns.arm)
        ):
            steps["evaluate-arms"].run()
        self.assertEqual(seen, ["F0", "FT1"])

    def test_report_step_passes_the_arm_subset(self) -> None:
        seen: list[list[str]] = []
        steps = {
            s.name: s
            for s in pipeline.build_steps(self.config, split="test", force=False, arms=["F0", "F1"])
        }
        with mock.patch.object(
            evaluation_commands, "command_report", lambda ns: seen.append(ns.arms)
        ):
            steps["report"].run()
        self.assertEqual(seen, [["F0", "F1"]])


class ExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.config = Config()
        self.config.experiment.output_dir = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def state(self) -> dict:
        return yaml.safe_load(paths.pipeline_state_path(self.config).read_text(encoding="utf-8"))

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
            pipeline.execute(self.config, [_step("one", calls=calls), _step("two", fails=True)])
        calls.clear()
        pipeline.execute(
            self.config, [_step("one", done=True, calls=calls), _step("two", calls=calls)]
        )
        self.assertEqual(calls, ["two"])
        self.assertEqual([s["status"] for s in self.state()["steps"]], ["skipped", "done"])


if __name__ == "__main__":
    unittest.main()
