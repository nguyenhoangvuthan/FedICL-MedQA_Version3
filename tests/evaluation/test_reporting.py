from __future__ import annotations

import unittest

from fedicl_mqa.evaluation.metrics import Prediction
from fedicl_mqa.evaluation.reporting import build_contrast_report


def values(*, correct: bool, likelihood_correct: bool, seed: int | None) -> list[Prediction]:
    return [
        Prediction(
            example_id=f"q-{index}",
            gold=index % 4,
            predicted=index % 4 if correct else (index + 1) % 4,
            stage="terminal_label",
            client_id=index % 2,
            subject="medicine",
            seed=seed,
            likelihood_predicted=index % 4 if likelihood_correct else (index + 1) % 4,
            likelihood_confidence=0.8,
        )
        for index in range(4)
    ]


class ReportingTests(unittest.TestCase):
    def test_reverse_evaluator_effect_is_flagged(self) -> None:
        seeds = (42, 43)
        trained_wrong = {
            seed: values(correct=False, likelihood_correct=True, seed=seed) for seed in seeds
        }
        trained_right = {
            seed: values(correct=True, likelihood_correct=False, seed=seed) for seed in seeds
        }
        arm_predictions = {
            "B0": {None: values(correct=False, likelihood_correct=True, seed=None)},
            "B1": {None: values(correct=True, likelihood_correct=False, seed=None)},
            "L0": trained_wrong,
            "L1": trained_right,
            "F0": trained_wrong,
            "F1": trained_right,
            "F2": trained_right,
            "C0": trained_right,
        }
        report = build_contrast_report(
            arm_predictions, samples=20, confidence=0.95, bootstrap_seed=1
        )
        self.assertTrue(report["primary"]["base_icl"]["evaluator_dependent"])
        self.assertTrue(report["primary"]["local_icl"]["evaluator_dependent"])
        self.assertFalse(report["primary"]["client_aware"]["evaluator_dependent"])


if __name__ == "__main__":
    unittest.main()
