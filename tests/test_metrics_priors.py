from __future__ import annotations

import unittest

from fedicl_mqa.metrics import (
    Prediction,
    accuracy,
    evaluation_summary,
    hierarchical_paired_bootstrap,
    holm_adjust,
    paired_item_bootstrap,
)
from fedicl_mqa.priors import leave_one_client_out_weakness


def prediction(index: int, predicted: int | None, *, seed: int | None = None) -> Prediction:
    return Prediction(
        example_id=f"q-{index}",
        gold=index % 4,
        predicted=predicted,
        stage="unresolved" if predicted is None else "terminal_label",
        client_id=index % 2,
        subject=f"s{index % 2}",
        seed=seed,
        likelihood_predicted=predicted,
        likelihood_confidence=0.8,
    )


class MetricsAndPriorTests(unittest.TestCase):
    def test_unresolved_remains_in_accuracy_denominator(self) -> None:
        values = [prediction(0, 0), prediction(1, None)]
        self.assertEqual(accuracy(values), 0.5)
        self.assertEqual(evaluation_summary(values)["unresolved_rate"], 0.5)

    def test_paired_bootstrap_effect(self) -> None:
        left = [prediction(i, None) for i in range(8)]
        right = [prediction(i, i % 4) for i in range(8)]
        result = paired_item_bootstrap(left, right, samples=200, seed=1)
        self.assertEqual(result["effect"], 1.0)

    def test_hierarchical_bootstrap_requires_paired_seeds(self) -> None:
        values = {42: [prediction(i, i % 4, seed=42) for i in range(4)]}
        with self.assertRaises(ValueError):
            hierarchical_paired_bootstrap(values, {43: values[42]}, samples=10)

    def test_bootstrap_rejects_duplicate_item_ids(self) -> None:
        values = [prediction(0, 0), prediction(0, None)]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            paired_item_bootstrap(values, values, samples=10)

    def test_holm_adjustment_is_monotone(self) -> None:
        adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertLessEqual(adjusted["a"], adjusted["c"])
        self.assertLessEqual(adjusted["c"], adjusted["b"])

    def test_leave_one_client_out_prior_excludes_target(self) -> None:
        rows = [
            {"prediction": {"client_id": 0, "subject": "cardio", "predicted": 0, "gold": 0}},
            {"prediction": {"client_id": 1, "subject": "cardio", "predicted": 1, "gold": 0}},
            {"prediction": {"client_id": 1, "subject": "neuro", "predicted": 0, "gold": 0}},
        ]
        priors = leave_one_client_out_weakness(rows, num_clients=2)
        self.assertGreater(priors[0]["cardio"], priors[0]["neuro"])


if __name__ == "__main__":
    unittest.main()
