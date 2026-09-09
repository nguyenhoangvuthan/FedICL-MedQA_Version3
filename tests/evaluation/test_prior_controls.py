from __future__ import annotations

import copy
import unittest
from collections import Counter

import numpy as np

from fedicl_mqa.core.config import Config
from fedicl_mqa.core.schema import MCQExample
from fedicl_mqa.evaluation.arms import ARMS, select_exemplars
from fedicl_mqa.evaluation.priors import (
    controlled_weakness,
    shuffled_subject_prior,
    validate_prior_variation,
)
from fedicl_mqa.evaluation.retrieval import RetrievedExample


def rows():
    return [
        {
            "prediction": {
                "example_id": f"{c}-{s}-{i}",
                "client_id": c,
                "subject": s,
                "gold": 0,
                "predicted": int(i < (c + 1 if s == "Anatomy" else 8 - c)),
            }
        }
        for c in range(5)
        for s in ("Anatomy", "Pathology")
        for i in range(10)
    ]


SUBJECTS = {c: {"Anatomy", "Pathology"} for c in range(5)}


class PriorControlTests(unittest.TestCase):
    def test_smoothed_loco_rate_and_target_client_exclusion(self):
        source = rows()
        prior, audit = controlled_weakness(source, support_subjects=SUBJECTS, min_count=10)
        self.assertAlmostEqual(prior[0]["Anatomy"], (2 + 3 + 4 + 5 + 1) / (40 + 2))
        self.assertEqual(audit["evidence"]["0"]["Anatomy"]["n"], 40)
        altered = copy.deepcopy(source)
        for row in altered:
            if row["prediction"]["example_id"] == "0-Anatomy-9":
                row["prediction"]["predicted"] = 1 - row["prediction"]["predicted"]
        changed, _ = controlled_weakness(altered, support_subjects=SUBJECTS, min_count=10)
        self.assertEqual(prior[0], changed[0])
        self.assertNotEqual(prior[1], changed[1])

    def test_degenerate_prior_fails_instead_of_inventing_variation(self):
        source = rows()
        for r in source:
            r["prediction"]["predicted"] = 0
        with self.assertRaisesRegex(ValueError, "constant"):
            controlled_weakness(source, support_subjects=SUBJECTS, min_count=10)
        with self.assertRaisesRegex(ValueError, "identical prior"):
            validate_prior_variation(
                {c: {"Anatomy": 0.2, "Pathology": 0.8} for c in range(5)}, SUBJECTS
            )

    def test_sparse_and_duplicate_predictions_fail(self):
        with self.assertRaisesRegex(ValueError, "only 40"):
            controlled_weakness(rows(), support_subjects=SUBJECTS, min_count=41)
        source = rows()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            controlled_weakness(source + [source[0]], support_subjects=SUBJECTS, min_count=10)

    def test_placebo_preserves_values_changes_alignment_and_is_reproducible(self):
        prior, _ = controlled_weakness(rows(), support_subjects=SUBJECTS, min_count=10)
        shuffled = shuffled_subject_prior(prior, seed=42)
        self.assertEqual(shuffled, shuffled_subject_prior(prior, seed=42))
        for c in prior:
            self.assertNotEqual(prior[c], shuffled[c])
            self.assertEqual(Counter(prior[c].values()), Counter(shuffled[c].values()))
        with self.assertRaisesRegex(ValueError, "constant"):
            shuffled_subject_prior({0: {"a": 0.5, "b": 0.5}}, seed=42)

    def test_factorial_arms_isolate_diversity_and_prior_on_same_candidates(self):
        config = Config()
        pool = []
        for i in range(6):
            q = MCQExample(
                str(i),
                "Q",
                ("a", "b", "c", "d"),
                0,
                "train",
                subject="Pathology" if i == 5 else "Anatomy",
            )
            pool.append(
                RetrievedExample(
                    q, 0.9 - i * 0.02, np.array([0.0, 1.0]) if i == 5 else np.array([1.0, 0.0])
                )
            )

        def ids(arm, weights):
            return [
                v.example.example_id for v in select_exemplars(pool, ARMS[arm], config, weights)
            ]

        weights = {"Anatomy": 0.0, "Pathology": 1.0}
        self.assertEqual(ids("F1", weights), ["0", "1", "2", "3", "4"])
        self.assertIn("5", ids("FD", {}))
        self.assertEqual(ids("FD", weights), ids("FD", {}))
        self.assertEqual(ids("FP", weights)[0], "5")
        config.retrieval.beta = 0
        self.assertEqual(ids("F2", weights), ids("FP", weights))
        config.retrieval.gamma = 0
        self.assertEqual(ids("F2", weights), ids("F1", weights))
