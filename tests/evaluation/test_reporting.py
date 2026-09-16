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


class ControlledReportingTests(unittest.TestCase):
    def test_train_icl_contrasts_are_paired_and_holm_adjusted_with_other_controls(self):
        from fedicl_mqa.evaluation.arms import ARMS
        from fedicl_mqa.evaluation.reporting import TRAIN_ICL_CONTRASTS

        predictions = {
            arm: {
                seed: values(
                    correct=arm in {"LT0", "LT1", "FT0", "FT1"}, likelihood_correct=True, seed=seed
                )
                for seed in ([None] if spec.checkpoint_family == "base" else [42, 43])
            }
            for arm, spec in ARMS.items()
        }
        report = build_contrast_report(
            predictions,
            samples=20,
            confidence=0.95,
            bootstrap_seed=1,
            controlled=True,
            train_icl=True,
        )
        self.assertEqual(len(report["primary"]), 22)
        for name in TRAIN_ICL_CONTRASTS:
            self.assertIn("holm_adjusted_p_value", report["primary"][name])
        self.assertEqual(report["primary"]["local_train_icl_eval_k0"]["effect"], 1)
        self.assertEqual(report["primary"]["fl_eval_icl_after_train_icl"]["effect"], 0)

    def test_centralized_factorial_mirrors_the_federated_one(self):
        from fedicl_mqa.evaluation.arms import ARMS
        from fedicl_mqa.evaluation.reporting import TRAIN_ICL_CONTRASTS

        self.assertEqual(TRAIN_ICL_CONTRASTS["central_icl"], ("C0", "C1"))
        self.assertEqual(TRAIN_ICL_CONTRASTS["central_train_icl_eval_k0"], ("C0", "CT0"))
        self.assertEqual(TRAIN_ICL_CONTRASTS["central_train_icl_eval_k5"], ("C1", "CT1"))
        self.assertEqual(TRAIN_ICL_CONTRASTS["central_eval_icl_after_train_icl"], ("CT0", "CT1"))
        self.assertEqual(TRAIN_ICL_CONTRASTS["fl_system"], ("F0", "FT1"))
        self.assertEqual(TRAIN_ICL_CONTRASTS["central_system"], ("C0", "CT1"))
        predictions = {
            arm: {
                seed: values(correct=arm in {"CT1", "FT1"}, likelihood_correct=True, seed=seed)
                for seed in ([None] if spec.checkpoint_family == "base" else [42])
            }
            for arm, spec in ARMS.items()
        }
        report = build_contrast_report(
            predictions,
            samples=20,
            confidence=0.95,
            bootstrap_seed=1,
            controlled=True,
            train_icl=True,
        )
        self.assertEqual(report["primary"]["central_system"]["effect"], 1)
        self.assertEqual(report["primary"]["fl_system"]["effect"], 1)
        # Federated versus Centralized is descriptive in every cell, as C0-F0 already was.
        self.assertEqual(
            set(report["descriptive"]),
            {"central", "central_eval_icl", "central_train_icl", "central_train_eval_icl"},
        )
        self.assertEqual(report["descriptive"]["central_train_eval_icl"]["effect"], 0)

    def test_full_controls_produce_separate_prior_diversity_and_matched_contrasts(self):
        from fedicl_mqa.evaluation.arms import ARMS
        from fedicl_mqa.evaluation.reporting import CONTROLLED_CONTRASTS

        predictions = {
            arm: {
                seed: values(correct=arm in {"F0", "F2", "FD"}, likelihood_correct=True, seed=seed)
                for seed in ([None] if spec.checkpoint_family == "base" else [42, 43])
            }
            for arm, spec in ARMS.items()
        }
        report = build_contrast_report(
            predictions, samples=20, confidence=0.95, bootstrap_seed=1, controlled=True
        )
        self.assertEqual(set(report["primary"]), set(CONTROLLED_CONTRASTS))
        self.assertEqual(report["primary"]["fl_matched"]["left"], "LM0")
        self.assertEqual(report["primary"]["prior_with_diversity"]["left"], "FD")
        self.assertEqual(report["primary"]["prior_with_diversity"]["effect"], 0)
        self.assertEqual(report["primary"]["prior_vs_shuffled"]["left"], "FS")
        del predictions["FP"][43]
        with self.assertRaisesRegex(ValueError, "identical training seed"):
            build_contrast_report(
                predictions, samples=20, confidence=0.95, bootstrap_seed=1, controlled=True
            )

    def test_arm_subset_reports_only_contrasts_with_both_sides_present(self):
        from fedicl_mqa.evaluation.reporting import TRAIN_ICL_CONTRASTS

        predictions = {
            arm: {42: values(correct=arm in {"FT0", "FT1"}, likelihood_correct=True, seed=42)}
            for arm in ("F0", "F1", "FT0", "FT1")
        }
        report = build_contrast_report(
            predictions,
            samples=20,
            confidence=0.95,
            bootstrap_seed=1,
            controlled=True,
            train_icl=True,
            arms=("F0", "F1", "FT0", "FT1"),
        )
        self.assertEqual(
            set(report["primary"]),
            {
                "fl_icl",
                "fl_train_icl_eval_k0",
                "fl_train_icl_eval_k5",
                "fl_eval_icl_after_train_icl",
                "fl_system",
            },
        )
        self.assertTrue(set(report["primary"]) < set(TRAIN_ICL_CONTRASTS) | {"fl_icl"})
        self.assertEqual(report["primary"]["fl_train_icl_eval_k0"]["effect"], 1)
        self.assertEqual(report["arms"], ["F0", "F1", "FT0", "FT1"])
        self.assertTrue(report["partial"])
        self.assertNotIn("central", report["descriptive"])

    def test_arm_subset_without_any_full_pair_is_rejected(self):
        predictions = {
            arm: {42: values(correct=True, likelihood_correct=True, seed=42)}
            for arm in ("F0", "LT1")
        }
        with self.assertRaisesRegex(ValueError, "no declared contrast"):
            build_contrast_report(
                predictions,
                samples=20,
                confidence=0.95,
                bootstrap_seed=1,
                controlled=True,
                train_icl=True,
                arms=("F0", "LT1"),
            )

    def test_full_arm_list_is_not_marked_partial(self):
        from fedicl_mqa.evaluation.arms import ARMS

        predictions = {
            arm: {
                seed: values(correct=True, likelihood_correct=True, seed=seed)
                for seed in ([None] if spec.checkpoint_family == "base" else [42])
            }
            for arm, spec in ARMS.items()
        }
        report = build_contrast_report(
            predictions,
            samples=20,
            confidence=0.95,
            bootstrap_seed=1,
            controlled=True,
            train_icl=True,
        )
        self.assertFalse(report["partial"])
        self.assertEqual(len(report["primary"]), 22)

    def test_pairing_rejects_changed_gold_labels(self):
        from dataclasses import replace

        from fedicl_mqa.evaluation.metrics import paired_item_bootstrap

        left = values(correct=True, likelihood_correct=True, seed=42)
        right = [replace(p, gold=(p.gold + 1) % 4) for p in left]
        with self.assertRaisesRegex(ValueError, "inconsistent gold"):
            paired_item_bootstrap(left, right, samples=20)


if __name__ == "__main__":
    unittest.main()
