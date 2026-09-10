from __future__ import annotations

import random
import unittest
from dataclasses import replace

from fedicl_mqa.core.schema import MCQExample
from fedicl_mqa.data.decontamination import assert_disjoint_splits, decontaminate_splits
from fedicl_mqa.data.leakage import LeakageIndex, audit_support_leakage


def item(name, text, split="train", **kwargs):
    return MCQExample(name, text, ("a", "b", "c", "d"), 0, split, **kwargs)


class LeakageIndexTests(unittest.TestCase):
    def test_index_matches_existing_audit_for_randomized_and_boundary_cases(self):
        rng = random.Random(17)
        words = [f"word{i}" for i in range(30)]
        candidates = [
            item(f"c{i}", " ".join(rng.sample(words, rng.randint(1, 25)))) for i in range(80)
        ]
        candidates += [item("empty", "!!!"), item("prov", "unique", provenance_group="group")]
        queries = [replace(q, example_id=f"q{i}", split="test") for i, q in enumerate(candidates)]
        queries += [
            item(f"new{i}", q.question + " extra", "test") for i, q in enumerate(candidates)
        ]
        queries += [item("c0", "different", "test"), item("other", "???", "test")]
        for threshold in (0.1, 0.5, 0.75, 0.85, 1.0):
            for provenance in (True, False):
                with self.subTest(threshold=threshold, provenance=provenance):
                    index = LeakageIndex(
                        candidates, lexical_threshold=threshold, check_provenance=provenance
                    )
                    expected = set(
                        audit_support_leakage(
                            candidates,
                            queries,
                            lexical_threshold=threshold,
                            check_provenance=provenance,
                            max_issues=1000000,
                        )
                    )
                    actual = {issue for q in queries for issue in index.matches(q)}
                    self.assertEqual(actual, expected)


class DecontaminationTests(unittest.TestCase):
    def sources(self):
        return {
            "test": [item("test", "one two three four five six seven", "test")],
            "validation": [
                item("val-overlap", "ONE two three four five six seven!", "validation"),
                item("val-keep", "alpha beta gamma delta epsilon zeta eta", "validation"),
            ],
            "train": [
                item("train-exact", "ONE two three four five six seven!"),
                item("train-near", "alpha beta gamma delta epsilon zeta eta theta"),
                item("train-keep", "unrelated"),
            ],
        }

    def test_preserves_test_and_cleans_all_training_before_client_assignment(self):
        sources = self.sources()
        clean, audit = decontaminate_splits(sources, lexical_threshold=0.85)
        self.assertEqual(clean["test"], sources["test"])
        self.assertEqual([q.example_id for q in clean["train"]], ["train-keep"])
        self.assertEqual([q.example_id for q in clean["validation"]], ["val-keep"])
        self.assertEqual(audit["exclusions"]["train"]["count"], 2)
        self.assertEqual(audit["exclusions"]["validation"]["count"], 1)
        assert_disjoint_splits(clean, lexical_threshold=0.85)
        # Independent pre-existing audit also sees no remaining cross-split pairs.
        self.assertEqual(
            audit_support_leakage(clean["train"], clean["validation"] + clean["test"]), []
        )
        self.assertEqual(
            audit_support_leakage(
                [replace(q, split="train") for q in clean["validation"]], clean["test"]
            ),
            [],
        )

    def test_order_answers_and_subjects_do_not_affect_exclusions(self):
        sources = self.sources()
        clean, audit = decontaminate_splits(sources, lexical_threshold=0.85)
        altered = {
            s: [replace(q, label=3, subject="different") for q in reversed(qs)]
            for s, qs in sources.items()
        }
        other, other_audit = decontaminate_splits(altered, lexical_threshold=0.85)
        self.assertEqual(audit, other_audit)
        self.assertEqual(
            {s: {q.example_id for q in qs} for s, qs in clean.items()},
            {s: {q.example_id for q in qs} for s, qs in other.items()},
        )
        repeated, second_audit = decontaminate_splits(clean, lexical_threshold=0.85)
        self.assertEqual(repeated, clean)
        self.assertEqual(second_audit["exclusions"]["train"]["count"], 0)

    def test_provenance_overlap_is_removed_even_when_text_differs(self):
        sources = self.sources()
        sources["train"].append(item("train-prov", "provenance only", provenance_group="same"))
        sources["test"][0] = replace(sources["test"][0], provenance_group="same")
        _, audit = decontaminate_splits(sources, lexical_threshold=0.85)
        records = {r["example_id"]: r for r in audit["exclusions"]["train"]["records"]}
        self.assertEqual(records["train-prov"]["kind"], "provenance_group")

    def test_empty_retained_split_fails_instead_of_weakening_threshold(self):
        sources = self.sources()
        sources["train"] = sources["train"][:2]
        with self.assertRaisesRegex(ValueError, "no train examples"):
            decontaminate_splits(sources, lexical_threshold=0.85)

    def test_global_guard_detects_fit_or_cross_client_overlap(self):
        with self.assertRaisesRegex(ValueError, "global split leakage"):
            assert_disjoint_splits(self.sources(), lexical_threshold=0.85)
