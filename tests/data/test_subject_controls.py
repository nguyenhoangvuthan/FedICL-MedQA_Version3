from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fedicl_mqa.core.schema import MCQExample
from fedicl_mqa.data.preparation import load_native_dataset
from fedicl_mqa.data.subjects import audit_subjects, development_split, filter_native_subjects


def item(i, subject="Anatomy", split="train"):
    return MCQExample(
        f"{split}-{subject}-{i}",
        f"question {i}",
        ("a", "b", "c", "d"),
        i % 4,
        split,
        subject=subject,
    )


def partitions():
    return {
        c: {
            role: [
                replace(
                    item(i, s, "train" if role in ("fit", "support") else role),
                    example_id=f"{c}-{role}-{s}-{i}",
                )
                for s, n in [("Anatomy", 20 + c), ("Pathology", 25 - c)]
                for i in range(n)
            ]
            for role in ("fit", "support", "validation", "test")
        }
        for c in range(5)
    }


class SubjectControlTests(unittest.TestCase):
    def test_split_is_disjoint_order_invariant_and_label_blind(self):
        train = [item(i, s) for s in ("Anatomy", "Pathology") for i in range(30)]
        official = [item(i, s, "validation") for s in ("Anatomy", "Pathology") for i in range(5)]
        first = development_split(train, official, fraction=0.2, seed=7, limits={})
        altered = [replace(q, label=(q.label + 1) % 4) for q in reversed(train)]
        second = development_split(
            altered, list(reversed(official)), fraction=0.2, seed=7, limits={}
        )
        ids = {r: {q.example_id for q in qs} for r, qs in first.items()}
        self.assertEqual(ids, {r: {q.example_id for q in qs} for r, qs in second.items()})
        self.assertEqual(len(ids["validation"]), 12)
        self.assertEqual(ids["test"], {q.example_id for q in official})
        self.assertFalse(
            ids["validation"] & ids["test"]
            | ids["train"] & ids["test"]
            | ids["train"] & ids["validation"]
        )
        self.assertTrue(all(q.split == r for r, qs in first.items() for q in qs))

    def test_loader_never_requests_masked_official_test(self):
        def row(i, split):
            return dict(
                id=f"{split}-{i}",
                question="Q",
                opa="a",
                opb="b",
                opc="c",
                opd="d",
                cop=i % 4,
                subject_name="Anatomy" if i % 2 else "Pathology",
            )

        def load(name, *, revision, split, trust_remote_code):
            self.assertNotEqual(split, "test")
            return [row(i, split) for i in range(20)]

        loader = Mock(side_effect=load)
        with patch.dict(sys.modules, {"datasets": SimpleNamespace(load_dataset=loader)}):
            result = load_native_dataset(
                "medmcqa", "native", revision="sha", development_fraction=0.2, data_seed=1
            )
        self.assertEqual(
            [c.kwargs["split"] for c in loader.call_args_list], ["train", "validation"]
        )
        self.assertEqual(len(result["test"]), 20)

    def test_controlled_loader_filters_before_holdout_and_limits_and_records_ids(self):
        def load(name, **kwargs):
            split = kwargs["split"]
            return [
                dict(
                    id=f"{split}-{i}",
                    question="Q",
                    opa="a",
                    opb="b",
                    opc="c",
                    opd="d",
                    cop=i % 4,
                    subject_name="Unknown" if i == 0 else "Anatomy",
                )
                for i in range(21)
            ]

        audit = {}
        with patch.dict(sys.modules, {"datasets": SimpleNamespace(load_dataset=load)}):
            result = load_native_dataset(
                "medmcqa",
                "native",
                revision="sha",
                development_fraction=0.2,
                limits={"train": 5, "test": 3},
                subject_filter_audit=audit,
            )
        self.assertEqual(
            {s: len(qs) for s, qs in result.items()}, {"train": 5, "validation": 4, "test": 3}
        )
        self.assertTrue(all(q.subject == "Anatomy" for qs in result.values() for q in qs))
        for split in ("train", "validation"):
            self.assertEqual(
                audit["source_splits"][split],
                {
                    "input_count": 21,
                    "retained_count": 20,
                    "excluded_count": 1,
                    "excluded_subject_counts": {"Unknown": 1},
                    "excluded_ids": [f"{split}-0"],
                },
            )

    def test_legacy_loader_keeps_unknown_subjects(self):
        row = dict(
            id="q", question="Q", opa="a", opb="b", opc="c", opd="d", cop=0, subject_name="Unknown"
        )
        loader = Mock(return_value={s: [row] for s in ("train", "validation", "test")})
        with patch.dict(sys.modules, {"datasets": SimpleNamespace(load_dataset=loader)}):
            result = load_native_dataset("medmcqa", "native", revision="sha")
        self.assertTrue(all(q.subject == "Unknown" for qs in result.values() for q in qs))

    def test_filter_is_label_blind_order_invariant_and_preserves_native_subjects(self):
        items = [
            item(i, s)
            for i, s in enumerate(
                ("Unknown", " unknown ", "", "General", "N/A", "None", "Medicine", "Anatomy")
            )
        ]
        kept, audit = filter_native_subjects({"train": items})
        altered = [replace(q, label=(q.label + 1) % 4) for q in reversed(items)]
        other, other_audit = filter_native_subjects({"train": altered})
        self.assertEqual(audit, other_audit)
        self.assertEqual(
            {q.example_id for q in kept["train"]}, {q.example_id for q in other["train"]}
        )
        self.assertEqual(kept["train"], items[-2:])
        self.assertEqual(audit["source_splits"]["train"]["excluded_count"], 6)

    def test_filter_rejects_source_without_usable_subjects(self):
        for items in ([], [item(0, "Unknown")]):
            with self.subTest(items=items), self.assertRaisesRegex(ValueError, "source train"):
                filter_native_subjects({"train": items})

    def test_subject_audit_records_other_client_coverage_and_skew(self):
        audit = audit_subjects(partitions(), min_validation_per_subject=10)
        self.assertEqual(audit["other_client_validation_counts"][0]["Anatomy"], 90)
        self.assertGreater(audit["max_pairwise_fit_subject_total_variation"], 0)

    def test_single_subject_support_is_rejected(self):
        parts = partitions()
        parts[0]["support"] = [q for q in parts[0]["support"] if q.subject == "Anatomy"]
        with self.assertRaisesRegex(ValueError, "at least two"):
            audit_subjects(parts, min_validation_per_subject=10)

    def test_unknown_metadata_and_sparse_loco_evidence_are_rejected(self):
        parts = partitions()
        parts[0]["fit"][0] = replace(parts[0]["fit"][0], subject="unknown")
        with self.assertRaisesRegex(ValueError, "missing native"):
            audit_subjects(parts, min_validation_per_subject=10)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            audit_subjects(partitions(), min_validation_per_subject=1000)

    def test_duplicate_id_across_source_splits_is_rejected(self):
        train = [item(i) for i in range(20)]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            development_split(
                train, [replace(train[0], split="validation")], fraction=0.1, seed=1, limits={}
            )
