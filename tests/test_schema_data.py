from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from fedicl_mqa.data import (
    adapt_medmcqa,
    adapt_medqa,
    build_partition,
    load_partition,
    materialize_partition,
)
from fedicl_mqa.schema import MCQExample, label_to_index, normalize_text


def example(index: int, split: str, subject: str = "medicine") -> MCQExample:
    return MCQExample(
        example_id=f"{split}-{index}",
        question=f"Clinical question number {index} for {subject}?",
        options=("alpha", "beta", "gamma", "delta"),
        label=index % 4,
        split=split,
        subject=subject,
    )


class SchemaAndDataTests(unittest.TestCase):
    def test_normalization_and_label_parsing(self) -> None:
        self.assertEqual(normalize_text("  Vitamin-C?! "), "vitamin c")
        self.assertEqual(label_to_index("D"), 3)
        self.assertEqual(label_to_index(3), 3)

    def test_adapt_medqa_nested_schema(self) -> None:
        item = adapt_medqa(
            {
                "id": "q1",
                "data": {
                    "Question": "Which answer?",
                    "Options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                    "Correct Option": "C",
                },
            },
            "dev",
        )
        self.assertEqual(item.label, 2)
        self.assertEqual(item.split, "validation")

    def test_adapt_medmcqa_zero_based_class_label(self) -> None:
        item = adapt_medmcqa(
            {
                "id": "q2",
                "question": "Which answer?",
                "opa": "a",
                "opb": "b",
                "opc": "c",
                "opd": "d",
                "cop": 1,
                "subject_name": "Pathology",
            },
            "train",
        )
        self.assertEqual(item.answer_label, "B")

    def test_partition_is_deterministic_and_has_support_capacity(self) -> None:
        splits = {
            "train": [example(i, "train", f"s{i % 4}") for i in range(120)],
            "validation": [example(i, "validation", f"s{i % 4}") for i in range(20)],
            "test": [example(i, "test", f"s{i % 4}") for i in range(20)],
        }
        first, first_weights = build_partition(
            splits, num_clients=5, alpha=0.5, fit_ratio=0.8, seed=42
        )
        second, second_weights = build_partition(
            splits, num_clients=5, alpha=0.5, fit_ratio=0.8, seed=42
        )
        self.assertEqual(first, second)
        self.assertEqual(first_weights, second_weights)
        for client in range(5):
            support = [
                value for value in first if value.client_id == client and value.role == "support"
            ]
            self.assertGreaterEqual(len(support), 5)
            self.assertTrue(
                any(value.client_id == client and value.role == "validation" for value in first)
            )
            self.assertTrue(
                any(value.client_id == client and value.role == "test" for value in first)
            )

    def test_partition_rejects_duplicate_ids_across_splits(self) -> None:
        splits = {
            "train": [example(i, "train") for i in range(30)],
            "validation": [
                replace(example(0, "validation"), example_id="train-0"),
                *[example(i, "validation") for i in range(1, 6)],
            ],
            "test": [example(i, "test") for i in range(6)],
        }
        with self.assertRaisesRegex(ValueError, "globally unique"):
            build_partition(splits, num_clients=5, alpha=0.5, fit_ratio=0.8, seed=42)

    def test_materialized_partition_is_hash_verified(self) -> None:
        splits = {
            "train": [example(i, "train", f"s{i % 3}") for i in range(60)],
            "validation": [example(i, "validation", f"s{i % 3}") for i in range(10)],
            "test": [example(i, "test", f"s{i % 3}") for i in range(10)],
        }
        assignments, weights = build_partition(
            splits, num_clients=5, alpha=0.5, fit_ratio=0.8, seed=42
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            materialize_partition(
                root,
                splits,
                assignments,
                dataset_id="dataset",
                dataset_revision="revision",
                data_seed=42,
                weights=weights,
                config_hash="config",
            )
            self.assertEqual(len(load_partition(root, expected_config_hash="config")), 5)
            target = next(root.glob("client_*/*.jsonl"))
            target.write_text(target.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_partition(root, expected_config_hash="config")


if __name__ == "__main__":
    unittest.main()
