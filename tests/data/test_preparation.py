from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from fedicl_mqa.core.io import file_sha256, read_json, write_json
from fedicl_mqa.core.schema import MCQExample
from fedicl_mqa.data.preparation import (
    _verify_partition_files,
    adapt_medmcqa,
    adapt_medqa,
    build_partition,
    load_partition,
    materialize_partition,
)


def example(index: int, split: str, subject: str = "medicine") -> MCQExample:
    return MCQExample(
        example_id=f"{split}-{index}",
        question=f"Clinical question number {index} for {subject}?",
        options=("alpha", "beta", "gamma", "delta"),
        label=index % 4,
        split=split,
        subject=subject,
    )


class PartitionTests(unittest.TestCase):
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


class PortableManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fit = self.root / "client_0" / "fit.jsonl"
        self.fit.parent.mkdir()
        self.fit.write_text("{}\n", encoding="utf-8")
        write_json(self.root / "partition_manifest.json", {"config_hash": "unchanged"})
        self.hashes_path = self.root / "file_hashes.json"

    def write_hashes(self, key: str) -> None:
        write_json(
            self.hashes_path,
            {
                "partition_manifest.json": file_sha256(self.root / "partition_manifest.json"),
                key: file_sha256(self.fit),
            },
        )

    def test_both_separator_styles_verify_without_rewriting_any_artifact(self) -> None:
        for key in ("client_0/fit.jsonl", "client_0\\fit.jsonl"):
            with self.subTest(key=key):
                self.write_hashes(key)
                before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
                _verify_partition_files(self.root)
                self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_windows_manifest_still_detects_content_tampering(self) -> None:
        self.write_hashes("client_0\\fit.jsonl")
        self.fit.write_text('{"tampered": true}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            _verify_partition_files(self.root)

    def test_aliases_cannot_hide_duplicate_manifest_entries(self) -> None:
        self.write_hashes("client_0/fit.jsonl")
        hashes = read_json(self.hashes_path)
        hashes["client_0\\fit.jsonl"] = hashes["client_0/fit.jsonl"]
        write_json(self.hashes_path, hashes)
        with self.assertRaisesRegex(ValueError, "duplicate normalized"):
            _verify_partition_files(self.root)

    def test_path_escape_is_rejected_in_both_styles(self) -> None:
        for key in (
            "../fit.jsonl",
            "..\\fit.jsonl",
            "/tmp/fit.jsonl",
            "C:\\data\\fit.jsonl",
            "C:fit.jsonl",
            "\\\\server\\share\\fit.jsonl",
        ):
            with self.subTest(key=key):
                self.write_hashes(key)
                with self.assertRaisesRegex(ValueError, "escapes root"):
                    _verify_partition_files(self.root)

    def test_missing_or_unlisted_files_still_fail(self) -> None:
        self.write_hashes("client_0\\fit.jsonl")
        extra = self.fit.with_name("test.jsonl")
        extra.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest does not match"):
            _verify_partition_files(self.root)
        extra.unlink()
        self.fit.unlink()
        with self.assertRaisesRegex(ValueError, "manifest does not match"):
            _verify_partition_files(self.root)


if __name__ == "__main__":
    unittest.main()
