from __future__ import annotations

import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - local minimal environment
    np = None

from fedicl_mqa.evaluation.audit import audit_retrieval_cohort
from fedicl_mqa.core.config import Config
from fedicl_mqa.core.io import read_json
from fedicl_mqa.evaluation.retrieval import ClosureConstrainedRetriever, client_aware_rerank
from fedicl_mqa.core.schema import MCQExample


class _HashEncoder:
    revision = "test"

    def encode(self, texts: list[str]):
        values = []
        for text in texts:
            vector = [0.0] * 16
            for token in text.casefold().split():
                vector[int(sha256(token.encode()).hexdigest(), 16) % len(vector)] += 1.0
            values.append(vector)
        return np.asarray(values, dtype=np.float32)


def item(index: int, question: str, *, subject: str = "general") -> MCQExample:
    return MCQExample(
        example_id=f"train-{index}",
        question=question,
        options=("a", "b", "c", "d"),
        label=index % 4,
        split="train",
        subject=subject,
    )


@unittest.skipIf(np is None, "numpy is not installed")
class RetrievalTests(unittest.TestCase):
    def test_exact_query_is_never_returned(self) -> None:
        support = [item(0, "duplicate question")] + [
            item(index, f"medical topic {index}", subject="priority" if index == 5 else "other")
            for index in range(1, 7)
        ]
        query = MCQExample(
            example_id="test",
            question="duplicate question",
            options=("a", "b", "c", "d"),
            label=0,
            split="test",
        )
        retriever = ClosureConstrainedRetriever(
            support,
            _HashEncoder(),
            duplicate_similarity_threshold=0.99,
            lexical_jaccard_threshold=0.99,
            initial_pool=7,
            expanded_pool=7,
        )
        results = retriever.retrieve(query, k=5)
        self.assertNotIn("train-0", {value.example.example_id for value in results})
        self.assertEqual(len(results), 5)

    def test_client_prior_can_promote_a_subject(self) -> None:
        support = [
            item(index, f"topic {index}", subject="priority" if index == 5 else "other")
            for index in range(6)
        ]
        query = MCQExample("test", "topic query", ("a", "b", "c", "d"), 0, "test")
        retriever = ClosureConstrainedRetriever(
            support,
            _HashEncoder(),
            duplicate_similarity_threshold=1.01,
            lexical_jaccard_threshold=1.01,
            initial_pool=6,
            expanded_pool=6,
        )
        pool = retriever.candidate_pool(query, minimum=5)
        selected = client_aware_rerank(
            pool,
            k=1,
            subject_weights={"priority": 100.0},
            alpha=0.0,
            beta=0.0,
            gamma=1.0,
        )
        self.assertEqual(selected[0].example.subject, "priority")

    def test_batched_retrieval_matches_single_query_retrieval(self) -> None:
        support = [item(index, f"medical topic {index}") for index in range(8)]
        queries = [
            MCQExample(f"test-{index}", f"query {index}", ("a", "b", "c", "d"), 0, "test")
            for index in range(2)
        ]
        retriever = ClosureConstrainedRetriever(
            support,
            _HashEncoder(),
            duplicate_similarity_threshold=1.01,
            lexical_jaccard_threshold=1.01,
            initial_pool=8,
            expanded_pool=8,
        )
        batched = retriever.retrieve_many(queries, k=5)
        singles = [retriever.retrieve(query, k=5) for query in queries]
        self.assertEqual(
            [[value.example.example_id for value in result] for result in batched],
            [[value.example.example_id for value in result] for result in singles],
        )

    def test_preflight_audit_persists_five_train_only_exemplars(self) -> None:
        config = Config()
        config.retrieval.duplicate_similarity_threshold = 1.01
        config.retrieval.lexical_jaccard_threshold = 1.01
        partitions = {}
        for client in range(5):
            support = [item(client * 10 + index, f"support {client} {index}") for index in range(6)]
            partitions[client] = {
                "fit": [],
                "support": support,
                "validation": [
                    MCQExample(
                        f"validation-{client}",
                        f"validation query {client}",
                        ("a", "b", "c", "d"),
                        0,
                        "validation",
                    )
                ],
                "test": [
                    MCQExample(
                        f"test-{client}",
                        f"test query {client}",
                        ("a", "b", "c", "d"),
                        0,
                        "test",
                    )
                ],
            }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "audit.json"
            result = audit_retrieval_cohort(config, partitions, output, encoder=_HashEncoder())
            self.assertEqual(result["query_count"], 10)
            self.assertTrue(
                all(len(row["top5_exemplar_ids"]) == 5 for row in read_json(output)["queries"])
            )


if __name__ == "__main__":
    unittest.main()
