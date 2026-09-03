from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .leakage import token_jaccard
from .schema import MCQExample


class TextEncoder(Protocol):
    revision: str

    def encode(self, texts: Sequence[str]) -> Any:
        """Return a 2-D float array whose rows have unit L2 norm."""


class SentenceTransformerEncoder:
    def __init__(
        self,
        model_id: str,
        *,
        revision: str,
        device: str = "cuda",
        batch_size: int = 128,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - full runtime only
            raise RuntimeError("sentence-transformers is required for dense retrieval") from exc
        self.revision = revision
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_id, revision=revision, device=device)

    def encode(self, texts: Sequence[str]) -> Any:
        return self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > self.batch_size,
        )


@dataclass(frozen=True, slots=True)
class RetrievedExample:
    example: MCQExample
    relevance: float
    vector: Any


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostics:
    query_id: str
    searched_candidates: int
    eligible_candidates: int
    expanded: bool
    exclusions: dict[str, int]


class CapacityFailure(RuntimeError):
    pass


class ClosureConstrainedRetriever:
    """One local dense index with exact, lexical and semantic closure checks."""

    def __init__(
        self,
        support: Sequence[MCQExample],
        encoder: TextEncoder,
        *,
        duplicate_similarity_threshold: float = 0.92,
        lexical_jaccard_threshold: float = 0.85,
        initial_pool: int = 50,
        expanded_pool: int = 100,
    ) -> None:
        if not support:
            raise ValueError("support repository cannot be empty")
        if any(item.split != "train" for item in support):
            raise ValueError("support repository may contain only original train items")
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - full runtime only
            raise RuntimeError("numpy is required for dense retrieval") from exc
        self._np = np
        self.support = tuple(support)
        self.encoder = encoder
        self.duplicate_similarity_threshold = duplicate_similarity_threshold
        self.lexical_jaccard_threshold = lexical_jaccard_threshold
        self.initial_pool = initial_pool
        self.expanded_pool = expanded_pool
        encoded = encoder.encode(
            [item.retrieval_text for item in self.support]
            + [item.question for item in self.support]
        )
        matrix = self._as_normalized(encoded, expected_rows=2 * len(self.support))
        self._relevance = matrix[: len(self.support)]
        self._duplicate = matrix[len(self.support) :]

    def _as_normalized(self, values: Any, *, expected_rows: int) -> Any:
        array = self._np.asarray(values, dtype=self._np.float32)
        if array.ndim != 2 or array.shape[0] != expected_rows:
            raise ValueError("encoder returned an invalid embedding matrix")
        norms = self._np.linalg.norm(array, axis=1, keepdims=True)
        return array / self._np.maximum(norms, 1e-12)

    def _top_indices(self, scores: Any, count: int) -> list[int]:
        count = min(count, len(scores))
        # Batched GEMM and single-query GEMV can differ in the last few bits.
        # Quantization plus index-order ties keeps Top-k invariant to batch shape.
        stable_scores = self._np.round(scores, decimals=6)
        if count == len(scores):
            indices = self._np.arange(len(stable_scores))
            return [int(index) for index in indices[self._np.lexsort((indices, -stable_scores))]]
        threshold = self._np.partition(stable_scores, len(stable_scores) - count)[
            len(stable_scores) - count
        ]
        above = self._np.flatnonzero(stable_scores > threshold)
        tied = self._np.flatnonzero(stable_scores == threshold)
        unordered = self._np.concatenate((above, tied[: count - len(above)]))
        ordered = unordered[self._np.lexsort((unordered, -stable_scores[unordered]))]
        return [int(index) for index in ordered]

    def candidate_pool(self, query: MCQExample, *, minimum: int = 5) -> list[RetrievedExample]:
        return self.candidate_pools([query], minimum=minimum)[0]

    def candidate_pools(
        self, queries: Sequence[MCQExample], *, minimum: int = 5
    ) -> list[list[RetrievedExample]]:
        """Retrieve many queries after one batched encoder invocation."""
        pools, _ = self.candidate_pools_with_diagnostics(queries, minimum=minimum)
        return pools

    def candidate_pools_with_diagnostics(
        self, queries: Sequence[MCQExample], *, minimum: int = 5
    ) -> tuple[list[list[RetrievedExample]], list[RetrievalDiagnostics]]:
        if not queries:
            return [], []
        encoded = self.encoder.encode(
            [query.retrieval_text for query in queries] + [query.question for query in queries]
        )
        matrix = self._as_normalized(encoded, expected_rows=2 * len(queries))
        relevance_vectors = matrix[: len(queries)]
        duplicate_vectors = matrix[len(queries) :]
        results = []
        # GEMM is substantially faster than thousands of individual matrix-vector
        # products. Chunking bounds the two score matrices for skewed large clients.
        score_batch_size = 256
        for offset in range(0, len(queries), score_batch_size):
            query_chunk = queries[offset : offset + score_batch_size]
            relevance_scores = (
                relevance_vectors[offset : offset + score_batch_size] @ self._relevance.T
            )
            duplicate_scores = (
                duplicate_vectors[offset : offset + score_batch_size] @ self._duplicate.T
            )
            results.extend(
                self._candidate_pool(
                    query,
                    relevance_scores[row],
                    duplicate_scores[row],
                    minimum=minimum,
                )
                for row, query in enumerate(query_chunk)
            )
        return [result[0] for result in results], [result[1] for result in results]

    def _candidate_pool(
        self,
        query: MCQExample,
        relevance_scores: Any,
        duplicate_scores: Any,
        *,
        minimum: int,
    ) -> tuple[list[RetrievedExample], RetrievalDiagnostics]:
        last_candidates: list[RetrievedExample] = []
        search_sizes = []
        for size in (self.initial_pool, self.expanded_pool, len(self.support)):
            bounded = min(size, len(self.support))
            if bounded not in search_sizes:
                search_sizes.append(bounded)
        for size in search_sizes:
            candidates: list[RetrievedExample] = []
            exclusions: Counter[str] = Counter()
            for index in self._top_indices(relevance_scores, size):
                item = self.support[index]
                reason = self._ineligibility_reason(query, item, float(duplicate_scores[index]))
                if reason is not None:
                    exclusions[reason] += 1
                    continue
                candidates.append(
                    RetrievedExample(
                        example=item,
                        relevance=float(relevance_scores[index]),
                        # A NumPy row view avoids materializing hundreds of Python
                        # float objects for every query-candidate pair.
                        vector=self._relevance[index],
                    )
                )
            last_candidates = candidates
            if len(candidates) >= minimum:
                return candidates, RetrievalDiagnostics(
                    query_id=query.example_id,
                    searched_candidates=size,
                    eligible_candidates=len(candidates),
                    expanded=size > min(self.initial_pool, len(self.support)),
                    exclusions=dict(sorted(exclusions.items())),
                )
        raise CapacityFailure(
            f"query {query.example_id} has {len(last_candidates)} eligible exemplar(s); "
            f"requires {minimum}"
        )

    def _ineligibility_reason(
        self, query: MCQExample, item: MCQExample, duplicate_score: float
    ) -> str | None:
        if item.example_id == query.example_id:
            return "example_id"
        if item.normalized_question == query.normalized_question:
            return "normalized_question"
        if item.question_options_hash == query.question_options_hash:
            return "question_options_hash"
        if (
            item.provenance_group
            and query.provenance_group
            and item.provenance_group == query.provenance_group
        ):
            return "provenance_group"
        lexical = token_jaccard(item.normalized_question, query.normalized_question)
        if lexical >= self.lexical_jaccard_threshold:
            return "lexical_near_duplicate"
        if duplicate_score >= self.duplicate_similarity_threshold:
            return "semantic_near_duplicate"
        return None

    def retrieve(self, query: MCQExample, *, k: int = 5) -> list[RetrievedExample]:
        return self.candidate_pool(query, minimum=k)[:k]

    def retrieve_many(
        self, queries: Sequence[MCQExample], *, k: int = 5
    ) -> list[list[RetrievedExample]]:
        return [pool[:k] for pool in self.candidate_pools(queries, minimum=k)]


def client_aware_rerank(
    candidates: Sequence[RetrievedExample],
    *,
    k: int,
    subject_weights: Mapping[str, float],
    alpha: float,
    beta: float,
    gamma: float,
) -> list[RetrievedExample]:
    """Greedy relevance-diversity-client-prior reranking over one candidate pool."""
    if len(candidates) < k:
        raise CapacityFailure(f"candidate pool has {len(candidates)} items; requires {k}")

    selected: list[RetrievedExample] = []
    remaining = list(candidates)
    while remaining and len(selected) < k:
        best_index = max(
            range(len(remaining)),
            key=lambda index: _rerank_score(
                remaining[index], selected, subject_weights, alpha, beta, gamma
            ),
        )
        selected.append(remaining.pop(best_index))
    return selected


def _rerank_score(
    candidate: RetrievedExample,
    selected: Sequence[RetrievedExample],
    subject_weights: Mapping[str, float],
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    redundancy = 0.0
    for prior in selected:
        dot = float(candidate.vector @ prior.vector)
        redundancy = max(redundancy, dot)
    subject_prior = float(subject_weights.get(candidate.example.subject, 0.0))
    return alpha * candidate.relevance - beta * redundancy + gamma * subject_prior
