from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import Config
from .io import write_json
from .leakage import assert_no_support_leakage
from .retrieval import ClosureConstrainedRetriever, SentenceTransformerEncoder, TextEncoder
from .schema import MCQExample


def audit_retrieval_cohort(
    config: Config,
    partitions: Mapping[int, Mapping[str, Sequence[MCQExample]]],
    output: str | Path,
    *,
    encoder: TextEncoder | None = None,
) -> dict[str, Any]:
    """Verify Top-5 capacity and persist the frozen candidate pools before training."""
    encoder = encoder or SentenceTransformerEncoder(
        config.retrieval.encoder_id,
        revision=config.retrieval.encoder_revision,
        device=config.hardware.device,
    )
    rows: list[dict[str, Any]] = []
    exclusion_counts: Counter[str] = Counter()
    expanded_queries = 0
    for client_id in sorted(partitions):
        support = list(partitions[client_id]["support"])
        queries = [
            *partitions[client_id]["validation"],
            *partitions[client_id]["test"],
        ]
        assert_no_support_leakage(
            support,
            queries,
            lexical_threshold=config.retrieval.lexical_jaccard_threshold,
        )
        retriever = ClosureConstrainedRetriever(
            support,
            encoder,
            duplicate_similarity_threshold=config.retrieval.duplicate_similarity_threshold,
            lexical_jaccard_threshold=config.retrieval.lexical_jaccard_threshold,
            initial_pool=config.retrieval.initial_pool,
            expanded_pool=config.retrieval.expanded_pool,
        )
        pools, diagnostics = retriever.candidate_pools_with_diagnostics(
            queries, minimum=config.retrieval.top_k
        )
        for query, pool, diagnostic in zip(queries, pools, diagnostics, strict=True):
            expanded_queries += int(diagnostic.expanded)
            exclusion_counts.update(diagnostic.exclusions)
            rows.append(
                {
                    "client_id": client_id,
                    "split": query.split,
                    "query_id": query.example_id,
                    "candidate_pool_ids": [value.example.example_id for value in pool],
                    "top5_exemplar_ids": [
                        value.example.example_id for value in pool[: config.retrieval.top_k]
                    ],
                    "diagnostics": asdict(diagnostic),
                }
            )

    payload = {
        "config_hash": config.hash,
        "dataset": config.data.dataset,
        "encoder_id": config.retrieval.encoder_id,
        "encoder_revision": config.retrieval.encoder_revision,
        "top_k": config.retrieval.top_k,
        "query_count": len(rows),
        "capacity_failures": 0,
        "expanded_queries": expanded_queries,
        "expanded_query_rate": expanded_queries / len(rows) if rows else 0.0,
        "exclusions": dict(sorted(exclusion_counts.items())),
        "queries": rows,
    }
    write_json(output, payload)
    return payload
