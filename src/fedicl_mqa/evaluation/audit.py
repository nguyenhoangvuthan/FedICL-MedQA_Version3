from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fedicl_mqa.core.config import Config
from fedicl_mqa.core.io import write_json
from fedicl_mqa.data.leakage import assert_no_support_leakage
from fedicl_mqa.evaluation.retrieval import (
    ClosureConstrainedRetriever,
    SentenceTransformerEncoder,
    TextEncoder,
)
from fedicl_mqa.core.schema import MCQExample
from fedicl_mqa.modeling.loader import chat_prefix
from fedicl_mqa.modeling.prompting import build_prompt, render_demonstration


def audit_retrieval_cohort(
    config: Config,
    partitions: Mapping[int, Mapping[str, Sequence[MCQExample]]],
    output: str | Path,
    *,
    encoder: TextEncoder | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Verify Top-5 capacity and the prompt budget, then freeze the candidate pools.

    Capacity alone is not enough. Five eligible exemplars can still build a prompt too
    long to evaluate, and that only surfaced per item during evaluation, halting a sweep
    after the training steps had already run. Measuring it here fails in minutes instead.
    """
    encoder = encoder or SentenceTransformerEncoder(
        config.retrieval.encoder_id,
        revision=config.retrieval.encoder_revision,
        device=config.hardware.device,
    )
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model.id,
            revision=config.model.revision,
            trust_remote_code=config.model.trust_remote_code,
            use_fast=True,
        )
    budget = config.model.max_seq_length - config.model.max_new_tokens
    over_budget: list[dict[str, Any]] = []
    longest = 0
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
            # F2 reranks inside the same pool, so it can pick a different five. Bound
            # every arm by measuring the longest five rather than the frozen Top-5.
            worst = sorted(pool, key=lambda value: len(render_demonstration(value.example)))[
                -config.retrieval.top_k :
            ]
            tokens = len(
                chat_prefix(tokenizer, build_prompt(query, worst), tokenize=True)
            )
            longest = max(longest, tokens)
            if tokens > budget:
                over_budget.append(
                    {"query_id": query.example_id, "split": query.split, "tokens": tokens}
                )
            rows.append(
                {
                    "client_id": client_id,
                    "worst_case_prompt_tokens": tokens,
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
        "prompt_budget": budget,
        "longest_prompt_tokens": longest,
        "over_budget_queries": len(over_budget),
        "expanded_queries": expanded_queries,
        "expanded_query_rate": expanded_queries / len(rows) if rows else 0.0,
        "exclusions": dict(sorted(exclusion_counts.items())),
        "queries": rows,
    }
    write_json(output, payload)
    if over_budget:
        worst = max(over_budget, key=lambda row: row["tokens"])
        raise ValueError(
            f"{len(over_budget)} of {len(rows)} queries build a prompt longer than the "
            f"{budget}-token budget (longest {worst['tokens']}, item={worst['query_id']}). "
            "Raise model.max_seq_length, or lower retrieval.top_k, before training: "
            f"the per-item counts are in {output}."
        )
    return payload
