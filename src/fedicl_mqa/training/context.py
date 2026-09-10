"""Frozen local exemplars and a common fit cohort for train-context ablations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from fedicl_mqa.core.config import Config
from fedicl_mqa.core.io import file_sha256, object_hash, read_json, write_json
from fedicl_mqa.evaluation.retrieval import (
    CapacityFailure,
    ClosureConstrainedRetriever,
    SentenceTransformerEncoder,
)
from fedicl_mqa.modeling.loader import chat_prefix
from fedicl_mqa.modeling.prompting import build_prompt, training_completion


def plan_path(config: Config) -> Path:
    return (
        Path(config.experiment.output_dir)
        / "data"
        / config.data.dataset
        / "training_icl_audit.json"
    )


def load_plan(config: Config) -> dict[str, Any]:
    path = plan_path(config)
    if not path.exists():
        raise FileNotFoundError("run audit-training-icl before training the shared fit cohort")
    plan = read_json(path)
    digest = object_hash({k: v for k, v in plan.items() if k != "sha256"})
    if (
        plan.get("config_hash") != config.hash
        or plan.get("sha256") != digest
        or plan.get("partition_sha256") != file_sha256(path.with_name("file_hashes.json"))
    ):
        raise ValueError("training exemplar plan is stale or modified; use a new output_dir")
    return plan


def audit_training_context(
    config: Config, partitions, *, encoder=None, tokenizer=None
) -> dict[str, Any]:
    if config.icl_training is None:
        raise ValueError("audit-training-icl requires icl_training in the configuration")
    if plan_path(config).exists():
        return load_plan(config)
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
    clients = {}
    for client, roles in sorted(partitions.items()):
        retriever = ClosureConstrainedRetriever(
            sorted(roles["support"], key=lambda q: q.example_id),
            encoder,
            duplicate_similarity_threshold=config.retrieval.duplicate_similarity_threshold,
            lexical_jaccard_threshold=config.retrieval.lexical_jaccard_threshold,
            initial_pool=config.retrieval.initial_pool,
            expanded_pool=config.retrieval.expanded_pool,
        )
        rows, excluded = [], []
        targets = sorted(roles["fit"], key=lambda q: q.example_id)
        for start in range(0, len(targets), 256):
            batch = targets[start : start + 256]
            try:
                pools = retriever.candidate_pools(batch, minimum=5)
            except CapacityFailure:
                pools = []
                for q in batch:
                    try:
                        pools.append(retriever.candidate_pool(q, minimum=5))
                    except CapacityFailure:
                        pools.append(None)
            for q, pool in zip(batch, pools, strict=True):
                if pool is None:
                    excluded.append({"query_id": q.example_id, "reason": "exemplar_capacity"})
                    continue
                demos = [r.example for r in pool[:5]]
                # Eligibility must not depend on the target's gold option.
                completion = max(
                    len(
                        tokenizer(
                            training_completion(replace(q, label=label)), add_special_tokens=False
                        )["input_ids"]
                    )
                    for label in range(4)
                ) + int(tokenizer.eos_token_id is not None)
                lengths = [
                    len(chat_prefix(tokenizer, build_prompt(q, examples), tokenize=True))
                    + completion
                    for examples in ((), demos)
                ]
                # The collator pads to a multiple of eight.
                if max((n + 7) // 8 * 8 for n in lengths) > config.model.max_seq_length:
                    excluded.append({"query_id": q.example_id, "reason": "context_budget"})
                    continue
                rows.append(
                    {
                        "query_id": q.example_id,
                        "exemplar_ids": [q.example_id for q in demos],
                        "tokens_k0": lengths[0],
                        "tokens_k5": lengths[1],
                    }
                )
            if start % 4096 == 0:
                print(
                    f"Training context client {client}: "
                    f"{min(start + 256, len(targets))}/{len(targets)}",
                    flush=True,
                )
        if not rows:
            raise ValueError(f"client {client}: no eligible common fit cohort")
        clients[str(client)] = {"queries": rows, "excluded": excluded, "input_count": len(targets)}
    plan = {
        "config_hash": config.hash,
        "partition_sha256": file_sha256(plan_path(config).with_name("file_hashes.json")),
        "policy": "local_dense_top5_common_fit_v1",
        "clients": clients,
    }
    plan["sha256"] = object_hash(plan)
    write_json(plan_path(config), plan)
    return plan


def training_inputs(config: Config, partitions):
    if config.icl_training is None:
        return {c: roles["fit"] for c, roles in partitions.items()}, None, None
    plan = load_plan(config)
    fit, exemplars = {}, {}
    if set(plan["clients"]) != {str(c) for c in partitions}:
        raise ValueError("training plan client coverage differs from partition")
    for c, roles in partitions.items():
        by_id = {q.example_id: q for q in roles["fit"]}
        support = {q.example_id: q for q in roles["support"]}
        rows = plan["clients"][str(c)]["queries"]
        fit[c], exemplars[c] = [], {}
        for row in rows:
            qid, ids = row["query_id"], row["exemplar_ids"]
            if qid not in by_id or qid in exemplars[c] or len(ids) != 5 or len(set(ids)) != 5:
                raise ValueError("invalid target or exemplar coverage in training plan")
            if any(i not in support or i == qid for i in ids):
                raise ValueError("training exemplar is not in the target client's support")
            fit[c].append(by_id[qid])
            exemplars[c][qid] = [support[i] for i in ids]
        excluded = {r["query_id"] for r in plan["clients"][str(c)]["excluded"]}
        if set(exemplars[c]) & excluded or set(exemplars[c]) | excluded != by_id.keys():
            raise ValueError("training plan does not account for the entire fit pool")
    return fit, exemplars, plan


def bind_protocol(config: Config, root: Path, plan, *, train_icl: bool, create: bool = False):
    if plan is None:
        return
    expected = {
        "config_hash": config.hash,
        "training_plan_sha256": plan["sha256"],
        "train_k": 5 if train_icl else 0,
    }
    path = root / "training_protocol.json"
    if path.exists():
        if read_json(path) != expected:
            raise ValueError("checkpoint training context differs from the sealed plan")
    elif create and not any(root.rglob("state.json")):
        write_json(path, expected)
    else:
        raise ValueError("checkpoint is missing its training context identity")
