from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fedicl_mqa.core.config import Config
from fedicl_mqa.core.io import atomic_write_text, write_json
from fedicl_mqa.data.leakage import assert_no_support_leakage
from fedicl_mqa.data.matching import AnswerMatcher
from fedicl_mqa.evaluation.metrics import Prediction, evaluation_summary, expected_calibration_error
from fedicl_mqa.modeling.loader import GenerationEngine, ModelBundle, chat_prefix, gpu_telemetry
from fedicl_mqa.modeling.prompting import Prompt, build_prompt
from fedicl_mqa.evaluation.retrieval import (
    ClosureConstrainedRetriever,
    SentenceTransformerEncoder,
    TextEncoder,
    client_aware_rerank,
)
from fedicl_mqa.core.schema import MCQExample


@dataclass(frozen=True, slots=True)
class ArmSpec:
    name: str
    checkpoint_family: str
    uses_icl: bool
    client_aware: bool = False


ARMS: dict[str, ArmSpec] = {
    "B0": ArmSpec("B0", "base", False),
    "B1": ArmSpec("B1", "base", True),
    "L0": ArmSpec("L0", "local", False),
    "L1": ArmSpec("L1", "local", True),
    "F0": ArmSpec("F0", "federated", False),
    "F1": ArmSpec("F1", "federated", True),
    "F2": ArmSpec("F2", "federated", True, True),
    "C0": ArmSpec("C0", "centralized", False),
}


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    prediction: Prediction
    generated_text: str
    likelihood_probabilities: tuple[float, float, float, float]
    prompt_tokens: int
    exemplar_ids: tuple[str, ...]
    retrieval_diagnostics: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["prediction"] = asdict(self.prediction)
        return payload


BeforeClient = Callable[[int, Any], None]


def evaluate_arm(
    bundle: ModelBundle,
    config: Config,
    partitions: Mapping[int, Mapping[str, Sequence[MCQExample]]],
    *,
    arm: str,
    split: str,
    output_dir: str | Path,
    seed: int | None,
    before_client: BeforeClient | None = None,
    subject_weights: Mapping[int, Mapping[str, float]] | None = None,
    encoder: TextEncoder | None = None,
) -> dict[str, Any]:
    arm_name = arm.upper()
    if arm_name not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
    if split not in {"validation", "test"}:
        raise ValueError("evaluation split must be validation or test")
    spec = ARMS[arm_name]
    if spec.client_aware and subject_weights is None:
        raise ValueError("F2 requires frozen client subject weights")

    if encoder is None:
        encoder = SentenceTransformerEncoder(
            config.retrieval.encoder_id,
            revision=config.retrieval.encoder_revision,
            device=config.hardware.device,
        )
    matcher = AnswerMatcher(
        encoder,
        semantic_threshold=config.evaluation.semantic_match_threshold,
        semantic_margin=config.evaluation.semantic_margin,
    )
    engine = GenerationEngine(bundle, config)
    all_records: list[EvaluationRecord] = []
    started = time.perf_counter()

    for client_id in sorted(partitions):
        if before_client:
            before_client(client_id, bundle.model)
        support = list(partitions[client_id]["support"])
        queries = list(partitions[client_id][split])
        if not queries:
            raise ValueError(f"client {client_id} has no {split} queries")
        assert_no_support_leakage(
            support,
            queries,
            lexical_threshold=config.retrieval.lexical_jaccard_threshold,
        )
        retriever = (
            ClosureConstrainedRetriever(
                support,
                encoder,
                duplicate_similarity_threshold=config.retrieval.duplicate_similarity_threshold,
                lexical_jaccard_threshold=config.retrieval.lexical_jaccard_threshold,
                initial_pool=config.retrieval.initial_pool,
                expanded_pool=config.retrieval.expanded_pool,
            )
            if spec.uses_icl
            else None
        )
        if retriever is not None:
            candidate_pools, retrieval_diagnostics = retriever.candidate_pools_with_diagnostics(
                queries, minimum=config.retrieval.top_k
            )
        else:
            candidate_pools = [[] for _ in queries]
            retrieval_diagnostics = [None for _ in queries]

        prompts: list[Prompt] = []
        exemplar_ids: list[tuple[str, ...]] = []
        prompt_tokens: list[int] = []
        for query, pool in zip(queries, candidate_pools, strict=True):
            exemplars = []
            if retriever is not None:
                if spec.client_aware:
                    exemplars = client_aware_rerank(
                        pool,
                        k=config.retrieval.top_k,
                        subject_weights=(subject_weights or {})[client_id],
                        alpha=config.retrieval.alpha,
                        beta=config.retrieval.beta,
                        gamma=config.retrieval.gamma,
                    )
                else:
                    exemplars = pool[: config.retrieval.top_k]
            prompt = build_prompt(query, exemplars)
            prompts.append(prompt)
            exemplar_ids.append(tuple(value.example.example_id for value in exemplars))
            token_count = len(chat_prefix(bundle.tokenizer, prompt, tokenize=True))
            if token_count + config.model.max_new_tokens > config.model.max_seq_length:
                raise ValueError(
                    "prompt exceeds the reserved generation context budget; seal a cohort "
                    f"without this item: item={query.example_id} tokens={token_count}"
                )
            prompt_tokens.append(token_count)

        batch_size = config.evaluation.generation_batch_size
        for offset in range(0, len(queries), batch_size):
            query_batch = queries[offset : offset + batch_size]
            prompt_batch = prompts[offset : offset + batch_size]
            generated_batch = engine.generate(prompt_batch)
            likelihood_batch = engine.score_options_batch(prompt_batch, query_batch)
            for local_index, (query, prompt, generated, likelihood) in enumerate(
                zip(query_batch, prompt_batch, generated_batch, likelihood_batch, strict=True)
            ):
                del prompt
                predicted_likelihood, likelihood_confidence, probabilities = likelihood
                match = matcher.match(generated, query)
                prediction = Prediction(
                    example_id=query.example_id,
                    gold=query.label,
                    predicted=match.label,
                    stage=match.stage,
                    client_id=client_id,
                    subject=query.subject,
                    seed=seed,
                    likelihood_predicted=predicted_likelihood,
                    likelihood_confidence=likelihood_confidence,
                )
                absolute_index = offset + local_index
                all_records.append(
                    EvaluationRecord(
                        prediction=prediction,
                        generated_text=generated,
                        likelihood_probabilities=tuple(probabilities),  # type: ignore[arg-type]
                        prompt_tokens=prompt_tokens[absolute_index],
                        exemplar_ids=exemplar_ids[absolute_index],
                        retrieval_diagnostics=(
                            asdict(retrieval_diagnostics[absolute_index])
                            if retrieval_diagnostics[absolute_index] is not None
                            else None
                        ),
                    )
                )

    elapsed = time.perf_counter() - started
    predictions = [record.prediction for record in all_records]
    if not predictions:
        raise ValueError(f"no predictions were produced for split {split}")
    summary = evaluation_summary(predictions)
    likelihood_correct = sum(item.likelihood_predicted == item.gold for item in predictions) / len(
        predictions
    )
    summary.update(
        {
            "arm": arm_name,
            "split": split,
            "seed": seed,
            "config_hash": config.hash,
            "conditional_likelihood_accuracy": likelihood_correct,
            "pipeline_likelihood_agreement": sum(
                item.predicted == item.likelihood_predicted for item in predictions
            )
            / len(predictions),
            "pipeline_likelihood_correctness_matrix": _correctness_agreement(predictions),
            "evaluator_disagreement_rate": sum(
                item.predicted != item.likelihood_predicted for item in predictions
            )
            / len(predictions),
            "ece": expected_calibration_error(predictions),
            "wall_clock_seconds": elapsed,
            "items_per_second": len(predictions) / elapsed,
            "prompt_tokens_p50": _percentile(
                prompt_tokens=[r.prompt_tokens for r in all_records], p=0.50
            ),
            "prompt_tokens_p95": _percentile(
                prompt_tokens=[r.prompt_tokens for r in all_records], p=0.95
            ),
            "prompt_tokens_max": max(record.prompt_tokens for record in all_records),
            "retrieval_health": _retrieval_health(all_records),
            "hardware": gpu_telemetry(),
        }
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) for record in all_records
    ]
    atomic_write_text(output / "predictions.jsonl", "\n".join(lines) + "\n")
    write_json(output / "summary.json", summary)
    return summary


def _percentile(*, prompt_tokens: Sequence[int], p: float) -> float:
    if not prompt_tokens:
        raise ValueError("cannot compute a percentile of an empty sequence")
    values = sorted(prompt_tokens)
    position = p * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _correctness_agreement(predictions: Sequence[Prediction]) -> dict[str, int]:
    matrix = {
        "both_correct": 0,
        "pipeline_only_correct": 0,
        "likelihood_only_correct": 0,
        "both_incorrect": 0,
    }
    for item in predictions:
        pipeline_correct = item.predicted == item.gold
        likelihood_correct = item.likelihood_predicted == item.gold
        if pipeline_correct and likelihood_correct:
            key = "both_correct"
        elif pipeline_correct:
            key = "pipeline_only_correct"
        elif likelihood_correct:
            key = "likelihood_only_correct"
        else:
            key = "both_incorrect"
        matrix[key] += 1
    return matrix


def _retrieval_health(records: Sequence[EvaluationRecord]) -> dict[str, Any]:
    diagnostics = [
        record.retrieval_diagnostics
        for record in records
        if record.retrieval_diagnostics is not None
    ]
    if not diagnostics:
        return {"retrieval_queries": 0}
    exclusion_counts: dict[str, int] = {}
    for diagnostic in diagnostics:
        for reason, count in diagnostic["exclusions"].items():
            exclusion_counts[reason] = exclusion_counts.get(reason, 0) + int(count)
    expanded = sum(bool(diagnostic["expanded"]) for diagnostic in diagnostics)
    return {
        "retrieval_queries": len(diagnostics),
        "expanded_queries": expanded,
        "expanded_query_rate": expanded / len(diagnostics),
        "capacity_failures": 0,
        "exclusions": dict(sorted(exclusion_counts.items())),
    }
