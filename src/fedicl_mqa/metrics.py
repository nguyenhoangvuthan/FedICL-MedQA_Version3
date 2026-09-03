from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Prediction:
    example_id: str
    gold: int
    predicted: int | None
    stage: str
    client_id: int
    subject: str
    seed: int | None = None
    likelihood_predicted: int | None = None
    likelihood_confidence: float | None = None

    @property
    def correct(self) -> bool:
        return self.predicted == self.gold


def accuracy(predictions: Sequence[Prediction]) -> float:
    if not predictions:
        raise ValueError("accuracy requires at least one prediction")
    return sum(item.correct for item in predictions) / len(predictions)


def macro_f1(predictions: Sequence[Prediction], labels: int = 4) -> float:
    if not predictions:
        raise ValueError("macro_f1 requires at least one prediction")
    scores: list[float] = []
    for label in range(labels):
        true_positive = sum(p.gold == label and p.predicted == label for p in predictions)
        false_positive = sum(p.gold != label and p.predicted == label for p in predictions)
        false_negative = sum(p.gold == label and p.predicted != label for p in predictions)
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / labels


def evaluation_summary(predictions: Sequence[Prediction]) -> dict[str, object]:
    stage_counts = Counter(item.stage for item in predictions)
    gold_positions = Counter(item.gold for item in predictions)
    predicted_positions = Counter(
        item.predicted for item in predictions if item.predicted is not None
    )
    by_client: dict[int, list[Prediction]] = defaultdict(list)
    by_subject: dict[str, list[Prediction]] = defaultdict(list)
    for item in predictions:
        by_client[item.client_id].append(item)
        by_subject[item.subject].append(item)
    client_accuracy = {str(key): accuracy(value) for key, value in sorted(by_client.items())}
    return {
        "n": len(predictions),
        "pipeline_accuracy": accuracy(predictions),
        "position_macro_f1": macro_f1(predictions),
        "exact_match_coverage": sum(
            stage_counts[stage] for stage in ("terminal_label", "exact_option")
        )
        / len(predictions),
        "semantic_fallback_rate": stage_counts["semantic_option"] / len(predictions),
        "unresolved_rate": stage_counts["unresolved"] / len(predictions),
        "gold_position_distribution": dict(sorted(gold_positions.items())),
        "predicted_position_distribution": dict(sorted(predicted_positions.items())),
        "client_accuracy": client_accuracy,
        "macro_client_accuracy": sum(client_accuracy.values()) / len(client_accuracy),
        "worst_client_accuracy": min(client_accuracy.values()),
        "subject_accuracy": {key: accuracy(value) for key, value in sorted(by_subject.items())},
    }


def expected_calibration_error(predictions: Sequence[Prediction], *, bins: int = 10) -> float:
    available = [
        item
        for item in predictions
        if item.likelihood_confidence is not None and item.likelihood_predicted is not None
    ]
    if not available:
        raise ValueError("ECE requires likelihood confidence values")
    total = len(available)
    error = 0.0
    for bin_index in range(bins):
        lower, upper = bin_index / bins, (bin_index + 1) / bins
        selected = [
            item
            for item in available
            if lower <= float(item.likelihood_confidence) <= upper
            and (bin_index == bins - 1 or float(item.likelihood_confidence) < upper)
        ]
        if not selected:
            continue
        confidence = sum(float(item.likelihood_confidence) for item in selected) / len(selected)
        observed = sum(item.likelihood_predicted == item.gold for item in selected) / len(selected)
        error += len(selected) / total * abs(observed - confidence)
    return error


def paired_item_bootstrap(
    left: Sequence[Prediction],
    right: Sequence[Prediction],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    _validate_bootstrap(samples, confidence)
    differences = _paired_differences(left, right)
    rng = random.Random(seed)
    draws = [
        sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(samples)
    ]
    draws.sort()
    tail = (1.0 - confidence) / 2.0
    return {
        "effect": sum(differences) / len(differences),
        "ci_low": _quantile(draws, tail),
        "ci_high": _quantile(draws, 1.0 - tail),
        "p_value": _two_sided_zero_p_value(draws),
    }


def hierarchical_paired_bootstrap(
    left: Mapping[int, Sequence[Prediction]],
    right: Mapping[int, Sequence[Prediction]],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    _validate_bootstrap(samples, confidence)
    if left.keys() != right.keys() or not left:
        raise ValueError("both arms must contain identical training seed IDs")
    paired: dict[int, list[float]] = {}
    for training_seed in left:
        try:
            paired[training_seed] = _paired_differences(left[training_seed], right[training_seed])
        except ValueError as exc:
            raise ValueError(f"invalid prediction pair for seed {training_seed}: {exc}") from exc
    rng = random.Random(seed)
    seeds = sorted(paired)
    draws: list[float] = []
    for _ in range(samples):
        sampled_seeds = [rng.choice(seeds) for _ in seeds]
        seed_effects: list[float] = []
        for selected_seed in sampled_seeds:
            values = paired[selected_seed]
            seed_effects.append(sum(rng.choice(values) for _ in values) / len(values))
        draws.append(sum(seed_effects) / len(seed_effects))
    draws.sort()
    observed = sum(sum(values) / len(values) for values in paired.values()) / len(paired)
    tail = (1.0 - confidence) / 2.0
    return {
        "effect": observed,
        "ci_low": _quantile(draws, tail),
        "ci_high": _quantile(draws, 1.0 - tail),
        "p_value": _two_sided_zero_p_value(draws),
    }


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        return math.nan
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _validate_bootstrap(samples: int, confidence: float) -> None:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("bootstrap confidence must be between zero and one")


def _paired_differences(left: Sequence[Prediction], right: Sequence[Prediction]) -> list[float]:
    if not left or not right:
        raise ValueError("paired bootstrap requires non-empty predictions")
    left_map = {item.example_id: item for item in left}
    right_map = {item.example_id: item for item in right}
    if len(left_map) != len(left) or len(right_map) != len(right):
        raise ValueError("prediction vectors contain duplicate item IDs")
    if left_map.keys() != right_map.keys():
        raise ValueError("paired bootstrap requires identical evaluation item IDs")
    return [
        float(right_map[key].correct) - float(left_map[key].correct) for key in sorted(left_map)
    ]


def _two_sided_zero_p_value(draws: Sequence[float]) -> float:
    if not draws:
        return math.nan
    non_positive = sum(value <= 0 for value in draws) / len(draws)
    non_negative = sum(value >= 0 for value in draws) / len(draws)
    return min(1.0, 2.0 * min(non_positive, non_negative))


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda key: p_values[key])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_values[key])
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted
