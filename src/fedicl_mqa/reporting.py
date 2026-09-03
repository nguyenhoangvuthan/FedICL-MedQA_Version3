from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .io import write_json
from .metrics import (
    Prediction,
    hierarchical_paired_bootstrap,
    holm_adjust,
    paired_item_bootstrap,
)

PRIMARY_CONTRASTS = {
    "base_icl": ("B0", "B1"),
    "local_icl": ("L0", "L1"),
    "fl_icl": ("F0", "F1"),
    "client_aware": ("F1", "F2"),
    "fl": ("L0", "F0"),
    "system": ("L1", "F2"),
}


def read_predictions(path: str | Path) -> list[Prediction]:
    result: list[Prediction] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            result.append(Prediction(**row.get("prediction", row)))
    return result


def build_contrast_report(
    arm_predictions: Mapping[str, Mapping[int | None, Sequence[Prediction]]],
    *,
    samples: int,
    confidence: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {"primary": {}, "descriptive": {}}
    for name, (left_arm, right_arm) in PRIMARY_CONTRASTS.items():
        left = arm_predictions[left_arm]
        right = arm_predictions[right_arm]
        result = _contrast(
            left,
            right,
            samples=samples,
            confidence=confidence,
            bootstrap_seed=bootstrap_seed,
        )
        likelihood = _contrast(
            _likelihood_view(left),
            _likelihood_view(right),
            samples=samples,
            confidence=confidence,
            bootstrap_seed=bootstrap_seed,
        )
        report["primary"][name] = {
            "left": left_arm,
            "right": right_arm,
            **result,
            "conditional_likelihood_effect": likelihood["effect"],
            "conditional_likelihood_ci_low": likelihood["ci_low"],
            "conditional_likelihood_ci_high": likelihood["ci_high"],
            "evaluator_dependent": result["effect"] * likelihood["effect"] < 0,
        }

    adjusted = holm_adjust({name: values["p_value"] for name, values in report["primary"].items()})
    for name, value in adjusted.items():
        report["primary"][name]["holm_adjusted_p_value"] = value

    if "C0" in arm_predictions and "F0" in arm_predictions:
        left, right = arm_predictions["F0"], arm_predictions["C0"]
        seeds = (set(left) & set(right)) - {None}
        report["descriptive"]["central"] = hierarchical_paired_bootstrap(
            {seed: left[seed] for seed in seeds},
            {seed: right[seed] for seed in seeds},
            samples=samples,
            confidence=confidence,
            seed=bootstrap_seed,
        )
    return report


def _contrast(
    left: Mapping[int | None, Sequence[Prediction]],
    right: Mapping[int | None, Sequence[Prediction]],
    *,
    samples: int,
    confidence: float,
    bootstrap_seed: int,
) -> dict[str, float]:
    if set(left) == {None} and set(right) == {None}:
        return paired_item_bootstrap(
            left[None],
            right[None],
            samples=samples,
            confidence=confidence,
            seed=bootstrap_seed,
        )
    trained_seeds = (set(left) - {None}) & (set(right) - {None})
    if not trained_seeds:
        trained_seeds = (set(left) | set(right)) - {None}
    left_seeded = {seed: left.get(seed, left.get(None, ())) for seed in sorted(trained_seeds)}
    right_seeded = {seed: right.get(seed, right.get(None, ())) for seed in sorted(trained_seeds)}
    return hierarchical_paired_bootstrap(
        left_seeded,
        right_seeded,
        samples=samples,
        confidence=confidence,
        seed=bootstrap_seed,
    )


def _likelihood_view(
    values: Mapping[int | None, Sequence[Prediction]],
) -> dict[int | None, list[Prediction]]:
    return {
        seed: [
            Prediction(
                example_id=item.example_id,
                gold=item.gold,
                predicted=item.likelihood_predicted,
                stage="conditional_likelihood",
                client_id=item.client_id,
                subject=item.subject,
                seed=item.seed,
                likelihood_predicted=item.likelihood_predicted,
                likelihood_confidence=item.likelihood_confidence,
            )
            for item in predictions
        ]
        for seed, predictions in values.items()
    }


def write_contrast_report(path: str | Path, report: Mapping[str, Any]) -> None:
    write_json(path, report)
