from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from fedicl_mqa.core.io import write_json
from fedicl_mqa.evaluation.metrics import (
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

CONTROLLED_CONTRASTS = {
    "base_icl": ("B0", "B1"),
    "matched_local_icl": ("LM0", "LM1"),
    "fl_icl": ("F0", "F1"),
    "fl_matched": ("LM0", "F0"),
    "system_matched": ("LM1", "F2"),
    "diversity_without_prior": ("F1", "FD"),
    "prior_without_diversity": ("F1", "FP"),
    "prior_with_diversity": ("FD", "F2"),
    "diversity_with_prior": ("FP", "F2"),
    "prior_vs_shuffled": ("FS", "F2"),
}

TRAIN_ICL_CONTRASTS = {
    "local_train_icl_eval_k0": ("L0", "LT0"),
    "local_train_icl_eval_k5": ("L1", "LT1"),
    "local_eval_icl_after_train_icl": ("LT0", "LT1"),
    "fl_train_icl_eval_k0": ("F0", "FT0"),
    "fl_train_icl_eval_k5": ("F1", "FT1"),
    "fl_eval_icl_after_train_icl": ("FT0", "FT1"),
    "central_icl": ("C0", "C1"),
    "central_train_icl_eval_k0": ("C0", "CT0"),
    "central_train_icl_eval_k5": ("C1", "CT1"),
    "central_eval_icl_after_train_icl": ("CT0", "CT1"),
    # Full train+eval exemplar system against the plain k=0 baseline of its family.
    "fl_system": ("F0", "FT1"),
    "central_system": ("C0", "CT1"),
}

# Federated versus Centralized in each train x eval cell. Descriptive, as C0-F0
# always was: the two differ in data pooling, not in the intervention under test.
DESCRIPTIVE_CENTRAL = {
    "central": ("F0", "C0"),
    "central_eval_icl": ("F1", "C1"),
    "central_train_icl": ("FT0", "CT0"),
    "central_train_eval_icl": ("FT1", "CT1"),
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
    controlled: bool = False,
    train_icl: bool = False,
    arms: Collection[str] | None = None,
) -> dict[str, Any]:
    """Bootstrap every declared contrast whose two arms are both available.

    arms restricts the report to a subset, e.g. a pilot that evaluated only the
    federated families. Holm correction then runs over the surviving contrasts
    only, so a partial report is not interchangeable with the full one; it is
    marked partial and lists the arms it covers.
    """
    contrasts = dict(CONTROLLED_CONTRASTS if controlled else PRIMARY_CONTRASTS)
    if train_icl:
        contrasts.update(TRAIN_ICL_CONTRASTS)
    declared = {arm for pair in contrasts.values() for arm in pair}
    available = sorted(declared if arms is None else set(arms))
    contrasts = {name: pair for name, pair in contrasts.items() if set(pair) <= set(available)}
    if not contrasts:
        raise ValueError(f"no declared contrast has both arms in {available}")
    report: dict[str, Any] = {
        "primary": {},
        "descriptive": {},
        "arms": available,
        "partial": set(available) != declared,
    }
    for name, (left_arm, right_arm) in contrasts.items():
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

    for name, (federated, central) in DESCRIPTIVE_CENTRAL.items():
        if federated not in arm_predictions or central not in arm_predictions:
            continue
        left, right = arm_predictions[federated], arm_predictions[central]
        if left.keys() != right.keys():
            raise ValueError("centralized and federated arms must have identical seed coverage")
        seeds = set(left) - {None}
        report["descriptive"][name] = hierarchical_paired_bootstrap(
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
    if None not in left and None not in right and left.keys() != right.keys():
        raise ValueError("both arms must contain identical training seed IDs")
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
