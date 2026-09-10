"""Native subject metadata and disjoint development data for controlled studies."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from fedicl_mqa.core.schema import MCQExample

MISSING_SUBJECTS = frozenset({"", "unknown", "general", "n/a", "none"})


def filter_native_subjects(
    sources: Mapping[str, Sequence[MCQExample]],
) -> tuple[dict[str, list[MCQExample]], dict[str, Any]]:
    """Exclude missing native labels before splitting, using metadata only."""
    retained = {}
    audit: dict[str, Any] = {
        "policy": "exclude_missing_native_subjects_v1",
        "missing_subject_values": sorted(MISSING_SUBJECTS),
        "source_splits": {},
    }
    for split, items in sources.items():
        excluded = [q for q in items if q.subject.strip().casefold() in MISSING_SUBJECTS]
        retained[split] = [q for q in items if q.subject.strip().casefold() not in MISSING_SUBJECTS]
        audit["source_splits"][split] = {
            "input_count": len(items),
            "retained_count": len(retained[split]),
            "excluded_count": len(excluded),
            "excluded_subject_counts": dict(sorted(Counter(q.subject for q in excluded).items())),
            "excluded_ids": sorted(q.example_id for q in excluded),
        }
        if not retained[split]:
            raise ValueError(f"source {split} has no examples with usable native subject labels")
    return retained, audit


def development_split(
    train: Sequence[MCQExample],
    official_validation: Sequence[MCQExample],
    *,
    fraction: float,
    seed: int,
    limits: Mapping[str, int | None],
) -> dict[str, list[MCQExample]]:
    """Hold out train questions by subject; reserve official validation for final test.

    Sampling uses IDs and subjects, never answers or model predictions. The official
    test split is not consumed: public copies can have masked answer labels.
    """
    if not 0 < fraction < 1:
        raise ValueError("development fraction must be between zero and one")
    all_ids = [q.example_id for q in [*train, *official_validation]]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("source splits contain duplicate example IDs")
    rng = random.Random(seed)
    grouped: dict[str, list[MCQExample]] = defaultdict(list)
    for item in sorted(train, key=lambda q: q.example_id):
        grouped[item.subject].append(item)
    result: dict[str, list[MCQExample]] = {"train": [], "validation": [], "test": []}
    for subject in sorted(grouped):
        items = grouped[subject]
        if len(items) < 2:
            raise ValueError(f"subject {subject!r} needs train and development examples")
        rng.shuffle(items)
        count = max(1, min(len(items) - 1, round(len(items) * fraction)))
        result["validation"].extend(replace(q, split="validation") for q in items[:count])
        result["train"].extend(replace(q, split="train") for q in items[count:])
    result["test"] = [
        replace(q, split="test") for q in sorted(official_validation, key=lambda q: q.example_id)
    ]
    for split, items in result.items():
        rng.shuffle(items)
        limit = limits.get(split)
        if limit is not None:
            if limit <= 0:
                raise ValueError("sample limits must be positive")
            result[split] = items[:limit]
    return result


def audit_subjects(
    partitions: Mapping[int, Mapping[str, Sequence[MCQExample]]],
    *,
    min_validation_per_subject: int,
) -> dict[str, Any]:
    """Reject placeholder subjects and priors unsupported by other-client validation."""
    counts: dict[int, dict[str, dict[str, int]]] = {}
    for client, roles in sorted(partitions.items()):
        counts[client] = {}
        for role in ("fit", "support", "validation", "test"):
            histogram = Counter(q.subject for q in roles[role])
            if not histogram:
                raise ValueError(f"client {client} has no {role} data")
            missing = {
                s: n for s, n in histogram.items() if s.strip().casefold() in MISSING_SUBJECTS
            }
            if missing:
                raise ValueError(
                    f"client {client}/{role} has missing native subject labels: {missing}"
                )
            counts[client][role] = dict(sorted(histogram.items()))
        if len(counts[client]["support"]) < 2:
            raise ValueError(f"client {client} needs at least two subjects in its support pool")
    loco_counts = {}
    for client in counts:
        available = Counter()
        for other in counts:
            if other != client:
                available.update(counts[other]["validation"])
        required = set(counts[client]["support"])
        insufficient = {
            s: available[s] for s in required if available[s] < min_validation_per_subject
        }
        if insufficient:
            raise ValueError(
                f"client {client}: insufficient other-client validation per subject: "
                f"{insufficient}; need {min_validation_per_subject}. "
                "Choose a larger predeclared development cohort; do not tune on test accuracy."
            )
        loco_counts[client] = {s: available[s] for s in sorted(required)}
    subjects = sorted({s for roles in counts.values() for s in roles["fit"]})
    proportions = {
        c: [roles["fit"].get(s, 0) / sum(roles["fit"].values()) for s in subjects]
        for c, roles in counts.items()
    }
    max_tv = max(
        sum(abs(a - b) for a, b in zip(proportions[c], proportions[d], strict=True)) / 2
        for c in counts
        for d in counts
    )
    if max_tv <= 1e-12:
        raise ValueError("client subject distributions are identical; no subject-skew contrast")
    return {
        "subject_source": "native subject_name metadata; no inferred labels",
        "counts": counts,
        "other_client_validation_counts": loco_counts,
        "max_pairwise_fit_subject_total_variation": max_tv,
        "min_validation_per_subject": min_validation_per_subject,
    }
