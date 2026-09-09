from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from fedicl_mqa.core.io import write_json


def load_prediction_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def leave_one_client_out_weakness(
    rows: Iterable[Mapping[str, Any]], *, num_clients: int
) -> dict[int, dict[str, float]]:
    """Build F2 subject priors using only aggregate validation results of other clients."""
    totals: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0])
    subjects: set[str] = set()
    for row in rows:
        prediction = row.get("prediction", row)
        client = int(prediction["client_id"])
        subject = str(prediction["subject"])
        correct = int(prediction.get("predicted") == prediction["gold"])
        totals[(client, subject)][0] += correct
        totals[(client, subject)][1] += 1
        subjects.add(subject)

    result: dict[int, dict[str, float]] = {}
    for held_out in range(num_clients):
        raw: dict[str, float] = {}
        for subject in subjects:
            correct = sum(
                totals[(client, subject)][0] for client in range(num_clients) if client != held_out
            )
            count = sum(
                totals[(client, subject)][1] for client in range(num_clients) if client != held_out
            )
            raw[subject] = 1.0 - (correct / count) if count else 0.0
        maximum = max(raw.values(), default=0.0)
        minimum = min(raw.values(), default=0.0)
        span = maximum - minimum
        result[held_out] = {
            subject: ((value - minimum) / span if span > 0 else 0.0)
            for subject, value in raw.items()
        }
    return result


def validate_prior_variation(
    priors: Mapping[int, Mapping[str, float]],
    support_subjects: Mapping[int, set[str]],
) -> None:
    if set(priors) != set(support_subjects):
        raise ValueError("prior client IDs differ from support client IDs")
    vectors = []
    for client, subjects in support_subjects.items():
        if not subjects <= priors[client].keys():
            raise ValueError(f"client {client}: prior lacks support subjects")
        values = [float(priors[client][s]) for s in subjects]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("prior weights must be finite and between zero and one")
        if len(values) < 2 or max(values) - min(values) <= 1e-12:
            raise ValueError(f"client {client}: prior is constant on its support subjects")
        vectors.append(tuple(sorted(priors[client].items())))
    if len(set(vectors)) < 2:
        raise ValueError("all clients have identical prior vectors; client variation is absent")


def controlled_weakness(
    rows: Sequence[Mapping[str, Any]],
    *,
    support_subjects: Mapping[int, set[str]],
    min_count: int,
) -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
    """LOCO error rates with Beta(1,1) smoothing, without min-max amplification.

    Only subjects that occur in the held-out client's support can affect ranking.
    Missing/sparse other-client evidence is an error, never a zero-valued prior.
    """
    totals: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0])
    ids: set[str] = set()
    for row in rows:
        p = row.get("prediction", row)
        if p["example_id"] in ids:
            raise ValueError("duplicate validation prediction ID")
        ids.add(p["example_id"])
        client = int(p["client_id"])
        if client not in support_subjects:
            raise ValueError("unknown validation client")
        entry = totals[client, str(p["subject"])]
        entry[0] += int(p["predicted"] != p["gold"])
        entry[1] += 1
    priors = {}
    evidence = {}
    for held_out, subjects in sorted(support_subjects.items()):
        priors[held_out] = {}
        evidence[str(held_out)] = {}
        for subject in sorted(subjects):
            errors = sum(totals[c, subject][0] for c in support_subjects if c != held_out)
            n = sum(totals[c, subject][1] for c in support_subjects if c != held_out)
            if n < min_count:
                raise ValueError(f"client {held_out}/{subject}: only {n} LOCO validation items")
            weight = (errors + 1) / (n + 2)
            priors[held_out][subject] = weight
            evidence[str(held_out)][subject] = {"errors": errors, "n": n, "weight": weight}
    validate_prior_variation(priors, support_subjects)
    return priors, {"estimator": "LOCO (errors+1)/(n+2)", "evidence": evidence}


def shuffled_subject_prior(
    priors: Mapping[int, Mapping[str, float]], *, seed: int
) -> dict[int, dict[str, float]]:
    """Deterministically permute subject weights within client, preserving their multiset."""
    result = {}
    for client, weights in sorted(priors.items()):
        subjects = sorted(weights)
        values = [weights[s] for s in subjects]
        shuffled = list(values)
        rng = random.Random(f"subject-placebo:{seed}:{client}")
        for _ in range(100):
            rng.shuffle(shuffled)
            if shuffled != values:
                break
        if shuffled == values:
            # Also guarantees a changed assignment with repeated values.
            shuffled = values[1:] + values[:1]
        if shuffled == values:
            raise ValueError(f"client {client}: constant prior cannot produce a shuffled placebo")
        result[client] = dict(zip(subjects, shuffled, strict=True))
    return result


def write_priors(path: str | Path, priors: Mapping[int, Mapping[str, float]]) -> None:
    write_json(path, {str(client): dict(values) for client, values in priors.items()})


def read_priors(path: str | Path) -> dict[int, dict[str, float]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        int(client): {str(subject): float(value) for subject, value in values.items()}
        for client, values in payload.items()
    }
