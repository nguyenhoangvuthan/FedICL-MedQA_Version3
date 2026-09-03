from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
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


def write_priors(path: str | Path, priors: Mapping[int, Mapping[str, float]]) -> None:
    write_json(path, {str(client): dict(values) for client, values in priors.items()})


def read_priors(path: str | Path) -> dict[int, dict[str, float]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        int(client): {str(subject): float(value) for subject, value in values.items()}
        for client, values in payload.items()
    }
