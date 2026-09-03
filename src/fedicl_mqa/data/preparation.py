from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fedicl_mqa.core.io import file_sha256, write_examples, write_json
from fedicl_mqa.core.schema import MCQExample, label_to_index, options_tuple


def adapt_medqa(row: Mapping[str, Any], split: str) -> MCQExample:
    """Adapt the nested schema used by openlifescienceai/medqa."""
    data = row.get("data", row)
    if not isinstance(data, Mapping):
        raise ValueError("MedQA row.data must be a mapping")
    question = data.get("Question", data.get("question"))
    options = data.get("Options", data.get("options"))
    correct = data.get("Correct Option", data.get("answer_idx", data.get("label")))
    if question is None or options is None or correct is None:
        raise ValueError(f"unsupported MedQA schema: keys={sorted(data.keys())}")
    example_id = str(row.get("id") or data.get("id") or _stable_row_id(question, options))
    return MCQExample(
        example_id=example_id,
        question=str(question),
        options=options_tuple(options),
        label=label_to_index(correct),
        split=_canonical_split(split),
        subject=str(row.get("subject_name") or data.get("subject_name") or "medicine"),
        topic=str(row.get("topic_name") or data.get("topic_name") or "unknown"),
        provenance_group=_optional_text(row.get("source") or data.get("source")),
        explanation=_optional_text(data.get("Explanation") or data.get("explanation")),
    )


def adapt_medmcqa(row: Mapping[str, Any], split: str) -> MCQExample:
    """Adapt openlifescienceai/medmcqa; its ClassLabel is zero-based."""
    options = (row["opa"], row["opb"], row["opc"], row["opd"])
    return MCQExample(
        example_id=str(row.get("id") or _stable_row_id(row["question"], options)),
        question=str(row["question"]),
        options=options_tuple(options),
        label=label_to_index(row["cop"], integer_base=0),
        split=_canonical_split(split),
        subject=str(row.get("subject_name") or "unknown"),
        topic=str(row.get("topic_name") or "unknown"),
        provenance_group=_optional_text(row.get("source") or row.get("exam")),
        explanation=_optional_text(row.get("exp")),
    )


def _canonical_split(split: str) -> str:
    normalized = split.casefold()
    aliases = {"dev": "validation", "valid": "validation", "val": "validation"}
    return aliases.get(normalized, normalized)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _stable_row_id(question: Any, options: Any) -> str:
    from hashlib import sha256

    raw = json.dumps([question, options], ensure_ascii=False, sort_keys=True, default=str)
    return sha256(raw.encode("utf-8")).hexdigest()[:24]


def resolve_hub_revision(dataset_id: str, revision: str | None) -> str:
    """Resolve a mutable Hub ref to an immutable commit SHA."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - exercised only in full environment
        raise RuntimeError("huggingface-hub is required to resolve dataset revisions") from exc
    info = HfApi().dataset_info(repo_id=dataset_id, revision=revision or "main")
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a commit SHA for {dataset_id}")
    return info.sha


def load_native_dataset(
    dataset_name: str,
    dataset_id: str,
    *,
    revision: str,
    limits: Mapping[str, int | None] | None = None,
) -> dict[str, list[MCQExample]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only in full environment
        raise RuntimeError("datasets is required; install the project dependencies") from exc

    raw = load_dataset(dataset_id, revision=revision, trust_remote_code=False)
    adapter = {"medqa": adapt_medqa, "medmcqa": adapt_medmcqa}.get(dataset_name.casefold())
    if adapter is None:
        raise ValueError(f"unsupported dataset: {dataset_name}")

    normalized: dict[str, list[MCQExample]] = {}
    for raw_split, records in raw.items():
        split = _canonical_split(raw_split)
        if split not in {"train", "validation", "test"}:
            continue
        limit = (limits or {}).get(split)
        selected = records if limit is None else records.select(range(min(limit, len(records))))
        normalized[split] = [adapter(row, split) for row in selected]
    missing = {"train", "validation", "test"} - normalized.keys()
    if missing:
        raise ValueError(f"dataset {dataset_id} is missing splits: {sorted(missing)}")
    return normalized


@dataclass(frozen=True, slots=True)
class Assignment:
    example_id: str
    client_id: int
    original_split: str
    role: str


def _dirichlet(rng: random.Random, count: int, alpha: float) -> tuple[float, ...]:
    if count < 1 or alpha <= 0:
        raise ValueError("count and alpha must be positive")
    draws = [rng.gammavariate(alpha, 1.0) for _ in range(count)]
    total = sum(draws)
    if total == 0:
        return tuple(1.0 / count for _ in draws)
    return tuple(draw / total for draw in draws)


def _weighted_choice(rng: random.Random, probabilities: Sequence[float]) -> int:
    needle = rng.random() * sum(probabilities)
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if needle <= cumulative:
            return index
    return len(probabilities) - 1


def build_partition(
    splits: Mapping[str, Sequence[MCQExample]],
    *,
    num_clients: int,
    alpha: float,
    fit_ratio: float,
    seed: int,
    min_support_per_client: int = 5,
) -> tuple[list[Assignment], dict[str, tuple[float, ...]]]:
    """Create a reproducible subject-skewed, disjoint client partition."""
    if not 0 < fit_ratio < 1:
        raise ValueError("fit_ratio must be between zero and one")
    all_ids = [example.example_id for values in splits.values() for example in values]
    duplicate_ids = [key for key, count in Counter(all_ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"example IDs must be globally unique; duplicates={duplicate_ids[:10]}")
    rng = random.Random(seed)
    subjects = sorted({example.subject for values in splits.values() for example in values})
    weights = {subject: _dirichlet(rng, num_clients, alpha) for subject in subjects}

    by_split_client: dict[tuple[str, int], list[MCQExample]] = defaultdict(list)
    for split in ("train", "validation", "test"):
        values = list(splits[split])
        rng.shuffle(values)
        for example in values:
            client = _weighted_choice(rng, weights[example.subject])
            by_split_client[(split, client)].append(example)

    _rebalance_split_clients(
        by_split_client,
        split="train",
        num_clients=num_clients,
        minimum=min_support_per_client + 1,
    )
    for split in ("validation", "test"):
        _rebalance_split_clients(
            by_split_client,
            split=split,
            num_clients=num_clients,
            minimum=1,
        )
    assignments: list[Assignment] = []
    for client in range(num_clients):
        train = by_split_client[("train", client)]
        rng.shuffle(train)
        if len(train) < min_support_per_client + 1:
            raise ValueError(
                f"client {client} has {len(train)} train items; need at least "
                f"{min_support_per_client + 1}"
            )
        support_count = max(min_support_per_client, round(len(train) * (1.0 - fit_ratio)))
        support_count = min(support_count, len(train) - 1)
        support_ids = {example.example_id for example in train[:support_count]}
        for example in train:
            role = "support" if example.example_id in support_ids else "fit"
            assignments.append(Assignment(example.example_id, client, "train", role))
        for split in ("validation", "test"):
            assignments.extend(
                Assignment(example.example_id, client, split, split)
                for example in by_split_client[(split, client)]
            )
    return assignments, weights


def _rebalance_split_clients(
    buckets: dict[tuple[str, int], list[MCQExample]],
    *,
    split: str,
    num_clients: int,
    minimum: int,
) -> None:
    if sum(len(buckets[(split, client)]) for client in range(num_clients)) < minimum * num_clients:
        raise ValueError(f"not enough {split} data to give every client {minimum} item(s)")
    for recipient in range(num_clients):
        while len(buckets[(split, recipient)]) < minimum:
            donor = max(range(num_clients), key=lambda client: len(buckets[(split, client)]))
            if len(buckets[(split, donor)]) <= minimum:
                raise ValueError(f"cannot rebalance {split} data across clients")
            buckets[(split, recipient)].append(buckets[(split, donor)].pop())


def materialize_partition(
    root: str | Path,
    splits: Mapping[str, Sequence[MCQExample]],
    assignments: Sequence[Assignment],
    *,
    dataset_id: str,
    dataset_revision: str,
    data_seed: int,
    weights: Mapping[str, Sequence[float]],
    config_hash: str,
) -> None:
    root = Path(root)
    by_id = {example.example_id: example for values in splits.values() for example in values}
    grouped: dict[tuple[int, str], list[MCQExample]] = defaultdict(list)
    for assignment in assignments:
        grouped[(assignment.client_id, assignment.role)].append(by_id[assignment.example_id])
    for (client, role), examples in grouped.items():
        write_examples(root / f"client_{client}" / f"{role}.jsonl", examples)

    assignment_payload = [asdict(assignment) for assignment in assignments]
    write_json(
        root / "partition_manifest.json",
        {
            "dataset_id": dataset_id,
            "dataset_revision": dataset_revision,
            "data_seed": data_seed,
            "config_hash": config_hash,
            "counts": dict(Counter(assignment.role for assignment in assignments)),
            "subject_client_weights": {key: list(value) for key, value in weights.items()},
            "assignments": assignment_payload,
        },
    )
    hashes = {
        str(path.relative_to(root)): file_sha256(path)
        for path in [root / "partition_manifest.json", *sorted(root.glob("client_*/*.jsonl"))]
    }
    write_json(root / "file_hashes.json", hashes)


def load_partition(
    root: str | Path, *, expected_config_hash: str | None = None
) -> dict[int, dict[str, list[MCQExample]]]:
    from fedicl_mqa.core.io import read_examples

    root = Path(root)
    _verify_partition_files(root)
    manifest = _read_manifest(root)
    if expected_config_hash is not None and manifest.get("config_hash") != expected_config_hash:
        raise ValueError("prepared partition config hash differs from the sealed experiment config")
    result: dict[int, dict[str, list[MCQExample]]] = {}
    for client_dir in sorted(root.glob("client_*")):
        client = int(client_dir.name.split("_", 1)[1])
        result[client] = {}
        for role in ("fit", "support", "validation", "test"):
            path = client_dir / f"{role}.jsonl"
            result[client][role] = read_examples(path) if path.exists() else []
    if not result:
        raise FileNotFoundError(f"no client partitions found under {root}")
    return result


def _read_manifest(root: Path) -> dict[str, Any]:
    from fedicl_mqa.core.io import read_json

    return read_json(root / "partition_manifest.json")


def _verify_partition_files(root: Path) -> None:
    from fedicl_mqa.core.io import read_json

    expected = read_json(root / "file_hashes.json")
    actual_files = {
        str(path.relative_to(root))
        for path in [root / "partition_manifest.json", *root.glob("client_*/*.jsonl")]
        if path.is_file()
    }
    if actual_files != set(expected):
        raise ValueError("prepared partition file manifest does not match files on disk")
    for relative, digest in expected.items():
        artifact = (root / relative).resolve()
        if root.resolve() not in artifact.parents:
            raise ValueError(f"partition manifest path escapes root: {relative}")
        if file_sha256(artifact) != digest:
            raise ValueError(f"prepared partition hash mismatch: {artifact}")
