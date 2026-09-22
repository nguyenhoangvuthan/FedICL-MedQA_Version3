"""Held-out answer loss and resumable selection of LoRA checkpoints."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

from fedicl_mqa.core.io import atomic_write_text, file_sha256, object_hash, read_json, write_json
from fedicl_mqa.core.schema import MCQExample
from fedicl_mqa.modeling.loader import ModelBundle
from fedicl_mqa.training.checkpointing import CheckpointManager

logger = logging.getLogger(__name__)


def epoch_checkpoint(
    manager: CheckpointManager, epoch: int, *, expected_kind: str, expected_seed: int
) -> tuple[str, dict[str, Any]]:
    """Require a saved end-of-epoch adapter; never substitute a later checkpoint."""
    if epoch < 1:
        raise ValueError("epoch must be positive")
    matches = []
    for path in sorted(manager.root.glob("checkpoint-*")):
        # Step checkpoints use in-progress epoch offsets, not completed epochs.
        if not path.name.startswith(("checkpoint-epoch-", "checkpoint-round-")):
            continue
        payload = read_json(path / "state.json")
        state = payload["trainer_state"]
        if state["epoch"] != epoch or state["batch_in_epoch"] != 0:
            continue
        manager.verify(path)
        if (
            payload["config_hash"] != manager.config_hash
            or payload["model_id"] != manager.model_id
            or payload["model_revision"] != manager.model_revision
            or state["kind"] != expected_kind
            or state["seed"] != expected_seed
        ):
            raise ValueError(f"epoch checkpoint has a different run/model/config identity: {path}")
        matches.append(
            (
                path.name,
                {
                    "checkpoint": path.name,
                    "checkpoint_sha256": file_sha256(path / "hashes.json"),
                    "epoch": state["epoch"],
                    "round": state["round_index"],
                    "global_step": state["global_step"],
                },
            )
        )
    if not matches:
        raise FileNotFoundError(
            f"no saved end-of-epoch {epoch} checkpoint in {manager.root}; "
            "restore that checkpoint from backup if available. No training or fallback performed."
        )
    if len(matches) != 1:
        raise ValueError(f"ambiguous end-of-epoch {epoch} checkpoints in {manager.root}")
    return matches[0]


def validation_identity(examples: Sequence[MCQExample]) -> str:
    if not examples or any(q.split != "validation" for q in examples):
        raise ValueError("checkpoint validation requires non-empty held-out validation data")
    if len({q.example_id for q in examples}) != len(examples):
        raise ValueError("duplicate validation example IDs")
    return object_hash(
        {"metric": "answer_token_loss_no_icl_v1", "examples": [q.to_dict() for q in examples]}
    )


def validate_fit_separation(fit: Sequence[MCQExample], validation: Sequence[MCQExample]) -> None:
    validation_identity(validation)
    ids = {q.example_id for q in fit}
    hashes = {q.question_options_hash for q in fit}
    if any(q.example_id in ids or q.question_options_hash in hashes for q in validation):
        raise ValueError("validation data overlaps training data")


def validation_loss(bundle: ModelBundle, examples: Sequence[MCQExample], max_length: int) -> float:
    """No ICL for any family; mean NLL over supervised answer/EOS tokens only."""
    import torch
    from torch.utils.data import DataLoader

    from fedicl_mqa.training.loop import AnswerCollator, AnswerOnlyDataset

    validation_identity(examples)
    model, tokenizer = bundle.model, bundle.tokenizer
    was_training, padding_side = model.training, tokenizer.padding_side
    use_cache = model.config.use_cache
    total_loss, total_tokens = 0.0, 0
    try:
        tokenizer.padding_side = "right"
        model.eval()
        model.config.use_cache = False
        # One item at a time bounds logits/activation memory on the training GPU.
        # A private generator prevents DataLoader from consuming the training RNG.
        loader = DataLoader(
            AnswerOnlyDataset(examples, tokenizer, max_length),
            batch_size=1,
            collate_fn=AnswerCollator(tokenizer),
            generator=torch.Generator().manual_seed(0),
        )
        with torch.inference_mode():
            for batch in loader:
                tokens = int((batch["labels"][:, 1:] != -100).sum().item())
                if not tokens:
                    raise ValueError("validation item has no supervised answer tokens")
                batch = {key: value.to(bundle.device) for key, value in batch.items()}
                loss = float(model(**batch).loss.item())
                if not math.isfinite(loss):
                    raise RuntimeError(f"non-finite validation loss: {loss}")
                total_loss += loss * tokens
                total_tokens += tokens
                del batch
        return total_loss / total_tokens
    finally:
        model.train(was_training)
        model.config.use_cache = use_cache
        tokenizer.padding_side = padding_side


def select_existing_checkpoints(
    bundle: ModelBundle,
    manager: CheckpointManager,
    examples: Sequence[MCQExample],
    *,
    max_length: int,
    expected_kind: str,
    expected_seed: int,
) -> dict[str, Any]:
    """Validate saved adapters without training or editing immutable checkpoints."""
    identity = {
        "config_hash": manager.config_hash,
        "validation_sha256": validation_identity(examples),
        "max_length": max_length,
        "kind": expected_kind,
        "seed": expected_seed,
    }
    report_path = manager.root / "validation_selection.json"
    previous = read_json(report_path) if report_path.exists() else {}
    if previous and previous.get("identity") != identity:
        raise ValueError("existing selection has a different validation cohort or run identity")
    candidates = []
    for path in sorted(manager.root.glob("checkpoint-*")):
        try:
            manager.verify(path)
            payload = read_json(path / "state.json")
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Skipping invalid checkpoint %s: %s", path, exc)
            continue
        state = payload["trainer_state"]
        if (
            payload["config_hash"] != manager.config_hash
            or payload["model_id"] != manager.model_id
            or payload["model_revision"] != manager.model_revision
            or state["kind"] != expected_kind
            or state["seed"] != expected_seed
        ):
            raise ValueError(f"checkpoint has a different run/model/config identity: {path}")
        if not (state["global_step"] or state["round_index"] or state["epoch"]):
            continue  # Round zero is the untrained initialization.
        candidates.append((path, state))
    if not candidates:
        raise ValueError(f"no valid trained checkpoints found in {manager.root}")
    logger.info("Validating %d saved checkpoints in %s", len(candidates), manager.root)
    candidates.sort(
        key=lambda item: (
            item[1]["round_index"],
            item[1]["global_step"],
            item[1]["epoch"],
            item[0].name,
        )
    )
    records = {}
    for path, state in candidates:
        checkpoint_hash = file_sha256(path / "hashes.json")
        cached = previous.get("checkpoints", {}).get(path.name, {})
        if cached.get("checkpoint_sha256") == checkpoint_hash:
            loss = float(cached["validation_loss"])
        else:
            manager.load(path, model=bundle.model, restore_rng=False)
            loss = validation_loss(bundle, examples, max_length)
        if not math.isfinite(loss):
            raise ValueError("non-finite validation loss cannot select a checkpoint")
        records[path.name] = {
            "checkpoint_sha256": checkpoint_hash,
            "validation_loss": loss,
            "epoch": state["epoch"],
            "round": state["round_index"],
            "global_step": state["global_step"],
        }
        logger.info("%s: validation_loss=%.6f", path.name, loss)
        # Keep partial scores for interrupted sweeps, without publishing a winner.
        write_json(report_path, {"identity": identity, "checkpoints": records, "complete": False})
    # Insertion order breaks ties in favor of earlier training progress.
    best = min(records, key=lambda name: records[name]["validation_loss"])
    report = {
        "identity": identity,
        "checkpoints": records,
        "complete": True,
        "best_checkpoint": best,
        "best_validation_loss": records[best]["validation_loss"],
    }
    write_json(report_path, report)
    # Keep the training-resume and original protocol-selection pointers untouched.
    atomic_write_text(manager.root / "best_validation_checkpoint.txt", f"{best}\n")
    logger.info("Best validation adapter: %s", manager.root / best / "adapter")
    return report


def validation_winner(
    manager: CheckpointManager,
    *,
    expected_kind: str | None = None,
    expected_seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve a completed sweep, fail closed on stale or modified adapters."""
    report = read_json(manager.root / "validation_selection.json")
    if not report.get("complete") or report["identity"]["config_hash"] != manager.config_hash:
        raise ValueError("validation selection is incomplete or has a different config")
    if (expected_kind is not None and report["identity"]["kind"] != expected_kind) or (
        expected_seed is not None and report["identity"]["seed"] != expected_seed
    ):
        raise ValueError("validation selection belongs to a different training family or seed")
    name = report["best_checkpoint"]
    pointer = (manager.root / "best_validation_checkpoint.txt").read_text().strip()
    if pointer != name:
        raise ValueError("validation winner pointer differs from the selection report; rerun sweep")
    path = manager.resolve(name)
    manager.verify(path)
    record = report["checkpoints"][name]
    if file_sha256(path / "hashes.json") != record["checkpoint_sha256"]:
        raise ValueError("selected checkpoint changed since validation")
    return name, {
        **record,
        "checkpoint": name,
        "validation_sha256": report["identity"]["validation_sha256"],
    }
