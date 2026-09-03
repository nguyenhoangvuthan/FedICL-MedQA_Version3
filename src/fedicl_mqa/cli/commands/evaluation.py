"""Baseline arms: single runs, per-arm sweeps and the contrast report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fedicl_mqa.training.checkpointing import CheckpointManager
from fedicl_mqa.core.config import Config
from fedicl_mqa.data.preparation import (
    build_partition,
    load_native_dataset,
    load_partition,
    materialize_partition,
    resolve_hub_revision,
)
from fedicl_mqa.evaluation.arms import ARMS, evaluate_arm
from fedicl_mqa.training.federated import adapter_state, set_adapter_state
from fedicl_mqa.core.io import read_json, write_json
from fedicl_mqa.modeling.loader import load_lora_bundle, resolve_model_revision
from fedicl_mqa.evaluation.priors import read_priors
from fedicl_mqa.evaluation.reporting import (
    build_contrast_report,
    read_predictions,
    write_contrast_report,
)
from fedicl_mqa.cli.paths import (
    arms_root,
    checkpoint_root,
    data_root,
    evaluation_dir,
    priors_path,
    report_path,
    seal_config,
    selected_round,
    summary_path,
)

def _load_arm_checkpoint(
    config: Config,
    arm: str,
    seed: int | None,
    round_index: int | None,
) -> tuple[Any, Any]:
    runtime_seed = seed if seed is not None else config.experiment.data_seed
    bundle = load_lora_bundle(config, seed=runtime_seed)
    initial = adapter_state(bundle.model)
    spec = ARMS[arm]

    def manager(root: Path, *, keep: int | None = None) -> CheckpointManager:
        return CheckpointManager(
            root,
            config_hash=config.hash,
            model_id=config.model.id,
            model_revision=config.model.revision,
            keep=keep,
        )

    if spec.checkpoint_family == "base":

        def before_client(client_id: int, model: Any) -> None:
            del client_id
            set_adapter_state(model, initial)

        return bundle, before_client
    if seed is None:
        raise ValueError(f"arm {arm} requires --seed")
    if spec.checkpoint_family == "local":

        def before_client(client_id: int, model: Any) -> None:
            root = checkpoint_root(config, "local") / f"seed-{seed}" / f"client-{client_id}"
            manager(root, keep=config.training.checkpoint_keep).load(
                "last", model=model, restore_rng=False
            )

        return bundle, before_client
    if spec.checkpoint_family == "federated":
        selected = round_index or selected_round(config)
        root = checkpoint_root(config, "federated") / f"seed-{seed}" / "global"
        manager(root).load(
            f"checkpoint-round-{selected:04d}", model=bundle.model, restore_rng=False
        )
        return bundle, None
    root = checkpoint_root(config, "centralized") / f"seed-{seed}"
    manager(root, keep=config.training.checkpoint_keep).load(
        "last", model=bundle.model, restore_rng=False
    )
    return bundle, None


def _effective_seeds(config: Config, arm: str) -> list[int | None]:
    """Seeds an arm is actually evaluated over.

    Base arms (B0/B1) do not consume a trained checkpoint, so every seed would produce
    an identical result; they run exactly once and record no seed. Every other family
    runs once per configured training seed.
    """
    if ARMS[arm].checkpoint_family == "base":
        return [None]
    return list(config.experiment.training_seeds)


def _run_single_evaluation(
    config: Config,
    arm: str,
    *,
    seed: int | None,
    split: str,
    round_index: int | None,
    subject_weights: str | None = None,
) -> dict[str, Any]:
    bundle, before_client = _load_arm_checkpoint(config, arm, seed, round_index)
    clients = load_partition(data_root(config), expected_config_hash=config.hash)
    prior_path = subject_weights
    if arm == "F2" and prior_path is None:
        selected = round_index or selected_round(config)
        prior_path = str(priors_path(config, seed=seed, round_index=selected))
    priors = read_priors(prior_path) if prior_path else None
    return evaluate_arm(
        bundle,
        config,
        clients,
        arm=arm,
        split=split,
        output_dir=evaluation_dir(
            config, arm, seed=seed, split=split, round_index=round_index
        ),
        seed=seed,
        before_client=before_client,
        subject_weights=priors,
    )


def _sweep_arm(
    config: Config, arm: str, *, split: str, round_index: int | None, force: bool
) -> list[float]:
    """Evaluate one arm over its effective seed set, skipping completed runs.

    evaluate_arm writes summary.json last, after predictions.jsonl, so its presence
    marks a run that finished rather than one that was interrupted part-way.
    """
    # --round only means anything for the federated family; ignore it elsewhere so a
    # single evaluate-all invocation can carry the flag without failing on B0/L0/C0.
    effective_round = round_index if ARMS[arm].checkpoint_family == "federated" else None
    accuracies: list[float] = []
    for seed in _effective_seeds(config, arm):
        label = f"seed-{seed}" if seed is not None else "deterministic"
        output = evaluation_dir(
            config, arm, seed=seed, split=split, round_index=effective_round
        )
        summary_path = output / "summary.json"
        if summary_path.exists() and not force:
            accuracy = float(read_json(summary_path)["pipeline_accuracy"])
            print(f"{arm} {label} {split}: skip, already evaluated, accuracy {accuracy:.6f}")
        else:
            summary = _run_single_evaluation(
                config, arm, seed=seed, split=split, round_index=effective_round
            )
            accuracy = float(summary["pipeline_accuracy"])
            print(f"{arm} {label} {split}: pipeline accuracy {accuracy:.6f}")
        accuracies.append(accuracy)
    return accuracies


def command_evaluate_arm(args: argparse.Namespace) -> None:
    config = seal_config(args.config)
    arm = args.arm.upper()
    accuracies = _sweep_arm(
        config, arm, split=args.split, round_index=args.round, force=args.force
    )
    mean = sum(accuracies) / len(accuracies)
    print(f"{arm} {args.split} mean over {len(accuracies)} run(s): {mean:.6f}")


def command_evaluate_all(args: argparse.Namespace) -> None:
    config = seal_config(args.config)
    means: dict[str, float] = {}
    for arm in sorted(ARMS):
        accuracies = _sweep_arm(
            config, arm, split=args.split, round_index=args.round, force=args.force
        )
        means[arm] = sum(accuracies) / len(accuracies)
    print(f"\n{args.split} pipeline accuracy by arm")
    for arm, mean in means.items():
        print(f"  {arm}: {mean:.6f}")


def command_evaluate(args: argparse.Namespace) -> None:
    config = seal_config(args.config)
    arm = args.arm.upper()
    family = ARMS[arm].checkpoint_family
    if family == "base" and args.seed is not None:
        raise ValueError(f"arm {arm} is deterministic and does not accept --seed")
    if family != "base" and args.seed is None:
        raise ValueError(f"arm {arm} requires --seed")
    if args.seed is not None and args.seed not in config.experiment.training_seeds:
        raise ValueError(f"seed must be one of {config.experiment.training_seeds}")
    if args.round is not None and family != "federated":
        raise ValueError("--round is only valid for F0/F1/F2")
    seed = None if family == "base" else args.seed
    bundle, before_client = _load_arm_checkpoint(config, arm, seed, args.round)
    clients = load_partition(data_root(config), expected_config_hash=config.hash)
    prior_path = args.subject_weights
    if arm == "F2" and prior_path is None:
        selected = args.round or selected_round(config)
        prior_path = str(priors_path(config, seed=seed, round_index=selected))
    priors = read_priors(prior_path) if prior_path else None
    summary = evaluate_arm(
        bundle,
        config,
        clients,
        arm=arm,
        split=args.split,
        output_dir=evaluation_dir(
            config, arm, seed=seed, split=args.split, round_index=args.round
        ),
        seed=seed,
        before_client=before_client,
        subject_weights=priors,
    )
    print(f"{arm} {args.split} pipeline accuracy: {summary['pipeline_accuracy']:.6f}")


def command_report(args: argparse.Namespace) -> None:
    config = seal_config(args.config)
    root = arms_root(config)
    arm_predictions: dict[str, dict[int | None, Any]] = {}
    for arm in ARMS:
        family = ARMS[arm].checkpoint_family
        if family == "base":
            path = root / arm / "deterministic" / "test" / "selected" / "predictions.jsonl"
            arm_predictions[arm] = {None: read_predictions(path)}
        else:
            arm_predictions[arm] = {
                seed: read_predictions(
                    root / arm / f"seed-{seed}" / "test" / "selected" / "predictions.jsonl"
                )
                for seed in config.experiment.training_seeds
            }
    report = build_contrast_report(
        arm_predictions,
        samples=config.evaluation.bootstrap_samples,
        confidence=config.evaluation.confidence_level,
        bootstrap_seed=config.experiment.data_seed,
    )
    output = report_path(config)
    write_contrast_report(output, report)
    print(f"Wrote primary contrast report to {output}")

