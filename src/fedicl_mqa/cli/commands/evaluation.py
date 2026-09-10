"""Baseline arms: single runs, per-arm sweeps and the contrast report."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from fedicl_mqa.cli import paths, results
from fedicl_mqa.cli.paths import (
    arms_root,
    checkpoint_root,
    data_root,
    evaluation_dir,
    priors_path,
    report_path,
    seal_config,
    selected_round,
)
from fedicl_mqa.core.config import Config
from fedicl_mqa.core.io import file_sha256, read_json
from fedicl_mqa.data.preparation import (
    load_partition,
)
from fedicl_mqa.evaluation.arms import ARMS, active_arms, evaluate_arm
from fedicl_mqa.evaluation.priors import read_priors
from fedicl_mqa.evaluation.reporting import (
    build_contrast_report,
    read_predictions,
    write_contrast_report,
)
from fedicl_mqa.modeling.loader import load_lora_bundle
from fedicl_mqa.training.checkpointing import CheckpointManager
from fedicl_mqa.training.context import bind_protocol, load_plan, training_inputs
from fedicl_mqa.training.federated import adapter_state, set_adapter_state


def _load_arm_checkpoint(
    config: Config,
    arm: str,
    seed: int | None,
    round_index: int | None,
) -> tuple[Any, Any]:
    if arm not in active_arms(config):
        raise ValueError(f"arm {arm} is not enabled by this configuration")
    plan = load_plan(config) if config.icl_training is not None else None
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
    if spec.checkpoint_family in {"local", "local-matched", "local-icl"}:
        matched = spec.checkpoint_family == "local-matched"
        rounds = selected_round(config) if matched else 1
        family_root = (
            paths.matched_local_root(config, rounds)
            if matched
            else checkpoint_root(config, spec.checkpoint_family)
        )
        partition = _training_partition(config) if matched or config.controls is not None else None

        def before_client(client_id: int, model: Any) -> None:
            root = family_root / f"seed-{seed}" / f"client-{client_id}"
            bind_protocol(
                config, root.parent, plan, train_icl=spec.checkpoint_family == "local-icl"
            )
            loaded = manager(root, keep=config.training.checkpoint_keep).load(
                "last", model=model, restore_rng=False
            )
            if partition is not None:
                state = loaded.trainer_state
                epochs = rounds * config.training.local_epochs
                if (
                    state.epoch != epochs
                    or state.batch_in_epoch != 0
                    or state.target_exposures != len(partition[client_id]["fit"]) * epochs
                ):
                    raise ValueError(f"{arm}: Local checkpoint has an incomplete training budget")

        return bundle, before_client
    if spec.checkpoint_family in {"federated", "federated-icl"}:
        selected = round_index or selected_round(config)
        root = checkpoint_root(config, spec.checkpoint_family) / f"seed-{seed}" / "global"
        bind_protocol(
            config, root.parent, plan, train_icl=spec.checkpoint_family == "federated-icl"
        )
        loaded = manager(root).load(
            f"checkpoint-round-{selected:04d}", model=bundle.model, restore_rng=False
        )
        if config.controls is not None:
            partition = _training_partition(config)
            exposures = sum(len(v["fit"]) for v in partition.values()) * selected
            if (
                loaded.trainer_state.round_index != selected
                or loaded.trainer_state.target_exposures != exposures
            ):
                raise ValueError("federated checkpoint does not match the selected training budget")
        return bundle, None
    root = checkpoint_root(config, "centralized") / f"seed-{seed}"
    bind_protocol(config, root, plan, train_icl=False)
    loaded = manager(root, keep=config.training.checkpoint_keep).load(
        "last", model=bundle.model, restore_rng=False
    )
    if config.controls is not None:
        epochs = selected_round(config) * config.training.local_epochs
        partition = _training_partition(config)
        exposures = sum(len(v["fit"]) for v in partition.values()) * epochs
        state = loaded.trainer_state
        if state.epoch != epochs or state.batch_in_epoch or state.target_exposures != exposures:
            raise ValueError("centralized checkpoint does not match the selected training budget")
    return bundle, None


def _training_partition(config: Config):
    clients = load_partition(data_root(config), expected_config_hash=config.hash)
    fit, _, _ = training_inputs(config, clients)
    return {c: {**roles, "fit": fit[c]} for c, roles in clients.items()}


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
    clients = load_partition(data_root(config), expected_config_hash=config.hash)
    prior_path = subject_weights
    if config.controls is not None and subject_weights is not None:
        raise ValueError("controlled arms require the audited validation prior, not an override")
    if ARMS[arm].client_aware and prior_path is None:
        selected = round_index or selected_round(config)
        prior_path = str(priors_path(config, seed=seed, round_index=selected))
    metadata: dict[str, Any] = {}
    if config.icl_training is not None:
        metadata["training_plan_sha256"] = load_plan(config)["sha256"]
    if config.controls is not None:
        metadata["selected_round"] = round_index or selected_round(config)
        if prior_path:
            audit = verified_prior_metadata(
                config, seed=seed, round_index=metadata["selected_round"]
            )
            metadata["prior_sha256"] = audit["prior_sha256"]
    priors = read_priors(prior_path) if prior_path else None
    bundle, before_client = _load_arm_checkpoint(config, arm, seed, round_index)
    return evaluate_arm(
        bundle,
        config,
        clients,
        arm=arm,
        split=split,
        output_dir=evaluation_dir(config, arm, seed=seed, split=split, round_index=round_index),
        seed=seed,
        before_client=before_client,
        subject_weights=priors,
        run_metadata=metadata,
    )


def _sweep_arm(
    config: Config, arm: str, *, split: str, round_index: int | None, force: bool
) -> list[float]:
    """Evaluate one arm over its effective seed set, skipping completed runs.

    evaluate_arm writes summary.json last, after predictions.jsonl, so its presence
    marks a run that finished rather than one that was interrupted part-way.

    Every run is traced into the arm's own log, and the cross-arm table is rebuilt once
    the arm finishes, so a long sweep stays readable while it is still going.
    """
    # --round only means anything for the federated family; ignore it elsewhere so a
    # single evaluate-all invocation can carry the flag without failing on B0/L0/C0.
    effective_round = round_index if ARMS[arm].checkpoint_family == "federated" else None
    accuracies: list[float] = []
    for seed in _effective_seeds(config, arm):
        label = paths.seed_label(seed)
        finished = paths.summary_path(
            config, arm, seed=seed, split=split, round_index=effective_round
        )
        started = time.perf_counter()
        if finished.exists() and not force:
            summary = read_json(finished)
            if config.controls is not None:
                validate_summary_identity(
                    config, summary, arm=arm, seed=seed, split=split, round_index=effective_round
                )
            accuracy = float(summary["pipeline_accuracy"])
            print(f"{arm} {label} {split}: skip, already evaluated, accuracy {accuracy:.6f}")
            status = "skipped"
        else:
            try:
                summary = _run_single_evaluation(
                    config, arm, seed=seed, split=split, round_index=effective_round
                )
            except Exception as exc:
                # Trace the failure before it propagates, or the log would end at the
                # last success and say nothing about why the sweep stopped here.
                results.record_arm_run(
                    config,
                    arm,
                    seed=seed,
                    split=split,
                    status="failed",
                    round_index=effective_round,
                    duration_seconds=time.perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            accuracy = float(summary["pipeline_accuracy"])
            print(f"{arm} {label} {split}: pipeline accuracy {accuracy:.6f}")
            status = "completed"
        results.record_arm_run(
            config,
            arm,
            seed=seed,
            split=split,
            status=status,
            accuracy=accuracy,
            round_index=effective_round,
            duration_seconds=time.perf_counter() - started,
        )
        accuracies.append(accuracy)
    results.update_comparison(config, split=split)
    return accuracies


def command_evaluate_arm(args: argparse.Namespace) -> None:
    config = seal_config(args.config)
    arm = args.arm.upper()
    accuracies = _sweep_arm(config, arm, split=args.split, round_index=args.round, force=args.force)
    mean = sum(accuracies) / len(accuracies)
    print(f"{arm} {args.split} mean over {len(accuracies)} run(s): {mean:.6f}")
    print()
    print(results.render_comparison(results.update_comparison(config, split=args.split)))


def command_evaluate_all(args: argparse.Namespace) -> None:
    config = seal_config(args.config)
    means: dict[str, float] = {}
    for arm in active_arms(config):
        accuracies = _sweep_arm(
            config, arm, split=args.split, round_index=args.round, force=args.force
        )
        means[arm] = sum(accuracies) / len(accuracies)
    print()
    print(results.render_comparison(results.update_comparison(config, split=args.split)))
    print(f"Comparison table: {paths.comparison_path(config, 'md')}")


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
    summary = _run_single_evaluation(
        config,
        arm,
        seed=seed,
        split=args.split,
        round_index=args.round,
        subject_weights=args.subject_weights,
    )
    print(f"{arm} {args.split} pipeline accuracy: {summary['pipeline_accuracy']:.6f}")


def command_report(args: argparse.Namespace) -> None:
    config = seal_config(args.config)
    root = arms_root(config)
    arm_predictions: dict[str, dict[int | None, Any]] = {}
    cohort = None
    source_hashes = {}
    if config.controls is not None:
        clients = load_partition(data_root(config), expected_config_hash=config.hash)
        cohort = {q.example_id: (c, q) for c, roles in clients.items() for q in roles["test"]}
    for arm in active_arms(config):
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
        if config.controls is not None:
            for seed in _effective_seeds(config, arm):
                summary = read_json(
                    paths.summary_path(config, arm, seed=seed, split="test", round_index=None)
                )
                validate_summary_identity(
                    config, summary, arm=arm, seed=seed, split="test", round_index=None
                )
                if len(arm_predictions[arm][seed]) != summary["n"]:
                    raise ValueError(f"{arm}/{seed}: prediction count differs from summary")
                predictions = arm_predictions[arm][seed]
                if {p.example_id for p in predictions} != cohort.keys() or len(predictions) != len(
                    cohort
                ):
                    raise ValueError(f"{arm}/{seed}: test cohort differs from prepared data")
                for p in predictions:
                    c, q = cohort[p.example_id]
                    if (p.gold, p.client_id, p.subject, p.seed) != (q.label, c, q.subject, seed):
                        raise ValueError(f"{arm}/{seed}: test prediction metadata differs")
                source_hashes[f"{arm}/{paths.seed_label(seed)}"] = summary["predictions_sha256"]
    report = build_contrast_report(
        arm_predictions,
        samples=config.evaluation.bootstrap_samples,
        confidence=config.evaluation.confidence_level,
        bootstrap_seed=config.experiment.data_seed,
        controlled=config.controls is not None,
        train_icl=config.icl_training is not None,
    )
    output = report_path(config)
    if config.controls is not None:
        report["protocol"] = {
            "config_hash": config.hash,
            "selected_round": selected_round(config),
            "test_source": "official MedMCQA validation",
            "prediction_sha256": source_hashes,
        }
    write_contrast_report(output, report)
    print(f"Wrote primary contrast report to {output}")


def validate_summary_identity(
    config: Config,
    summary: dict[str, Any],
    *,
    arm: str,
    seed: int | None,
    split: str,
    round_index: int | None,
) -> None:
    expected = (config.hash, arm, seed, split, round_index or selected_round(config))
    actual = (
        summary.get("config_hash"),
        summary.get("arm"),
        summary.get("seed"),
        summary.get("split"),
        summary.get("run_metadata", {}).get("selected_round"),
    )
    if actual != expected:
        raise ValueError(f"{arm}/{seed}: stale or incompatible summary; rerun with --force")
    if config.icl_training is not None:
        if (
            summary.get("run_metadata", {}).get("training_plan_sha256")
            != load_plan(config)["sha256"]
        ):
            raise ValueError("evaluation training cohort differs from the frozen plan")
    predictions = (
        evaluation_dir(config, arm, seed=seed, split=split, round_index=round_index)
        / "predictions.jsonl"
    )
    if file_sha256(predictions) != summary.get("predictions_sha256"):
        raise ValueError(f"{arm}/{seed}: predictions hash differs from summary")
    if ARMS[arm].client_aware:
        audit = verified_prior_metadata(config, seed=seed, round_index=expected[-1])
        if summary["run_metadata"].get("prior_sha256") != audit["prior_sha256"]:
            raise ValueError(f"{arm}/{seed}: evaluation prior has changed")


def verified_prior_metadata(config: Config, *, seed: int, round_index: int) -> dict[str, Any]:
    prior = priors_path(config, seed=seed, round_index=round_index)
    audit = read_json(prior.with_suffix(".audit.json"))
    if (
        audit.get("config_hash"),
        audit.get("seed"),
        audit.get("round"),
        audit.get("source_split"),
    ) != (config.hash, seed, round_index, "validation"):
        raise ValueError("prior audit identity does not match this evaluation")
    source = (
        evaluation_dir(config, "F0", seed=seed, split="validation", round_index=round_index)
        / "predictions.jsonl"
    )
    for path, key in [
        (prior, "prior_sha256"),
        (source, "predictions_sha256"),
        (source.with_name("summary.json"), "summary_sha256"),
    ]:
        if file_sha256(path) != audit.get(key):
            raise ValueError(f"prior provenance hash mismatch: {path}")
    return audit
