from __future__ import annotations

import argparse
import importlib.metadata
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .audit import audit_retrieval_cohort
from .auth import apply_hf_token, find_token_file
from .checkpointing import CheckpointManager
from .config import Config
from .data import (
    build_partition,
    load_native_dataset,
    load_partition,
    materialize_partition,
    resolve_hub_revision,
)
from .evaluation import ARMS, evaluate_arm
from .federated import adapter_state, set_adapter_state
from .io import read_json, write_json
from .leakage import assert_no_support_leakage
from .modeling import load_lora_bundle, resolve_model_revision
from .priors import leave_one_client_out_weakness, load_prediction_rows, read_priors, write_priors
from .reporting import build_contrast_report, read_predictions, write_contrast_report
from .workflows import train_centralized, train_federated, train_local_clients

_SHA = re.compile(r"^[0-9a-f]{40,64}$", re.IGNORECASE)


def _seal_config(path: str | Path) -> Config:
    config = Config.from_file(path)
    if not _SHA.fullmatch(config.data.revision):
        config.data.revision = resolve_hub_revision(config.dataset_id, config.data.revision)
    if not _SHA.fullmatch(config.model.revision):
        config.model.revision = resolve_model_revision(config.model.id, config.model.revision)
    if not _SHA.fullmatch(config.retrieval.encoder_revision):
        config.retrieval.encoder_revision = resolve_model_revision(
            config.retrieval.encoder_id, config.retrieval.encoder_revision
        )
    config.validate()
    output = Path(config.experiment.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sealed = output / "sealed_config.json"
    if sealed.exists():
        existing = Config.from_file(sealed)
        if existing.hash != config.hash:
            raise ValueError(
                f"sealed config differs from current config: {sealed}; use a new output_dir"
            )
    else:
        write_json(sealed, config.to_dict())
    return config


def _data_root(config: Config) -> Path:
    return Path(config.experiment.output_dir) / "data" / config.data.dataset


def _checkpoint_root(config: Config, family: str) -> Path:
    return Path(config.experiment.output_dir) / "checkpoints" / config.data.dataset / family


def command_prepare_data(args: argparse.Namespace) -> None:
    config = _seal_config(args.config)
    limits = {
        "train": config.data.max_train_samples,
        "validation": config.data.max_validation_samples,
        "test": config.data.max_test_samples,
    }
    splits = load_native_dataset(
        config.data.dataset,
        config.dataset_id,
        revision=config.data.revision,
        limits=limits,
    )
    assignments, weights = build_partition(
        splits,
        num_clients=config.data.num_clients,
        alpha=config.data.dirichlet_alpha,
        fit_ratio=config.fit_ratio,
        seed=config.experiment.data_seed,
        min_support_per_client=config.data.min_support_per_client,
    )
    root = _data_root(config)
    materialize_partition(
        root,
        splits,
        assignments,
        dataset_id=config.dataset_id,
        dataset_revision=config.data.revision,
        data_seed=config.experiment.data_seed,
        weights=weights,
        config_hash=config.hash,
    )
    clients = load_partition(root, expected_config_hash=config.hash)
    for values in clients.values():
        assert_no_support_leakage(
            values["support"],
            [*values["validation"], *values["test"]],
            lexical_threshold=config.retrieval.lexical_jaccard_threshold,
        )
    print(f"Prepared and audited {config.data.dataset} at {root}")


def command_audit_retrieval(args: argparse.Namespace) -> None:
    config = _seal_config(args.config)
    output = _data_root(config) / "retrieval_audit.json"
    result = audit_retrieval_cohort(
        config,
        load_partition(_data_root(config), expected_config_hash=config.hash),
        output,
    )
    print(
        f"Audited Top-{config.retrieval.top_k} capacity for {result['query_count']} queries; "
        f"manifest: {output}"
    )


def _requested_seeds(config: Config, args: argparse.Namespace) -> list[int]:
    if getattr(args, "all_seeds", False):
        return list(config.experiment.training_seeds)
    if args.seed is None:
        raise ValueError("provide --seed or --all-seeds")
    if args.seed not in config.experiment.training_seeds:
        raise ValueError(f"seed must be one of {config.experiment.training_seeds}")
    return [args.seed]


def command_train(args: argparse.Namespace) -> None:
    config = _seal_config(args.config)
    if args.fl_round is not None and args.mode != "centralized":
        raise ValueError("--fl-round is only valid for centralized training")
    clients = load_partition(_data_root(config), expected_config_hash=config.hash)
    client_fit = {client: values["fit"] for client, values in clients.items()}
    for seed in _requested_seeds(config, args):
        if args.mode == "local":
            telemetry = train_local_clients(
                config,
                client_fit,
                seed=seed,
                output_root=_checkpoint_root(config, "local"),
                resume=args.resume,
            )
        elif args.mode == "federated":
            trainer = train_federated(
                config,
                client_fit,
                seed=seed,
                output_root=_checkpoint_root(config, "federated"),
                resume=args.resume,
            )
            if trainer.final_state is None:
                raise RuntimeError("federated trainer returned without final state")
            telemetry = {
                "checkpoint_root": str(trainer.checkpoints.root),
                "total_wall_clock_seconds": trainer.final_state.elapsed_seconds,
                "total_target_exposures": trainer.final_state.target_exposures,
                "total_optimizer_updates": trainer.final_state.optimizer_updates,
                "effective_batch_size": config.training.train_micro_batch_size
                * config.training.gradient_accumulation_steps,
                "total_uplink_bytes": sum(
                    int(row["communication"]["uplink_bytes"]) for row in trainer.history
                ),
                "total_downlink_bytes": sum(
                    int(row["communication"]["downlink_bytes"]) for row in trainer.history
                ),
            }
        else:
            fl_round = args.fl_round or _selected_round(config)
            telemetry = train_centralized(
                config,
                client_fit,
                seed=seed,
                fl_rounds=fl_round,
                output_root=_checkpoint_root(config, "centralized"),
                resume=args.resume,
            )
        write_json(
            _checkpoint_root(config, args.mode) / f"seed-{seed}" / "telemetry.json",
            telemetry,
        )
        print(f"Completed {args.mode} training for seed {seed}")


def _selected_round(config: Config) -> int:
    path = _checkpoint_root(config, "federated") / "selected_round.json"
    if not path.exists():
        raise FileNotFoundError("no selected FL round; pass --fl-round or run select-round")
    return int(read_json(path)["round"])


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
            root = _checkpoint_root(config, "local") / f"seed-{seed}" / f"client-{client_id}"
            manager(root, keep=config.training.checkpoint_keep).load(
                "last", model=model, restore_rng=False
            )

        return bundle, before_client
    if spec.checkpoint_family == "federated":
        selected = round_index or _selected_round(config)
        root = _checkpoint_root(config, "federated") / f"seed-{seed}" / "global"
        manager(root).load(
            f"checkpoint-round-{selected:04d}", model=bundle.model, restore_rng=False
        )
        return bundle, None
    root = _checkpoint_root(config, "centralized") / f"seed-{seed}"
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


def _evaluation_output_dir(
    config: Config, arm: str, *, seed: int | None, split: str, round_index: int | None
) -> Path:
    return (
        Path(config.experiment.output_dir)
        / "evaluations"
        / config.data.dataset
        / arm
        / (f"seed-{seed}" if seed is not None else "deterministic")
        / split
        / (f"round-{round_index}" if round_index is not None else "selected")
    )


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
    clients = load_partition(_data_root(config), expected_config_hash=config.hash)
    prior_path = subject_weights
    if arm == "F2" and prior_path is None:
        selected = round_index or _selected_round(config)
        prior_path = str(
            Path(config.experiment.output_dir)
            / "priors"
            / config.data.dataset
            / f"seed-{seed}"
            / f"round-{selected}.json"
        )
    priors = read_priors(prior_path) if prior_path else None
    return evaluate_arm(
        bundle,
        config,
        clients,
        arm=arm,
        split=split,
        output_dir=_evaluation_output_dir(
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
        output = _evaluation_output_dir(
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
    config = _seal_config(args.config)
    arm = args.arm.upper()
    accuracies = _sweep_arm(
        config, arm, split=args.split, round_index=args.round, force=args.force
    )
    mean = sum(accuracies) / len(accuracies)
    print(f"{arm} {args.split} mean over {len(accuracies)} run(s): {mean:.6f}")


def command_evaluate_all(args: argparse.Namespace) -> None:
    config = _seal_config(args.config)
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
    config = _seal_config(args.config)
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
    clients = load_partition(_data_root(config), expected_config_hash=config.hash)
    prior_path = args.subject_weights
    if arm == "F2" and prior_path is None:
        selected = args.round or _selected_round(config)
        prior_path = str(
            Path(config.experiment.output_dir)
            / "priors"
            / config.data.dataset
            / f"seed-{seed}"
            / f"round-{selected}.json"
        )
    priors = read_priors(prior_path) if prior_path else None
    summary = evaluate_arm(
        bundle,
        config,
        clients,
        arm=arm,
        split=args.split,
        output_dir=_evaluation_output_dir(
            config, arm, seed=seed, split=args.split, round_index=args.round
        ),
        seed=seed,
        before_client=before_client,
        subject_weights=priors,
    )
    print(f"{arm} {args.split} pipeline accuracy: {summary['pipeline_accuracy']:.6f}")


def command_select_round(args: argparse.Namespace) -> None:
    config = _seal_config(args.config)
    scores: dict[int, float] = {}
    per_seed: dict[str, dict[str, float]] = {}
    for round_index in config.training.fl_round_candidates:
        round_scores: list[float] = []
        for seed in config.experiment.training_seeds:
            summary = read_json(
                Path(config.experiment.output_dir)
                / "evaluations"
                / config.data.dataset
                / "F0"
                / f"seed-{seed}"
                / "validation"
                / f"round-{round_index}"
                / "summary.json"
            )
            value = float(summary["pipeline_accuracy"])
            round_scores.append(value)
            per_seed.setdefault(str(seed), {})[str(round_index)] = value
        scores[round_index] = sum(round_scores) / len(round_scores)
    selected = max(sorted(scores), key=lambda value: scores[value])
    for seed in config.experiment.training_seeds:
        manager = CheckpointManager(
            _checkpoint_root(config, "federated") / f"seed-{seed}" / "global",
            config_hash=config.hash,
            model_id=config.model.id,
            model_revision=config.model.revision,
        )
        manager.mark_best(f"checkpoint-round-{selected:04d}")
    write_json(
        _checkpoint_root(config, "federated") / "selected_round.json",
        {"round": selected, "mean_validation_accuracy": scores, "per_seed": per_seed},
    )
    print(f"Selected global round {selected}: mean validation accuracy {scores[selected]:.6f}")


def command_build_priors(args: argparse.Namespace) -> None:
    config = _seal_config(args.config)
    if args.seed not in config.experiment.training_seeds:
        raise ValueError(f"seed must be one of {config.experiment.training_seeds}")
    round_index = args.round or _selected_round(config)
    predictions = (
        Path(config.experiment.output_dir)
        / "evaluations"
        / config.data.dataset
        / "F0"
        / f"seed-{args.seed}"
        / "validation"
        / f"round-{round_index}"
        / "predictions.jsonl"
    )
    priors = leave_one_client_out_weakness(
        load_prediction_rows([predictions]), num_clients=config.data.num_clients
    )
    output = (
        Path(config.experiment.output_dir)
        / "priors"
        / config.data.dataset
        / f"seed-{args.seed}"
        / f"round-{round_index}.json"
    )
    write_priors(output, priors)
    print(f"Wrote leave-one-client-out prior to {output}")


def command_report(args: argparse.Namespace) -> None:
    config = _seal_config(args.config)
    root = Path(config.experiment.output_dir) / "evaluations" / config.data.dataset
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
    output = Path(config.experiment.output_dir) / "reports" / config.data.dataset / "contrasts.json"
    write_contrast_report(output, report)
    print(f"Wrote primary contrast report to {output}")


def _cuda_failure_reason(*, torch_version: str, cuda_build: str | None) -> str:
    """Explain why CUDA is unavailable, distinguishing the two very different causes.

    A CPU-only wheel and a driver problem both surface as is_available() == False, but
    they need opposite fixes. On Windows the default PyPI torch wheel carries no CUDA at
    all, while on Linux it does, so the same dependency pin produces different builds per
    platform and this is the failure a Windows setup hits first.
    """
    if cuda_build is None:
        return (
            f"PyTorch {torch_version} is a CPU-only build with no CUDA support. "
            "The default PyPI wheel for Windows excludes CUDA; reinstall from the "
            "PyTorch index, for example: uv pip install torch --force-reinstall "
            "--index-url https://download.pytorch.org/whl/cu128"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    scope = (
        f" This process is restricted to CUDA_VISIBLE_DEVICES={visible!r}, so check that "
        "--gpu names a device that exists."
        if visible is not None
        else " Check the driver with nvidia-smi."
    )
    return (
        f"PyTorch {torch_version} was built against CUDA {cuda_build} but no device is "
        f"usable, which points at the driver rather than the install.{scope}"
    )


def command_doctor(args: argparse.Namespace) -> None:
    """Fail-fast hardware/dependency check before a costly experiment run."""
    config = Config.from_file(args.config)
    config.validate()
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed in the active environment") from exc
    if config.hardware.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            _cuda_failure_reason(
                torch_version=torch.__version__, cuda_build=torch.version.cuda
            )
        )

    packages = {}
    for package in ("torch", "transformers", "peft", "datasets", "sentence-transformers"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "missing"
    if "missing" in packages.values():
        missing = [name for name, version in packages.items() if version == "missing"]
        raise RuntimeError(f"missing runtime packages: {', '.join(missing)}")

    token_file = find_token_file()
    report: dict[str, Any] = {
        "packages": packages,
        "torch_build": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        # Presence and origin only. The token value must never be printed or written
        # to any artifact, since this project hashes and seals its provenance records.
        "hf_token": {
            "source": apply_hf_token(),
            "file": str(token_file) if token_file else None,
        },
        "model": config.model.id,
        "dtype": config.model.dtype,
        "attention": config.model.attention,
        "max_seq_length": config.model.max_seq_length,
        "train_effective_batch_size": config.training.train_micro_batch_size
        * config.training.gradient_accumulation_steps,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        report["gpu"] = {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "vram_gib": round(properties.total_memory / 1024**3, 2),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        }
        if config.hardware.bf16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("the selected GPU/PyTorch build does not support BF16")
    import json

    print(json.dumps(report, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fedicl-mqa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared so --gpu can be written after the subcommand rather than before it.
    gpu = argparse.ArgumentParser(add_help=False)
    gpu.add_argument(
        "--gpu",
        type=int,
        choices=[0, 1],
        help="run on this physical GPU; leaves the sealed config hash untouched",
    )

    prepare = subparsers.add_parser(
        "prepare-data", parents=[gpu], help="download, normalize and partition data"
    )
    prepare.add_argument("--config", required=True)
    prepare.set_defaults(func=command_prepare_data)

    audit = subparsers.add_parser(
        "audit-retrieval",
        parents=[gpu],
        help="verify Top-5 capacity and freeze candidate manifests",
    )
    audit.add_argument("--config", required=True)
    audit.set_defaults(func=command_audit_retrieval)

    training = subparsers.add_parser(
        "train", parents=[gpu], help="train Local, Federated or Central LoRA"
    )
    training.add_argument("--config", required=True)
    training.add_argument("--mode", choices=["local", "federated", "centralized"], required=True)
    seed_group = training.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument("--all-seeds", action="store_true")
    training.add_argument("--fl-round", type=int, choices=[4, 6, 8])
    training.add_argument("--resume", default="auto")
    training.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("evaluate", parents=[gpu], help="run one baseline arm")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--arm", choices=sorted(ARMS), required=True)
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--split", choices=["validation", "test"], default="test")
    evaluate.add_argument("--round", type=int, choices=[4, 6, 8])
    evaluate.add_argument("--subject-weights")
    evaluate.set_defaults(func=command_evaluate)

    select = subparsers.add_parser(
        "select-round", parents=[gpu], help="select FL round using validation only"
    )
    select.add_argument("--config", required=True)
    select.set_defaults(func=command_select_round)

    priors = subparsers.add_parser(
        "build-priors", parents=[gpu], help="build F2 leave-one-client-out prior"
    )
    priors.add_argument("--config", required=True)
    priors.add_argument("--seed", type=int, required=True)
    priors.add_argument("--round", type=int, choices=[4, 6, 8])
    priors.set_defaults(func=command_build_priors)

    report = subparsers.add_parser(
        "report", parents=[gpu], help="bootstrap the six primary contrasts"
    )
    report.add_argument("--config", required=True)
    report.set_defaults(func=command_report)

    sweep = subparsers.add_parser(
        "evaluate-arm",
        parents=[gpu],
        help="evaluate one arm across every configured seed on the test split",
    )
    sweep.add_argument("--config", required=True)
    sweep.add_argument("--arm", choices=sorted(ARMS), required=True)
    sweep.add_argument("--split", choices=["validation", "test"], default="test")
    sweep.add_argument("--round", type=int, choices=[4, 6, 8])
    sweep.add_argument("--force", action="store_true", help="re-run evaluations already on disk")
    sweep.set_defaults(func=command_evaluate_arm)

    sweep_all = subparsers.add_parser(
        "evaluate-all",
        parents=[gpu],
        help="evaluate every arm across every configured seed on the test split",
    )
    sweep_all.add_argument("--config", required=True)
    sweep_all.add_argument("--split", choices=["validation", "test"], default="test")
    sweep_all.add_argument("--round", type=int, choices=[4, 6, 8])
    sweep_all.add_argument(
        "--force", action="store_true", help="re-run evaluations already on disk"
    )
    sweep_all.set_defaults(func=command_evaluate_all)

    doctor = subparsers.add_parser(
        "doctor", parents=[gpu], help="check the local GPU and ML dependencies"
    )
    doctor.add_argument("--config", required=True)
    doctor.set_defaults(func=command_doctor)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        # Restrict the process to one GPU before anything creates a CUDA context. The
        # selected device then appears as cuda:0, so config.hardware.device stays the
        # literal string "cuda" and the sealed config hash is unaffected.
        if getattr(args, "gpu", None) is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        # Authenticate before any command reaches the Hub. Every Hub client in this
        # project reads the token from the environment, so this single call covers
        # model, dataset and encoder downloads alike.
        apply_hf_token()
        args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
