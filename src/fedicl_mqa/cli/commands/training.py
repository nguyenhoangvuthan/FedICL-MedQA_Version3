"""LoRA training, FL round selection and the F2 prior."""

from __future__ import annotations

import argparse
from pathlib import Path

from fedicl_mqa.training.checkpointing import CheckpointManager
from fedicl_mqa.core.config import Config
from fedicl_mqa.data.preparation import (
    build_partition,
    load_native_dataset,
    load_partition,
    materialize_partition,
    resolve_hub_revision,
)
from fedicl_mqa.core.io import read_json, write_json
from fedicl_mqa.evaluation.priors import (
    leave_one_client_out_weakness,
    load_prediction_rows,
    write_priors,
)
from fedicl_mqa.training.workflows import train_centralized, train_federated, train_local_clients
from fedicl_mqa.cli.paths import (
    checkpoint_root,
    data_root,
    evaluation_dir,
    priors_path,
    seal_config,
    selected_round,
    summary_path,
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
    config = seal_config(args.config)
    if args.fl_round is not None and args.mode != "centralized":
        raise ValueError("--fl-round is only valid for centralized training")
    clients = load_partition(data_root(config), expected_config_hash=config.hash)
    client_fit = {client: values["fit"] for client, values in clients.items()}
    for seed in _requested_seeds(config, args):
        if args.mode == "local":
            telemetry = train_local_clients(
                config,
                client_fit,
                seed=seed,
                output_root=checkpoint_root(config, "local"),
                resume=args.resume,
            )
        elif args.mode == "federated":
            trainer = train_federated(
                config,
                client_fit,
                seed=seed,
                output_root=checkpoint_root(config, "federated"),
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
            fl_round = args.fl_round or selected_round(config)
            telemetry = train_centralized(
                config,
                client_fit,
                seed=seed,
                fl_rounds=fl_round,
                output_root=checkpoint_root(config, "centralized"),
                resume=args.resume,
            )
        write_json(
            checkpoint_root(config, args.mode) / f"seed-{seed}" / "telemetry.json",
            telemetry,
        )
        print(f"Completed {args.mode} training for seed {seed}")


def command_select_round(args: argparse.Namespace) -> None:
    config = seal_config(args.config)
    scores: dict[int, float] = {}
    per_seed: dict[str, dict[str, float]] = {}
    for round_index in config.training.fl_round_candidates:
        round_scores: list[float] = []
        for seed in config.experiment.training_seeds:
            summary = read_json(
                summary_path(
                    config,
                    "F0",
                    seed=seed,
                    split="validation",
                    round_index=round_index,
                )
            )
            value = float(summary["pipeline_accuracy"])
            round_scores.append(value)
            per_seed.setdefault(str(seed), {})[str(round_index)] = value
        scores[round_index] = sum(round_scores) / len(round_scores)
    selected = max(sorted(scores), key=lambda value: scores[value])
    for seed in config.experiment.training_seeds:
        manager = CheckpointManager(
            checkpoint_root(config, "federated") / f"seed-{seed}" / "global",
            config_hash=config.hash,
            model_id=config.model.id,
            model_revision=config.model.revision,
        )
        manager.mark_best(f"checkpoint-round-{selected:04d}")
    write_json(
        checkpoint_root(config, "federated") / "selected_round.json",
        {"round": selected, "mean_validation_accuracy": scores, "per_seed": per_seed},
    )
    print(f"Selected global round {selected}: mean validation accuracy {scores[selected]:.6f}")


def command_build_priors(args: argparse.Namespace) -> None:
    config = seal_config(args.config)
    if args.seed not in config.experiment.training_seeds:
        raise ValueError(f"seed must be one of {config.experiment.training_seeds}")
    round_index = args.round or selected_round(config)
    predictions = (
        evaluation_dir(
            config, "F0", seed=args.seed, split="validation", round_index=round_index
        )
        / "predictions.jsonl"
    )
    priors = leave_one_client_out_weakness(
        load_prediction_rows([predictions]), num_clients=config.data.num_clients
    )
    output = priors_path(config, seed=args.seed, round_index=round_index)
    write_priors(output, priors)
    print(f"Wrote leave-one-client-out prior to {output}")

