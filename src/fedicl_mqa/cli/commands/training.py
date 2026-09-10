"""LoRA training, FL round selection and the F2 prior."""

from __future__ import annotations

import argparse

from fedicl_mqa.cli.paths import (
    checkpoint_root,
    data_root,
    evaluation_dir,
    matched_local_root,
    priors_path,
    seal_config,
    selected_round,
    summary_path,
)
from fedicl_mqa.core.config import Config
from fedicl_mqa.core.io import file_sha256, read_json, write_json
from fedicl_mqa.data.preparation import (
    load_partition,
)
from fedicl_mqa.evaluation.priors import (
    controlled_weakness,
    leave_one_client_out_weakness,
    load_prediction_rows,
    write_priors,
)
from fedicl_mqa.training.checkpointing import CheckpointManager
from fedicl_mqa.training.context import bind_protocol, training_inputs
from fedicl_mqa.training.workflows import train_centralized, train_federated, train_local_clients


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
    if args.fl_round is not None and args.mode not in {"centralized", "local-matched"}:
        raise ValueError("--fl-round is only valid for centralized or local-matched training")
    clients = load_partition(data_root(config), expected_config_hash=config.hash)
    train_icl = args.mode in {"local-icl", "federated-icl"}
    if train_icl and config.icl_training is None:
        raise ValueError("ICL training modes require icl_training in a fresh configuration")
    client_fit, client_exemplars, plan = training_inputs(config, clients)
    for seed in _requested_seeds(config, args):
        telemetry_root = checkpoint_root(config, args.mode)
        if args.mode == "local-matched":
            telemetry_root = matched_local_root(config, args.fl_round or selected_round(config))
        bind_protocol(
            config, telemetry_root / f"seed-{seed}", plan, train_icl=train_icl, create=True
        )
        if args.mode in {"local", "local-icl"}:
            telemetry = train_local_clients(
                config,
                client_fit,
                seed=seed,
                output_root=telemetry_root,
                resume=args.resume,
                **({"client_exemplars": client_exemplars} if train_icl else {}),
            )
        elif args.mode == "local-matched":
            fl_round = args.fl_round or selected_round(config)
            telemetry_root = matched_local_root(config, fl_round)
            telemetry = train_local_clients(
                config,
                client_fit,
                seed=seed,
                output_root=telemetry_root,
                resume=args.resume,
                fl_rounds=fl_round,
            )
        elif args.mode in {"federated", "federated-icl"}:
            trainer = train_federated(
                config,
                client_fit,
                seed=seed,
                output_root=telemetry_root,
                resume=args.resume,
                **(
                    {"client_exemplars": client_exemplars, "rounds": selected_round(config)}
                    if train_icl
                    else {}
                ),
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
            telemetry_root / f"seed-{seed}" / "telemetry.json",
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
            if config.controls is not None:
                from fedicl_mqa.cli.commands.evaluation import validate_summary_identity

                validate_summary_identity(
                    config,
                    summary,
                    arm="F0",
                    seed=seed,
                    split="validation",
                    round_index=round_index,
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
        evaluation_dir(config, "F0", seed=args.seed, split="validation", round_index=round_index)
        / "predictions.jsonl"
    )
    output = priors_path(config, seed=args.seed, round_index=round_index)
    rows = load_prediction_rows([predictions])
    audit = None
    if config.controls is not None:
        clients = load_partition(data_root(config), expected_config_hash=config.hash)
        summary_file = predictions.with_name("summary.json")
        summary = read_json(summary_file)
        if (
            summary.get("config_hash"),
            summary.get("arm"),
            summary.get("split"),
            summary.get("seed"),
        ) != (config.hash, "F0", "validation", args.seed):
            raise ValueError("prior source must be this seed's F0 validation summary")
        from fedicl_mqa.cli.commands.evaluation import validate_summary_identity

        validate_summary_identity(
            config, summary, arm="F0", seed=args.seed, split="validation", round_index=round_index
        )
        expected = {
            q.example_id: (c, q) for c, roles in clients.items() for q in roles["validation"]
        }
        if {r["prediction"]["example_id"] for r in rows} != expected.keys():
            raise ValueError("prior predictions do not match the complete validation cohort")
        for row in rows:
            p = row["prediction"]
            client, q = expected[p["example_id"]]
            if (p["client_id"], p["subject"], p["gold"], p["seed"]) != (
                client,
                q.subject,
                q.label,
                args.seed,
            ):
                raise ValueError("prior validation metadata differs from prepared data")
        priors, audit = controlled_weakness(
            rows,
            support_subjects={
                c: {q.subject for q in roles["support"]} for c, roles in clients.items()
            },
            min_count=config.controls.min_validation_per_subject,
        )
        audit.update(
            {
                "config_hash": config.hash,
                "seed": args.seed,
                "round": round_index,
                "source_split": "validation",
                "predictions_sha256": file_sha256(predictions),
                "summary_sha256": file_sha256(summary_file),
            }
        )
    else:
        priors = leave_one_client_out_weakness(rows, num_clients=config.data.num_clients)
    write_priors(output, priors)
    if audit is not None:
        audit["prior_sha256"] = file_sha256(output)
        write_json(output.with_suffix(".audit.json"), audit)
    print(f"Wrote leave-one-client-out prior to {output}")
