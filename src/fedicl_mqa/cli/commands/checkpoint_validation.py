"""Select existing adapters using held-out validation; never run training."""

from __future__ import annotations

import argparse

from fedicl_mqa.cli.commands.training import _requested_seeds
from fedicl_mqa.cli.paths import checkpoint_root, data_root, matched_local_root, seal_config
from fedicl_mqa.data.preparation import load_partition
from fedicl_mqa.modeling.loader import load_lora_bundle
from fedicl_mqa.training.checkpointing import CheckpointManager
from fedicl_mqa.training.context import bind_protocol, training_inputs
from fedicl_mqa.training.validation import select_existing_checkpoints, validate_fit_separation


def command_validate_checkpoints(args: argparse.Namespace) -> None:
    if args.mode == "all":
        for mode in ("local", "federated", "centralized"):
            command_validate_checkpoints(argparse.Namespace(**{**vars(args), "mode": mode}))
        return
    config = seal_config(args.config)
    clients = load_partition(data_root(config), expected_config_hash=config.hash)
    train_icl = args.mode.endswith("-icl")
    if train_icl and config.icl_training is None:
        raise ValueError("ICL checkpoints require the original icl_training configuration")
    if args.fl_round is not None and args.mode != "local-matched":
        raise ValueError("--fl-round is only needed for the local-matched checkpoint directory")
    _, _, plan = training_inputs(config, clients)
    family_root = (
        matched_local_root(config, args.fl_round)
        if args.mode == "local-matched"
        else checkpoint_root(config, args.mode)
    )
    all_validation = [q for c in sorted(clients) for q in clients[c]["validation"]]
    all_training = [
        q for c in sorted(clients) for role in ("fit", "support") for q in clients[c][role]
    ]
    validate_fit_separation(all_training, all_validation)
    for seed in _requested_seeds(config, args):
        run_root = family_root / f"seed-{seed}"
        bind_protocol(config, run_root, plan, train_icl=train_icl)
        if args.mode.startswith("local"):
            targets = [
                (
                    run_root / f"client-{c}",
                    clients[c]["validation"],
                    f"{args.mode}-client-{c}",
                    seed * 1_000 + c,
                )
                for c in sorted(clients)
            ]
        else:
            root = run_root / "global" if args.mode.startswith("federated") else run_root
            targets = [(root, all_validation, args.mode, seed)]
        # Check inputs before loading a potentially large model.
        for root, validation, _, _ in targets:
            if not root.is_dir():
                raise FileNotFoundError(f"checkpoint directory does not exist: {root}")
            validate_fit_separation(all_training, validation)
        bundle = load_lora_bundle(config, seed=seed)
        for root, validation, kind, checkpoint_seed in targets:
            manager = CheckpointManager(
                root,
                config_hash=config.hash,
                model_id=config.model.id,
                model_revision=config.model.revision,
            )
            report = select_existing_checkpoints(
                bundle,
                manager,
                validation,
                max_length=config.model.max_seq_length,
                expected_kind=kind,
                expected_seed=checkpoint_seed,
            )
            adapter = root / report["best_checkpoint"] / "adapter"
            print(f"Best adapter: {adapter} (validation_loss={report['best_validation_loss']:.6f})")
        del bundle
