from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from .checkpointing import CheckpointManager
from .config import Config
from .federated import FederatedTrainer, adapter_state, set_adapter_state
from .modeling import configure_runtime, load_lora_bundle
from .schema import MCQExample
from .training import train


def train_local_clients(
    config: Config,
    client_fit: Mapping[int, Sequence[MCQExample]],
    *,
    seed: int,
    output_root: str | Path,
    resume: str | None = "auto",
) -> dict[int, dict[str, float | str]]:
    bundle = load_lora_bundle(config, seed=seed)
    initial_adapter = adapter_state(bundle.model)
    telemetry: dict[int, dict[str, float | str]] = {}
    for client_id in range(config.data.num_clients):
        set_adapter_state(bundle.model, initial_adapter)
        local_seed = seed * 1_000 + client_id
        configure_runtime(config, local_seed)
        manager = CheckpointManager(
            Path(output_root) / f"seed-{seed}" / f"client-{client_id}",
            config_hash=config.hash,
            model_id=config.model.id,
            model_revision=config.model.revision,
            keep=config.training.checkpoint_keep,
        )
        _, run_metrics = train(
            bundle,
            client_fit[client_id],
            config,
            seed=local_seed,
            epochs=config.training.local_epochs,
            kind=f"local-client-{client_id}",
            checkpoint_manager=manager,
            resume=resume,
        )
        telemetry[client_id] = run_metrics
    return telemetry


def train_centralized(
    config: Config,
    client_fit: Mapping[int, Sequence[MCQExample]],
    *,
    seed: int,
    fl_rounds: int,
    output_root: str | Path,
    resume: str | None = "auto",
) -> dict[str, float | str]:
    bundle = load_lora_bundle(config, seed=seed)
    pooled = [example for client in sorted(client_fit) for example in client_fit[client]]
    manager = CheckpointManager(
        Path(output_root) / f"seed-{seed}",
        config_hash=config.hash,
        model_id=config.model.id,
        model_revision=config.model.revision,
        keep=config.training.checkpoint_keep,
    )
    _, telemetry = train(
        bundle,
        pooled,
        config,
        seed=seed,
        epochs=fl_rounds * config.training.local_epochs,
        kind="centralized",
        checkpoint_manager=manager,
        resume=resume,
    )
    return telemetry


def train_federated(
    config: Config,
    client_fit: Mapping[int, Sequence[MCQExample]],
    *,
    seed: int,
    output_root: str | Path,
    resume: str | None = "auto",
) -> FederatedTrainer:
    bundle = load_lora_bundle(config, seed=seed)
    trainer = FederatedTrainer(
        bundle,
        config,
        seed=seed,
        run_root=Path(output_root) / f"seed-{seed}",
    )
    trainer.run(
        client_fit,
        rounds=max(config.training.fl_round_candidates),
        resume=resume,
    )
    return trainer
