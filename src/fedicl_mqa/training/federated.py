from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from fedicl_mqa.training.checkpointing import CheckpointManager, TrainerState
from fedicl_mqa.core.config import Config
from fedicl_mqa.core.io import file_sha256, read_json, write_json
from fedicl_mqa.modeling.loader import ModelBundle, configure_runtime
from fedicl_mqa.core.schema import MCQExample
from fedicl_mqa.training.loop import train


def adapter_state(model: Any) -> dict[str, Any]:
    try:
        from peft import get_peft_model_state_dict
    except ImportError as exc:  # pragma: no cover - full runtime only
        raise RuntimeError("peft is required to access adapter state") from exc
    return {
        key: value.detach().cpu().contiguous().clone()
        for key, value in get_peft_model_state_dict(model).items()
    }


def set_adapter_state(model: Any, state: Mapping[str, Any]) -> None:
    try:
        from peft import set_peft_model_state_dict
    except ImportError as exc:  # pragma: no cover - full runtime only
        raise RuntimeError("peft is required to restore adapter state") from exc
    result = set_peft_model_state_dict(model, dict(state))
    unexpected = getattr(result, "unexpected_keys", ())
    if unexpected:
        raise ValueError(f"unexpected adapter keys: {unexpected}")


def adapter_nbytes(state: Mapping[str, Any]) -> int:
    return sum(int(value.numel() * value.element_size()) for value in state.values())


def weighted_fedavg(
    states: Sequence[Mapping[str, Any]], weights: Sequence[int | float]
) -> dict[str, Any]:
    if not states or len(states) != len(weights):
        raise ValueError("states and weights must be non-empty and have equal length")
    import torch

    keys = set(states[0])
    if any(set(state) != keys for state in states[1:]):
        raise ValueError("all adapter states must have identical keys")
    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError("FedAvg weights must sum to a positive value")
    averaged: dict[str, Any] = {}
    for key in sorted(keys):
        reference = states[0][key]
        accumulator = torch.zeros_like(reference, dtype=torch.float32, device="cpu")
        for state, weight in zip(states, weights, strict=True):
            value = state[key]
            if value.shape != reference.shape:
                raise ValueError(f"shape mismatch for adapter tensor {key}")
            accumulator.add_(value.float().cpu(), alpha=float(weight) / total_weight)
        averaged[key] = accumulator.to(dtype=reference.dtype).contiguous()
    return averaged


def save_adapter_state(path: str | Path, state: Mapping[str, Any]) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - full runtime only
        raise RuntimeError("safetensors is required to save FL client updates") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(dict(state), str(temporary))
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_adapter_state(path: str | Path) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - full runtime only
        raise RuntimeError("safetensors is required to load FL client updates") from exc
    return load_file(str(path), device="cpu")


ValidationCallback = Callable[[Any, int], Mapping[str, float]]


class FederatedTrainer:
    """Sequential single-GPU FedAvg with round and completed-client recovery."""

    def __init__(
        self,
        bundle: ModelBundle,
        config: Config,
        *,
        seed: int,
        run_root: str | Path,
    ) -> None:
        self.bundle = bundle
        self.config = config
        self.seed = seed
        self.run_root = Path(run_root)
        self.run_root.mkdir(parents=True, exist_ok=True)
        # Preserve all round checkpoints because rounds 4/6/8 are validation candidates.
        self.checkpoints = CheckpointManager(
            self.run_root / "global",
            config_hash=config.hash,
            model_id=config.model.id,
            model_revision=config.model.revision,
            keep=None,
        )
        self.final_state: TrainerState | None = None
        self.history: list[dict[str, Any]] = []

    def run(
        self,
        client_fit: Mapping[int, Sequence[MCQExample]],
        *,
        rounds: int,
        resume: str | None = "auto",
        validation_callback: ValidationCallback | None = None,
    ) -> tuple[TrainerState, list[dict[str, Any]]]:
        if set(client_fit) != set(range(self.config.data.num_clients)):
            raise ValueError("full participation requires fit data for every client")
        if any(not values for values in client_fit.values()):
            raise ValueError("every client must have non-empty fit data")

        state = TrainerState(kind="federated", seed=self.seed)
        latest = self.checkpoints.latest()
        if resume and latest is not None:
            loaded = self.checkpoints.load(
                latest if resume == "auto" else resume,
                model=self.bundle.model,
                optimizer=None,
                restore_rng=True,
            )
            state = loaded.trainer_state
        elif latest is None:
            self.checkpoints.save(
                "checkpoint-round-0000",
                model=self.bundle.model,
                optimizer=None,
                trainer_state=state,
                extra={"communication": {"uplink_bytes": 0, "downlink_bytes": 0}},
            )

        initial_elapsed_seconds = state.elapsed_seconds
        started = time.perf_counter()
        history_path = self.run_root / "history.json"
        history = list(read_json(history_path).get("rounds", [])) if history_path.exists() else []
        original_history_count = len(history)
        history = self._reconcile_history(history, completed_round=state.round_index)
        if state.round_index and len(history) != original_history_count:
            write_json(history_path, {"rounds": history})
        for round_index in range(state.round_index + 1, rounds + 1):
            round_started = time.perf_counter()
            global_state = adapter_state(self.bundle.model)
            global_bytes = adapter_nbytes(global_state)
            work = self.run_root / f"round-work-{round_index:04d}"
            work.mkdir(parents=True, exist_ok=True)
            round_manifest = work / "round_state.json"
            previous_checkpoint = self.checkpoints.latest()
            identity = {
                "round": round_index,
                "source_checkpoint": previous_checkpoint.name if previous_checkpoint else None,
                "seed": self.seed,
                "config_hash": self.config.hash,
            }
            if round_manifest.exists() and read_json(round_manifest) != identity:
                raise ValueError(f"stale or incompatible partial round state: {round_manifest}")
            write_json(round_manifest, identity)

            client_states: list[dict[str, Any]] = []
            client_weights: list[int] = []
            client_metrics: dict[str, Any] = {}
            for client_id in range(self.config.data.num_clients):
                update_path = work / f"client-{client_id}.safetensors"
                metadata_path = work / f"client-{client_id}.json"
                if update_path.exists() and metadata_path.exists():
                    metadata = read_json(metadata_path)
                    if metadata.get("client_id") != client_id or metadata.get("examples") != len(
                        client_fit[client_id]
                    ):
                        raise ValueError(f"partial client metadata mismatch: {metadata_path}")
                    if file_sha256(update_path) != metadata["sha256"]:
                        raise ValueError(f"partial client update hash mismatch: {update_path}")
                    local_state = load_adapter_state(update_path)
                    telemetry = metadata["telemetry"]
                else:
                    set_adapter_state(self.bundle.model, global_state)
                    local_seed = self.seed * 1_000_000 + round_index * 1_000 + client_id
                    configure_runtime(self.config, local_seed)
                    _, telemetry = train(
                        self.bundle,
                        client_fit[client_id],
                        self.config,
                        seed=local_seed,
                        epochs=self.config.training.local_epochs,
                        kind=f"federated-client-{client_id}",
                    )
                    local_state = adapter_state(self.bundle.model)
                    save_adapter_state(update_path, local_state)
                    write_json(
                        metadata_path,
                        {
                            "client_id": client_id,
                            "examples": len(client_fit[client_id]),
                            "sha256": file_sha256(update_path),
                            "telemetry": telemetry,
                        },
                    )
                client_states.append(local_state)
                client_weights.append(len(client_fit[client_id]))
                client_metrics[str(client_id)] = telemetry
                if self.config.hardware.empty_cache_between_clients:
                    try:
                        import torch

                        torch.cuda.empty_cache()
                    except (ImportError, RuntimeError):
                        pass

            aggregated = weighted_fedavg(client_states, client_weights)
            set_adapter_state(self.bundle.model, aggregated)
            metrics = (
                dict(validation_callback(self.bundle.model, round_index))
                if validation_callback
                else {}
            )
            state.round_index = round_index
            state.epoch += self.config.training.local_epochs
            state.target_exposures += sum(client_weights) * self.config.training.local_epochs
            state.optimizer_updates += sum(
                int(client_metrics[str(client)]["total_optimizer_updates"])
                for client in range(self.config.data.num_clients)
            )
            communication = {
                "uplink_bytes": sum(adapter_nbytes(value) for value in client_states),
                "downlink_bytes": global_bytes * self.config.data.num_clients,
            }
            record = {
                "round": round_index,
                "metrics": metrics,
                "communication": communication,
                "clients": client_metrics,
                "round_wall_clock_seconds": time.perf_counter() - round_started,
            }
            history.append(record)
            state.elapsed_seconds = initial_elapsed_seconds + (time.perf_counter() - started)
            self.checkpoints.save(
                f"checkpoint-round-{round_index:04d}",
                model=self.bundle.model,
                optimizer=None,
                trainer_state=state,
                metrics=metrics,
                extra={
                    "communication": communication,
                    "round_wall_clock_seconds": record["round_wall_clock_seconds"],
                },
                is_best=False,
            )
            write_json(history_path, {"rounds": history})
        state.elapsed_seconds = initial_elapsed_seconds + (time.perf_counter() - started)
        self.final_state = state
        self.history = history
        return state, history

    def _reconcile_history(
        self, history: Sequence[Mapping[str, Any]], *, completed_round: int
    ) -> list[dict[str, Any]]:
        """Recover history after a checkpoint-commit/history-write interruption."""
        by_round = {int(record["round"]): dict(record) for record in history}
        if any(round_index > completed_round for round_index in by_round):
            raise ValueError("federated history is ahead of the last committed checkpoint")
        for round_index in range(1, completed_round + 1):
            if round_index in by_round:
                continue
            checkpoint = self.checkpoints.resolve(f"checkpoint-round-{round_index:04d}")
            self.checkpoints.verify(checkpoint)
            state = read_json(checkpoint / "state.json")
            work = self.run_root / f"round-work-{round_index:04d}"
            client_metrics = {
                str(client_id): read_json(work / f"client-{client_id}.json")["telemetry"]
                for client_id in range(self.config.data.num_clients)
            }
            by_round[round_index] = {
                "round": round_index,
                "metrics": state.get("metrics", {}),
                "communication": state.get("extra", {}).get("communication", {}),
                "clients": client_metrics,
                "round_wall_clock_seconds": state.get("extra", {}).get(
                    "round_wall_clock_seconds", 0.0
                ),
            }
        return [by_round[index] for index in sorted(by_round)]


def select_best_round(
    checkpoint_root: str | Path,
    *,
    candidates: Sequence[int] = (4, 6, 8),
    metric: str = "pipeline_accuracy",
) -> int:
    root = Path(checkpoint_root)
    scores: dict[int, float] = {}
    for round_index in candidates:
        state = read_json(root / f"checkpoint-round-{round_index:04d}" / "state.json")
        if metric not in state.get("metrics", {}):
            raise ValueError(f"checkpoint round {round_index} has no validation metric {metric!r}")
        scores[round_index] = float(state["metrics"][metric])
    # Deterministic tie-break: select the earlier round.
    return max(sorted(scores), key=lambda value: scores[value])
