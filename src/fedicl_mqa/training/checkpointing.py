from __future__ import annotations

import os
import random
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fedicl_mqa.core.io import atomic_write_text, file_sha256, read_json, write_json


@dataclass(slots=True)
class TrainerState:
    kind: str
    seed: int
    global_step: int = 0
    epoch: int = 0
    batch_in_epoch: int = 0
    round_index: int = 0
    optimizer_updates: int = 0
    target_exposures: int = 0
    elapsed_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    path: Path
    trainer_state: TrainerState
    metrics: dict[str, float]
    extra: dict[str, Any]


class CheckpointManager:
    """Atomic PEFT checkpoints with optimizer, RNG, hashes and best/last pointers."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        root: str | Path,
        *,
        config_hash: str,
        model_id: str,
        model_revision: str,
        keep: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_hash = config_hash
        self.model_id = model_id
        self.model_revision = model_revision
        self.keep = keep

    def save(
        self,
        name: str,
        *,
        model: Any,
        optimizer: Any | None,
        trainer_state: TrainerState,
        metrics: Mapping[str, float] | None = None,
        extra: Mapping[str, Any] | None = None,
        is_best: bool = False,
    ) -> Path:
        self._validate_name(name)
        destination = self.root / name
        if destination.exists():
            raise FileExistsError(f"checkpoint already exists: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=self.root))
        try:
            adapter_dir = temporary / "adapter"
            model.save_pretrained(adapter_dir, safe_serialization=True)
            if optimizer is not None:
                torch = _torch()
                torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
            self._save_rng(temporary)
            write_json(
                temporary / "state.json",
                {
                    "format_version": self.FORMAT_VERSION,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "config_hash": self.config_hash,
                    "model_id": self.model_id,
                    "model_revision": self.model_revision,
                    "trainer_state": asdict(trainer_state),
                    "metrics": dict(metrics or {}),
                    "extra": dict(extra or {}),
                },
            )
            hashes = {
                str(path.relative_to(temporary)): file_sha256(path)
                for path in sorted(temporary.rglob("*"))
                if path.is_file() and path.name != "hashes.json"
            }
            write_json(temporary / "hashes.json", hashes)
            os.replace(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        atomic_write_text(self.root / "last_checkpoint.txt", f"{name}\n")
        if is_best:
            atomic_write_text(self.root / "best_checkpoint.txt", f"{name}\n")
        self._prune(protected={name, self._read_pointer("best_checkpoint.txt")})
        return destination

    def load(
        self,
        reference: str | Path,
        *,
        model: Any,
        optimizer: Any | None = None,
        restore_rng: bool = True,
    ) -> LoadedCheckpoint:
        path = self.resolve(reference)
        self.verify(path)
        state = read_json(path / "state.json")
        if state.get("format_version") != self.FORMAT_VERSION:
            raise ValueError(
                f"unsupported checkpoint format version: {state.get('format_version')!r}"
            )
        if state["config_hash"] != self.config_hash:
            raise ValueError(
                "config hash mismatch: "
                f"checkpoint={state['config_hash']} current={self.config_hash}"
            )
        if state["model_id"] != self.model_id or state["model_revision"] != self.model_revision:
            raise ValueError("base model identity or revision differs from checkpoint")

        try:
            from peft import load_peft_weights, set_peft_model_state_dict
        except ImportError as exc:  # pragma: no cover - full runtime only
            raise RuntimeError("peft is required to restore adapter checkpoints") from exc
        adapter_state = load_peft_weights(str(path / "adapter"), device="cpu")
        result = set_peft_model_state_dict(model, adapter_state)
        unexpected = getattr(result, "unexpected_keys", ())
        if unexpected:
            raise ValueError(f"unexpected adapter keys: {unexpected}")

        if optimizer is not None and (path / "optimizer.pt").exists():
            torch = _torch()
            optimizer.load_state_dict(
                torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
            )
            _optimizer_to_device(optimizer, next(model.parameters()).device)
        if restore_rng:
            self._restore_rng(path)

        trainer_state = TrainerState(**state["trainer_state"])
        return LoadedCheckpoint(
            path=path,
            trainer_state=trainer_state,
            metrics={key: float(value) for key, value in state.get("metrics", {}).items()},
            extra=dict(state.get("extra", {})),
        )

    def resolve(self, reference: str | Path) -> Path:
        reference_text = str(reference)
        if reference_text in {"auto", "last", "best"}:
            pointer = "best_checkpoint.txt" if reference_text == "best" else "last_checkpoint.txt"
            name = self._read_pointer(pointer)
            if not name:
                raise FileNotFoundError(f"checkpoint pointer is missing: {self.root / pointer}")
            path = self.root / name
        else:
            candidate = Path(reference)
            path = candidate if candidate.is_absolute() else self.root / candidate
        resolved = path.resolve()
        if self.root.resolve() not in resolved.parents:
            raise ValueError(f"checkpoint must be inside {self.root}")
        if not resolved.is_dir():
            raise FileNotFoundError(resolved)
        return resolved

    def latest(self) -> Path | None:
        pointed_name = self._read_pointer("last_checkpoint.txt")
        pointed = self.root / pointed_name if pointed_name else None
        candidates = [
            path
            for path in self.root.iterdir()
            if path.is_dir() and path.name.startswith("checkpoint-")
        ]
        valid: list[tuple[tuple[int, int, int, int], Path]] = []
        for candidate in candidates:
            try:
                self.verify(candidate)
                state = read_json(candidate / "state.json")
            except (FileNotFoundError, ValueError):
                continue
            if (
                state.get("format_version") == self.FORMAT_VERSION
                and state.get("config_hash") == self.config_hash
                and state.get("model_id") == self.model_id
                and state.get("model_revision") == self.model_revision
            ):
                trainer_state = state.get("trainer_state", {})
                progress = (
                    int(trainer_state.get("round_index", 0)),
                    int(trainer_state.get("epoch", 0)),
                    int(trainer_state.get("global_step", 0)),
                    candidate.stat().st_mtime_ns,
                )
                valid.append((progress, candidate))
        if not valid:
            return None
        candidate = max(valid, key=lambda value: value[0])[1]
        if pointed is None or candidate.resolve() != pointed.resolve():
            atomic_write_text(self.root / "last_checkpoint.txt", f"{candidate.name}\n")
        return candidate

    def mark_best(self, reference: str | Path) -> Path:
        path = self.resolve(reference)
        self.verify(path)
        atomic_write_text(self.root / "best_checkpoint.txt", f"{path.name}\n")
        return path

    def verify(self, checkpoint: str | Path) -> None:
        path = Path(checkpoint).resolve()
        expected = read_json(path / "hashes.json")
        actual_files = {
            str(value.relative_to(path))
            for value in path.rglob("*")
            if value.is_file() and value.name != "hashes.json"
        }
        if actual_files != set(expected):
            missing = sorted(set(expected) - actual_files)
            unexpected = sorted(actual_files - set(expected))
            raise ValueError(
                f"checkpoint artifact manifest mismatch: missing={missing}, unexpected={unexpected}"
            )
        for relative, digest in expected.items():
            artifact = (path / relative).resolve()
            if path not in artifact.parents:
                raise ValueError(f"checkpoint manifest path escapes root: {relative}")
            if not artifact.is_file():
                raise ValueError(f"checkpoint artifact is missing: {artifact}")
            actual = file_sha256(artifact)
            if actual != digest:
                raise ValueError(f"checkpoint hash mismatch: {artifact}")

    def _save_rng(self, root: Path) -> None:
        torch = _torch()
        tensor_state = {"cpu": torch.get_rng_state()}
        if torch.cuda.is_available():
            for index, value in enumerate(torch.cuda.get_rng_state_all()):
                tensor_state[f"cuda_{index}"] = value.cpu()
        torch.save(tensor_state, root / "torch_rng.pt")

        payload: dict[str, Any] = {"python": _jsonable(random.getstate())}
        try:
            import numpy as np

            numpy_state = np.random.get_state()
            payload["numpy"] = {
                "algorithm": numpy_state[0],
                "keys": numpy_state[1].tolist(),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            }
        except ImportError:
            pass
        write_json(root / "host_rng.json", payload)

    def _restore_rng(self, root: Path) -> None:
        torch = _torch()
        tensor_state = torch.load(root / "torch_rng.pt", map_location="cpu", weights_only=True)
        torch.set_rng_state(tensor_state["cpu"])
        if torch.cuda.is_available():
            cuda_states = [
                tensor_state[f"cuda_{index}"]
                for index in range(torch.cuda.device_count())
                if f"cuda_{index}" in tensor_state
            ]
            if cuda_states:
                torch.cuda.set_rng_state_all(cuda_states)
        payload = read_json(root / "host_rng.json")
        random.setstate(_tuple_tree(payload["python"]))
        if "numpy" in payload:
            try:
                import numpy as np

                state = payload["numpy"]
                np.random.set_state(
                    (
                        state["algorithm"],
                        np.asarray(state["keys"], dtype=np.uint32),
                        state["position"],
                        state["has_gauss"],
                        state["cached_gaussian"],
                    )
                )
            except ImportError:
                pass

    def _read_pointer(self, filename: str) -> str | None:
        path = self.root / filename
        return path.read_text(encoding="utf-8").strip() if path.exists() else None

    @staticmethod
    def _validate_name(name: str) -> None:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"invalid checkpoint name: {name!r}")

    def _prune(self, *, protected: set[str | None]) -> None:
        if self.keep is None or self.keep <= 0:
            return
        checkpoints = sorted(
            [
                path
                for path in self.root.iterdir()
                if path.is_dir() and path.name.startswith("checkpoint-")
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        keep_names = {path.name for path in checkpoints[: self.keep]} | {
            name for name in protected if name
        }
        for path in checkpoints:
            if path.name not in keep_names and path.parent.resolve() == self.root.resolve():
                shutil.rmtree(path)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - full runtime only
        raise RuntimeError("torch is required for checkpoints") from exc
    return torch


def _optimizer_to_device(optimizer: Any, device: Any) -> None:
    torch = _torch()
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value
