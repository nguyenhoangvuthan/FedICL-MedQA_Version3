"""Environment checks run before a costly experiment."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from typing import Any

from fedicl_mqa.core.auth import apply_hf_token, find_token_file
from fedicl_mqa.core.config import Config
from fedicl_mqa.cli.paths import require_existing_config

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
    config = Config.from_file(require_existing_config(args.config))
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

