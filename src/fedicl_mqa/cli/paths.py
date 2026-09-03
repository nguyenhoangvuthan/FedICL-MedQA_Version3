"""Filesystem layout for every artifact the CLI reads or writes."""

from __future__ import annotations

import re
from pathlib import Path

from fedicl_mqa.core.config import Config
from fedicl_mqa.data.preparation import (
    build_partition,
    load_native_dataset,
    load_partition,
    materialize_partition,
    resolve_hub_revision,
)
from fedicl_mqa.core.io import read_json, write_json
from fedicl_mqa.modeling.loader import load_lora_bundle, resolve_model_revision

_SHA = re.compile(r"^[0-9a-f]{40,64}$", re.IGNORECASE)


def _require_existing_config(path: str | Path) -> Path:
    """Fail with the command that produces a missing configuration file.

    outputs/ is git-ignored, so a fresh clone has no sealed configuration and the bare
    "No such file or directory" gives no hint that prepare-data is what writes one.
    """
    resolved = Path(path)
    if resolved.exists():
        return resolved
    if resolved.name == "sealed_config.json":
        # Configs follow outputs/<name> <-> configs/<name>.yaml.
        suggestion = f"configs/{resolved.parent.name}.yaml"
        raise FileNotFoundError(
            f"{resolved} does not exist. The sealed configuration is written by "
            f"prepare-data from the YAML config, and outputs/ is not in git. Run: "
            f"fedicl-mqa prepare-data --config {suggestion}"
        )
    raise FileNotFoundError(f"configuration file not found: {resolved}")


def _seal_config(path: str | Path) -> Config:
    config = Config.from_file(_require_existing_config(path))
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


def _selected_round(config: Config) -> int:
    path = _checkpoint_root(config, "federated") / "selected_round.json"
    if not path.exists():
        raise FileNotFoundError("no selected FL round; pass --fl-round or run select-round")
    return int(read_json(path)["round"])


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

