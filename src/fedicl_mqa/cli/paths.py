"""Filesystem layout for every artifact the CLI reads or writes.

Every path in the output tree is built here and nowhere else, so changing the layout is
a single-file edit. The tree is:

    outputs/<name>/
    ├── sealed_config.json          frozen configuration, hashed
    ├── pipeline_state.yaml         per-step status of a pipeline run
    ├── arms_comparison.json|.md    accuracy of every arm evaluated so far
    ├── data/<dataset>/             partitioned clients and the manifest
    ├── training/<dataset>/<family>/seed-<n>/
    ├── arms/<dataset>/<ARM>/
    │   ├── run_log.json            one trace record per run of this arm
    │   ├── priors/seed-<n>/round-<r>.json
    │   └── seed-<n>|deterministic/<split>/round-<r>|selected/
    └── reports/<dataset>/contrasts.json

Checkpoints live under training/<family> rather than under each arm because a checkpoint
belongs to a training run, not to an arm: L0 and L1 read the same local adapter, and F0,
F1 and F2 read the same federated one. Copying them per arm would let the arms drift
apart under --resume and would break the very comparison the experiment exists to make.
"""

from __future__ import annotations

import re
from pathlib import Path

from fedicl_mqa.core.config import Config
from fedicl_mqa.core.io import read_json, write_json
from fedicl_mqa.data.preparation import resolve_hub_revision
from fedicl_mqa.modeling.loader import resolve_model_revision

_SHA = re.compile(r"^[0-9a-f]{40,64}$", re.IGNORECASE)


def require_existing_config(path: str | Path) -> Path:
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


def seal_config(path: str | Path) -> Config:
    config = Config.from_file(require_existing_config(path))
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


def output_root(config: Config) -> Path:
    return Path(config.experiment.output_dir)


def data_root(config: Config) -> Path:
    return output_root(config) / "data" / config.data.dataset


def checkpoint_root(config: Config, family: str) -> Path:
    return output_root(config) / "training" / config.data.dataset / family


def arms_root(config: Config) -> Path:
    return output_root(config) / "arms" / config.data.dataset


def arm_dir(config: Config, arm: str) -> Path:
    return arms_root(config) / arm


def arm_log_path(config: Config, arm: str) -> Path:
    """Per-arm trace log: one record per run, appended as runs complete."""
    return arm_dir(config, arm) / "run_log.json"


def seed_label(seed: int | None) -> str:
    """Base arms consume no checkpoint, so their results are not seed-specific."""
    return f"seed-{seed}" if seed is not None else "deterministic"


def round_label(round_index: int | None) -> str:
    return f"round-{round_index}" if round_index is not None else "selected"


def evaluation_dir(
    config: Config, arm: str, *, seed: int | None, split: str, round_index: int | None
) -> Path:
    return arm_dir(config, arm) / seed_label(seed) / split / round_label(round_index)


def summary_path(
    config: Config, arm: str, *, seed: int | None, split: str, round_index: int | None
) -> Path:
    """Marks a finished run: evaluate_arm writes it after predictions.jsonl."""
    return (
        evaluation_dir(config, arm, seed=seed, split=split, round_index=round_index)
        / "summary.json"
    )


def priors_path(config: Config, *, seed: int, round_index: int) -> Path:
    """The leave-one-client-out prior, stored under the arm that consumes it."""
    return arm_dir(config, "F2") / "priors" / f"seed-{seed}" / f"round-{round_index}.json"


def comparison_path(config: Config, suffix: str) -> Path:
    """Cross-arm accuracy table, rewritten whenever an arm finishes."""
    return output_root(config) / f"arms_comparison.{suffix}"


def pipeline_state_path(config: Config) -> Path:
    return output_root(config) / "pipeline_state.yaml"


def report_path(config: Config) -> Path:
    return output_root(config) / "reports" / config.data.dataset / "contrasts.json"


def selected_round(config: Config) -> int:
    path = checkpoint_root(config, "federated") / "selected_round.json"
    if not path.exists():
        raise FileNotFoundError("no selected FL round; pass --fl-round or run select-round")
    return int(read_json(path)["round"])
