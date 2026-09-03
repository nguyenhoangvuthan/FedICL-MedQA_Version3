"""The whole experiment as one resumable sequence.

Ten steps run in dependency order, from downloading the dataset to writing the contrast
report. Each one knows how to tell whether it has already been done, so re-running the
command after an interruption continues where it stopped instead of repeating hours of
GPU work.

Progress is mirrored to pipeline_state.yaml after every status change, which is what
makes a long run watchable: the file is written before the first step starts and again
whenever a step finishes, so tailing it shows where the run actually is.

The step list is deliberately explicit rather than derived. The dependencies between
these steps are not obvious from their names -- build-priors needs the F0 validation
evaluations, not the test ones, and select-round must precede anything that resolves the
"selected" round -- so the order is written down where it can be read and checked.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from fedicl_mqa.cli import paths
from fedicl_mqa.cli.commands import data as data_commands
from fedicl_mqa.cli.commands import evaluation as evaluation_commands
from fedicl_mqa.cli.commands import training as training_commands
from fedicl_mqa.core.config import Config
from fedicl_mqa.core.io import atomic_write_text

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Step:
    name: str
    run: Callable[[], None]
    is_done: Callable[[], bool]


def _namespace(config_path: str | Path, **fields: Any) -> argparse.Namespace:
    """The command handlers take argparse namespaces; build them rather than duplicate."""
    return argparse.Namespace(config=str(config_path), **fields)


def build_steps(
    config: Config, *, split: str, force: bool, config_path: str | Path | None = None
) -> list[Step]:
    """The ten steps, in the order their outputs become available.

    config_path is the file the user invoked with. Only prepare-data needs it, since
    it is what turns a YAML config into the sealed one every later step reads.
    """
    sealed = paths.output_root(config) / "sealed_config.json"
    source = config_path if config_path is not None else sealed
    seeds = list(config.experiment.training_seeds)
    rounds = list(config.training.fl_round_candidates)

    def done(predicate: Callable[[], bool]) -> Callable[[], bool]:
        # --force re-runs everything, so nothing may report itself as already done.
        return (lambda: False) if force else predicate

    def train(mode: str, **extra: Any) -> Callable[[], None]:
        def run() -> None:
            training_commands.command_train(
                _namespace(
                    sealed,
                    mode=mode,
                    seed=None,
                    all_seeds=True,
                    fl_round=None,
                    resume="auto",
                    **extra,
                )
            )

        return run

    def evaluate_f0_validation() -> None:
        for seed in seeds:
            for round_index in rounds:
                target = paths.summary_path(
                    config, "F0", seed=seed, split="validation", round_index=round_index
                )
                if target.exists() and not force:
                    continue
                evaluation_commands.command_evaluate(
                    _namespace(
                        sealed,
                        arm="F0",
                        seed=seed,
                        split="validation",
                        round=round_index,
                        subject_weights=None,
                    )
                )

    def build_priors() -> None:
        for seed in seeds:
            training_commands.command_build_priors(
                _namespace(sealed, seed=seed, round=None)
            )

    def f0_validation_complete() -> bool:
        return all(
            paths.summary_path(
                config, "F0", seed=seed, split="validation", round_index=round_index
            ).exists()
            for seed in seeds
            for round_index in rounds
        )

    def priors_complete() -> bool:
        try:
            selected = paths.selected_round(config)
        except FileNotFoundError:
            return False
        return all(
            paths.priors_path(config, seed=seed, round_index=selected).exists() for seed in seeds
        )

    def evaluations_complete() -> bool:
        for arm in evaluation_commands.ARMS:
            for seed in evaluation_commands._effective_seeds(config, arm):
                if not paths.summary_path(
                    config, arm, seed=seed, split=split, round_index=None
                ).exists():
                    return False
        return True

    return [
        Step(
            "prepare-data",
            # Only this step reads the YAML; prepare-data is what writes the sealed file.
            lambda: data_commands.command_prepare_data(_namespace(source)),
            done(lambda: (paths.data_root(config) / "partition_manifest.json").exists()),
        ),
        Step(
            "audit-retrieval",
            lambda: data_commands.command_audit_retrieval(_namespace(sealed)),
            done(lambda: (paths.data_root(config) / "retrieval_audit.json").exists()),
        ),
        # Training resumes internally from its own checkpoints, so these always run and
        # return quickly when there is nothing left to do.
        Step("train-local", train("local"), lambda: False),
        Step("train-federated", train("federated"), lambda: False),
        Step("evaluate-f0-validation", evaluate_f0_validation, done(f0_validation_complete)),
        Step(
            "select-round",
            lambda: training_commands.command_select_round(_namespace(sealed)),
            done(
                lambda: (
                    paths.checkpoint_root(config, "federated") / "selected_round.json"
                ).exists()
            ),
        ),
        Step("train-centralized", train("centralized"), lambda: False),
        Step("build-priors", build_priors, done(priors_complete)),
        Step(
            "evaluate-all",
            lambda: evaluation_commands.command_evaluate_all(
                _namespace(sealed, split=split, round=None, force=force)
            ),
            done(evaluations_complete),
        ),
        Step(
            "report",
            lambda: evaluation_commands.command_report(_namespace(sealed)),
            done(lambda: paths.report_path(config).exists()),
        ),
    ]


def _write_state(config: Config, records: list[dict[str, Any]]) -> None:
    payload = {
        "config_hash": config.hash,
        "dataset": config.data.dataset,
        "output_dir": str(paths.output_root(config)),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "steps": records,
    }
    atomic_write_text(
        paths.pipeline_state_path(config),
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
    )


def execute(config: Config, steps: list[Step]) -> list[dict[str, Any]]:
    """Run the steps, mirroring status to pipeline_state.yaml as it goes."""
    records: list[dict[str, Any]] = [
        {
            "index": index,
            "name": step.name,
            "status": "pending",
            "started_at": None,
            "finished_at": None,
            "duration_seconds": None,
            "error": None,
        }
        for index, step in enumerate(steps, start=1)
    ]
    total = len(steps)
    for record, step in zip(records, steps, strict=True):
        if step.is_done():
            record["status"] = "skipped"
            logger.info("[%d/%d] %s: already complete", record["index"], total, step.name)
            _write_state(config, records)
            continue
        record["status"] = "running"
        record["started_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        # Written before the step starts so a step that runs for hours is still visible.
        _write_state(config, records)
        logger.info("[%d/%d] %s: running", record["index"], total, step.name)
        started = time.perf_counter()
        try:
            step.run()
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["duration_seconds"] = round(time.perf_counter() - started, 3)
            record["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            _write_state(config, records)
            logger.error("[%d/%d] %s: failed", record["index"], total, step.name)
            raise
        record["status"] = "done"
        record["duration_seconds"] = round(time.perf_counter() - started, 3)
        record["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _write_state(config, records)
        logger.info(
            "[%d/%d] %s: done in %.1fs",
            record["index"],
            total,
            step.name,
            record["duration_seconds"],
        )
    return records


def command_pipeline(args: argparse.Namespace) -> None:
    config = paths.seal_config(args.config)
    steps = build_steps(
        config, split=args.split, force=args.force, config_path=args.config
    )
    execute(config, steps)
    print(f"\nPipeline state: {paths.pipeline_state_path(config)}")
    print(f"Comparison table: {paths.comparison_path(config, 'md')}")
