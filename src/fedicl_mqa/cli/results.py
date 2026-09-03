"""Cross-arm comparison table and per-arm trace logs.

Two records with different jobs. The comparison table answers "how do the arms compare
so far" and is rebuilt from the summaries on disk every time an arm finishes, so it stays
truthful after an interrupted sweep and never drifts from the underlying results. The
per-arm log answers "what happened when this arm ran" and accumulates one record per run,
including the runs that were skipped or that failed, which leave no summary behind.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any

from fedicl_mqa.cli import paths
from fedicl_mqa.core.config import Config
from fedicl_mqa.core.io import read_json, write_json
from fedicl_mqa.evaluation.arms import ARMS

# Reported for every arm. pipeline_accuracy is the primary endpoint; the rest are the
# secondary metrics that distinguish arms scoring alike on it.
COMPARISON_METRICS = (
    "pipeline_accuracy",
    "position_macro_f1",
    "conditional_likelihood_accuracy",
    "ece",
    "exact_match_coverage",
    "semantic_fallback_rate",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_arm_run(
    config: Config,
    arm: str,
    *,
    seed: int | None,
    split: str,
    status: str,
    accuracy: float | None = None,
    round_index: int | None = None,
    duration_seconds: float | None = None,
    error: str | None = None,
) -> None:
    """Append one trace record for a run of `arm`.

    Status is "completed", "skipped" or "failed". Skipped and failed runs are recorded
    too: they produce no summary.json, so without this the log would silently omit the
    runs most worth tracing.
    """
    path = paths.arm_log_path(config, arm)
    payload: dict[str, Any] = (
        read_json(path)
        if path.exists()
        else {"arm": arm, "config_hash": config.hash, "dataset": config.data.dataset, "records": []}
    )
    payload["updated_at"] = _now()
    payload["records"].append(
        {
            "seed": seed,
            "split": split,
            "round": round_index,
            "status": status,
            "accuracy": accuracy,
            "duration_seconds": duration_seconds,
            "error": error,
            "recorded_at": _now(),
        }
    )
    write_json(path, payload)


def _summaries_for(config: Config, arm: str, split: str) -> dict[str, dict[str, Any]]:
    """Every finished run of one arm on one split, keyed by its seed label."""
    found: dict[str, dict[str, Any]] = {}
    directory = paths.arm_dir(config, arm)
    if not directory.is_dir():
        return found
    for seed_dir in sorted(directory.iterdir()):
        if not seed_dir.is_dir() or seed_dir.name == "priors":
            continue
        for summary in sorted((seed_dir / split).glob("*/summary.json")):
            found[seed_dir.name] = read_json(summary)
    return found


def update_comparison(config: Config, *, split: str) -> dict[str, Any]:
    """Rebuild the cross-arm table from disk and write both renderings.

    Reading the summaries back rather than accumulating in memory keeps the table
    correct across separate invocations and after a crash: what it reports is exactly
    what is on disk.
    """
    arms: dict[str, Any] = {}
    for arm in sorted(ARMS):
        summaries = _summaries_for(config, arm, split)
        if not summaries:
            continue
        mean: dict[str, float] = {}
        spread: dict[str, float] = {}
        for metric in COMPARISON_METRICS:
            values = [
                float(summary[metric]) for summary in summaries.values() if metric in summary
            ]
            if not values:
                continue
            mean[metric] = statistics.fmean(values)
            spread[metric] = statistics.stdev(values) if len(values) > 1 else 0.0
        arms[arm] = {
            "runs": len(summaries),
            "seeds": summaries,
            "mean": mean,
            "std": spread,
        }

    table = {
        "config_hash": config.hash,
        "dataset": config.data.dataset,
        "split": split,
        "updated_at": _now(),
        "metrics": list(COMPARISON_METRICS),
        "arms": arms,
    }
    write_json(paths.comparison_path(config, "json"), table)
    paths.comparison_path(config, "md").write_text(render_comparison(table), encoding="utf-8")
    return table


def render_comparison(table: dict[str, Any]) -> str:
    """Markdown table, best accuracy first, so a partial sweep is still readable."""
    header = ["arm", "runs", *COMPARISON_METRICS]
    lines = [
        f"# Arm comparison ({table['dataset']}, {table['split']})",
        "",
        f"Updated {table['updated_at']} | config {table['config_hash'][:12]}",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    ranked = sorted(
        table["arms"].items(),
        key=lambda item: item[1]["mean"].get("pipeline_accuracy", float("-inf")),
        reverse=True,
    )
    for arm, entry in ranked:
        cells = [arm, str(entry["runs"])]
        for metric in COMPARISON_METRICS:
            if metric not in entry["mean"]:
                cells.append("-")
                continue
            value = entry["mean"][metric]
            deviation = entry["std"].get(metric, 0.0)
            cells.append(
                f"{value:.4f}" if entry["runs"] == 1 else f"{value:.4f} ± {deviation:.4f}"
            )
        lines.append("| " + " | ".join(cells) + " |")
    if not ranked:
        lines.append("| (no arm evaluated yet) |" + " |" * len(COMPARISON_METRICS))
    return "\n".join(lines) + "\n"
