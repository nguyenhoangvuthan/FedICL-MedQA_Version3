"""Re-stamp a sealed run after an evaluation-only configuration change.

Config.hash covers every configuration field, so raising model.max_seq_length rejects
checkpoints that the change provably cannot affect. max_seq_length is a fail-closed
guard everywhere it is used and never truncates: the tokenizer runs with
truncation=False, and training, generation and likelihood scoring each raise rather than
shorten an input. Anything that fitted the old limit therefore produces identical results
under a larger one.

This script applies such a change to an already-sealed run and re-stamps the artifacts
that record the hash. It refuses any field outside ALLOWED_FIELDS, so it cannot be used
to smuggle through a change that would alter training.

    python scripts/migrate_context_budget.py \\
        --config outputs/a5000/sealed_config.json \\
        --set model.max_seq_length=4096            # dry run, prints the plan
    python scripts/migrate_context_budget.py ... --apply

Two artifacts hash their own manifest: a checkpoint's hashes.json covers its state.json,
and a partition's file_hashes.json covers its partition_manifest.json. Editing the inner
file without recomputing the outer one would leave every checkpoint failing verification,
so both are rewritten together.

Applying this leaves a record under outputs/<name>/migrations/ naming the old and new
hash and every file touched, so the run's provenance still states what happened.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from fedicl_mqa.core.config import Config
from fedicl_mqa.core.io import file_sha256, read_json, write_json

# Fields this migration may change. Each one must be provably unable to alter a training
# input or a completed evaluation; max_seq_length qualifies because every use of it
# raises rather than truncates.
ALLOWED_FIELDS = frozenset({"model.max_seq_length"})


def _parse_override(text: str) -> tuple[str, int]:
    field, _, raw = text.partition("=")
    if not raw:
        raise SystemExit(f"expected field=value, got {text!r}")
    if field not in ALLOWED_FIELDS:
        raise SystemExit(
            f"{field} is not migratable. This script only re-stamps fields that cannot "
            f"change a training input or a completed evaluation: {sorted(ALLOWED_FIELDS)}. "
            "Any other change needs a fresh output_dir and a new run."
        )
    return field, int(raw)


def _apply_override(payload: dict[str, Any], field: str, value: int) -> None:
    section, _, name = field.partition(".")
    payload[section][name] = value


def _checkpoint_dirs(root: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in root.rglob("state.json")
        if (path.parent / "hashes.json").exists()
    )


def _rewrite_json(path: Path, key: str, value: str) -> bool:
    if not path.exists():
        return False
    payload = read_json(path)
    if payload.get(key) == value:
        return False
    payload[key] = value
    write_json(path, payload)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="the run's sealed_config.json")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        required=True,
        metavar="FIELD=VALUE",
        help=f"one of {sorted(ALLOWED_FIELDS)}",
    )
    parser.add_argument("--apply", action="store_true", help="write the changes")
    args = parser.parse_args()

    sealed_path = Path(args.config)
    if sealed_path.name != "sealed_config.json":
        raise SystemExit(f"expected a sealed_config.json, got {sealed_path}")

    payload = read_json(sealed_path)
    before = Config.from_mapping(payload)
    old_hash = before.hash

    for override in args.overrides:
        field, value = _parse_override(override)
        _apply_override(payload, field, value)
    after = Config.from_mapping(payload)
    after.validate()
    new_hash = after.hash

    if new_hash == old_hash:
        raise SystemExit("configuration is unchanged; nothing to migrate")

    root = Path(after.experiment.output_dir)
    data_root = root / "data" / after.data.dataset
    checkpoints = _checkpoint_dirs(root / "training")
    round_states = sorted((root / "training").rglob("round_state.json"))
    summaries = sorted((root / "arms").rglob("summary.json")) if (root / "arms").exists() else []
    logs = sorted((root / "arms").rglob("run_log.json")) if (root / "arms").exists() else []

    print(f"config     : {sealed_path}")
    for override in args.overrides:
        print(f"change     : {override}")
    print(f"old hash   : {old_hash}")
    print(f"new hash   : {new_hash}")
    print(f"checkpoints: {len(checkpoints)} (state.json and hashes.json rewritten)")
    print(f"round work : {len(round_states)}")
    print(f"summaries  : {len(summaries)}")
    print(f"arm logs   : {len(logs)}")
    print(f"partition  : {data_root / 'partition_manifest.json'} (+ file_hashes.json)")
    if not args.apply:
        print("\ndry run; pass --apply to write these changes")
        return

    touched: list[str] = []

    write_json(sealed_path, after.to_dict())
    touched.append(str(sealed_path))

    manifest = data_root / "partition_manifest.json"
    if _rewrite_json(manifest, "config_hash", new_hash):
        touched.append(str(manifest))
        # file_hashes.json covers partition_manifest.json, so it must follow it.
        hashes_path = data_root / "file_hashes.json"
        hashes = read_json(hashes_path)
        hashes["partition_manifest.json"] = file_sha256(manifest)
        write_json(hashes_path, hashes)
        touched.append(str(hashes_path))

    for checkpoint in checkpoints:
        if not _rewrite_json(checkpoint / "state.json", "config_hash", new_hash):
            continue
        # hashes.json covers state.json; recompute it or verification fails.
        digests = read_json(checkpoint / "hashes.json")
        digests["state.json"] = file_sha256(checkpoint / "state.json")
        write_json(checkpoint / "hashes.json", digests)
        touched.append(str(checkpoint))

    for path in [*round_states, *summaries, *logs, data_root / "retrieval_audit.json",
                 root / "arms_comparison.json"]:
        if _rewrite_json(path, "config_hash", new_hash):
            touched.append(str(path))

    state_path = root / "pipeline_state.yaml"
    if state_path.exists():
        state = yaml.safe_load(state_path.read_text(encoding="utf-8"))
        state["config_hash"] = new_hash
        state_path.write_text(
            yaml.safe_dump(state, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        touched.append(str(state_path))

    record = root / "migrations" / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    write_json(
        record,
        {
            "migrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "old_config_hash": old_hash,
            "new_config_hash": new_hash,
            "changes": list(args.overrides),
            "rationale": (
                "max_seq_length is a fail-closed guard, never a truncation limit, so "
                "inputs that fitted the previous value are unchanged by a larger one. "
                "Training sequences were measured at max 1014 tokens against a 2048 "
                "limit, with no truncation."
            ),
            "files": touched,
        },
    )
    print(f"\nmigrated {len(touched)} artifacts; record written to {record}")
    print(json.dumps({"old": old_hash, "new": new_hash}, indent=2))


if __name__ == "__main__":
    main()
