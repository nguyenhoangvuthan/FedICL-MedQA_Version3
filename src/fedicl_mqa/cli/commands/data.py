"""Dataset preparation and retrieval-cohort auditing."""

from __future__ import annotations

import argparse

from fedicl_mqa.cli.paths import data_root, seal_config
from fedicl_mqa.core.io import write_json
from fedicl_mqa.data.leakage import assert_no_support_leakage
from fedicl_mqa.data.preparation import (
    build_partition,
    load_native_dataset,
    load_partition,
    materialize_partition,
)
from fedicl_mqa.data.subjects import audit_subjects
from fedicl_mqa.evaluation.audit import audit_retrieval_cohort


def command_prepare_data(args: argparse.Namespace) -> None:
    config = seal_config(args.config)
    limits = {
        "train": config.data.max_train_samples,
        "validation": config.data.max_validation_samples,
        "test": config.data.max_test_samples,
    }
    subject_filter_audit = {}
    splits = load_native_dataset(
        config.data.dataset,
        config.dataset_id,
        revision=config.data.revision,
        limits=limits,
        **(
            {
                "development_fraction": config.controls.medmcqa_validation_fraction,
                "data_seed": config.experiment.data_seed,
                "subject_filter_audit": subject_filter_audit,
            }
            if config.controls is not None
            else {}
        ),
    )
    assignments, weights = build_partition(
        splits,
        num_clients=config.data.num_clients,
        alpha=config.data.dirichlet_alpha,
        fit_ratio=config.fit_ratio,
        seed=config.experiment.data_seed,
        min_support_per_client=config.data.min_support_per_client,
    )
    root = data_root(config)
    protocol_metadata = None
    if config.controls is not None:
        by_id = {q.example_id: q for values in splits.values() for q in values}
        preview = {
            c: {r: [] for r in ("fit", "support", "validation", "test")}
            for c in range(config.data.num_clients)
        }
        for assignment in assignments:
            preview[assignment.client_id][assignment.role].append(by_id[assignment.example_id])
        subject_audit = audit_subjects(
            preview, min_validation_per_subject=config.controls.min_validation_per_subject
        )
        subject_audit["source_subject_filter"] = subject_filter_audit
        for split, counts in subject_filter_audit["source_splits"].items():
            print(
                f"Native subjects ({split}): retained {counts['retained_count']}/"
                f"{counts['input_count']}; excluded {counts['excluded_count']} "
                f"missing labels {counts['excluded_subject_counts']}",
                flush=True,
            )
        protocol_metadata = {
            "name": "controlled",
            "source_splits": {
                "fit": "train",
                "support": "train",
                "validation": "train holdout",
                "test": "official validation",
            },
            "official_test_used": False,
            "subject_audit": subject_audit,
        }
    materialize_partition(
        root,
        splits,
        assignments,
        dataset_id=config.dataset_id,
        dataset_revision=config.data.revision,
        data_seed=config.experiment.data_seed,
        weights=weights,
        config_hash=config.hash,
        protocol_metadata=protocol_metadata,
    )
    clients = load_partition(root, expected_config_hash=config.hash)
    for values in clients.values():
        assert_no_support_leakage(
            values["support"],
            [*values["validation"], *values["test"]],
            lexical_threshold=config.retrieval.lexical_jaccard_threshold,
        )
    if protocol_metadata is not None:
        write_json(
            root / "subject_audit.json",
            {"config_hash": config.hash, **protocol_metadata["subject_audit"]},
        )
    print(f"Prepared and audited {config.data.dataset} at {root}")


def command_audit_retrieval(args: argparse.Namespace) -> None:
    config = seal_config(args.config)
    output = data_root(config) / "retrieval_audit.json"
    result = audit_retrieval_cohort(
        config,
        load_partition(data_root(config), expected_config_hash=config.hash),
        output,
    )
    print(
        f"Audited Top-{config.retrieval.top_k} capacity for {result['query_count']} queries; "
        f"manifest: {output}"
    )
