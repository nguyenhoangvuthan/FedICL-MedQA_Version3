"""Dataset preparation and retrieval-cohort auditing."""

from __future__ import annotations

import argparse

from fedicl_mqa.evaluation.audit import audit_retrieval_cohort
from fedicl_mqa.data.preparation import (
    build_partition,
    load_native_dataset,
    load_partition,
    materialize_partition,
    resolve_hub_revision,
)
from fedicl_mqa.data.leakage import assert_no_support_leakage
from fedicl_mqa.cli.paths import data_root, seal_config

def command_prepare_data(args: argparse.Namespace) -> None:
    config = seal_config(args.config)
    limits = {
        "train": config.data.max_train_samples,
        "validation": config.data.max_validation_samples,
        "test": config.data.max_test_samples,
    }
    splits = load_native_dataset(
        config.data.dataset,
        config.dataset_id,
        revision=config.data.revision,
        limits=limits,
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
    materialize_partition(
        root,
        splits,
        assignments,
        dataset_id=config.dataset_id,
        dataset_revision=config.data.revision,
        data_seed=config.experiment.data_seed,
        weights=weights,
        config_hash=config.hash,
    )
    clients = load_partition(root, expected_config_hash=config.hash)
    for values in clients.values():
        assert_no_support_leakage(
            values["support"],
            [*values["validation"], *values["test"]],
            lexical_threshold=config.retrieval.lexical_jaccard_threshold,
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

