"""Argument parsing. Wiring only: every handler lives under cli.commands."""

from __future__ import annotations

import argparse

from fedicl_mqa.cli.commands.data import command_audit_retrieval, command_prepare_data
from fedicl_mqa.cli.commands.diagnostics import command_doctor
from fedicl_mqa.cli.commands.evaluation import (
    command_evaluate,
    command_evaluate_all,
    command_evaluate_arm,
    command_report,
)
from fedicl_mqa.cli.pipeline import command_pipeline
from fedicl_mqa.cli.commands.training import (
    command_build_priors,
    command_select_round,
    command_train,
)
from fedicl_mqa.evaluation.arms import ARMS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fedicl-mqa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared so --gpu can be written after the subcommand rather than before it.
    gpu = argparse.ArgumentParser(add_help=False)
    gpu.add_argument(
        "--gpu",
        type=int,
        choices=[0, 1],
        help="run on this physical GPU; leaves the sealed config hash untouched",
    )
    gpu.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-batch training progress; warnings and errors still print",
    )

    prepare = subparsers.add_parser(
        "prepare-data", parents=[gpu], help="download, normalize and partition data"
    )
    prepare.add_argument("--config", required=True)
    prepare.set_defaults(func=command_prepare_data)

    audit = subparsers.add_parser(
        "audit-retrieval",
        parents=[gpu],
        help="verify Top-5 capacity and freeze candidate manifests",
    )
    audit.add_argument("--config", required=True)
    audit.set_defaults(func=command_audit_retrieval)

    training = subparsers.add_parser(
        "train", parents=[gpu], help="train Local, Federated or Central LoRA"
    )
    training.add_argument("--config", required=True)
    training.add_argument("--mode", choices=["local", "federated", "centralized"], required=True)
    seed_group = training.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument("--all-seeds", action="store_true")
    training.add_argument("--fl-round", type=int, choices=[4, 6, 8])
    training.add_argument("--resume", default="auto")
    training.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("evaluate", parents=[gpu], help="run one baseline arm")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--arm", choices=sorted(ARMS), required=True)
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--split", choices=["validation", "test"], default="test")
    evaluate.add_argument("--round", type=int, choices=[4, 6, 8])
    evaluate.add_argument("--subject-weights")
    evaluate.set_defaults(func=command_evaluate)

    select = subparsers.add_parser(
        "select-round", parents=[gpu], help="select FL round using validation only"
    )
    select.add_argument("--config", required=True)
    select.set_defaults(func=command_select_round)

    priors = subparsers.add_parser(
        "build-priors", parents=[gpu], help="build F2 leave-one-client-out prior"
    )
    priors.add_argument("--config", required=True)
    priors.add_argument("--seed", type=int, required=True)
    priors.add_argument("--round", type=int, choices=[4, 6, 8])
    priors.set_defaults(func=command_build_priors)

    report = subparsers.add_parser(
        "report", parents=[gpu], help="bootstrap the six primary contrasts"
    )
    report.add_argument("--config", required=True)
    report.set_defaults(func=command_report)

    sweep = subparsers.add_parser(
        "evaluate-arm",
        parents=[gpu],
        help="evaluate one arm across every configured seed on the test split",
    )
    sweep.add_argument("--config", required=True)
    sweep.add_argument("--arm", choices=sorted(ARMS), required=True)
    sweep.add_argument("--split", choices=["validation", "test"], default="test")
    sweep.add_argument("--round", type=int, choices=[4, 6, 8])
    sweep.add_argument("--force", action="store_true", help="re-run evaluations already on disk")
    sweep.set_defaults(func=command_evaluate_arm)

    sweep_all = subparsers.add_parser(
        "evaluate-all",
        parents=[gpu],
        help="evaluate every arm across every configured seed on the test split",
    )
    sweep_all.add_argument("--config", required=True)
    sweep_all.add_argument("--split", choices=["validation", "test"], default="test")
    sweep_all.add_argument("--round", type=int, choices=[4, 6, 8])
    sweep_all.add_argument(
        "--force", action="store_true", help="re-run evaluations already on disk"
    )
    sweep_all.set_defaults(func=command_evaluate_all)

    whole = subparsers.add_parser(
        "pipeline",
        parents=[gpu],
        help="run every step from data preparation to the contrast report, resumably",
    )
    whole.add_argument("--config", required=True)
    whole.add_argument("--split", choices=["validation", "test"], default="test")
    whole.add_argument(
        "--force", action="store_true", help="re-run every step, ignoring finished work"
    )
    whole.set_defaults(func=command_pipeline)

    doctor = subparsers.add_parser(
        "doctor", parents=[gpu], help="check the local GPU and ML dependencies"
    )
    doctor.add_argument("--config", required=True)
    doctor.set_defaults(func=command_doctor)
    return parser
