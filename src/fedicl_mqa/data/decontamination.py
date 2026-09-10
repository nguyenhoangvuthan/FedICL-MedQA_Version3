"""Remove cross-split question overlap before controlled client assignment."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from fedicl_mqa.core.schema import MCQExample
from fedicl_mqa.data.leakage import LeakageIndex


def decontaminate_splits(
    splits: Mapping[str, Sequence[MCQExample]],
    *,
    lexical_threshold: float,
) -> tuple[dict[str, list[MCQExample]], dict[str, Any]]:
    """Preserve final test; remove overlap from validation, then from train.

    Protects both fit and support, globally across all future clients. Existing
    within-split duplicates are not removed. Exclusion uses only question/options,
    IDs and provenance, never correct answers, subjects or model predictions.
    """
    retained = {s: list(splits[s]) for s in ("train", "validation", "test")}
    audit: dict[str, Any] = {
        "policy": "global_cross_split_exclusion_v1",
        "priority": ["test", "validation", "train"],
        "lexical_threshold": lexical_threshold,
        "check_provenance": True,
        "input_counts": {s: len(qs) for s, qs in retained.items()},
        "exclusions": {},
    }
    for split, protected_splits in (("validation", ("test",)), ("train", ("test", "validation"))):
        protected = [q for s in protected_splits for q in retained[s]]
        index = LeakageIndex(protected, lexical_threshold=lexical_threshold)
        protected_roles = {q.example_id: q.split for q in protected}
        excluded = []
        kept = []
        for q in retained[split]:
            issue = next(index.matches(q), None)
            if issue is None:
                kept.append(q)
            else:
                excluded.append(
                    {
                        "example_id": q.example_id,
                        "matched_id": issue.support_id,
                        "matched_split": protected_roles[issue.support_id],
                        "kind": issue.kind,
                    }
                )
        retained[split] = kept
        audit["exclusions"][split] = {
            "count": len(excluded),
            "reason_counts": dict(sorted(Counter(row["kind"] for row in excluded).items())),
            "records": sorted(excluded, key=lambda row: row["example_id"]),
        }
        if not kept:
            raise ValueError(
                f"decontamination leaves no {split} examples; use a larger predeclared cohort"
            )
    audit["retained_counts"] = {s: len(qs) for s, qs in retained.items()}
    return retained, audit


def assert_disjoint_splits(
    splits: Mapping[str, Sequence[MCQExample]],
    *,
    lexical_threshold: float,
) -> None:
    """Verify the global boundary again after filtering or loading artifacts."""
    for split, protected_splits in (("validation", ("test",)), ("train", ("test", "validation"))):
        index = LeakageIndex(
            [q for s in protected_splits for q in splits[s]],
            lexical_threshold=lexical_threshold,
        )
        for q in splits[split]:
            issue = next(index.matches(q), None)
            if issue is not None:
                raise ValueError(
                    f"global split leakage: {split}/{q.example_id} -> "
                    f"{issue.support_id} ({issue.kind})"
                )
