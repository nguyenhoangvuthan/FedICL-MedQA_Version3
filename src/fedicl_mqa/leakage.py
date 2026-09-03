from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .schema import MCQExample


@dataclass(frozen=True, slots=True)
class LeakageIssue:
    kind: str
    support_id: str
    evaluation_id: str
    value: str


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def audit_support_leakage(
    support: Sequence[MCQExample],
    evaluation: Iterable[MCQExample],
    *,
    lexical_threshold: float = 0.85,
    check_provenance: bool = True,
    max_issues: int = 1_000,
) -> list[LeakageIssue]:
    issues: list[LeakageIssue] = []
    support_by_id: dict[str, list[MCQExample]] = defaultdict(list)
    support_by_question: dict[str, list[MCQExample]] = defaultdict(list)
    support_by_hash: dict[str, list[MCQExample]] = defaultdict(list)
    support_by_provenance: dict[str, list[MCQExample]] = defaultdict(list)
    token_sets: list[frozenset[str]] = []
    postings: dict[str, list[int]] = defaultdict(list)
    for candidate in support:
        if candidate.split != "train":
            issues.append(
                LeakageIssue("support_not_train", candidate.example_id, "", candidate.split)
            )
        support_by_id[candidate.example_id].append(candidate)
        support_by_question[candidate.normalized_question].append(candidate)
        support_by_hash[candidate.question_options_hash].append(candidate)
        if candidate.provenance_group:
            support_by_provenance[candidate.provenance_group].append(candidate)
        tokens = frozenset(candidate.normalized_question.split())
        token_sets.append(tokens)
        for token in tokens:
            postings[token].append(len(token_sets) - 1)

    evaluation_values = list(evaluation)
    for query in evaluation_values:
        for candidate in support_by_id.get(query.example_id, ()):
            issues.append(
                LeakageIssue("id", candidate.example_id, query.example_id, query.example_id)
            )
        for candidate in support_by_question.get(query.normalized_question, ()):
            issues.append(
                LeakageIssue(
                    "normalized_question",
                    candidate.example_id,
                    query.example_id,
                    query.normalized_question,
                )
            )
        for candidate in support_by_hash.get(query.question_options_hash, ()):
            issues.append(
                LeakageIssue(
                    "question_options_hash",
                    candidate.example_id,
                    query.example_id,
                    query.question_options_hash,
                )
            )
        if check_provenance and query.provenance_group:
            for candidate in support_by_provenance.get(query.provenance_group, ()):
                issues.append(
                    LeakageIssue(
                        "provenance_group",
                        candidate.example_id,
                        query.example_id,
                        query.provenance_group,
                    )
                )

        query_tokens = frozenset(query.normalized_question.split())
        intersections: Counter[int] = Counter()
        for token in query_tokens:
            intersections.update(postings.get(token, ()))
        for index, intersection in intersections.items():
            candidate_tokens = token_sets[index]
            # Exact lower bound for Jaccard >= threshold; avoids scoring almost all pairs.
            minimum = math.ceil(
                lexical_threshold
                / (1.0 + lexical_threshold)
                * (len(query_tokens) + len(candidate_tokens))
            )
            if intersection < minimum:
                continue
            similarity = intersection / len(query_tokens | candidate_tokens)
            candidate = support[index]
            if (
                similarity >= lexical_threshold
                and candidate.normalized_question != query.normalized_question
            ):
                issues.append(
                    LeakageIssue(
                        "lexical_near_duplicate",
                        candidate.example_id,
                        query.example_id,
                        f"{similarity:.6f}",
                    )
                )
            if len(issues) >= max_issues:
                return issues
    return issues


def assert_no_support_leakage(
    support: Sequence[MCQExample],
    evaluation: Iterable[MCQExample],
    *,
    lexical_threshold: float = 0.85,
    check_provenance: bool = True,
) -> None:
    issues = audit_support_leakage(
        support,
        evaluation,
        lexical_threshold=lexical_threshold,
        check_provenance=check_provenance,
        max_issues=1_000,
    )
    if issues:
        preview = ", ".join(
            f"{issue.kind}:{issue.support_id}->{issue.evaluation_id}" for issue in issues[:10]
        )
        raise ValueError(f"support leakage audit failed with {len(issues)} issue(s): {preview}")
