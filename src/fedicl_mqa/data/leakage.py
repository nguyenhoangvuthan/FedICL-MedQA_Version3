from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from fedicl_mqa.core.schema import MCQExample


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


class LeakageIndex:
    """Exact metadata checks and a lossless candidate index for token Jaccard.

    A match at threshold t shares at least ceil(t * len(query)) query tokens.
    Thus it must share one of any len(query) - ceil(t * len(query)) + 1 tokens.
    Query the rarest such prefix to avoid scanning common-word postings, then
    compute the exact score. No approximate search or answer labels are used.
    """

    def __init__(
        self,
        examples: Sequence[MCQExample],
        *,
        lexical_threshold: float = 0.85,
        check_provenance: bool = True,
    ) -> None:
        if not 0 < lexical_threshold <= 1:
            raise ValueError("lexical threshold must be in (0, 1]")
        self.examples = sorted(examples, key=lambda q: q.example_id)
        self.threshold = lexical_threshold
        self.check_provenance = check_provenance
        self.exact: dict[str, dict[str, list[int]]] = {
            kind: defaultdict(list)
            for kind in ("id", "normalized_question", "question_options_hash", "provenance_group")
        }
        self.tokens: list[frozenset[str]] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        for index, q in enumerate(self.examples):
            for kind, value in self._keys(q):
                self.exact[kind][value].append(index)
            tokens = frozenset(q.normalized_question.split())
            self.tokens.append(tokens)
            for token in tokens:
                self.postings[token].append(index)

    def _keys(self, q: MCQExample) -> Iterator[tuple[str, str]]:
        yield "id", q.example_id
        yield "normalized_question", q.normalized_question
        yield "question_options_hash", q.question_options_hash
        if self.check_provenance and q.provenance_group:
            yield "provenance_group", q.provenance_group

    def matches(self, query: MCQExample) -> Iterator[LeakageIssue]:
        for kind, value in self._keys(query):
            for index in self.exact[kind].get(value, ()):
                yield LeakageIssue(kind, self.examples[index].example_id, query.example_id, value)
        normalized = query.normalized_question
        tokens = frozenset(normalized.split())
        # floor instead of ceil is deliberately conservative at float boundaries.
        prefix_length = min(len(tokens), len(tokens) - math.floor(self.threshold * len(tokens)) + 1)
        prefix = sorted(tokens, key=lambda t: (len(self.postings.get(t, ())), t))[:prefix_length]
        candidates: set[int] = set()
        for token in prefix:
            candidates.update(self.postings.get(token, ()))
        for index in sorted(candidates):
            other = self.tokens[index]
            if min(len(tokens), len(other)) / max(len(tokens), len(other)) < self.threshold:
                continue
            similarity = len(tokens & other) / len(tokens | other)
            if (
                similarity >= self.threshold
                and self.examples[index].normalized_question != normalized
            ):
                yield LeakageIssue(
                    "lexical_near_duplicate",
                    self.examples[index].example_id,
                    query.example_id,
                    f"{similarity:.6f}",
                )


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
