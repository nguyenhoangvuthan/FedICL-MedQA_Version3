from __future__ import annotations

import unittest

from fedicl_mqa.leakage import audit_support_leakage
from fedicl_mqa.matching import AnswerMatcher
from fedicl_mqa.schema import MCQExample


def item(example_id: str, question: str, split: str) -> MCQExample:
    return MCQExample(
        example_id=example_id,
        question=question,
        options=("Vitamin A", "Vitamin B", "Vitamin C", "Vitamin D"),
        label=2,
        split=split,
    )


class LeakageAndMatchingTests(unittest.TestCase):
    def test_leakage_detects_normalized_and_lexical_duplicates(self) -> None:
        support = [item("train-1", "What deficiency causes scurvy?", "train")]
        exact = item("test-1", "WHAT deficiency causes scurvy!", "test")
        near = item("test-2", "What deficiency causes severe scurvy?", "test")
        issues = audit_support_leakage(support, [exact, near], lexical_threshold=0.75)
        kinds = {issue.kind for issue in issues}
        self.assertIn("normalized_question", kinds)
        self.assertIn("lexical_near_duplicate", kinds)

    def test_matcher_state_machine(self) -> None:
        query = item("test-1", "What deficiency causes scurvy?", "test")
        matcher = AnswerMatcher()
        terminal = matcher.match("Reasoning...\nFinal answer: C", query)
        exact = matcher.match("Vitamin C", query)
        unresolved = matcher.match("I am unsure", query)
        self.assertEqual((terminal.label, terminal.stage), (2, "terminal_label"))
        self.assertEqual((exact.label, exact.stage), (2, "exact_option"))
        self.assertFalse(unresolved.resolved)


if __name__ == "__main__":
    unittest.main()
