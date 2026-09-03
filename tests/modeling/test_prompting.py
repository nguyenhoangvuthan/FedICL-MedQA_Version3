from __future__ import annotations

import unittest

from fedicl_mqa.core.schema import MCQExample
from fedicl_mqa.modeling.prompting import build_prompt


class PromptTests(unittest.TestCase):
    def test_prompt_contains_exact_terminal_contract(self) -> None:
        item = MCQExample(
            example_id="q",
            question="Question?",
            options=("a", "b", "c", "d"),
            label=1,
            split="test",
        )
        prompt = build_prompt(item)
        self.assertIn("Final answer: X", prompt.system)
        self.assertIn("B. b", prompt.user)


if __name__ == "__main__":
    unittest.main()
