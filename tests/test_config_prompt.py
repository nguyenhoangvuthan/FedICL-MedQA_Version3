from __future__ import annotations

import unittest

from fedicl_mqa.config import Config
from fedicl_mqa.prompting import build_prompt
from fedicl_mqa.schema import MCQExample


class ConfigAndPromptTests(unittest.TestCase):
    def test_a5000_defaults_use_qwen3_small_model(self) -> None:
        config = Config()
        config.validate()
        self.assertEqual(config.model.id, "Qwen/Qwen3-0.6B")
        self.assertEqual(config.training.train_micro_batch_size, 8)
        self.assertFalse(config.model.gradient_checkpointing)

    def test_unknown_configuration_key_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            Config.from_mapping({"model": {"unknown": True}})

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
