from __future__ import annotations

import unittest

from fedicl_mqa.core.config import Config


class ConfigTests(unittest.TestCase):
    def test_a5000_defaults_use_qwen3_small_model(self) -> None:
        config = Config()
        config.validate()
        self.assertEqual(config.model.id, "Qwen/Qwen3-0.6B")
        self.assertEqual(config.training.train_micro_batch_size, 8)
        self.assertFalse(config.model.gradient_checkpointing)

    def test_unknown_configuration_key_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            Config.from_mapping({"model": {"unknown": True}})


if __name__ == "__main__":
    unittest.main()
