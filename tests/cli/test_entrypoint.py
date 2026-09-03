from __future__ import annotations

import os
import unittest
from typing import Any
from unittest import mock

import fedicl_mqa.cli as entry
from fedicl_mqa.cli import parser

class GpuSelectionTests(unittest.TestCase):
    def test_main_restricts_the_process_to_the_chosen_gpu(self) -> None:
        recorded: dict[str, str | None] = {}

        def fake_command(args: Any) -> None:
            recorded["visible"] = os.environ.get("CUDA_VISIBLE_DEVICES")

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(parser, "command_evaluate_all", fake_command),
            mock.patch.object(entry, "apply_hf_token", lambda *a, **k: "none"),
        ):
            entry.main(["evaluate-all", "--config", "c.yaml", "--gpu", "1"])
        self.assertEqual(recorded["visible"], "1")

    def test_main_leaves_the_variable_alone_without_the_flag(self) -> None:
        recorded: dict[str, str | None] = {}

        def fake_command(args: Any) -> None:
            recorded["visible"] = os.environ.get("CUDA_VISIBLE_DEVICES")

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(parser, "command_evaluate_all", fake_command),
            mock.patch.object(entry, "apply_hf_token", lambda *a, **k: "none"),
        ):
            entry.main(["evaluate-all", "--config", "c.yaml"])
        self.assertIsNone(recorded["visible"])


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
