from __future__ import annotations

import os
import unittest
from typing import Any
from unittest import mock

import fedicl_mqa.cli as entry
from fedicl_mqa.cli import parser
from fedicl_mqa.core.config import Config


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


class CudaDebugTests(unittest.TestCase):
    def test_debug_is_configured_before_authentication_and_dispatch(self) -> None:
        config = Config()
        original_hash = config.hash
        stages: list[str] = []

        def check_environment(stage: str) -> None:
            self.assertEqual(os.environ["CUDA_LAUNCH_BLOCKING"], "1")
            self.assertEqual(os.environ["TORCH_SHOW_CPP_STACKTRACES"], "1")
            self.assertEqual(os.environ["PYTHONFAULTHANDLER"], "1")
            self.assertEqual(config.hash, original_hash)
            stages.append(stage)

        with (
            mock.patch.dict(os.environ, {"CUDA_LAUNCH_BLOCKING": "0"}, clear=True),
            mock.patch.object(entry.faulthandler, "enable") as enable,
            mock.patch.object(entry, "apply_hf_token", lambda: check_environment("auth")),
            mock.patch.object(parser, "command_pipeline", lambda _: check_environment("command")),
        ):
            entry.main(["pipeline", "--config", "c.yaml", "--cuda-debug"])
        self.assertEqual(stages, ["auth", "command"])
        enable.assert_called_once_with()

    def test_runtime_error_keeps_its_original_traceback_in_debug_mode(self) -> None:
        error = RuntimeError("CUDA error: test failure")
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(entry.faulthandler, "enable"),
            mock.patch.object(entry, "apply_hf_token", lambda: "none"),
            mock.patch.object(parser, "command_doctor", side_effect=error),
            self.assertRaises(RuntimeError) as caught,
        ):
            entry.main(["doctor", "--config", "c.yaml", "--cuda-debug"])
        self.assertIs(caught.exception, error)

    def test_normal_run_does_not_enable_cuda_debugging(self) -> None:
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(entry.faulthandler, "enable") as enable,
            mock.patch.object(entry, "apply_hf_token", lambda: "none"),
            mock.patch.object(parser, "command_doctor", lambda _: None),
        ):
            entry.main(["doctor", "--config", "c.yaml"])
            self.assertNotIn("CUDA_LAUNCH_BLOCKING", os.environ)
            self.assertNotIn("TORCH_SHOW_CPP_STACKTRACES", os.environ)
        enable.assert_not_called()


if __name__ == "__main__":
    unittest.main()
