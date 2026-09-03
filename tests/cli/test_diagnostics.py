from __future__ import annotations

import os
import unittest
from unittest import mock

from fedicl_mqa.cli.commands import diagnostics

class CudaDiagnosisTests(unittest.TestCase):
    """doctor must name the actual cause, not just report that CUDA is missing."""

    def test_cpu_only_build_is_named_explicitly(self) -> None:
        message = diagnostics._cuda_failure_reason(torch_version="2.14.0+cpu", cuda_build=None)
        self.assertIn("CPU-only", message)
        self.assertIn("2.14.0+cpu", message)
        self.assertIn("download.pytorch.org", message)

    def test_cuda_build_without_a_visible_device_reports_the_driver_side(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            message = diagnostics._cuda_failure_reason(
                torch_version="2.14.0+cu128", cuda_build="12.8"
            )
        self.assertNotIn("CPU-only", message)
        self.assertIn("driver", message.lower())

    def test_message_surfaces_a_restrictive_visible_device_list(self) -> None:
        """--gpu 3 on a two-GPU box must not look like a broken driver."""
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}, clear=True):
            message = diagnostics._cuda_failure_reason(
                torch_version="2.14.0+cu128", cuda_build="12.8"
            )
        self.assertIn("CUDA_VISIBLE_DEVICES", message)
        self.assertIn("3", message)


if __name__ == "__main__":
    unittest.main()
