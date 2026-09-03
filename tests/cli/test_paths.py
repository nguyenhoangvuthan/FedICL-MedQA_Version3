from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fedicl_mqa.cli import paths


class MissingConfigTests(unittest.TestCase):
    """A sealed config that was never written must say which command writes it."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_missing_sealed_config_names_prepare_data(self) -> None:
        missing = self.root / "outputs" / "a5000" / "sealed_config.json"
        with self.assertRaises(FileNotFoundError) as caught:
            paths._require_existing_config(missing)
        message = str(caught.exception)
        self.assertIn("prepare-data", message)
        self.assertIn("configs/a5000.yaml", message)

    def test_suggestion_follows_the_output_directory_name(self) -> None:
        missing = self.root / "outputs" / "a5000-smoke" / "sealed_config.json"
        with self.assertRaises(FileNotFoundError) as caught:
            paths._require_existing_config(missing)
        self.assertIn("configs/a5000-smoke.yaml", str(caught.exception))

    def test_missing_yaml_config_does_not_suggest_prepare_data(self) -> None:
        with self.assertRaises(FileNotFoundError) as caught:
            paths._require_existing_config(self.root / "configs" / "nope.yaml")
        self.assertNotIn("prepare-data", str(caught.exception))

    def test_existing_file_is_returned_unchanged(self) -> None:
        present = self.root / "a5000.yaml"
        present.write_text("experiment: {}\n", encoding="utf-8")
        self.assertEqual(paths._require_existing_config(present), present)


if __name__ == "__main__":
    unittest.main()
