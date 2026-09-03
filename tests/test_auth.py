from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fedicl_mqa.auth import (
    TOKEN_ENV_VARS,
    apply_hf_token,
    find_repo_root,
    find_token_file,
    read_token_file,
    resolve_hf_token,
    token_from_env,
)


class TokenFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_find_token_file_returns_none_when_absent(self) -> None:
        self.assertIsNone(find_token_file(self.root))

    def test_find_token_file_accepts_both_spellings(self) -> None:
        target = self.root / "HF_Access_Token"
        target.write_text("hf_bare\n", encoding="utf-8")
        self.assertEqual(find_token_file(self.root), target)

    def test_txt_spelling_wins_when_both_exist(self) -> None:
        (self.root / "HF_Access_Token").write_text("hf_bare\n", encoding="utf-8")
        preferred = self.root / "HF_Access_Token.txt"
        preferred.write_text("hf_txt\n", encoding="utf-8")
        self.assertEqual(find_token_file(self.root), preferred)

    def test_read_token_file_strips_trailing_newline(self) -> None:
        path = self.root / "HF_Access_Token.txt"
        path.write_text("  hf_secret_value\n", encoding="utf-8")
        self.assertEqual(read_token_file(path), "hf_secret_value")

    def test_reads_token_written_by_windows_notepad(self) -> None:
        """Notepad writes UTF-8 with a BOM and CRLF; neither may reach the token."""
        path = self.root / "HF_Access_Token.txt"
        path.write_bytes("\ufeffhf_windows_value\r\n".encode("utf-8"))
        self.assertEqual(read_token_file(path), "hf_windows_value")

    def test_empty_token_file_fails_closed(self) -> None:
        path = self.root / "HF_Access_Token.txt"
        path.write_text("\n   \n", encoding="utf-8")
        with self.assertRaises(ValueError):
            read_token_file(path)

    def test_multiline_token_file_fails_closed(self) -> None:
        path = self.root / "HF_Access_Token.txt"
        path.write_text("hf_one\nhf_two\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            read_token_file(path)

    def test_find_repo_root_walks_up_to_marker(self) -> None:
        (self.root / "pyproject.toml").write_text("", encoding="utf-8")
        nested = self.root / "a" / "b"
        nested.mkdir(parents=True)
        self.assertEqual(find_repo_root(nested), self.root.resolve())


class TokenEnvTests(unittest.TestCase):
    def test_token_from_env_ignores_blank_values(self) -> None:
        with mock.patch.dict(os.environ, {"HF_TOKEN": "   "}, clear=True):
            self.assertIsNone(token_from_env())

    def test_token_from_env_reads_legacy_variable(self) -> None:
        with mock.patch.dict(os.environ, {"HUGGING_FACE_HUB_TOKEN": "hf_legacy"}, clear=True):
            self.assertEqual(token_from_env(), "hf_legacy")


class ResolveTokenTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_no_token_anywhere_reports_none_without_raising(self) -> None:
        """Public datasets and models must still work anonymously."""
        with mock.patch.dict(os.environ, {}, clear=True):
            token, source = resolve_hf_token(self.root)
        self.assertIsNone(token)
        self.assertEqual(source, "none")

    def test_file_only_is_used(self) -> None:
        (self.root / "HF_Access_Token.txt").write_text("hf_from_file\n", encoding="utf-8")
        with mock.patch.dict(os.environ, {}, clear=True):
            token, source = resolve_hf_token(self.root)
        self.assertEqual(token, "hf_from_file")
        self.assertEqual(source, "file")

    def test_env_only_is_used(self) -> None:
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_from_env"}, clear=True):
            token, source = resolve_hf_token(self.root)
        self.assertEqual(token, "hf_from_env")
        self.assertEqual(source, "environment")

    def test_file_wins_over_environment(self) -> None:
        """A stale machine-scope HF_TOKEN must not shadow the token in the checkout."""
        (self.root / "HF_Access_Token.txt").write_text("hf_from_file\n", encoding="utf-8")
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_stale_env"}, clear=True):
            token, source = resolve_hf_token(self.root)
        self.assertEqual(token, "hf_from_file")
        self.assertEqual(source, "file")

    def test_apply_exports_every_supported_variable(self) -> None:
        (self.root / "HF_Access_Token.txt").write_text("hf_exported\n", encoding="utf-8")
        with mock.patch.dict(os.environ, {}, clear=True):
            source = apply_hf_token(self.root)
            exported = {name: os.environ.get(name) for name in TOKEN_ENV_VARS}
        self.assertEqual(source, "file")
        self.assertEqual(set(exported.values()), {"hf_exported"})

    def test_apply_leaves_environment_clean_when_no_token(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            source = apply_hf_token(self.root)
            present = [name for name in TOKEN_ENV_VARS if name in os.environ]
        self.assertEqual(source, "none")
        self.assertEqual(present, [])


if __name__ == "__main__":
    unittest.main()
