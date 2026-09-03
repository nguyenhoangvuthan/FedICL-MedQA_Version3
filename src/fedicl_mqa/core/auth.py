from __future__ import annotations

import os
from pathlib import Path

# The token file lives in the repository root and is git-ignored. Both spellings are
# accepted so a token saved without an extension still works.
TOKEN_FILENAMES = ("HF_Access_Token.txt", "HF_Access_Token")

# huggingface_hub reads HF_TOKEN; transformers, datasets and sentence-transformers all
# authenticate through huggingface_hub, so exporting this one variable covers every
# Hub call in the project. HUGGING_FACE_HUB_TOKEN is the legacy name still honoured by
# older releases of the same libraries.
TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

# Marker files that identify the repository root when the CLI is invoked from a
# subdirectory.
_ROOT_MARKERS = ("pyproject.toml", ".git")


def find_repo_root(start: str | Path | None = None) -> Path:
    """Walk upward from `start` to the directory holding pyproject.toml or .git."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    return current


def find_token_file(root: str | Path | None = None) -> Path | None:
    """Return the token file in the repository root, or None if absent."""
    base = Path(root) if root is not None else find_repo_root()
    for name in TOKEN_FILENAMES:
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


def read_token_file(path: str | Path) -> str:
    """Read a token from disk, tolerating a trailing newline or surrounding blanks.

    Uses utf-8-sig because Notepad on Windows Server writes UTF-8 with a byte order
    mark. A BOM is not whitespace, so str.strip() would leave it attached to the token
    and the Hub would reject the request as unauthorized. utf-8-sig removes a BOM when
    present and behaves exactly like utf-8 when it is not.

    Raises ValueError on an empty file so a half-finished setup fails loudly instead of
    silently falling through to anonymous access.
    """
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"token file is empty: {path}")
    if len(text.splitlines()) > 1:
        raise ValueError(f"token file must contain a single line: {path}")
    return text


def token_from_env() -> str | None:
    """Return the first non-empty token set in the environment, if any."""
    for name in TOKEN_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def resolve_hf_token(root: str | Path | None = None) -> tuple[str | None, str]:
    """Resolve the Hugging Face access token for this run.

    Returns a `(token, source)` pair, where `source` is one of "environment", "file" or
    "none". The token itself must never be logged; callers report only the source.

    The file in the repository root wins over the environment. The deployment target is
    Windows Server, where HF_TOKEN is typically set permanently at machine or user
    scope rather than per command: a stale value set months ago would otherwise
    silently override the token sitting visibly in the checkout. File-first also keeps
    the checkout self-describing, which matters here because this project seals its
    configuration for reproducibility.

    Returning (None, "none") is a normal outcome, not an error. Every dataset and model
    in the default configs is public, so anonymous access still works; only gated repos
    need a token.
    """
    path = find_token_file(root)
    if path is not None:
        return read_token_file(path), "file"
    from_env = token_from_env()
    if from_env is not None:
        return from_env, "environment"
    return None, "none"


def apply_hf_token(root: str | Path | None = None) -> str:
    """Resolve the token and export it so every Hub client authenticates.

    Returns the source string for reporting. Existing environment variables are
    overwritten only when the resolved token differs, so an already-correct
    environment is left untouched.
    """
    token, source = resolve_hf_token(root)
    if token is None:
        return source
    for name in TOKEN_ENV_VARS:
        os.environ[name] = token
    return source
