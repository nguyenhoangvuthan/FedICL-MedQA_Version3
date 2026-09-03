"""Command-line entry point."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

from fedicl_mqa.cli.parser import build_parser
from fedicl_mqa.core.auth import apply_hf_token

__all__ = ["build_parser", "main"]


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Progress logging is off by default in the library, so configure a handler here.
    # Training runs for hours with no other output; without this the console cannot
    # distinguish a working run from a hung one.
    logging.basicConfig(
        level=logging.WARNING if getattr(args, "quiet", False) else logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        # Restrict the process to one GPU before anything creates a CUDA context. The
        # selected device then appears as cuda:0, so config.hardware.device stays the
        # literal string "cuda" and the sealed config hash is unaffected.
        if getattr(args, "gpu", None) is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        # Authenticate before any command reaches the Hub. Every Hub client in this
        # project reads the token from the environment, so this single call covers
        # model, dataset and encoder downloads alike.
        apply_hf_token()
        args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
