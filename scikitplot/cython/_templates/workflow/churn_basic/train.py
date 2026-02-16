"""Train script template.

This is an **executable-friendly** workflow script shipped as package data.

Rules
-----
- Defines ``parse_args`` and ``main``.
- Uses logging and returns an int exit code.
- Safe defaults: does not require external services.

Replace the toy model with your real pipeline.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse training arguments."""
    p = argparse.ArgumentParser(
        prog="train", description="Train a toy model and write an artifact."
    )
    p.add_argument(
        "--data", type=Path, required=False, default=None, help="Path to a CSV dataset"
    )
    p.add_argument(
        "--out",
        type=Path,
        required=False,
        default=Path("model_artifact.txt"),
        help="Output artifact path",
    )
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return p.parse_args(list(argv) if argv is not None else None)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    ns = parse_args(argv)
    _configure_logging(ns.log_level)
    log = logging.getLogger("workflow.train")

    # Minimal placeholder logic: write a small artifact.
    if ns.data is not None:
        log.info("Using data: %s", ns.data)
        if not ns.data.exists():
            log.error("Data not found: %s", ns.data)
            return 2

    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text("trained=true\n", encoding="utf-8")
    log.info("Wrote artifact: %s", ns.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
