"""Predict script template.

Executable-friendly script with strict IO handling.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="predict", description="Load a toy artifact and write predictions."
    )
    p.add_argument(
        "--model",
        type=Path,
        default=Path("model_artifact.txt"),
        help="Path to model artifact",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("predictions.txt"),
        help="Output predictions path",
    )
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return p.parse_args(list(argv) if argv is not None else None)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )


def main(argv: Sequence[str] | None = None) -> int:
    ns = parse_args(argv)
    _configure_logging(ns.log_level)
    log = logging.getLogger("workflow.predict")

    if not ns.model.exists():
        log.error("Model artifact not found: %s", ns.model)
        return 2

    # Placeholder: write deterministic predictions.
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text("id,pred\n0,0.5\n", encoding="utf-8")
    log.info("Wrote predictions: %s", ns.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
