"""Hyperparameter optimization (HPO) script template.

Executable-friendly script with ``parse_args`` and ``main``.

This template uses a deterministic grid to stay strict and reproducible.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="hpo", description="Run a deterministic hyperparameter grid search."
    )
    p.add_argument(
        "--out", type=Path, default=Path("hpo_results.json"), help="Output results path"
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
    log = logging.getLogger("workflow.hpo")

    # Deterministic grid (strict, reproducible)
    grid = [
        {"lr": 0.1, "l2": 0.0},
        {"lr": 0.1, "l2": 0.01},
        {"lr": 0.01, "l2": 0.0},
        {"lr": 0.01, "l2": 0.01},
    ]

    results = []
    for hp in grid:
        # Placeholder score; replace with real evaluation.
        score = 1.0 / (1.0 + hp["l2"]) * (hp["lr"] / 0.1)
        results.append({"params": hp, "score": float(score)})

    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    log.info("Wrote HPO results: %s", ns.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
