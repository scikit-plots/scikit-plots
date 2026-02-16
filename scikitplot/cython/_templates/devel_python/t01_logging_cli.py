"""Devel Python template: logging + argparse skeleton."""

from __future__ import annotations

import argparse
import logging
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="tool", description="Skeleton CLI")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    ns = parse_args(argv)
    logging.basicConfig(level=getattr(logging, ns.log_level.upper(), logging.INFO))
    logging.getLogger(__name__).info("Hello from template")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
