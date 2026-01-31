"""Hard Python template: CLI skeleton with argparse + logging.

This demonstrates canonical "executable-friendly" structure:
- parse_args(argv)
- main(argv) returning an exit code
"""

from __future__ import annotations

import argparse
import logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="hard_cli")
    parser.add_argument("--level", default="INFO", help="Logging level")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.level).upper(), logging.INFO))
    logging.getLogger(__name__).info("Hello from CLI skeleton")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
