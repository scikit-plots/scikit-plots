"""Test helper: a submodule-style entry point exposing main(argv) -> int."""
from __future__ import annotations
import argparse
import json
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="fake-delegate")
    parser.add_argument("--name", default="world")
    parser.add_argument("--fail", action="store_true")
    args = parser.parse_args(argv)  # raises SystemExit on --help / bad args
    if args.fail:
        return 3
    json.dump({"name": args.name, "argv": list(argv or [])}, sys.stdout)
    sys.stdout.write("\n")
    return 0
