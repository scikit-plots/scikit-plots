#!/usr/bin/env python3
"""Check _annoy maintenance; --update reconciles derived inventories only."""
import sys
sys.dont_write_bytecode = True
from cli import main

if __name__ == "__main__":
    raise SystemExit(main(allow_update=True))
