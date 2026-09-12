#!/usr/bin/env python3
"""Sample-compatible entrypoint backed by one maintenance implementation."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cli import main
if __name__ == "__main__":
    raise SystemExit(main(allow_update=True))
