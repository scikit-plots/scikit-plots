#!/usr/bin/env python3
"""Print the observed _annoy graph without writing files."""
import sys
sys.dont_write_bytecode = True
from cli import main

if __name__ == "__main__":
    raise SystemExit(main(graph_only=True))
