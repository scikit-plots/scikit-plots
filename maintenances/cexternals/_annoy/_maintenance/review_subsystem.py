#!/usr/bin/env python3
"""Run fixed _annoy review lanes; --release requires all native gates."""
import sys
sys.dont_write_bytecode = True
from cli import main

if __name__ == "__main__":
    raise SystemExit(main())
