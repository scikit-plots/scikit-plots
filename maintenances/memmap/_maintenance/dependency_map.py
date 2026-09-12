#!/usr/bin/env python3
from __future__ import annotations
import runpy, sys
from pathlib import Path
if '--dependencies' not in sys.argv: sys.argv.append('--dependencies')
runpy.run_path(str(Path(__file__).parent/'tools'/'check_contract.py'),run_name='__main__')
