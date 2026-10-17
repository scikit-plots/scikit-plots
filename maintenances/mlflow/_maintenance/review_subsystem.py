#!/usr/bin/env python3
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).parent/'tools/review_subsystem.py'),run_name='__main__')
