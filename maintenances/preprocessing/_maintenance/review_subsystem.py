#!/usr/bin/env python3
from __future__ import annotations
import runpy
from pathlib import Path
TARGET=Path(__file__).resolve().parent/'tools'/'review_subsystem.py'
runpy.run_path(str(TARGET),run_name='__main__')
