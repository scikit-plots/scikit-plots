#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
from pathlib import Path

def main():
    tool=Path(__file__).with_name('check_contract.py')
    spec=importlib.util.spec_from_file_location('annoy_contract',tool); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.main()
if __name__=='__main__': raise SystemExit(main())
