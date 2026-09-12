from pathlib import Path
import runpy
ns=runpy.run_path(str(Path(__file__).with_name('tools')/'review_subsystem.py'))
if __name__=='__main__': raise SystemExit(ns['main']())
