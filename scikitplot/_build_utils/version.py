#!/usr/bin/env python3
"""
Extract version number from __init__.py file.
"""
import os
try:
  scikitplot_init = os.path.join(os.path.dirname(__file__), "../__init__.py")  
  data            = open(scikitplot_init).readlines()  
  version_line    = next(line for line in data if line.startswith("__version__"))  
  version         = version_line.strip().split(" = ")[1].replace('"', "").replace("'", "")
except:
  version = "0.0.0"
if __name__ == "__main__":
  print(version)