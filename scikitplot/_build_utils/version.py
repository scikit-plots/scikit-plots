#!/usr/bin/env python
"""Extract version number from __init__.py file."""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

import os

try:
    scikitplot_init = os.path.join(os.path.dirname(__file__), "../__init__.py")
    # data            = open(scikitplot_init).readlines()
    with open(scikitplot_init, encoding="utf-8") as f:
        data = f.readlines()
    version_line = next(line for line in data if line.startswith("__version__"))
    # print(re.search(r'__version__\s*=\s*[\"\\'](.*?)[\"\\']', f.read()).group(1))
    version = (
        version_line.strip().split(" = ")[1].replace('"', "").replace("'", "").strip()
    )
except:
    version = "0.0.0"
if __name__ == "__main__":
    print(version)
