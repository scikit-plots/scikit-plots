# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Index (cython) python-api benchmark with examples
=================================================

An example showing the :py:class:`~scikitplot.annoy._annoy.Index` class.
"""

# %%

import pytest
import numpy as np
import random; random.seed(0)
from pprint import pprint

# from annoy import Annoy, AnnoyIndex
# from scikitplot.cexternals._annoy import Annoy, AnnoyIndex
# from scikitplot.annoy import Annoy, AnnoyIndex, Index
from scikitplot.annoy._annoy import Index

# print(Index.__doc__)

# %%

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# pytest ../../../scikitplot/annoy/_annoy/tests/test_benchmark_dtype_combinations.py::test_benchmark_summary_table
cmd = [
    sys.executable,
    "-m",
    "pytest",
    # "../../../scikitplot/annoy/_annoy/tests/test_benchmark_dtype_combinations.py::test_benchmark_summary_table",
    "scikitplot/annoy/_annoy/tests/test_benchmark_dtype_combinations.py::test_benchmark_summary_table",
    "-vv",
]

result = subprocess.run(
    cmd,
    cwd=PROJECT_ROOT,
    check=False,
    capture_output=True,
    text=True,
)

print("Return code:", result.returncode)
print(result.stdout)
print(result.stderr)

# %%
#
# .. tags::
#
#    level: beginner
#    purpose: showcase
