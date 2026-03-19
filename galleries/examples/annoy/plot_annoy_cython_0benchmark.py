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
import inspect
import joblib
from pathlib import Path

def get_current_script_dir() -> Path:
    """
    Returns the directory of the current script in a robust way.

    Returns
    -------
    Path
        Absolute directory path.

    Raises
    ------
    RuntimeError
        If location cannot be determined.
    """
    # Case 1: normal Python execution
    if "__file__" in globals():
        return Path(__file__).resolve().parent

    # Case 2: fallback for Sphinx-gallery / exec environments
    frame = inspect.currentframe()
    if frame is not None:
        file = inspect.getfile(frame)
        return Path(file).resolve().parent

    raise RuntimeError("Cannot determine script directory.")

BASE_DIR = get_current_script_dir()
PROJECT_ROOT = BASE_DIR.parents[2]

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
#    model-workflow: vector-db
#    level: beginner
#    purpose: showcase
