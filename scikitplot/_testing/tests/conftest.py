# scikitplot/_testing/tests/conftest.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
pytest configuration for the _testing sub-package test suite.

Path injection
--------------
Ensures that ``_testing`` is importable as a package whether the tests are
executed:

* As part of the full library  —  ``pytest --pyargs scikitplot._testing``
* From the repository root     —  ``pytest _testing/tests/``
* From the _testing directory  —  ``cd _testing && pytest tests/``

The injection is guarded by a membership check so it is effectively a no-op
when scikitplot is already installed or already on ``sys.path``.

No fixtures are defined here.  Shared state is deliberately avoided to keep
each test hermetic and order-independent.

Notes
-----
Developer note: conftest.py is discovered by pytest before any test module is
imported, so path manipulation here takes effect before relative imports in
test modules are resolved.  Do NOT move path manipulation into individual test
modules — it would run too late and relative imports would fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path injection
# ---------------------------------------------------------------------------
# Directory layout (relative to this file):
#
#   <repo_root>/
#     _testing/              ← the package we want importable
#       tests/
#         conftest.py        ← this file
#
# We need <repo_root> on sys.path so that ``import _testing`` resolves to
# the package directory, which in turn makes ``from .. import X`` work inside
# the test modules.

# _HERE = Path(__file__).resolve()
# _REPO_ROOT = _HERE.parent.parent.parent  # tests/ -> _testing/ -> <repo_root>

# if str(_REPO_ROOT) not in sys.path:
#     sys.path.insert(0, str(_REPO_ROOT))
