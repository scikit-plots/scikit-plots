"""
Fixtures shared by every test of :mod:`scikitplot.annoy`.

Notes
-----
**Developer notes.** The ``scikit-plots-annoy`` distribution ships this tree
without the root ``scikitplot/conftest.py``, so it registers the package's one
log capture itself (Rule 139). The registration lives here, not in
``tests/conftest.py``: ``tests/test_no_residue.py`` copies that file into a
temporary directory and runs it as a suite of its own, where a relative import
of the package cannot resolve.
"""

from __future__ import annotations

import pytest

from ..logging import _capture

captured_logger = pytest.fixture(_capture.captured_logger, name="captured_logger")
