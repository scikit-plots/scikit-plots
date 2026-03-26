# scikitplot/misc/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.misc
===============

Color-table utilities and miscellaneous plotting helpers.

Public functions
----------------
closest_color_name      Find the closest named color(s) to a given hex/named color.
display_colors          Render a list of colors as horizontal bars.
plot_colortable         Display a table of color swatches with names and hex codes.
plot_overlapping_colors Plot CSS4 vs XKCD overlapping color names side by side.

Usage
-----
Run the test suite for this submodule from Python::

    import scikitplot.misc

    scikitplot.misc.test()

Or with extra arguments::

    scikitplot.misc.test(label="full", verbose=2, coverage=True)
"""  # noqa: D205, D400

from __future__ import annotations

from . import _plot_colortable
from ._plot_colortable import *  # noqa: F403

__all__ = []
__all__ += _plot_colortable.__all__

# ---------------------------------------------------------------------------
# Re-point __module__ on every public symbol to 'scikitplot.misc' so that
# help(), repr(), and IDE tooling show the public path, not the private one.
#
# Developer note: without this, functions imported via ``from ._plot_colortable
# import *`` keep __module__ = 'scikitplot.misc._plot_colortable'.
# export_all() walks __all__ and sets __module__ = __name__ on each object.
# ---------------------------------------------------------------------------
# from .._utils._export import export_all as _export_all  # noqa: E402
# _export_all(globals(), public_module=__name__)
# del _export_all

# ---------------------------------------------------------------------------
# numpy-style test runner
# ---------------------------------------------------------------------------
# Developer note
#   PytestTester is imported from the sibling ``_testing`` package (one level
#   up) using a relative import.  This preserves the private-submodule path
#   and avoids any absolute-import dependency on the installed package name.
#
#   Pattern mirrors NumPy / SciPy conventions:
#       from numpy._pytesttester import PytestTester
#       test = PytestTester(__name__)
#       del PytestTester
#
#   If _testing is unavailable (e.g., a stripped distribution), the ``test``
#   attribute is silently omitted rather than breaking the import.
# ---------------------------------------------------------------------------
try:
    from .._testing._pytesttester import PytestTester as _PytestTester

    test = _PytestTester(__name__)
    del _PytestTester
except ImportError:
    pass
