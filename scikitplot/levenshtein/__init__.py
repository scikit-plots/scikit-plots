# scikitplot/levenshtein/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Levenshtein distance facade with safe optional acceleration.

``scikitplot.levenshtein`` is always importable.  The default backend chain is
bundled Cython -> RapidFuzz -> dependency-free Python.  The GPL-licensed
``Levenshtein`` package is available only when explicitly requested.
"""

from __future__ import annotations

from . import _core
from ._core import *  # noqa: F403

__all__ = []
__all__ += _core.__all__
