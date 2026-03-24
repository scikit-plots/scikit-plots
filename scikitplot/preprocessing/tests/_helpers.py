# preprocessing/tests/_helpers.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Shared test helpers for the ``preprocessing`` submodule test suite.

These helpers satisfy two requirements:

1. **Isolated execution** — the ``preprocessing`` submodule depends on
   ``scikitplot`` only for version-URL construction.  This module installs a
   minimal in-process stub so tests can run without a full ``scikitplot``
   installation (CI, Docker, bare-repo notebooks).

2. **Fixture factory** — reusable ``pandas.DataFrame`` builders that cover
   common edge cases (NaN, whitespace, mixed-sep, numeric columns, sparse).

Notes
-----
Developer note
    Import ``install_scikitplot_stub()`` **before** importing any module from
    ``preprocessing``.  The stub must be in ``sys.modules`` at import time
    because ``_encoders.py`` executes ``from scikitplot import __version__``
    at module level.

    The helpers are intentionally framework-agnostic: they work with both
    ``unittest`` and ``pytest``.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# scikitplot stub
# ---------------------------------------------------------------------------

def install_scikitplot_stub(version: str = "0.5.0.dev0") -> None:
    """
    Install a minimal ``scikitplot`` stub into ``sys.modules``.

    Only the attributes consumed by ``preprocessing._encoders`` at import time
    are provided: ``scikitplot.__version__`` and
    ``scikitplot.externals._packaging.version.parse``.

    This function is idempotent — calling it multiple times is safe.

    Parameters
    ----------
    version : str, default ``"0.5.0.dev0"``
        Version string forwarded to the stub's ``__version__`` attribute.
    """
    if "scikitplot" in sys.modules:
        return  # already installed

    class _FakeVersion:
        """Minimal version object returned by the stub ``parse()``."""

        def __init__(self, s: str) -> None:
            base = s.split(".dev")[0]
            parts = base.split(".")
            self.major = int(parts[0]) if len(parts) > 0 else 0
            self.minor = int(parts[1]) if len(parts) > 1 else 0
            self.dev = "dev" if ".dev" in s else None

    # Build module hierarchy
    _sk = types.ModuleType("scikitplot")
    _sk.__version__ = version

    _ext = types.ModuleType("scikitplot.externals")
    _pkg = types.ModuleType("scikitplot.externals._packaging")
    _pv = types.ModuleType("scikitplot.externals._packaging.version")
    _pv.parse = lambda s: _FakeVersion(s)

    sys.modules["scikitplot"] = _sk
    sys.modules["scikitplot.externals"] = _ext
    sys.modules["scikitplot.externals._packaging"] = _pkg
    sys.modules["scikitplot.externals._packaging.version"] = _pv


# ---------------------------------------------------------------------------
# DataFrame fixture factory
# ---------------------------------------------------------------------------

def make_tags_df() -> pd.DataFrame:
    """
    Return a standard multi-label DataFrame used across most tests.

    Returns
    -------
    df : pd.DataFrame
        Columns: ``tags`` (multi-label, comma-sep), ``color`` (single-label),
        ``value`` (int).

    Examples
    --------
    >>> df = make_tags_df()
    >>> df.shape
    (4, 3)
    """
    return pd.DataFrame(
        {
            "tags": ["a,b,", " A , b", "a,B,C", None],
            "color": ["red", "blue", "green", "Red"],
            "value": [1, 2, 3, 4],
        }
    )


def make_simple_df() -> pd.DataFrame:
    """
    Return a minimal 3-row DataFrame with no NaN values.

    Returns
    -------
    df : pd.DataFrame
        Columns: ``tags`` (multi-label), ``val`` (int).
    """
    return pd.DataFrame(
        {
            "tags": ["a,b", "b,c", "a"],
            "val": [1, 2, 3],
        }
    )


def make_multi_col_df() -> pd.DataFrame:
    """
    Return a DataFrame with two multi-label string columns.

    Returns
    -------
    df : pd.DataFrame
        Columns: ``tags``, ``color``, ``val``.
    """
    return pd.DataFrame(
        {
            "tags": ["a,b", "b,c", "a"],
            "color": ["red", "blue", "red"],
            "val": [1, 2, 3],
        }
    )


def make_nan_df() -> pd.DataFrame:
    """
    Return a DataFrame with NaN values in the label column.

    Returns
    -------
    df : pd.DataFrame
        Columns: ``tags`` (contains ``None``), ``val``.
    """
    return pd.DataFrame(
        {
            "tags": ["a,b", None, "b,c"],
            "val": [1, 2, 3],
        }
    )


def make_prefix_collision_df() -> pd.DataFrame:
    """
    Return a DataFrame whose column names produce abbreviated-prefix collisions.

    Both ``"ta"`` and ``"tags"`` abbreviate to ``"ta"`` under the first-2-chars
    rule, triggering the collision-fallback path.

    Returns
    -------
    df : pd.DataFrame
        Columns: ``ta``, ``tags``, ``v``.
    """
    return pd.DataFrame(
        {
            "ta": ["a,b", "b"],
            "tags": ["x,y", "y"],
            "v": [1, 2],
        }
    )


def make_mixed_type_df() -> pd.DataFrame:
    """
    Return a DataFrame with a string passthrough column alongside dummy columns.

    This exercises the sparse-output path on mixed-type DataFrames (Bug 3).

    Returns
    -------
    df : pd.DataFrame
        Columns: ``tags`` (multi-label), ``name`` (string passthrough), ``val``.
    """
    return pd.DataFrame(
        {
            "tags": ["a,b", "b,c", "a"],
            "name": ["alice", "bob", "carol"],
            "val": [1, 2, 3],
        }
    )


def make_infrequent_df() -> pd.DataFrame:
    """
    Return a DataFrame suited for testing infrequent-category grouping.

    ``"a"`` appears 3×, ``"b"`` once, ``"c"`` once — so with
    ``min_frequency=2``, only ``"a"`` is frequent.

    Returns
    -------
    df : pd.DataFrame
        Columns: ``tags``, ``val``.
    """
    return pd.DataFrame(
        {
            "tags": ["a", "a", "a", "b", "c"],
            "val": [1, 1, 1, 2, 3],
        }
    )
