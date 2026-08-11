# scikitplot/_cli/_commands/info.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""`scikitplot info` - runtime and installation information."""

from __future__ import annotations

import platform
import sys

from ..context import Context
from ..output import emit


def run(ctx: Context, *, fmt: str = "text") -> int:
    """Emit version/python/platform information in the requested format."""
    try:
        from ... import __version__  # ruff: ignore[import-outside-top-level]
    except Exception:  # pragma: no cover  # ruff: ignore[blind-except]
        __version__ = "unknown"
    data = {
        "scikitplot": {"version": __version__},
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }
    emit(ctx, data)
    return 0
