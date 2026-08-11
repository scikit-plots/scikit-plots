# scikitplot/_cli/_commands/show_versions.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""`scikitplot show-versions` - wraps the library ``show_versions`` function.

Adapter pattern: for ``text`` output, delegate to the library's own renderer;
for ANY structured format (json/yaml/toml/...), fetch the data once as a dict and
let :func:`output.emit` render it. The handler never enumerates output formats,
so adding a new format to ``output.py`` requires no change here
(root-cause fix for the historical ``KeyError: 'toml'``; see FINDINGS CLI-FE-011).
"""

from __future__ import annotations

from ..context import Context
from ..output import emit


def run(ctx: Context, *, fmt: str = "text") -> int:
    """Render version/dependency info via the library ``show_versions``."""
    # lazy: real library function
    from ... import show_versions  # ruff: ignore[import-outside-top-level]

    if fmt == "text":
        show_versions(mode="stdout")  # native human-readable rendering
        return 0
    data = show_versions(mode="dict")  # side-effect free structured data
    emit(ctx, data)
    return 0
