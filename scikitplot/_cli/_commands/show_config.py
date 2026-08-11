# scikitplot/_cli/_commands/show_config.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""`scikitplot show-config` - build and runtime configuration.

For human ``text`` output this delegates to the library ``show_config``. For
structured output (``json``/``yaml``/``toml``) it reads the authoritative
``CONFIG`` mapping directly, which is side-effect free and works regardless of
whether the library ``show_config(mode="dicts")`` fix is applied. See
``_maintenance/FINDINGS.md`` (CLI-FE-010) and ``show_config_fix.diff``.
"""

from __future__ import annotations

import copy

from ..context import Context
from ..output import emit


def run(ctx: Context, *, fmt: str = "text") -> int:
    """Render build/runtime configuration in the requested format."""
    if fmt == "text":
        # lazy: library's human-readable printer
        from ... import show_config  # ruff: ignore[import-outside-top-level]

        show_config(mode="stdout")
        return 0
    # Structured output: read the authoritative configuration mapping directly.
    # This avoids the historical ``show_config(mode="dicts")`` behavior that
    # pretty-printed to stdout and returned None, which corrupted machine output.
    # single data source of truth
    from ...config.__config__ import CONFIG  # ruff: ignore[import-outside-top-level]

    emit(ctx, copy.deepcopy(CONFIG))
    return 0
