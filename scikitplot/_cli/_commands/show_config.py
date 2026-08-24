# scikitplot/_cli/_commands/show_config.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""`scikitplot show-config` - build and runtime configuration.

Two output controls:

* ``--mode`` (default ``stdout``) selects the library render mode
  (``stdout`` | ``dicts``).
* ``--format`` (default ``text``) selects serialization for structured data.

Precedence: an explicit structured ``--format`` (json/yaml/toml) always emits the
configuration mapping. Otherwise ``--mode`` drives: ``stdout`` delegates to the
library's printer; ``dicts`` emits the mapping as ``text``. Structured output is
read from the authoritative ``CONFIG`` mapping directly (side-effect free), which
is robust whether or not the library ``show_config(mode="dicts")`` fix is applied
(see FINDINGS CLI-FE-010).
"""

from __future__ import annotations

import copy

from ..context import Context
from ..output import emit


def _config_dict() -> dict:
    # Structured output: read the authoritative configuration mapping directly.
    # This avoids the historical ``show_config(mode="dicts")`` behavior that
    # pretty-printed to stdout and returned None, which corrupted machine output.
    # single data source of truth, authoritative data source
    from ...config.__config__ import CONFIG  # ruff: ignore[import-outside-top-level]

    return copy.deepcopy(CONFIG)


def run(ctx: Context, *, mode: str = "stdout", fmt: str = "text") -> int:
    """Render configuration per ``--mode``/``--format`` (see module docstring)."""
    if fmt != "text":  # explicit serialization wins
        emit(ctx, _config_dict())
        return 0
    if mode == "dicts":  # structured data as text
        emit(ctx, _config_dict())
        return 0
    # lazy: library's human-readable printer
    # mode == "stdout": library printer
    from ... import show_config  # ruff: ignore[import-outside-top-level]

    show_config(mode="stdout")
    return 0
