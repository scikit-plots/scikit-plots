# scikitplot/_cli/_commands/show_versions.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""`scikitplot show-versions` - version and dependency information.

Two output controls:

* ``--mode`` (default ``stdout``) selects the library render mode
  (``stdout`` | ``dict`` | ``yaml`` | ``rich``).
* ``--format`` (default ``text``) selects serialization for structured data.

Precedence: an explicit structured ``--format`` (json/yaml/toml) always emits the
version mapping. Otherwise ``--mode`` drives: ``stdout``/``rich`` delegate to the
library's printer; ``dict`` emits the mapping as ``text``; ``yaml`` uses the
library's native YAML rendering. The handler never enumerates ``--format`` values
(FINDINGS CLI-FE-011).
"""

from __future__ import annotations

from ..context import Context
from ..errors import CapabilityMissingError
from ..output import emit


def run(ctx: Context, *, mode: str = "stdout", fmt: str = "text") -> int:
    """Render versions per ``--mode``/``--format`` (see module docstring)."""
    # lazy: real library function, library source of truth
    from ...utils._show_versions import (  # ruff: ignore[import-outside-top-level]
        show_versions,
    )

    if fmt != "text":  # explicit serialization wins
        emit(ctx, show_versions(mode="dict"))
        return 0
    if mode in ("stdout", "rich"):  # library prints (rich if available)
        show_versions(mode=mode)  # native human-readable rendering
        return 0
    if mode == "dict":  # structured data as text
        emit(ctx, show_versions(mode="dict"))  # side-effect free structured data
        return 0
    if mode == "yaml":  # library's native YAML string
        try:
            out = show_versions(mode="yaml")
        except ModuleNotFoundError as exc:
            raise CapabilityMissingError(
                "yaml", install_hint="Install PyYAML: pip install pyyaml"
            ) from exc
        if isinstance(out, str):
            ctx.stdout.write(out if out.endswith("\n") else out + "\n")
        return 0
    return 0  # unreachable: choices are constrained by the registry
