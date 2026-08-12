# scikitplot/_cli/output.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Output rendering with a stable machine contract.

``text`` and ``json`` render with the standard library only. ``yaml`` and
``toml`` require an optional writer (Tier 2) and fail with an actionable
:class:`CapabilityMissing` error when absent, rather than an import traceback.

Notes
-----
TOML has no null type and requires a table (mapping) at the top level. ``None``
values are therefore dropped when rendering TOML (an absent key denotes an unset
value); ``json``/``yaml`` preserve them. This asymmetry is intentional and does
not affect frontend parity, since both frontends share this renderer.
"""

from __future__ import annotations

import json
from typing import Any

from .context import Context
from .errors import CapabilityMissingError


def emit(ctx: Context, data: Any) -> None:
    """Render ``data`` to ``ctx.stdout`` in ``ctx.fmt``.

    Parameters
    ----------
    ctx : Context
        Invocation context providing the output stream and format.
    data : any
        Serializable mapping or value. Structured formats (json/yaml/toml)
        expect JSON-compatible data; ``text`` renders ``key: value`` lines.

    Raises
    ------
    CapabilityMissing
        If ``fmt`` is ``yaml`` or ``toml`` but no writer is installed.
    ValueError
        If ``fmt="toml"`` and ``data`` is not a mapping (TOML top-level must be
        a table).
    """
    if ctx.fmt == "json":
        json.dump(data, ctx.stdout, indent=2, sort_keys=True)
        ctx.stdout.write("\n")
        return
    if ctx.fmt == "yaml":
        yaml = _require("yaml", "PyYAML", "pip install pyyaml")
        yaml.safe_dump(data, ctx.stdout, sort_keys=False)
        return
    if ctx.fmt == "toml":
        _emit_toml(ctx, data)
        return
    # text
    if isinstance(data, dict):
        for key, value in data.items():
            ctx.stdout.write(f"{key}: {value}\n")
    else:
        ctx.stdout.write(f"{data}\n")


def _emit_toml(ctx: Context, data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError(  # ruff: ignore[type-check-without-type-error]
            "TOML output requires a mapping at the top level; "
            f"got {type(data).__name__}. Use --format json for this data."
        )
    writer, dumps = _toml_writer()
    text = dumps(writer, _toml_safe(data))
    ctx.stdout.write(text if text.endswith("\n") else text + "\n")


def _toml_writer():
    """Return ``(module, dumps_callable)`` for the first available TOML writer.

    Tries ``tomli_w`` then ``toml``. The standard-library ``tomllib`` is
    read-only and cannot be used here.
    """
    try:
        import tomli_w  # noqa: PLC0415

        return tomli_w, lambda mod, obj: mod.dumps(obj)
    except ImportError:
        pass
    try:
        import toml  # noqa: PLC0415

        return toml, lambda mod, obj: mod.dumps(obj)
    except ImportError as exc:
        raise CapabilityMissingError(
            "toml", install_hint="Install a TOML writer: pip install tomli-w"
        ) from exc


def _toml_safe(value: Any) -> Any:
    """Recursively drop ``None`` values, which TOML cannot represent."""
    if isinstance(value, dict):
        return {k: _toml_safe(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_toml_safe(v) for v in value if v is not None]
    return value


def _require(module_name: str, dist_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise CapabilityMissingError(dist_name, install_hint=install_hint) from exc


__all__ = ["emit"]
