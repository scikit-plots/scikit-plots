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


def _reject_non_finite(data: Any, path: str = "") -> None:
    """Refuse a value no structured format can express, naming where it is.

    Parameters
    ----------
    data : object
        Value about to be rendered.
    path : str, optional
        Dotted location within the document, used in the message.

    Raises
    ------
    ValueError
        If any float in the document is ``NaN`` or an infinity.

    Notes
    -----
    **Developer.** ``json.dump`` defaults to ``allow_nan=True`` and writes bare
    ``NaN`` and ``Infinity`` tokens, which are not JSON. Failing here rather
    than at the writer lets the message name the key, so the caller knows which
    value to fix rather than which library complained.
    """
    if isinstance(data, float):
        if data != data or data in (  # ruff: ignore[comparison-with-itself]
            float("inf"),
            float("-inf"),
        ):
            where = path or "value"
            raise ValueError(
                f"{where} is {data!r}, which no structured format can express; "
                "use null, a string, or a finite number."
            )
    elif isinstance(data, dict):
        for key, value in data.items():
            _reject_non_finite(value, f"{path}.{key}" if path else str(key))
    elif isinstance(data, (list, tuple)):
        for position, value in enumerate(data):
            _reject_non_finite(value, f"{path}[{position}]")


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
        _reject_non_finite(data)
        # allow_nan=False: NaN and Infinity are not JSON. Emitting them produced
        # output that this package's own strict reader refuses, and that other
        # implementations are not required to accept.
        # sort_keys=False: the caller's order is data. json sorted while yaml
        # preserved, so one machine format reordered the document and the other
        # did not; they now agree.
        json.dump(data, ctx.stdout, indent=2, sort_keys=False, allow_nan=False)
        ctx.stdout.write("\n")
        return
    if ctx.fmt == "yaml":
        yaml = _require("yaml", "PyYAML", "pip install pyyaml")
        _reject_non_finite(data)
        yaml.safe_dump(data, ctx.stdout, sort_keys=False)
        return
    if ctx.fmt == "toml":
        _emit_toml(ctx, data)
        return
    # text: a human format, escaped and flattened so its own line contract holds
    if isinstance(data, (dict, list, tuple)):
        for key, text in _text_rows(data):
            # Keys are escaped with the same contract as values. Escaping one
            # side only moved the injection rather than removing it: a key
            # containing a newline still emitted a second apparent field.
            ctx.stdout.write(f"{_text_scalar(key)}: {text}\n")
    else:
        ctx.stdout.write(f"{_text_scalar(data)}\n")


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


def _toml_safe(value: Any, path: str = "") -> Any:
    """Drop ``None`` mapping entries, and refuse ``None`` inside a list.

    Parameters
    ----------
    value : object
        Value about to be rendered as TOML.
    path : str, optional
        Dotted location within the document, used in the refusal message.

    Returns
    -------
    object
        The value with ``None`` mapping entries removed.

    Raises
    ------
    ValueError
        If a list contains ``None``.

    Notes
    -----
    **Developer.** Dropping a *key* is documented and position-free: the key is
    absent, and no other key moves. Dropping a list *element* shifts every later
    position, so a caller reading ``xs[1]`` silently receives what used to be
    ``xs[2]`` -- a different value, not a missing one. That is refused rather
    than performed quietly.

    **User.** Replace the ``None``, or choose a format that has a null.
    """
    if isinstance(value, dict):
        return {
            key: _toml_safe(item, f"{path}.{key}" if path else str(key))
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, (list, tuple)):
        for position, item in enumerate(value):
            if item is None:
                where = path or "value"
                raise ValueError(
                    f"{where}[{position}] is None, which TOML cannot express. "
                    "Removing it would shift every later element, so a caller "
                    "reading a position would get a different value rather than "
                    "a missing one. Replace it, or use a format with a null."
                )
        return [
            _toml_safe(item, f"{path}[{position}]")
            for position, item in enumerate(value)
        ]
    return value


def _text_scalar(value: Any) -> str:
    """Return ``value`` as one line, escaping what would break the contract.

    Notes
    -----
    **Developer.** ``text`` is documented as ``key: value`` lines, and plain
    interpolation let a value containing a newline emit extra lines a consumer
    could not tell from real fields. Only the line-breaking characters are
    escaped, so the value stays readable.
    """
    text = "" if value is None else str(value)
    return (
        text.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _text_rows(value: Any, path: str = ""):
    """Yield ``(key, text)`` rows, flattening nested data by dotted key.

    Notes
    -----
    **Developer.** Nested data previously rendered as a Python repr, which is
    neither the documented shape nor parseable. Flattening keeps every value
    reachable under a name that says where it came from.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _text_rows(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for position, item in enumerate(value):
            yield from _text_rows(item, f"{path}[{position}]")
    else:
        yield path or "value", _text_scalar(value)


def _require(module_name: str, dist_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise CapabilityMissingError(dist_name, install_hint=install_hint) from exc


__all__ = ["emit"]
