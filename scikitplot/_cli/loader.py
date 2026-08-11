# scikitplot/_cli/loader.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Lazy handler loading and dispatch.

The loader imports a command's handler only on invocation and normalizes any
import/attribute failure into the CLI error taxonomy, so a broken command yields
an actionable error instead of a raw traceback (FINDINGS CLI-FE-004).
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

from ._spec import CommandSpec
from .context import Context
from .errors import HandlerLoadError


def load_handler(target: str) -> Callable[..., int]:
    """Import ``"module:attr"`` and return the resolved callable.

    Parameters
    ----------
    target : str
        Import target of the form ``package.module:attribute``.

    Returns
    -------
    callable
        The handler ``run(ctx, **params) -> int``.

    Raises
    ------
    HandlerLoadError
        If the target is malformed, the module cannot be imported, or the
        attribute is missing or not callable.
    """
    module_name, sep, attr = target.partition(":")
    if not sep or not module_name or not attr:
        raise HandlerLoadError(
            f"Malformed handler target {target!r}; expected 'module:attribute'."
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise HandlerLoadError(
            f"Could not import command module {module_name!r}: {exc}"
        ) from exc
    handler = getattr(module, attr, None)
    if not callable(handler):
        raise HandlerLoadError(
            f"Handler target {target!r} did not resolve to a callable."
        )
    return handler


def dispatch(spec: CommandSpec, params: dict[str, Any], ctx: Context) -> int:
    """Load ``spec``'s handler and invoke it, returning its exit code."""
    if spec.deprecated:
        ctx.stderr.write(f"warning: command {spec.name!r} is deprecated.\n")
    handler = load_handler(spec.handler)
    return int(handler(ctx, **params))


__all__ = ["dispatch", "load_handler"]
