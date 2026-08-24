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

# Bare module form: execute like `python -m <target>`.
import runpy
import sys
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


__all__ = ["dispatch", "load_handler", "run_delegate"]


def _exit_code(code: object) -> int:
    """Normalize a SystemExit code (int, None, or str) to a process exit code."""
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1  # a string message implies failure


def run_delegate(
    target: str, argv: list[str], *, install_hint: str | None = None
) -> int:
    """Forward ``argv`` verbatim to a submodule's own entry point.

    Two target forms are supported:

    * ``"module:attr"`` - import ``module`` lazily and call
      ``attr(argv) -> int`` (the recommended contract, e.g.
      ``"scikitplot.mcp.__main__:main"``).
    * ``"module"`` - execute the module like ``python -m module`` via
      :mod:`runpy` (for submodules that only provide a ``__main__`` guard).

    The submodule owns all argument parsing (including ``--help``). A
    :class:`SystemExit` raised by the submodule (argparse ``--help`` or
    validation errors) is converted to an exit code. A missing submodule or
    dependency becomes an actionable :class:`CapabilityMissing`, never a raw
    traceback.

    Parameters
    ----------
    target : str
        ``"module:attr"`` or ``"module"``.
    argv : list of str
        Arguments to forward, already stripped of the CLI command name.
    install_hint : str, optional
        Shown if the submodule/dependency is unavailable.

    Returns
    -------
    int
        Process exit code.
    """
    from .errors import (  # ruff: ignore[import-outside-top-level]
        CapabilityMissingError,
        HandlerLoadError,
    )

    argv = list(argv)
    if ":" in target:
        module_name, _, attr = target.partition(":")
        if not module_name or not attr:
            raise HandlerLoadError(
                f"Malformed delegate target {target!r}; expected 'module:attr'."
            )
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise CapabilityMissingError(
                module_name,
                install_hint=install_hint or f"Ensure {module_name!r} is installed.",
            ) from exc
        entry = getattr(module, attr, None)
        if not callable(entry):
            raise HandlerLoadError(
                f"Delegate target {target!r} did not resolve to a callable."
            )
        try:
            result = entry(argv)
        except SystemExit as exc:  # submodule argparse --help / validation
            return _exit_code(exc.code)
        return int(result) if isinstance(result, int) else 0

    old_argv = sys.argv
    sys.argv = [target, *argv]
    try:
        runpy.run_module(target, run_name="__main__", alter_sys=True)
        return 0
    except ImportError as exc:
        raise CapabilityMissingError(
            target, install_hint=install_hint or f"Ensure {target!r} is installed."
        ) from exc
    except SystemExit as exc:
        return _exit_code(exc.code)
    finally:
        sys.argv = old_argv
