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

from .._distributions import IMPORT_NAME, install_hint, provider_of
from ._spec import CommandSpec
from .context import Context
from .errors import CapabilityMissingError, HandlerLoadError


def _missing_part(exc: ImportError) -> str | None:
    """Return the ``scikitplot`` module whose absence caused ``exc``, if any.

    Parameters
    ----------
    exc : ImportError
        The import failure to classify.

    Returns
    -------
    str or None
        The absolute module name when ``exc`` reports that a module of the
        ``scikitplot`` package itself could not be found; ``None`` when the
        missing module belongs to another project, or when the failure is not
        a missing module at all.

    Notes
    -----
    **Developer.** The two cases need different advice. A missing third-party
    module is fixed by installing that dependency. A missing ``scikitplot``
    module means that part of the package was not installed, which happens
    whenever a partial distribution (see ``scikitplot._distributions``) is in
    use, and is fixed by installing the distribution that ships the part.
    """
    if not isinstance(exc, ModuleNotFoundError) or not exc.name:
        return None
    name = exc.name
    if name == IMPORT_NAME or name.startswith(IMPORT_NAME + "."):
        return name
    return None


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
    """Load ``spec``'s handler and invoke it, returning its exit code.

    Parameters
    ----------
    spec : CommandSpec
        The native command to run.
    params : dict
        Parsed parameters, passed to the handler as keyword arguments.
    ctx : Context
        Invocation context.

    Returns
    -------
    int
        The handler's exit code.

    Raises
    ------
    HandlerLoadError
        If the handler cannot be loaded.
    CapabilityMissingError
        If the handler needs a part of ``scikitplot`` that is not installed.
        The hint names the distribution that ships it.

    Notes
    -----
    **Developer.** Handlers import the library function they wrap inside
    ``run`` (so top-level help stays cheap). With a partial distribution that
    function's module may not be installed, and the resulting
    ``ModuleNotFoundError`` used to reach ``main`` as an "Internal error". It
    is an unavailable capability with a known remedy, so it is reported as
    one. A missing third-party module is left to the handler, which knows what
    it was for.
    """
    if spec.deprecated:
        ctx.stderr.write(f"warning: command {spec.name!r} is deprecated.\n")
    handler = load_handler(spec.handler)
    try:
        return int(handler(ctx, **params))
    except ModuleNotFoundError as exc:
        missing = _missing_part(exc)
        if missing is None:
            raise
        raise CapabilityMissingError(
            missing,
            install_hint=(
                "This part of scikitplot is not installed. "
                f"Install it with: {install_hint(missing)}"
            ),
        ) from exc


__all__ = ["dispatch", "load_handler", "run_delegate"]


def _delegate_hint(exc: ImportError, module_name: str, install_hint: str | None) -> str:
    """Choose the advice shown when a delegated submodule cannot be imported.

    Parameters
    ----------
    exc : ImportError
        The import failure.
    module_name : str
        The delegate's module.
    install_hint : str or None
        The command's own hint, which describes its optional dependencies.

    Returns
    -------
    str
        The installation command for the partial distribution that ships the
        missing part, when the failure is that part being absent and a partial
        distribution ships it; otherwise ``install_hint``; otherwise a generic
        instruction naming ``module_name``.

    Notes
    -----
    **Developer.** A command's ``install_hint`` is written for the case where
    the submodule is present and an optional dependency is not. When the
    submodule itself is absent that advice is wrong: installing an extra of a
    distribution that is not installed changes nothing the user can see. The
    distribution map is consulted first for exactly that case.
    """
    missing = _missing_part(exc)
    if missing is not None:
        provider = provider_of(missing)
        if provider is not None:
            return (
                "This part of scikitplot is not installed. "
                f"Install it with: pip install {provider}"
            )
    return install_hint or f"Ensure {module_name!r} is installed."


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
                install_hint=_delegate_hint(exc, module_name, install_hint),
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
            target, install_hint=_delegate_hint(exc, target, install_hint)
        ) from exc
    except SystemExit as exc:
        return _exit_code(exc.code)
    finally:
        sys.argv = old_argv
