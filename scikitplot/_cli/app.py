# scikitplot/_cli/app.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""CLI application root: frontend selection and top-level error handling."""

from __future__ import annotations

import logging
import os
import sys
from typing import Mapping, Sequence

from . import exit_codes
from .errors import CliError

#: The frontends this entry point can honour. One set, named in the refusal so
#: a caller who mistyped learns what is accepted rather than silently getting
#: the default.
_FRONTENDS = frozenset({"argparse", "click"})


def _version_string() -> str:
    try:
        from .. import __version__  # ruff: ignore[import-outside-top-level]
    except Exception:  # pragma: no cover - defensive  # ruff: ignore[blind-except]
        __version__ = "unknown"
    return f"scikitplot {__version__}"


def _default_frontend() -> str:
    """Return the frontend used when the environment expresses no preference."""
    return "argparse"


def _select_frontend(env: Mapping[str, str] | None = None) -> str:
    """Return the frontend to use, refusing a value that cannot be honoured.

    Parameters
    ----------
    env : mapping, optional
        Environment to read ``SCIKITPLOT_CLI_FRONTEND`` from. Defaults to the
        process environment.

    Returns
    -------
    str
        ``"click"`` or ``"argparse"``.

    Raises
    ------
    SystemExit
        If the variable holds a value that is neither, naming what was given
        and what is accepted.

    Notes
    -----
    **Developer.** Any value other than ``click`` previously resolved to
    argparse in silence, so a typo was indistinguishable from the default and
    the caller never learned their setting had no effect. Unset and empty are
    still the default: absence is not a mistake, only an unusable value is.
    """
    environ = os.environ if env is None else env
    raw = environ.get("SCIKITPLOT_CLI_FRONTEND", "")
    requested = raw.strip().lower()
    if not requested:
        return _default_frontend()
    if requested not in _FRONTENDS:
        raise SystemExit(
            f"SCIKITPLOT_CLI_FRONTEND={raw!r} is not a frontend; "
            f"accepted values are {', '.join(sorted(_FRONTENDS))}."
        )
    from ._frontends import (  # ruff: ignore[import-outside-top-level]
        is_click_available,
    )

    if requested == "click" and not is_click_available():
        sys.stderr.write(
            "Warning: click frontend requested but click is not installed; "
            "using argparse.\n"
        )
        return "argparse"
    return requested


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``scikitplot`` console script and ``python -m``.

    Returns
    -------
    int
        Process exit code.
    """
    try:
        frontend = _select_frontend()
        if frontend == "click":
            from ._frontends import _click  # ruff: ignore[import-outside-top-level]

            return _click.run(argv)
        from ._frontends import _argparse  # ruff: ignore[import-outside-top-level]

        return _argparse.run(argv)
    except CliError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        if exc.hint:
            sys.stderr.write(f"Hint: {exc.hint}\n")
        return exc.exit_code
    except KeyboardInterrupt:  # pragma: no cover - interactive
        sys.stderr.write("Interrupted.\n")
        return exit_codes.INTERRUPTED
    except BrokenPipeError:
        # `scikitplot ... | head` closes the reader as soon as it has enough.
        # That is the normal end of the invocation, not a failure, but Python
        # would otherwise print a traceback and a second complaint at shutdown
        # when it flushes stdout. Redirect the remaining stdout to devnull so
        # that flush cannot raise again.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return exit_codes.OK
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - the last boundary before the OS
        # exit_codes.SOFTWARE exists for exactly this and was never assigned:
        # an unexpected exception escaped main() and the process exited 1 with a
        # raw traceback, which is neither the documented code nor a useful
        # report. stdout is the result channel and stays empty.
        sys.stderr.write(f"Internal error: {type(exc).__name__}: {exc}\n")
        logging.getLogger(__name__).debug("unhandled CLI failure", exc_info=exc)
        return exit_codes.SOFTWARE


__all__ = ["main"]
