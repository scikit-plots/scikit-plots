# scikitplot/_cli/app.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""CLI application root: frontend selection and top-level error handling."""

from __future__ import annotations

import errno
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
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - the last boundary before the OS
        # `scikitplot ... | head` closes the reader as soon as it has enough.
        # That is the normal end of the invocation, not a failure.
        if isinstance(exc, OSError) and _is_closed_reader(exc):
            return _reader_closed()
        return _internal_error(exc)


#: Whether the process runs on Windows. A module constant so that a test can
#: take either branch of ``_is_closed_reader`` on any platform.
_IS_WINDOWS = os.name == "nt"


def _is_closed_reader(exc: OSError) -> bool:
    """
    Return whether an ``OSError`` means "the reader of stdout has gone away".

    Parameters
    ----------
    exc : OSError
        The error that escaped a command.

    Returns
    -------
    bool
        ``True`` for a broken pipe in either platform's form; ``False`` for any
        other error, which is then reported as an internal error.

    Notes
    -----
    **Developer.** On POSIX a write to a pipe whose reader closed raises
    :class:`BrokenPipeError` (``EPIPE``). On
    Windows the same write raises a plain ``OSError`` with ``errno.EINVAL``
    ("Invalid argument"), so ``scikitplot ... | more`` followed by ``q`` was
    reported as "Internal error" with exit status 70.

    ``EINVAL`` alone is not proof, since many things raise it. The proof is
    stdout itself: the bytes that could not be written are still pending, so
    flushing fails again if, and only if, the pipe is the cause.
    """
    if isinstance(exc, BrokenPipeError):
        return True
    if not (_IS_WINDOWS and exc.errno == errno.EINVAL):
        return False
    try:
        sys.stdout.flush()
    except (OSError, ValueError):
        return True
    return False


def _reader_closed() -> int:
    """
    End quietly after the reader of stdout closed; return the exit status.

    Notes
    -----
    **Developer.** Python flushes stdout once more at shutdown and would
    print a second complaint then. Pointing the descriptor at the null device
    makes that flush succeed. A stdout without a descriptor (a replaced
    stream object) has nothing to redirect.
    """
    try:
        descriptor = sys.stdout.fileno()
    except (OSError, ValueError, AttributeError):
        return exit_codes.OK
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, descriptor)
    finally:
        os.close(devnull)
    return exit_codes.OK


def _internal_error(exc: BaseException) -> int:
    """
    Report an exception that escaped ``main`` and return ``SOFTWARE``.

    Notes
    -----
    **Developer.** ``exit_codes.SOFTWARE`` exists for exactly this: without it
    the process exited 1 with a raw traceback, which is neither the documented
    code nor a useful report. stdout is the result channel and stays empty.
    """
    sys.stderr.write(f"Internal error: {type(exc).__name__}: {exc}\n")
    logging.getLogger(__name__).debug("unhandled CLI failure", exc_info=exc)
    return exit_codes.SOFTWARE


__all__ = ["main"]
