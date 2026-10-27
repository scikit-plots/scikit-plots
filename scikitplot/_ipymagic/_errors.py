# scikitplot/_ipymagic/_errors.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Errors of the notebook frontend, each with a stable code and a hint.

Notes
-----
**User notes.** In IPython an error of this module is shown in two lines,
the message and the hint, instead of a traceback through the magic's code:
the traceback says nothing about your notebook.

**Developer notes.** Every class also derives from the built-in a caller
would catch (``ValueError`` for usage, ``RuntimeError`` for failures,
``ImportError`` for a missing optional package). ``_render_traceback_`` is
IPython's documented hook for custom exception display.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CapabilityMissingError",
    "CommandFailedError",
    "IpyMagicError",
    "MagicUsageError",
    "ProfilerBusyError",
]


class IpyMagicError(Exception):
    """
    Base class of every error raised by :py:mod:`scikitplot._ipymagic`.

    Parameters
    ----------
    message : str
        What went wrong.
    hint : str or None, default=None
        What to do about it.
    code : str or None, default=None
        Stable reason code; the class default when None.
    """

    code = "ipymagic_error"

    def __init__(
        self, message: str, *, hint: str | None = None, code: str | None = None
    ) -> None:  # noqa: D107
        super().__init__(message)
        self.message = message
        self.hint = hint
        if code is not None:
            self.code = code

    def to_dict(self) -> dict[str, Any]:
        """
        Return the error as plain data.

        Returns
        -------
        dict
            ``error`` (class name), ``code``, ``message`` and ``hint``.
        """
        return {
            "error": type(self).__name__,
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
        }

    def _render_traceback_(self) -> list[str]:
        """
        Return the lines IPython shows instead of a traceback.

        Returns
        -------
        list of str
            ``"<Class> [<code>]: <message>"`` and, with a hint, ``"  hint: ..."``.
        """
        lines = [f"{type(self).__name__} [{self.code}]: {self.message}"]
        if self.hint:
            lines.append(f"  hint: {self.hint}")
        return lines


class MagicUsageError(IpyMagicError, ValueError):
    """The magic was called with arguments it cannot run (nothing was run)."""

    code = "usage"


class CommandFailedError(IpyMagicError, RuntimeError):
    """
    A command exited with a non-zero status (raised only when asked to).

    Parameters
    ----------
    message : str
        What failed.
    result : CommandResult
        The full result (exit status, output, timing).
    hint : str or None, default=None
        What to do about it.
    """

    code = "command_failed"

    def __init__(self, message: str, *, result: Any, hint: str | None = None) -> None:
        super().__init__(message, hint=hint)
        self.result = result


class ProfilerBusyError(IpyMagicError, RuntimeError):
    """Another profiler is active in this process (Rule 133); nothing was run."""

    code = "profiler_busy"


class CapabilityMissingError(IpyMagicError, ImportError):
    """An optional package the request needs is not installed."""

    code = "capability_missing"
