# scikitplot/_cli/errors.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""CLI error taxonomy.

Every recoverable CLI failure is one of these, each carrying a stable exit code
and an actionable message. Raw tracebacks are shown only in debug mode. This
lets command/alias resolution distinguish *not found* from *import failed* from
*capability missing* instead of swallowing everything (see FINDINGS CLI-FE-004).
"""

from __future__ import annotations

from . import exit_codes


class CliError(Exception):
    """Base class for actionable CLI errors.

    Parameters
    ----------
    message : str
        Human-readable, actionable description.
    exit_code : int, optional
        Process exit code; defaults to :data:`exit_codes.ERROR`.
    hint : str, optional
        Optional next-step suggestion shown on a separate line.
    """

    exit_code: int = exit_codes.ERROR

    def __init__(
        self, message: str, *, exit_code: int | None = None, hint: str | None = None
    ) -> None:
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code
        self.hint = hint


class UsageError(CliError):
    """Invalid arguments or unknown command."""

    exit_code = exit_codes.USAGE


class CommandNotFoundError(UsageError):
    """The requested command (or alias) does not exist."""

    def __init__(self, name: str) -> None:
        super().__init__(
            f"Unknown command {name!r}.",
            hint="Run `scikitplot --help` to list available commands.",
        )


class HandlerLoadError(CliError):
    """A known command failed to import or did not resolve to a callable."""

    exit_code = exit_codes.SOFTWARE


class CapabilityMissingError(CliError):
    """An optional dependency or platform capability is unavailable."""

    exit_code = exit_codes.UNAVAILABLE

    def __init__(self, capability: str, *, install_hint: str | None = None) -> None:
        super().__init__(
            f"Required capability {capability!r} is not available.",
            hint=install_hint or "Run `scikitplot doctor` for details.",
        )


__all__ = [
    "CapabilityMissingError",
    "CliError",
    "CommandNotFoundError",
    "HandlerLoadError",
    "UsageError",
]
