# scikitplot/_env/_core/_errors.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Error types of the environment builder, each with a stable reason code.

Notes
-----
**User notes.** Every error has a ``code`` (``"approval_required"``,
``"backend_unavailable"`` ...) and, where one exists, a ``hint`` naming what
to change. :meth:`EnvError.to_dict` gives both, so a CI job or an agent can
branch on the code instead of parsing the message.

**Developer notes.** Each class also derives from the built-in type a caller
would expect (``ValueError`` for bad input, ``PermissionError`` for a missing
approval, ``RuntimeError`` for a failed run), so generic handlers keep working.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ApprovalRequiredError",
    "BackendUnavailableError",
    "ConfigError",
    "ConversionRefusedError",
    "EnvError",
    "ExecutionError",
    "LockBusyError",
    "NetworkForbiddenError",
    "PlanningError",
    "SourceError",
    "TargetUnsupportedError",
]


class EnvError(Exception):
    """
    Base class of every error raised by the environment builder.

    Parameters
    ----------
    message : str
        What happened.
    code : str or None, default=None
        Stable reason code; the class default when None.
    hint : str or None, default=None
        What to change to succeed.
    """

    code = "env_error"

    def __init__(
        self, message: str, *, code: str | None = None, hint: str | None = None
    ) -> None:  # noqa: D107
        super().__init__(message)
        if code is not None:
            self.code = code
        self.hint = hint

    def to_dict(self) -> dict[str, Any]:
        """
        Return the error as a mapping for machine output.

        Returns
        -------
        dict
            ``error`` (class name), ``code``, ``message`` and ``hint``.
        """
        return {
            "error": type(self).__name__,
            "code": self.code,
            "message": str(self),
            "hint": self.hint,
        }


class ConfigError(EnvError, ValueError):
    """A configuration document or argument is off contract."""

    code = "config_invalid"


class SourceError(EnvError, ValueError):
    """A dependency source cannot be read or has an unsupported entry."""

    code = "source_invalid"


class PlanningError(EnvError, RuntimeError):
    """No plan can satisfy the request on this host."""

    code = "plan_impossible"


class BackendUnavailableError(PlanningError):
    """The requested backend is missing or cannot do what was asked."""

    code = "backend_unavailable"


class TargetUnsupportedError(PlanningError):
    """The target cannot be built or executed from this host."""

    code = "target_unsupported"


class NetworkForbiddenError(PlanningError):
    """The plan would use the network and the network policy forbids it."""

    code = "network_forbidden"


class ConversionRefusedError(PlanningError):
    """A dependency would change ecosystem inexactly and the policy forbids it."""

    code = "conversion_refused"


class ApprovalRequiredError(EnvError, PermissionError):
    """
    A step needs an approval that was not given.

    Parameters
    ----------
    message : str
        What needs approval.
    kind : str
        Approval kind (``"bootstrap"``, ``"shell_init"``, ``"remove"``).
    code : str or None, default=None
        Reason code.
    hint : str or None, default=None
        How to approve.
    """

    code = "approval_required"

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        code: str | None = None,
        hint: str | None = None,
    ) -> None:  # noqa: D107
        super().__init__(message, code=code, hint=hint)
        self.kind = kind

    def to_dict(self) -> dict[str, Any]:
        """
        Return the error as a mapping, with the approval kind.

        Returns
        -------
        dict
            As :meth:`EnvError.to_dict`, plus ``kind``.
        """
        data = super().to_dict()
        data["kind"] = self.kind
        return data


class ExecutionError(EnvError, RuntimeError):
    """
    A step of a plan failed while running.

    Parameters
    ----------
    message : str
        What failed.
    step : str
        Id of the failed step.
    returncode : int or None, default=None
        Exit status of the command, when it was a command.
    code : str or None, default=None
        Reason code.
    hint : str or None, default=None
        What to try.
    """

    code = "step_failed"

    def __init__(
        self,
        message: str,
        *,
        step: str,
        returncode: int | None = None,
        code: str | None = None,
        hint: str | None = None,
    ) -> None:  # noqa: D107
        super().__init__(message, code=code, hint=hint)
        self.step = step
        self.returncode = returncode

    def to_dict(self) -> dict[str, Any]:
        """
        Return the error as a mapping, with the step and its exit status.

        Returns
        -------
        dict
            As :meth:`EnvError.to_dict`, plus ``step`` and ``returncode``.
        """
        data = super().to_dict()
        data.update(step=self.step, returncode=self.returncode)
        return data


class LockBusyError(EnvError, RuntimeError):
    """Another process holds the lock of the environment being changed."""

    code = "lock_busy"
