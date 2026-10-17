# scikitplot/agent/_errors.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Error records and the codes an agent run reports.

A run never loses an error in its loop: a denied action, a tool that raised,
a limit that stopped the run, each becomes an :class:`AgentError` in the
run's state and its audit trail (invariant I2).
"""

from __future__ import annotations

import dataclasses
from enum import Enum, unique
from typing import Any, Mapping

__all__ = [
    "AgentError",
    "ErrorCode",
    "SchemaError",
    "StateError",
]


class _StrEnum(str, Enum):
    """A string enumeration whose ``str()`` is its value."""

    def __str__(self) -> str:
        return str(self.value)


@unique
class ErrorCode(_StrEnum):
    """What went wrong, as a stable code a caller may branch on."""

    INVALID_ARGUMENTS = "invalid_arguments"
    """The arguments do not match the tool's schema (Validation gate)."""

    UNKNOWN_TOOL = "unknown_tool"
    """The planner named a tool the toolset does not hold (Validation gate)."""

    PERMISSION_DENIED = "permission_denied"
    """The policy does not allow the tool or its effect (Permissions gate)."""

    SCOPE_DENIED = "scope_denied"
    """The tool needs a scope the policy does not grant (Scope gate)."""

    BUDGET_EXCEEDED = "budget_exceeded"
    """One more call would pass a limit (Budget gate)."""

    APPROVAL_DENIED = "approval_denied"
    """An approver refused the action (Approval gate)."""

    APPROVAL_STALE = "approval_stale"
    """The action changed since it was approved (Approval gate, at resume)."""

    TOOL_FAILED = "tool_failed"
    """The tool raised."""

    TOOL_RESULT_NOT_JSON = "tool_result_not_json"
    """The tool returned a value strict JSON cannot hold."""

    TOOL_INTERRUPTED = "tool_interrupted"
    """A non-idempotent call was in flight when the run stopped; whether it
    took effect is unknown, so it is not repeated."""

    PLANNER_FAILED = "planner_failed"
    """The planner raised."""

    INVALID_DECISION = "invalid_decision"
    """The planner returned something that is not a valid :class:`Decision`."""

    USAGE_NOT_REPORTED = "usage_not_reported"
    """A token limit is set and the planner reported no usage (invariant I7)."""

    ANSWER_REJECTED = "answer_rejected"
    """The done criterion refused the planner's answer."""

    DONE_CHECK_FAILED = "done_check_failed"
    """The done criterion raised."""

    LIMIT_REACHED = "limit_reached"
    """A run limit stopped the run before the next step."""


class SchemaError(ValueError):
    """A tool's parameter schema uses something outside the supported subset."""


class StateError(ValueError):
    """A saved run state is malformed, or of a schema version not supported."""


@dataclasses.dataclass(frozen=True)
class AgentError:
    """
    One error, as recorded in a run's state and audit trail.

    Parameters
    ----------
    code : ErrorCode
        What went wrong.
    message : str
        A sentence a person can act on.
    details : mapping, default={}
        JSON-compatible facts (the gate, the limit, the tool, ...).

    Raises
    ------
    TypeError
        If ``code`` is not an :class:`ErrorCode` (or its value) or ``message``
        is not a string.

    Examples
    --------
    >>> from scikitplot.agent import AgentError, ErrorCode
    >>> err = AgentError(ErrorCode.SCOPE_DENIED, "needs 'corpus:read'")
    >>> err.to_dict()["code"]
    'scope_denied'
    >>> AgentError.from_dict(err.to_dict()) == err
    True
    """

    code: ErrorCode
    message: str
    details: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "code", ErrorCode(self.code))
        except ValueError:
            raise TypeError(f"AgentError.code: unknown code {self.code!r}") from None
        if not isinstance(self.message, str):
            raise TypeError("AgentError.message must be a string.")
        if not isinstance(self.details, Mapping):
            raise TypeError("AgentError.details must be a mapping.")
        object.__setattr__(self, "details", dict(self.details))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {
            "code": self.code.value,
            "message": self.message,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AgentError:
        """Rebuild an error written by :meth:`to_dict`."""
        if not isinstance(data, Mapping) or set(data) != {"code", "message", "details"}:
            raise StateError(f"an error record needs code, message, details: {data!r}")
        try:
            return cls(data["code"], data["message"], data["details"])
        except TypeError as exc:
            raise StateError(f"a malformed error record: {exc}") from None
