# scikitplot/agent/_policy.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The policy, the limits, and the gate chain every action passes.

Before a tool runs, the action passes five gates, in order:

1. **Validation** - the tool exists and the arguments match its schema;
2. **Permissions** - the policy allows this tool and its kind of effect;
3. **Scope** - the policy grants every scope the tool needs;
4. **Budget** - one more call stays within the run's limits;
5. **Approval** - when the policy asks for it, a person approved this exact
   action (see :class:`~scikitplot.agent.Approval`).

The first gate that refuses stops the chain; the refusal is recorded as the
step's outcome (``denied``) and the planner sees it at its next decision.

Notes
-----
**Developer.** Least privilege is the default (invariant I3): ``Policy()``
allows ``read`` tools that need no scope, and nothing else. Every check is a
comparison of declared facts (names, effects, scopes, counters); no gate asks
a model, so a gate's answer is reproducible from the state alone.
"""

from __future__ import annotations

import dataclasses
import math
from enum import unique
from typing import Any, Iterable

from ._canonical import plain
from ._errors import AgentError, ErrorCode, _StrEnum
from ._schema import validate
from ._state import Action, Spent
from ._tools import Tool, ToolEffect, Toolset

__all__ = ["GateKind", "GateResult", "Limits", "Policy"]


@unique
class GateKind(_StrEnum):
    """The five gates, in the order an action passes them."""

    VALIDATION = "validation"
    PERMISSIONS = "permissions"
    SCOPE = "scope"
    BUDGET = "budget"
    APPROVAL = "approval"


def _frozen(value: Any, what: str, kind: type | tuple[type, ...] = str) -> frozenset:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f"{what} must be an iterable, got {type(value).__name__}.")
    items = frozenset(value)
    bad = sorted(repr(i) for i in items if not isinstance(i, kind))
    if bad:
        raise TypeError(f"{what}: {bad} are not of the expected type.")
    return items


def _effects(value: Any, what: str) -> frozenset[ToolEffect]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f"{what} must be an iterable of ToolEffect.")
    try:
        return frozenset(ToolEffect(v) for v in value)
    except ValueError as exc:
        raise ValueError(f"{what}: {exc}") from None


@dataclasses.dataclass(frozen=True)
class Policy:
    """
    What an agent may do: least privilege unless stated otherwise.

    Parameters
    ----------
    allow_effects : iterable of ToolEffect or str, default=("read",)
        Kinds of effect the agent may cause.
    granted_scopes : iterable of str, default=()
        Scopes the agent holds; a tool runs only if it needs no other.
    allow_tools : iterable of str or None, default=None
        If set, only these tools may run. ``None`` allows any registered tool
        the other rules allow.
    deny_tools : iterable of str, default=()
        Tools that never run, whatever else allows them.
    approval_effects : iterable of ToolEffect or str, default=("write", "external")
        Kinds of effect that need a person's approval before each call.
    approval_tools : iterable of str, default=()
        Tools that need approval before each call, whatever their effect.

    Raises
    ------
    TypeError, ValueError
        If a field is not an iterable of the right kind.

    Examples
    --------
    >>> from scikitplot.agent import Policy
    >>> Policy().allows_effect("write")
    False
    >>> Policy(allow_effects=["read", "write"]).needs_approval_for("write")
    True
    """

    allow_effects: frozenset[ToolEffect] = frozenset({ToolEffect.READ})
    granted_scopes: frozenset[str] = frozenset()
    allow_tools: frozenset[str] | None = None
    deny_tools: frozenset[str] = frozenset()
    approval_effects: frozenset[ToolEffect] = frozenset(
        {ToolEffect.WRITE, ToolEffect.EXTERNAL}
    )
    approval_tools: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "allow_effects", _effects(self.allow_effects, "Policy.allow_effects")
        )
        object.__setattr__(
            self,
            "approval_effects",
            _effects(self.approval_effects, "Policy.approval_effects"),
        )
        for name in ("granted_scopes", "deny_tools", "approval_tools"):
            object.__setattr__(
                self, name, _frozen(getattr(self, name), f"Policy.{name}")
            )
        if self.allow_tools is not None:
            object.__setattr__(
                self, "allow_tools", _frozen(self.allow_tools, "Policy.allow_tools")
            )

    def allows_effect(self, effect: ToolEffect | str) -> bool:
        """Whether the policy allows ``effect``."""
        return ToolEffect(effect) in self.allow_effects

    def needs_approval_for(self, effect: ToolEffect | str) -> bool:
        """Whether a call with ``effect`` needs approval."""
        return ToolEffect(effect) in self.approval_effects

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping, every set sorted."""
        return {
            "allow_effects": sorted(e.value for e in self.allow_effects),
            "granted_scopes": sorted(self.granted_scopes),
            "allow_tools": (
                None if self.allow_tools is None else sorted(self.allow_tools)
            ),
            "deny_tools": sorted(self.deny_tools),
            "approval_effects": sorted(e.value for e in self.approval_effects),
            "approval_tools": sorted(self.approval_tools),
        }


def _limit(value: Any, what: str, *, integer: bool) -> Any:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{what} must be a number or None, got {value!r}.")
    if integer and not isinstance(value, int):
        raise TypeError(f"{what} must be an integer or None, got {value!r}.")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{what} must be finite and > 0, got {value!r}.")
    return value if integer else float(value)


@dataclasses.dataclass(frozen=True)
class Limits:
    """
    Stopping conditions: when a run stops, whatever the planner proposes.

    Parameters
    ----------
    max_steps : int, default=16
        Planner decisions. Required and finite: it is what makes every run
        end (invariant I1).
    max_tool_calls : int or None, default=None
        Tool calls; an action beyond them is denied at the Budget gate.
    max_cost : float or None, default=None
        Budget units: tool costs plus the cost the planner reports. An action
        whose tool would pass it is denied; at or past it, no further
        decision is asked for.
    max_tokens : int or None, default=None
        Model tokens the planner reports. Every decision must then report
        its :class:`~scikitplot.agent.Usage` (invariant I7).
    max_seconds : float or None, default=None
        Running time, summed over the run's sessions (time waiting for an
        approval does not count). Checked before each decision and each tool
        call; a call already running is not interrupted.

    Raises
    ------
    TypeError, ValueError
        If a limit is not a positive finite number of the right kind.

    Examples
    --------
    >>> from scikitplot.agent import Limits
    >>> Limits(max_steps=8, max_cost=10.0).max_cost
    10.0
    """

    max_steps: int = 16
    max_tool_calls: int | None = None
    max_cost: float | None = None
    max_tokens: int | None = None
    max_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_steps is None:
            raise ValueError("Limits.max_steps is required: it is what ends every run.")
        object.__setattr__(
            self, "max_steps", _limit(self.max_steps, "Limits.max_steps", integer=True)
        )
        for name, integer in (
            ("max_tool_calls", True),
            ("max_cost", False),
            ("max_tokens", True),
            ("max_seconds", False),
        ):
            value = _limit(getattr(self, name), f"Limits.{name}", integer=integer)
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class GateResult:
    """One gate's answer for one action."""

    gate: GateKind
    passed: bool
    error: AgentError | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        out: dict[str, Any] = {"gate": self.gate.value, "passed": self.passed}
        if self.error is not None:
            out["code"] = self.error.code.value
            out["message"] = self.error.message
        return out


def _deny(gate: GateKind, code: ErrorCode, message: str, **details: Any) -> GateResult:
    return GateResult(
        gate,
        False,
        AgentError(code, message, plain({"gate": gate.value, **details})),
    )


def check_action(  # noqa: PLR0911
    action: Action,
    tools: Toolset,
    policy: Policy,
    limits: Limits,
    spent: Spent,
) -> tuple[Tool | None, list[GateResult]]:
    """
    Run the Validation, Permissions, Scope and Budget gates for ``action``.

    Parameters
    ----------
    action : Action
        The proposed call.
    tools : Toolset
        The tools the agent holds.
    policy : Policy
        What the agent may do.
    limits : Limits
        The run's limits.
    spent : Spent
        What the run has spent before this call (time included).

    Returns
    -------
    tool : Tool or None
        The tool named by the action, ``None`` if there is none.
    results : list of GateResult
        One per gate run; the last has ``passed=False`` when one refused.

    Notes
    -----
    The Approval gate is the loop's, because it may pause the run; it runs
    after these four pass.
    """
    results: list[GateResult] = []
    tool = tools.get(action.tool)
    if tool is None:
        results.append(
            _deny(
                GateKind.VALIDATION,
                ErrorCode.UNKNOWN_TOOL,
                f"no tool named {action.tool!r}; the tools are {list(tools.names())}",
                tool=action.tool,
            )
        )
        return None, results
    problems = validate(tool.parameters, action.arguments)
    if problems:
        results.append(
            _deny(
                GateKind.VALIDATION,
                ErrorCode.INVALID_ARGUMENTS,
                f"arguments of {tool.name!r} do not match its schema: "
                + "; ".join(problems),
                tool=tool.name,
                problems=problems,
            )
        )
        return tool, results
    results.append(GateResult(GateKind.VALIDATION, True))

    if tool.name in policy.deny_tools:
        results.append(
            _deny(
                GateKind.PERMISSIONS,
                ErrorCode.PERMISSION_DENIED,
                f"the policy denies the tool {tool.name!r}",
                tool=tool.name,
                rule="deny_tools",
            )
        )
        return tool, results
    if policy.allow_tools is not None and tool.name not in policy.allow_tools:
        results.append(
            _deny(
                GateKind.PERMISSIONS,
                ErrorCode.PERMISSION_DENIED,
                f"the policy allows only {sorted(policy.allow_tools)}",
                tool=tool.name,
                rule="allow_tools",
            )
        )
        return tool, results
    if not policy.allows_effect(tool.effect):
        results.append(
            _deny(
                GateKind.PERMISSIONS,
                ErrorCode.PERMISSION_DENIED,
                f"the policy does not allow {tool.effect.value!r} effects "
                f"(allowed: {sorted(e.value for e in policy.allow_effects)})",
                tool=tool.name,
                rule="allow_effects",
                effect=tool.effect.value,
            )
        )
        return tool, results
    results.append(GateResult(GateKind.PERMISSIONS, True))

    missing = sorted(tool.scopes - policy.granted_scopes)
    if missing:
        results.append(
            _deny(
                GateKind.SCOPE,
                ErrorCode.SCOPE_DENIED,
                f"{tool.name!r} needs scope(s) {missing}, which the policy does not grant",
                tool=tool.name,
                missing=missing,
            )
        )
        return tool, results
    results.append(GateResult(GateKind.SCOPE, True))

    over = _over_budget(tool, limits, spent)
    if over is not None:
        name, limit, would = over
        results.append(
            _deny(
                GateKind.BUDGET,
                ErrorCode.BUDGET_EXCEEDED,
                f"calling {tool.name!r} would pass {name}={limit!r} ({would!r})",
                tool=tool.name,
                limit=name,
                value=limit,
                would_be=would,
            )
        )
        return tool, results
    results.append(GateResult(GateKind.BUDGET, True))
    return tool, results


def _over_budget(
    tool: Tool, limits: Limits, spent: Spent
) -> tuple[str, Any, Any] | None:
    if (
        limits.max_tool_calls is not None
        and spent.tool_calls + 1 > limits.max_tool_calls
    ):
        return "max_tool_calls", limits.max_tool_calls, spent.tool_calls + 1
    if limits.max_cost is not None and spent.cost + tool.cost > limits.max_cost:
        return "max_cost", limits.max_cost, spent.cost + tool.cost
    if limits.max_seconds is not None and spent.seconds >= limits.max_seconds:
        return "max_seconds", limits.max_seconds, spent.seconds
    if limits.max_tokens is not None and spent.tokens > limits.max_tokens:
        # The decision proposing this call already passed the limit.
        return "max_tokens", limits.max_tokens, spent.tokens
    return None


def approval_reason(tool: Tool, policy: Policy) -> str | None:
    """Why ``tool`` needs approval under ``policy``, or ``None`` if it does not."""
    if tool.name in policy.approval_tools:
        return f"the policy requires approval for the tool {tool.name!r}"
    if policy.needs_approval_for(tool.effect):
        return f"the policy requires approval for {tool.effect.value!r} effects"
    return None
