# scikitplot/agent/_delegate.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Sub-runs: a tool that runs another agent inside the parent's budget.

Notes
-----
**User.** ``delegate("research", child_agent, max_steps=6)`` is a tool the
parent's planner calls with ``{"goal": ...}``. The child runs to its end and
the tool returns ``{"run_id", "status", "stop_reason", "answer", "spent"}``;
the parent sees only that, not the child's steps, so the child's work does not
fill the parent's state.

**Developer.** Three rules, each tested:

*No privilege the parent lacks.* The delegate's effect is the strongest effect
among the child's tools and its scopes are the union of theirs, so the
parent's Permissions, Scope and Approval gates judge a delegation as they would
judge the child's tools called directly.

*Budget is conserved.* For each limit the parent has (tool calls, cost,
tokens, seconds) the child's limit is the smaller of what the delegate asks for
and what the parent has left, and everything the child spends is charged to
the parent. A parent therefore never spends more than its limits through its
children; tokens carry the same guarantee as one run (checked before each
decision, not inside it).

*One trail.* The child writes to the parent's audit sink under its own run id
-- the digest of the parent's run id and the call key, so it is determined by
the call -- and the parent records a ``subrun`` event with the child's run id,
status and last audit digest. A delegate is not idempotent: a parent that
stopped during a sub-run marks the step interrupted rather than running the
child again.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

from ._canonical import digest
from ._policy import Limits
from ._tools import Tool, ToolEffect

__all__ = ["DelegateTool", "delegate"]

_ORDER = {ToolEffect.READ: 0, ToolEffect.WRITE: 1, ToolEffect.EXTERNAL: 2}
_GOAL = {
    "type": "object",
    "properties": {"goal": {"type": "string", "minLength": 1}},
    "required": ["goal"],
    "additionalProperties": False,
}


def _not_called(goal: str) -> Any:  # pragma: no cover - the loop runs the child
    raise RuntimeError("a DelegateTool is run by the agent loop, not called directly")


@dataclasses.dataclass(frozen=True, eq=False)
class DelegateTool(Tool):
    """
    A tool whose call is a bounded run of :attr:`child`.

    Build it with :func:`delegate`.
    """

    child: Any = None
    child_limits: Limits | None = None


def delegate(
    name: str,
    child: Any,
    *,
    description: str = "",
    max_steps: int = 8,
    max_tool_calls: int | None = None,
    max_cost: float | None = None,
    max_tokens: int | None = None,
    max_seconds: float | None = None,
) -> DelegateTool:
    """
    Return a tool that runs ``child`` toward the goal it is called with.

    Parameters
    ----------
    name : str
        The tool's name.
    child : Agent
        The agent to run. Its tools, planner and policy are used as they are;
        its limits are replaced by these, capped by the parent's remaining
        budget at each call; its audit sink and state store become the
        parent's.
    description : str, default=""
        For the parent's planner.
    max_steps : int, default=8
        Decisions the child may make. Steps are not shared: each run counts
        its own decisions.
    max_tool_calls, max_cost, max_tokens, max_seconds : number or None
        The child's own caps; ``None`` leaves only the parent's remaining
        budget.

    Returns
    -------
    DelegateTool

    Raises
    ------
    TypeError
        If ``child`` is not an :class:`~scikitplot.agent.Agent`.
    ValueError
        For a limit outside its domain, or a child with no tools.
    """
    from ._loop import Agent  # noqa: PLC0415 - _loop imports this module

    if not isinstance(child, Agent):
        raise TypeError(f"delegate {name!r}: child must be an Agent.")
    tools = list(child.tools)
    if not tools:
        raise ValueError(f"delegate {name!r}: the child has no tools.")
    limits = Limits(
        max_steps=max_steps,
        max_tool_calls=max_tool_calls,
        max_cost=max_cost,
        max_tokens=max_tokens,
        max_seconds=max_seconds,
    )
    effect = max((t.effect for t in tools), key=_ORDER.__getitem__)
    scopes = frozenset().union(*(t.scopes for t in tools))
    return DelegateTool(
        name,
        _not_called,
        description=description,
        parameters=_GOAL,
        effect=effect,
        scopes=scopes,
        cost=0.0,
        idempotent=False,
        child=child,
        child_limits=limits,
    )


def child_run_id(parent_run_id: str, key: str) -> str:
    """Return the child's run ID determined by the parent's run and call."""
    return digest({"parent": parent_run_id, "call": key})


def capped_limits(requested: Limits, parent: Limits, spent: Any) -> Limits | None:
    """
    Cap the child's limits by what the parent has left.

    Returns
    -------
    Limits or None
        ``None`` when the parent has nothing left of a limited resource, so
        no child may start.
    """
    values: dict[str, Any] = {"max_steps": requested.max_steps}
    for field, used in (
        ("max_tool_calls", spent.tool_calls),
        ("max_cost", spent.cost),
        ("max_tokens", spent.tokens),
        ("max_seconds", spent.seconds),
    ):
        own = getattr(requested, field)
        cap = getattr(parent, field)
        left = None if cap is None else cap - used
        if left is not None and left <= 0:
            return None
        choices = [v for v in (own, left) if v is not None]
        values[field] = min(choices) if choices else None
    return Limits(**values)


def child_agent(tool: DelegateTool, parent: Any, limits: Limits) -> Any:
    """Bind a copy of the child agent to the parent's sink, store and limits."""
    agent = copy.copy(tool.child)
    agent.limits = limits
    agent.audit = parent.audit
    agent.store = parent.store
    if agent.artifacts is None:
        agent.artifacts = parent.artifacts
        agent.offload_results_over = parent.offload_results_over
    return agent
