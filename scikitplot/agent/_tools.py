# scikitplot/agent/_tools.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tools and the toolset an agent may call.

A :class:`Tool` is a Python function together with what the runtime needs to
decide whether a call may run: a schema for its arguments (the *Validation*
gate), the kind of effect it has (*Permissions*), the scopes it touches
(*Scope*), what one call costs (*Budget*), and whether a call that may already
have run can be repeated safely (idempotency).
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import re
from enum import unique
from typing import Any, Callable, Iterable, Iterator, Mapping

from ._canonical import plain
from ._errors import SchemaError, _StrEnum
from ._schema import check_schema

__all__ = ["Tool", "ToolEffect", "Toolset"]

#: Tool names: what model tool-calling interfaces accept.
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}")
#: Scope names: ``corpus:read``, ``files/reports:write``, ...
_SCOPE = re.compile(r"[A-Za-z0-9_.:/-]{1,128}")


@unique
class ToolEffect(_StrEnum):
    """The kind of effect a tool has, which the policy permits or not."""

    READ = "read"
    """Reads only; the world is the same after the call."""

    WRITE = "write"
    """Changes state the caller owns (a file, a store, an index)."""

    EXTERNAL = "external"
    """Reaches outside the process: the network, a message, a payment."""


def _closed_object() -> dict[str, Any]:
    return {"type": "object", "properties": {}, "additionalProperties": False}


@dataclasses.dataclass(frozen=True, eq=False)
class Tool:
    """
    One callable tool and the facts the gate chain decides on.

    Parameters
    ----------
    name : str
        1 to 64 characters, ``[A-Za-z_][A-Za-z0-9_-]*``: the name a planner
        uses to call it.
    function : callable
        Called with the validated arguments as keyword arguments. It must
        return a JSON value (see :func:`~scikitplot.agent._canonical.plain`);
        anything else is reported as ``tool_result_not_json``.
    description : str, default=""
        What the tool does, for the planner.
    parameters : mapping or None, default=None
        The JSON Schema of the arguments, in the subset of
        :mod:`scikitplot.agent._schema`; its root is a closed object.
        ``None`` is a tool without arguments.
    effect : ToolEffect or str, default="read"
        Its kind of effect.
    scopes : iterable of str, default=()
        Scopes a policy must grant before the tool runs (``"corpus:read"``).
    cost : float, default=1.0
        Budget units one call costs, in the caller's unit (Budget gate).
    idempotent : bool or None, default=None
        Whether a call may be repeated with the same effect. ``None`` is
        ``True`` for a ``read`` tool and ``False`` otherwise. A call of a
        non-idempotent tool that was in flight when a run stopped is not
        repeated at resume (``tool_interrupted``).

    Raises
    ------
    TypeError
        If a field has the wrong type.
    ValueError
        If ``name`` or a scope is malformed, or ``cost`` is negative or not
        finite.
    SchemaError
        If ``parameters`` is outside the supported subset, or does not fit the
        function's signature: a property the function cannot take as a
        keyword, or a parameter without a default that the schema does not
        require.

    Examples
    --------
    >>> from scikitplot.agent import Tool
    >>> def add(a, b=0):
    ...     return a + b
    >>> tool = Tool(
    ...     "add",
    ...     add,
    ...     parameters={
    ...         "type": "object",
    ...         "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    ...         "required": ["a"],
    ...         "additionalProperties": False,
    ...     },
    ... )
    >>> tool.effect.value, tool.idempotent
    ('read', True)
    """

    name: str
    function: Callable[..., Any]
    description: str = ""
    parameters: Mapping[str, Any] | None = None
    effect: ToolEffect = ToolEffect.READ
    scopes: frozenset[str] = frozenset()
    cost: float = 1.0
    idempotent: bool | None = None

    def __post_init__(self) -> None:  # noqa: PLR0912
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            raise ValueError(
                f"Tool.name must match [A-Za-z_][A-Za-z0-9_-]{{0,63}}, got {self.name!r}"
            )
        if not callable(self.function):
            raise TypeError(f"Tool {self.name!r}: function must be callable.")
        if not isinstance(self.description, str):
            raise TypeError(f"Tool {self.name!r}: description must be a string.")
        schema = _closed_object() if self.parameters is None else self.parameters
        object.__setattr__(
            self,
            "parameters",
            check_schema(schema, what=f"Tool {self.name!r} parameters"),
        )
        try:
            object.__setattr__(self, "effect", ToolEffect(self.effect))
        except ValueError:
            raise ValueError(
                f"Tool {self.name!r}: effect must be one of "
                f"{[e.value for e in ToolEffect]}, got {self.effect!r}"
            ) from None
        if isinstance(self.scopes, str) or not isinstance(self.scopes, Iterable):
            raise TypeError(f"Tool {self.name!r}: scopes must be an iterable of str.")
        scopes = frozenset(self.scopes)
        bad = sorted(
            repr(s) for s in scopes if not isinstance(s, str) or not _SCOPE.fullmatch(s)
        )
        if bad:
            raise ValueError(
                f"Tool {self.name!r}: scopes must match [A-Za-z0-9_.:/-]{{1,128}}: {bad}"
            )
        object.__setattr__(self, "scopes", scopes)
        if isinstance(self.cost, bool) or not isinstance(self.cost, (int, float)):
            raise TypeError(f"Tool {self.name!r}: cost must be a number.")
        if not math.isfinite(self.cost) or self.cost < 0:
            raise ValueError(f"Tool {self.name!r}: cost must be finite and >= 0.")
        object.__setattr__(self, "cost", float(self.cost))
        if self.idempotent is None:
            object.__setattr__(self, "idempotent", self.effect is ToolEffect.READ)
        elif not isinstance(self.idempotent, bool):
            raise TypeError(f"Tool {self.name!r}: idempotent must be a bool or None.")
        self._check_signature()

    def _check_signature(self) -> None:
        """Refuse a schema the function cannot be called with."""
        try:
            signature = inspect.signature(self.function)
        except (TypeError, ValueError):
            return  # a builtin without a signature: the call itself decides
        params = signature.parameters
        var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
        keyword = {
            name
            for name, p in params.items()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        }
        declared = set(self.parameters.get("properties", {}))
        required = set(self.parameters.get("required", ()))
        cannot = sorted(declared - keyword) if not var_kw else []
        if cannot:
            raise SchemaError(
                f"Tool {self.name!r}: the function takes no keyword argument {cannot}"
            )
        needs = sorted(
            name
            for name, p in params.items()
            if p.default is p.empty
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            and name not in required
        )
        if needs:
            raise SchemaError(
                f"Tool {self.name!r}: the function needs {needs}, which the schema "
                "does not require"
            )
        positional_only = sorted(
            name
            for name, p in params.items()
            if p.kind is p.POSITIONAL_ONLY and p.default is p.empty
        )
        if positional_only:
            raise SchemaError(
                f"Tool {self.name!r}: positional-only parameters {positional_only} "
                "cannot be passed as arguments"
            )

    def describe(self) -> dict[str, Any]:
        """Return what a planner needs to call this tool, as JSON."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": plain(self.parameters),
            "effect": self.effect.value,
            "scopes": sorted(self.scopes),
            "cost": self.cost,
            "idempotent": self.idempotent,
        }


class Toolset:
    """
    The tools one agent may call, by name.

    Parameters
    ----------
    tools : iterable of Tool, default=()
        Names must be distinct.

    Raises
    ------
    TypeError
        If an item is not a :class:`Tool`.
    ValueError
        If two tools share a name.

    Examples
    --------
    >>> from scikitplot.agent import Tool, Toolset
    >>> tools = Toolset([Tool("ping", lambda: "pong")])
    >>> "ping" in tools, len(tools), tools.names()
    (True, 1, ('ping',))
    """

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        by_name: dict[str, Tool] = {}
        for tool in tools:
            if not isinstance(tool, Tool):
                raise TypeError(f"Toolset: {tool!r} is not a Tool.")
            if tool.name in by_name:
                raise ValueError(f"Toolset: two tools are named {tool.name!r}.")
            by_name[tool.name] = tool
        self._tools = dict(sorted(by_name.items()))

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __getitem__(self, name: str) -> Tool:
        return self._tools[name]

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"Toolset({list(self._tools)})"

    def get(self, name: str) -> Tool | None:
        """Return the tool named ``name``, or ``None``."""
        return self._tools.get(name)

    def names(self) -> tuple[str, ...]:
        """Return the tool names, sorted."""
        return tuple(self._tools)

    def describe(self) -> list[dict[str, Any]]:
        """Return :meth:`Tool.describe` of every tool, sorted by name."""
        return [tool.describe() for tool in self._tools.values()]
