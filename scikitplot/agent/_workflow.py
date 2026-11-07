# scikitplot/agent/_workflow.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Declarative workflow graphs, run by the agent loop as its planner.

A :class:`WorkflowGraph` names the steps of a task and the order between them:
:class:`Task` nodes call one tool, :class:`FanOut` nodes call several (one
after another, or at the same time), :class:`Finish` nodes answer, and
:class:`Edge` objects choose the next node, optionally on a condition. Results
are folded into named *channels* by declared reducers. The graph is checked
when it is built and compiled into an ordinary planner for
:class:`~scikitplot.agent.Agent`.

.. code-block:: python

    from scikitplot.agent import Agent, Edge, Finish, Task, WorkflowGraph

    graph = WorkflowGraph.sequence(
        Task("find", search, arguments=lambda view: {"query": view.goal}, into="hits"),
        Finish("answer", answer=lambda view: view.channels["hits"]),
        channels={"hits": "append"},
    )
    agent = Agent(graph.tools(), graph.planner())
    state = agent.run("which chunker keeps spans?")

Notes
-----
**User.** A simple task is a :meth:`WorkflowGraph.sequence`; a complex one adds
conditional edges, loops and fan-outs. Every tool call still passes the five
gates (Validation, Permissions, Scope, Budget, Approval) and is written to the
audit trail, because the graph only *decides*; the agent loop *runs*. A loop
is allowed only through a node with ``max_visits``, and a run that reaches the
bound continues at that node's ``on_exhausted`` node, so every graph that can
be built terminates.

A :class:`FanOut` runs its branches either ``"ordered"`` -- one audited step
per branch, each gated on its own, in declared order -- or ``"concurrent"`` --
one step whose tool runs every branch at the same time on a thread pool. A
concurrent fan-out is one action to the gates: its effect is the strongest of
its branches', its scopes their union and its cost their sum, so a policy and
an approver judge it as they would the branches together. Its result is
``{branch: result}`` in declared order whatever order the branches finished
in; if any branch raises, the whole step fails (all or nothing).

**Developer.** The planner is stateless: at every decision it *replays* the
recorded steps from the start node -- re-evaluating each node's arguments and
checking them against the recorded action -- to find where the run is. The
graph position is therefore never saved separately, so resume after a pause or
a crash needs nothing new, the state schema does not change, and a state that
this graph did not produce is refused (:class:`WorkflowError`) instead of being
continued from a wrong place. Replay costs one pass over the steps per
decision, bounded by ``Limits.max_steps``. Callbacks (``arguments``,
``update``, ``when``, ``answer``, custom reducers) must be deterministic
functions of the view they receive; the replay check detects an
``arguments`` callback that is not.

Termination is proved statically. With every node that has ``max_visits``
removed, the remaining transitions (edges, ``on_exhausted``, ``on_rejected``)
must form a DAG, so every cycle passes through a bounded node; and no node's
``on_exhausted`` target may reach that node again, so an exhausted node is
never re-entered. Each bounded node is entered at most ``max_visits`` times
plus once to divert, and the unbounded part is acyclic, so the number of node
entries is finite.

Only the standard library and this package's own modules are imported
(invariant I9).
"""

from __future__ import annotations

import dataclasses
import re
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from ._canonical import canonical_json, digest, plain
from ._delegate import _ORDER
from ._state import Decision, RunState, StepOutcome, Usage
from ._tools import Tool

__all__ = [
    "REDUCERS",
    "Branch",
    "Edge",
    "FanOut",
    "Finish",
    "StepSummary",
    "Task",
    "WorkflowBranchError",
    "WorkflowError",
    "WorkflowGraph",
    "WorkflowView",
]

#: Node names, as tool names: a concurrent fan-out's tool takes its node's name.
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}")
_MODES = ("ordered", "concurrent")
#: Fewest branches a fan-out has; one branch is a Task.
_MIN_BRANCHES = 2


class WorkflowError(ValueError):
    """A graph that cannot be built, or a run the graph cannot continue."""


class WorkflowBranchError(RuntimeError):
    """A branch of a concurrent fan-out raised; the fan-out step failed."""


# -- reducers -------------------------------------------------------------


def _replace(old: Any, new: Any) -> Any:
    return new


def _append(old: Any, new: Any) -> Any:
    if not isinstance(old, list):
        raise WorkflowError(f"reducer 'append' needs a list, got {type(old).__name__}")
    return [*old, new]


def _extend(old: Any, new: Any) -> Any:
    if not isinstance(old, list) or not isinstance(new, (list, tuple)):
        raise WorkflowError("reducer 'extend' needs a list and a list of new items")
    return [*old, *new]


def _add(old: Any, new: Any) -> Any:
    for value in (old, new):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WorkflowError(
                f"reducer 'add' needs numbers, got {type(value).__name__}"
            )
    return old + new


def _merge(old: Any, new: Any) -> Any:
    if not isinstance(old, dict) or not isinstance(new, dict):
        raise WorkflowError("reducer 'merge' needs two objects")
    return {**old, **new}


#: Built-in reducers: ``(old, new) -> value``, each with its empty value.
REDUCERS: Mapping[str, tuple[Callable[[Any, Any], Any], Any]] = MappingProxyType(
    {
        "replace": (_replace, None),
        "append": (_append, []),
        "extend": (_extend, []),
        "add": (_add, 0),
        "merge": (_merge, {}),
    }
)


# -- nodes and edges --------------------------------------------------------


def _check_name(value: Any, what: str) -> None:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise WorkflowError(
            f"{what} must match [A-Za-z_][A-Za-z0-9_-]{{0,63}}, got {value!r}"
        )


def _check_optional_name(value: Any, what: str) -> None:
    if value is not None:
        _check_name(value, what)


def _check_arguments(value: Any, what: str) -> None:
    if value is None or callable(value):
        return
    if not isinstance(value, Mapping):
        raise WorkflowError(f"{what}: arguments must be a mapping, a callable or None")
    plain(dict(value), what=f"{what} arguments")


def _check_bound(node: Any) -> None:
    bound = node.max_visits
    if bound is not None and (
        isinstance(bound, bool) or not isinstance(bound, int) or bound < 1
    ):
        raise WorkflowError(f"node {node.name!r}: max_visits must be an int >= 1")
    if node.on_exhausted is not None and bound is None:
        raise WorkflowError(f"node {node.name!r}: on_exhausted needs max_visits")
    _check_optional_name(node.on_exhausted, f"node {node.name!r} on_exhausted")


def _check_update(node: Any) -> None:
    _check_optional_name(node.into, f"node {node.name!r} into")
    if node.update is not None and not callable(node.update):
        raise WorkflowError(f"node {node.name!r}: update must be callable")
    if node.into is not None and node.update is not None:
        raise WorkflowError(f"node {node.name!r}: give into or update, not both")


@dataclasses.dataclass(frozen=True)
class Task:
    """
    A node that calls one tool.

    Parameters
    ----------
    name : str
        Node name, unique in the graph.
    tool : Tool
        The tool to call.
    arguments : mapping, callable or None, default=None
        The call's arguments, or ``arguments(view) -> mapping``. ``None`` is
        no arguments.
    into : str or None, default=None
        Channel the result is reduced into when the call executed.
    update : callable or None, default=None
        ``update(view, result) -> {channel: value}``, each value reduced into
        its channel, instead of ``into``.
    max_visits : int or None, default=None
        How many times the node may be entered; required on every loop.
    on_exhausted : str or None, default=None
        Node entered instead once ``max_visits`` is reached. ``None`` makes
        reaching the bound an error.

    Raises
    ------
    WorkflowError
        If a field is malformed.
    """

    name: str
    tool: Tool
    arguments: Any = None
    into: str | None = None
    update: Callable[..., Mapping[str, Any]] | None = None
    max_visits: int | None = None
    on_exhausted: str | None = None

    def __post_init__(self) -> None:
        _check_name(self.name, "Task.name")
        if not isinstance(self.tool, Tool):
            raise WorkflowError(f"Task {self.name!r}: tool must be a Tool")
        _check_arguments(self.arguments, f"Task {self.name!r}")
        _check_update(self)
        _check_bound(self)


@dataclasses.dataclass(frozen=True)
class Branch:
    """
    One branch of a :class:`FanOut`: a tool and its arguments.

    Parameters
    ----------
    name : str
        Branch name, unique in its fan-out; the key of its result.
    tool : Tool
        The tool to call.
    arguments : mapping, callable or None, default=None
        As for :class:`Task`.
    """

    name: str
    tool: Tool
    arguments: Any = None

    def __post_init__(self) -> None:
        _check_name(self.name, "Branch.name")
        if not isinstance(self.tool, Tool):
            raise WorkflowError(f"Branch {self.name!r}: tool must be a Tool")
        _check_arguments(self.arguments, f"Branch {self.name!r}")


@dataclasses.dataclass(frozen=True)
class FanOut:
    """
    A node that calls several tools, then joins their results.

    Parameters
    ----------
    name : str
        Node name, unique in the graph; also the tool name of a concurrent
        fan-out, so it must not be the name of any tool in the graph.
    branches : sequence of Branch
        At least two, with unique names.
    mode : {"ordered", "concurrent"}, default="ordered"
        One audited step per branch in declared order, or one step running
        every branch at the same time.
    max_workers : int or None, default=None
        Threads for a concurrent fan-out; ``None`` is one per branch.
    into, update, max_visits, on_exhausted
        As for :class:`Task`; the result is ``{branch: result}`` of the
        branches that executed, in declared order.

    Raises
    ------
    WorkflowError
        If a field is malformed.
    """

    name: str
    branches: tuple[Branch, ...]
    mode: str = "ordered"
    max_workers: int | None = None
    into: str | None = None
    update: Callable[..., Mapping[str, Any]] | None = None
    max_visits: int | None = None
    on_exhausted: str | None = None

    def __post_init__(self) -> None:
        _check_name(self.name, "FanOut.name")
        if isinstance(self.branches, (str, bytes)) or not isinstance(
            self.branches, Sequence
        ):
            raise WorkflowError(f"FanOut {self.name!r}: branches must be a sequence")
        branches = tuple(self.branches)
        object.__setattr__(self, "branches", branches)
        if len(branches) < _MIN_BRANCHES or not all(
            isinstance(b, Branch) for b in branches
        ):
            raise WorkflowError(f"FanOut {self.name!r}: needs two or more Branch")
        names = [b.name for b in branches]
        if len(set(names)) != len(names):
            raise WorkflowError(f"FanOut {self.name!r}: branch names must be unique")
        if self.mode not in _MODES:
            raise WorkflowError(
                f"FanOut {self.name!r}: mode must be one of {_MODES}, got {self.mode!r}"
            )
        workers = self.max_workers
        if workers is not None and (
            isinstance(workers, bool) or not isinstance(workers, int) or workers < 1
        ):
            raise WorkflowError(f"FanOut {self.name!r}: max_workers must be >= 1")
        _check_update(self)
        _check_bound(self)


@dataclasses.dataclass(frozen=True)
class Finish:
    """
    A node that ends the run with an answer.

    Parameters
    ----------
    name : str
        Node name, unique in the graph.
    answer : callable or None, default=None
        ``answer(view) -> JSON value``; ``None`` answers with every channel.
    on_rejected : str or None, default=None
        Node entered when the agent's done criterion rejects the answer.
        ``None`` makes a rejection end the workflow with an error.

    Raises
    ------
    WorkflowError
        If a field is malformed.
    """

    name: str
    answer: Callable[..., Any] | None = None
    on_rejected: str | None = None
    max_visits: int | None = None
    on_exhausted: str | None = None

    def __post_init__(self) -> None:
        _check_name(self.name, "Finish.name")
        if self.answer is not None and not callable(self.answer):
            raise WorkflowError(f"Finish {self.name!r}: answer must be callable")
        _check_optional_name(self.on_rejected, f"Finish {self.name!r} on_rejected")
        _check_bound(self)


@dataclasses.dataclass(frozen=True)
class Edge:
    """
    A transition from ``source`` to ``target``, taken when ``when`` is true.

    Parameters
    ----------
    source, target : str
        Node names.
    when : callable or None, default=None
        ``when(view) -> bool``; ``None`` always matches. A source's edges are
        tried in declaration order and the first match is taken.
    """

    source: str
    target: str
    when: Callable[..., bool] | None = None

    def __post_init__(self) -> None:
        _check_name(self.source, "Edge.source")
        _check_name(self.target, "Edge.target")
        if self.when is not None and not callable(self.when):
            raise WorkflowError(
                f"Edge {self.source!r}->{self.target!r}: when must be callable"
            )


# -- what a callback sees ---------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StepSummary:
    """
    What one recorded step came to, as a callback sees it.

    Parameters
    ----------
    node : str
        The node (``node.branch`` for a branch of an ordered fan-out).
    outcome : StepOutcome
        The step's outcome.
    result : object
        The tool's result, when it executed; otherwise ``None``.
    error_code : str or None
        The step's error code, when it has one.
    """

    node: str
    outcome: StepOutcome
    result: Any = None
    error_code: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the tool executed."""
        return self.outcome is StepOutcome.EXECUTED


@dataclasses.dataclass(frozen=True)
class WorkflowView:
    """
    Where a run stands in its workflow, rebuilt from the recorded steps.

    Parameters
    ----------
    goal : str
        The run's goal.
    node : str or None
        The node the run is at; ``None`` once it finished.
    channels : mapping
        Every channel's current value (a read-only copy).
    visits : mapping of str to int
        How many times each node has been entered.
    last : StepSummary or None
        The most recent step.
    branches : mapping of str to StepSummary
        The branches of the most recent fan-out, by branch name.
    finished : bool
        Whether a :class:`Finish` node's answer was accepted.
    """

    goal: str
    node: str | None
    channels: Mapping[str, Any]
    visits: Mapping[str, int]
    last: StepSummary | None = None
    branches: Mapping[str, StepSummary] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )
    finished: bool = False

    def digest(self) -> str:
        """
        Return the digest of this position: node, channels and visit counts.

        Returns
        -------
        str
            64 hexadecimal digits; equal for equal positions.
        """
        return digest(
            {
                "node": self.node,
                "channels": dict(self.channels),
                "visits": dict(self.visits),
                "finished": self.finished,
            }
        )


@dataclasses.dataclass
class _Position:
    """The mutable replay cursor behind a :class:`WorkflowView`."""

    goal: str
    node: str | None
    channels: dict[str, Any]
    visits: dict[str, int]
    last: StepSummary | None = None
    branches: dict[str, StepSummary] = dataclasses.field(default_factory=dict)
    branch_index: int = 0
    finished: bool = False

    def view(self) -> WorkflowView:
        return WorkflowView(
            goal=self.goal,
            node=self.node,
            channels=MappingProxyType(plain(self.channels, what="channels")),
            visits=MappingProxyType(dict(self.visits)),
            last=self.last,
            branches=MappingProxyType(dict(self.branches)),
            finished=self.finished,
        )


# -- the graph --------------------------------------------------------------


def _strongly_connected(nodes: Sequence[str], succ: Mapping[str, list[str]]) -> list:
    """Return the strongly connected components, iteratively (Tarjan)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set = set()
    stack: list[str] = []
    components: list[list[str]] = []
    counter = 0
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(succ[root]))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(succ[child])))
                    advanced = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(component)
    return components


def _reachable(start: str, succ: Mapping[str, list[str]]) -> set:
    seen = {start}
    todo = [start]
    while todo:
        for child in succ[todo.pop()]:
            if child not in seen:
                seen.add(child)
                todo.append(child)
    return seen


class WorkflowGraph:
    """
    A checked, immutable workflow: nodes, edges and channels.

    Parameters
    ----------
    start : str
        The first node.
    nodes : sequence of Task, FanOut or Finish
        Every node, with unique names.
    edges : sequence of Edge
        Transitions; a source's edges are tried in order. At most one edge of a
        source is unconditional, and it is the last.
    channels : mapping of str to str or callable, default=None
        Channel name to reducer: a name in :data:`REDUCERS` or a callable
        ``(old, new) -> value``.
    initial : mapping, default=None
        Initial channel values (JSON); a channel not given starts at its
        reducer's empty value (``None`` for a custom reducer).

    Raises
    ------
    WorkflowError
        If the graph is malformed, has a node unreachable from ``start`` or one
        from which no :class:`Finish` is reachable, has a cycle through no
        node with ``max_visits``, or an ``on_exhausted`` target that leads back
        to its node.

    See Also
    --------
    WorkflowGraph.sequence : A graph that runs its nodes in order.

    Examples
    --------
    >>> from scikitplot.agent import Edge, Finish, Task, Tool, WorkflowGraph
    >>> count = Tool("count", lambda: 1)
    >>> graph = WorkflowGraph(
    ...     "tick",
    ...     [
    ...         Task("tick", count, into="n", max_visits=3, on_exhausted="done"),
    ...         Finish("done", answer=lambda view: view.channels["n"]),
    ...     ],
    ...     [Edge("tick", "tick")],
    ...     channels={"n": "add"},
    ... )
    >>> [tool.name for tool in graph.tools()]
    ['count']
    """

    def __init__(
        self,
        start: str,
        nodes: Sequence[Any],
        edges: Sequence[Edge] = (),
        *,
        channels: Mapping[str, Any] | None = None,
        initial: Mapping[str, Any] | None = None,
    ) -> None:
        _check_name(start, "WorkflowGraph.start")
        nodes = tuple(nodes)
        edges = tuple(edges)
        for node in nodes:
            if not isinstance(node, (Task, FanOut, Finish)):
                raise WorkflowError(
                    f"a node must be a Task, FanOut or Finish, got {type(node).__name__}"
                )
        for edge in edges:
            if not isinstance(edge, Edge):
                raise WorkflowError(
                    f"an edge must be an Edge, got {type(edge).__name__}"
                )
        by_name: dict[str, Any] = {}
        for node in nodes:
            if node.name in by_name:
                raise WorkflowError(f"node name {node.name!r} is used twice")
            by_name[node.name] = node
        if start not in by_name:
            raise WorkflowError(f"start node {start!r} is not a node")
        self._start = start
        self._nodes = MappingProxyType(by_name)
        self._reducers = self._check_channels(channels, by_name)
        self._initial = self._check_initial(initial)
        self._edges = self._check_edges(edges, by_name)
        self._tools = self._collect_tools(nodes)
        self._check_termination(by_name)

    # -- construction checks ------------------------------------------------

    @staticmethod
    def _check_channels(channels: Any, nodes: Mapping[str, Any]) -> Mapping:
        channels = {} if channels is None else channels
        if not isinstance(channels, Mapping):
            raise WorkflowError("channels must be a mapping of name to reducer")
        reducers: dict[str, tuple[Callable, Any, str]] = {}
        for name, reducer in channels.items():
            _check_name(name, "channel name")
            if isinstance(reducer, str):
                if reducer not in REDUCERS:
                    raise WorkflowError(
                        f"channel {name!r}: reducer must be one of "
                        f"{sorted(REDUCERS)} or a callable, got {reducer!r}"
                    )
                function, empty = REDUCERS[reducer]
                reducers[name] = (function, empty, reducer)
            elif callable(reducer):
                reducers[name] = (reducer, None, "custom")
            else:
                raise WorkflowError(
                    f"channel {name!r}: reducer must be a name or callable"
                )
        for node in nodes.values():
            into = getattr(node, "into", None)
            if into is not None and into not in reducers:
                raise WorkflowError(
                    f"node {node.name!r}: into={into!r} is not a channel"
                )
        return MappingProxyType(reducers)

    def _check_initial(self, initial: Any) -> Mapping[str, Any]:
        initial = {} if initial is None else initial
        if not isinstance(initial, Mapping):
            raise WorkflowError("initial must be a mapping of channel to value")
        unknown = sorted(set(initial) - set(self._reducers))
        if unknown:
            raise WorkflowError(f"initial names unknown channel(s) {unknown}")
        values = {
            name: plain(initial.get(name, empty), what=f"channel {name!r}")
            for name, (_f, empty, _n) in self._reducers.items()
        }
        return MappingProxyType(values)

    @staticmethod
    def _check_edges(edges: Sequence[Edge], nodes: Mapping[str, Any]) -> Mapping:
        out: dict[str, list[Edge]] = {name: [] for name in nodes}
        for edge in edges:
            for end in (edge.source, edge.target):
                if end not in nodes:
                    raise WorkflowError(
                        f"edge {edge.source!r}->{edge.target!r}: {end!r} is not a node"
                    )
            if isinstance(nodes[edge.source], Finish):
                raise WorkflowError(
                    f"edge {edge.source!r}->{edge.target!r}: a Finish node has no "
                    "outgoing edges (use on_rejected)"
                )
            if out[edge.source] and out[edge.source][-1].when is None:
                raise WorkflowError(
                    f"edge {edge.source!r}->{edge.target!r} follows an unconditional "
                    "edge of the same source and could never be taken"
                )
            out[edge.source].append(edge)
        for name, node in nodes.items():
            if not isinstance(node, Finish) and not out[name]:
                raise WorkflowError(f"node {name!r} has no outgoing edge")
            for target in (node.on_exhausted, getattr(node, "on_rejected", None)):
                if target is not None and target not in nodes:
                    raise WorkflowError(f"node {name!r}: {target!r} is not a node")
        return MappingProxyType({name: tuple(es) for name, es in out.items()})

    @staticmethod
    def _collect_tools(nodes: Sequence[Any]) -> tuple[Tool, ...]:
        tools: dict[str, Tool] = {}

        def add(tool: Tool) -> None:
            known = tools.get(tool.name)
            if known is not None and known is not tool:
                raise WorkflowError(
                    f"two different tools are named {tool.name!r}; a toolset holds "
                    "one tool per name"
                )
            tools[tool.name] = tool

        for node in nodes:
            if isinstance(node, Task):
                add(node.tool)
            elif isinstance(node, FanOut) and node.mode == "ordered":
                for branch in node.branches:
                    add(branch.tool)
        for node in nodes:
            if isinstance(node, FanOut) and node.mode == "concurrent":
                if node.name in tools:
                    raise WorkflowError(
                        f"concurrent FanOut {node.name!r}: its tool name is already "
                        "used by another tool"
                    )
                tools[node.name] = _concurrent_tool(node)
        return tuple(tools.values())

    def _check_termination(self, nodes: Mapping[str, Any]) -> None:
        succ: dict[str, list[str]] = {name: [] for name in nodes}
        for name, node in nodes.items():
            succ[name].extend(edge.target for edge in self._edges[name])
            for target in (node.on_exhausted, getattr(node, "on_rejected", None)):
                if target is not None:
                    succ[name].append(target)
        reached = _reachable(self._start, succ)
        unreachable = [name for name in nodes if name not in reached]
        if unreachable:
            raise WorkflowError(f"node(s) {unreachable} cannot be reached from start")
        finishes = {name for name, node in nodes.items() if isinstance(node, Finish)}
        stuck = [name for name in nodes if not (_reachable(name, succ) & finishes)]
        if stuck:
            raise WorkflowError(f"no Finish node can be reached from {stuck}")
        unbounded = [name for name, node in nodes.items() if node.max_visits is None]
        inner = {
            n: [c for c in succ[n] if nodes[c].max_visits is None] for n in unbounded
        }
        for component in _strongly_connected(unbounded, inner):
            if len(component) > 1 or component[0] in inner[component[0]]:
                raise WorkflowError(
                    f"cycle through {sorted(component)} has no node with max_visits; "
                    "every loop needs a bound"
                )
        for name, node in nodes.items():
            if node.on_exhausted is not None and name in _reachable(
                node.on_exhausted, succ
            ):
                raise WorkflowError(
                    f"node {name!r}: on_exhausted={node.on_exhausted!r} leads back to "
                    "it, so the bound would not end the loop"
                )

    # -- public surface -----------------------------------------------------

    @classmethod
    def sequence(
        cls,
        *nodes: Any,
        channels: Mapping[str, Any] | None = None,
        initial: Mapping[str, Any] | None = None,
    ) -> WorkflowGraph:
        """
        Return a graph that runs ``nodes`` in order and ends at the last one.

        Parameters
        ----------
        *nodes : Task, FanOut or Finish
            At least one; the last must be a :class:`Finish`.
        channels, initial
            As for :class:`WorkflowGraph`.

        Returns
        -------
        WorkflowGraph

        Raises
        ------
        WorkflowError
            If no node is given or the last is not a :class:`Finish`.
        """
        if not nodes:
            raise WorkflowError("sequence needs at least one node")
        if not isinstance(nodes[-1], Finish):
            raise WorkflowError("the last node of a sequence must be a Finish")
        edges = [Edge(a.name, b.name) for a, b in zip(nodes, nodes[1:])]
        return cls(nodes[0].name, nodes, edges, channels=channels, initial=initial)

    @property
    def start(self) -> str:
        """The first node's name."""
        return self._start

    @property
    def nodes(self) -> Mapping[str, Any]:
        """Every node, by name."""
        return self._nodes

    @property
    def fingerprint(self) -> str:
        """
        Digest of the graph's structure.

        Notes
        -----
        **Developer.** Names, kinds, tools, modes, bounds and edge order are
        covered; callbacks are not (a function has no stable digest), so two
        graphs that differ only in a callback share a fingerprint.
        """

        def describe(node: Any) -> dict[str, Any]:
            entry: dict[str, Any] = {
                "kind": type(node).__name__,
                "max_visits": node.max_visits,
                "on_exhausted": node.on_exhausted,
            }
            if isinstance(node, Task):
                entry.update(tool=node.tool.name, into=node.into)
            elif isinstance(node, FanOut):
                entry.update(
                    mode=node.mode,
                    into=node.into,
                    branches=[[b.name, b.tool.name] for b in node.branches],
                )
            else:
                entry.update(on_rejected=node.on_rejected)
            return entry

        return digest(
            {
                "start": self._start,
                "nodes": {name: describe(node) for name, node in self._nodes.items()},
                "edges": {
                    name: [[e.target, e.when is not None] for e in edges]
                    for name, edges in self._edges.items()
                },
                "channels": {name: r[2] for name, r in self._reducers.items()},
                "initial": dict(self._initial),
            }
        )

    def tools(self) -> list[Tool]:
        """
        Return every tool the graph calls, for the agent's toolset.

        Returns
        -------
        list of Tool
            Task and ordered-branch tools, then one tool per concurrent
            fan-out, each name once.
        """
        return list(self._tools)

    def planner(self, *, artifacts: Any = None) -> Callable[..., Decision]:
        """
        Return the planner that walks this graph.

        Parameters
        ----------
        artifacts : ArtifactStore or None, default=None
            Where offloaded results are read from; needed when the agent
            offloads results (``offload_results_over``).

        Returns
        -------
        callable
            ``planner(state, tools) -> Decision``, as
            :class:`~scikitplot.agent.Agent` expects.
        """

        def plan(state: RunState, tools: Any = None) -> Decision:
            position = self._replay(state, artifacts)
            if position.finished:
                raise WorkflowError("the workflow has already finished")
            return self._expected(position)

        return plan

    def replay(self, state: RunState, *, artifacts: Any = None) -> WorkflowView:
        """
        Rebuild where ``state`` stands in this graph.

        Parameters
        ----------
        state : RunState
            A run's state.
        artifacts : ArtifactStore or None, default=None
            As for :meth:`planner`.

        Returns
        -------
        WorkflowView

        Raises
        ------
        WorkflowError
            If a recorded step is not the one this graph would have taken.
        """
        return self._replay(state, artifacts).view()

    # -- replay -------------------------------------------------------------

    def _enter(self, position: _Position, name: str) -> None:
        node = self._nodes[name]
        while (
            node.max_visits is not None
            and position.visits.get(name, 0) >= node.max_visits
        ):
            if node.on_exhausted is None:
                raise WorkflowError(
                    f"node {name!r} reached max_visits={node.max_visits} and has no "
                    "on_exhausted node"
                )
            name = node.on_exhausted
            node = self._nodes[name]
        position.visits[name] = position.visits.get(name, 0) + 1
        position.node = name
        position.branch_index = 0
        if isinstance(node, FanOut):
            # A new fan-out starts with no branch outcomes; any other node keeps
            # the last fan-out's, so its callbacks can read them.
            position.branches = {}

    def _next(self, position: _Position) -> None:
        view = position.view()
        for edge in self._edges[position.node]:
            if edge.when is None:
                self._enter(position, edge.target)
                return
            verdict = edge.when(view)
            if not isinstance(verdict, bool):
                raise WorkflowError(
                    f"edge {edge.source!r}->{edge.target!r}: when returned "
                    f"{type(verdict).__name__}, not bool"
                )
            if verdict:
                self._enter(position, edge.target)
                return
        raise WorkflowError(f"node {position.node!r}: no outgoing edge matched")

    def _reduce(self, position: _Position, updates: Any, node: str) -> None:
        if not isinstance(updates, Mapping):
            raise WorkflowError(f"node {node!r}: update must return a mapping")
        for name, value in updates.items():
            if name not in self._reducers:
                raise WorkflowError(
                    f"node {node!r}: update names unknown channel {name!r}"
                )
            function = self._reducers[name][0]
            new = function(position.channels[name], plain(value, what=name))
            position.channels[name] = plain(new, what=f"channel {name!r}")

    def _apply(self, position: _Position, node: Any, result: Any) -> None:
        if node.into is not None:
            self._reduce(position, {node.into: result}, node.name)
        elif node.update is not None:
            self._reduce(position, node.update(position.view(), result), node.name)

    @staticmethod
    def _result(step: Any, artifacts: Any) -> Any:
        if step.outcome is not StepOutcome.EXECUTED:
            return None
        if step.result_ref is None:
            return step.result
        if artifacts is None:
            raise WorkflowError(
                f"step {step.index}: its result was offloaded; pass artifacts= to "
                "planner() or replay()"
            )
        return artifacts.get(step.result_ref["digest"])

    def _arguments(self, spec: Any, view: WorkflowView, what: str) -> dict[str, Any]:
        value = spec(view) if callable(spec) else spec
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise WorkflowError(f"{what}: arguments must give a mapping")
        return plain(dict(value), what=f"{what} arguments")

    def _expected(self, position: _Position) -> Decision:
        node = self._nodes[position.node]
        view = position.view()
        rationale = f"workflow node {node.name!r}"
        if isinstance(node, Finish):
            answer = dict(view.channels) if node.answer is None else node.answer(view)
            return Decision.finish(answer, rationale=rationale, usage=Usage())
        if isinstance(node, Task):
            arguments = self._arguments(node.arguments, view, f"node {node.name!r}")
            return Decision.act(
                node.tool.name, arguments, rationale=rationale, usage=Usage()
            )
        if node.mode == "concurrent":
            arguments = {
                branch.name: self._arguments(
                    branch.arguments, view, f"branch {node.name}.{branch.name}"
                )
                for branch in node.branches
            }
            return Decision.act(
                node.name, arguments, rationale=rationale, usage=Usage()
            )
        branch = node.branches[position.branch_index]
        arguments = self._arguments(
            branch.arguments, view, f"branch {node.name}.{branch.name}"
        )
        return Decision.act(
            branch.tool.name,
            arguments,
            rationale=f"workflow node {node.name!r} branch {branch.name!r}",
            usage=Usage(),
        )

    def _replay(self, state: RunState, artifacts: Any) -> _Position:
        if not isinstance(state, RunState):
            raise WorkflowError(f"replay needs a RunState, got {type(state).__name__}")
        position = _Position(
            goal=state.goal,
            node=None,
            channels=plain(dict(self._initial), what="channels"),
            visits={},
        )
        self._enter(position, self._start)
        for step in state.steps:
            if position.finished:
                raise WorkflowError(f"step {step.index} follows the workflow's finish")
            expected = self._expected(position)
            recorded = step.decision
            if canonical_json(
                {"kind": recorded.kind, "action": _action(recorded)}
            ) != canonical_json({"kind": expected.kind, "action": _action(expected)}):
                raise WorkflowError(
                    f"step {step.index}: the run did not take this graph's step at "
                    f"node {position.node!r}; the state was not produced by this graph"
                )
            self._advance(position, step, artifacts)
        return position

    def _advance(self, position: _Position, step: Any, artifacts: Any) -> None:
        node = self._nodes[position.node]
        error = None if step.error is None else str(step.error.code)
        if isinstance(node, Finish):
            position.last = StepSummary(node.name, step.outcome, None, error)
            if step.outcome is StepOutcome.FINISHED:
                position.finished = True
                position.node = None
                return
            if node.on_rejected is None:
                raise WorkflowError(
                    f"the answer of Finish {node.name!r} was rejected and it has no "
                    "on_rejected node"
                )
            self._enter(position, node.on_rejected)
            return
        result = self._result(step, artifacts)
        if isinstance(node, Task):
            position.last = StepSummary(node.name, step.outcome, result, error)
            if step.outcome is StepOutcome.EXECUTED:
                self._apply(position, node, result)
            self._next(position)
            return
        if node.mode == "concurrent":
            position.last = StepSummary(node.name, step.outcome, result, error)
            position.branches = {
                b.name: StepSummary(
                    f"{node.name}.{b.name}",
                    step.outcome,
                    None if result is None else result.get(b.name),
                    error,
                )
                for b in node.branches
            }
            if step.outcome is StepOutcome.EXECUTED:
                self._apply(position, node, dict(result))
            self._next(position)
            return
        branch = node.branches[position.branch_index]
        summary = StepSummary(f"{node.name}.{branch.name}", step.outcome, result, error)
        position.last = summary
        position.branches[branch.name] = summary
        position.branch_index += 1
        if position.branch_index < len(node.branches):
            return
        joined = {name: s.result for name, s in position.branches.items() if s.ok}
        self._apply(position, node, joined)
        self._next(position)


def _action(decision: Decision) -> Any:
    if decision.action is None:
        return None
    return {"tool": decision.action.tool, "arguments": decision.action.arguments}


def _concurrent_tool(node: FanOut) -> Tool:
    """Compile a concurrent fan-out into one tool that runs every branch."""
    branches = node.branches
    workers = node.max_workers or len(branches)

    def run(**arguments: Any) -> dict[str, Any]:
        def call(branch: Branch) -> Any:
            return branch.tool.function(**arguments[branch.name])

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(call, branch) for branch in branches]
        results: dict[str, Any] = {}
        for branch, future in zip(branches, futures):
            exc = future.exception()
            if exc is not None:
                raise WorkflowBranchError(
                    f"fan-out {node.name!r}: branch {branch.name!r} raised "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            results[branch.name] = future.result()
        return results

    return Tool(
        node.name,
        run,
        description=(
            f"Runs {', '.join(b.name for b in branches)} at the same time "
            f"(workflow fan-out {node.name!r})."
        ),
        parameters={
            "type": "object",
            "properties": {b.name: dict(b.tool.parameters) for b in branches},
            "required": [b.name for b in branches],
            "additionalProperties": False,
        },
        effect=max((b.tool.effect for b in branches), key=_ORDER.__getitem__),
        scopes=frozenset().union(*(b.tool.scopes for b in branches)),
        cost=sum(b.tool.cost for b in branches),
        idempotent=all(b.tool.idempotent for b in branches),
    )
