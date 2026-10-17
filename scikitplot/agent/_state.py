# scikitplot/agent/_state.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Typed, versioned, JSON-serialisable run state.

The state is everything a run is: the goal, each step taken (the planner's
decision, the gates it passed, the tool's result or the error), what has been
spent, and, when the run is paused, the action waiting for approval. It is
saved after every step, so a run resumes from its last saved state, in this
process or another.

Notes
-----
**Developer.** :meth:`RunState.from_dict` is strict: unknown or missing keys,
another schema version, or a field of the wrong type raise :class:`StateError`.
A state file is read back exactly or not at all; a guessed repair would resume
a run from a state it never had.
"""

from __future__ import annotations

import dataclasses
import math
import re
from enum import unique
from typing import Any, Mapping

from ._canonical import digest, plain
from ._errors import AgentError, StateError, _StrEnum

__all__ = [
    "Action",
    "Approval",
    "ApprovalRequest",
    "Decision",
    "RunState",
    "RunStatus",
    "Spent",
    "Step",
    "StepOutcome",
    "StopReason",
    "Usage",
]

#: Version of :meth:`RunState.to_dict`; bumped on any incompatible change.
SCHEMA_VERSION = 1
#: A run id: 64 hexadecimal digits (decision D3).
RUN_ID = re.compile(r"[0-9a-f]{64}")


@unique
class RunStatus(_StrEnum):
    """Where a run is."""

    RUNNING = "running"
    """Steps remain; ``run`` or ``resume`` continues it."""

    AWAITING_APPROVAL = "awaiting_approval"
    """Paused before an action that needs approval (a human checkpoint)."""

    COMPLETED = "completed"
    """The planner finished and the done criterion accepted the answer."""

    STOPPED = "stopped"
    """A limit stopped the run; ``stop_reason`` names it."""

    FAILED = "failed"
    """The planner or the done criterion failed; ``error`` says how."""


@unique
class StopReason(_StrEnum):
    """Why a run is no longer running. Set whenever the status is not running."""

    DONE = "done"
    MAX_STEPS = "max_steps"
    MAX_COST = "max_cost"
    MAX_TOKENS = "max_tokens"
    TIMEOUT = "timeout"
    APPROVAL_REQUIRED = "approval_required"
    PLANNER_FAILED = "planner_failed"
    INVALID_DECISION = "invalid_decision"
    USAGE_NOT_REPORTED = "usage_not_reported"
    DONE_CHECK_FAILED = "done_check_failed"


@unique
class StepOutcome(_StrEnum):
    """What one step came to."""

    EXECUTED = "executed"
    """The tool ran and returned a JSON value."""

    FAILED = "failed"
    """The tool raised, or returned something that is not JSON."""

    DENIED = "denied"
    """A gate refused the action; the tool did not run."""

    INTERRUPTED = "interrupted"
    """The call was in flight when the run stopped, and was not repeated."""

    FINISHED = "finished"
    """The planner's answer was accepted."""

    REJECTED = "rejected"
    """The done criterion refused the planner's answer."""


def _check_text(value: Any, what: str, *, empty: bool = True) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a string, got {type(value).__name__}.")
    if not empty and not value.strip():
        raise ValueError(f"{what} must not be empty.")
    return value


def _check_count(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{what} must be a non-negative integer, got {value!r}.")
    return int(value)


def _check_amount(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{what} must be a number, got {value!r}.")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{what} must be finite and >= 0, got {value!r}.")
    return float(value)


@dataclasses.dataclass(frozen=True)
class Usage:
    """
    What producing one decision cost, as the planner reports it.

    Parameters
    ----------
    tokens : int, default=0
        Model tokens (input and output) used for the decision.
    cost : float, default=0.0
        Cost in the caller's budget unit (the same unit as ``Tool.cost``).
    """

    tokens: int = 0
    cost: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", _check_count(self.tokens, "Usage.tokens"))
        object.__setattr__(self, "cost", _check_amount(self.cost, "Usage.cost"))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {"tokens": self.tokens, "cost": self.cost}


@dataclasses.dataclass(frozen=True, eq=False)
class Action:
    """
    A call the planner proposes: a tool name and its arguments.

    Parameters
    ----------
    tool : str
        The tool's name.
    arguments : mapping, default={}
        JSON arguments; converted to plain JSON (a tuple becomes a list).
    """

    tool: str
    arguments: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_text(self.tool, "Action.tool", empty=False)
        if not isinstance(self.arguments, Mapping):
            raise TypeError("Action.arguments must be a mapping.")
        object.__setattr__(
            self, "arguments", plain(dict(self.arguments), what="Action.arguments")
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Action):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    __hash__ = None  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {"tool": self.tool, "arguments": plain(self.arguments)}


@dataclasses.dataclass(frozen=True, eq=False)
class Decision:
    """
    What the planner decided: call a tool, or finish with an answer.

    Build one with :meth:`act` or :meth:`finish`.

    Parameters
    ----------
    kind : {"act", "finish"}
        Call a tool, or finish with an answer.
    action : Action or None, default=None
        The call, for an ``act`` decision.
    answer : object, default=None
        The answer, a JSON value, for a ``finish`` decision.
    rationale : str, default=""
        Why, in the planner's words; recorded, never acted on.
    usage : Usage or None, default=None
        What the decision cost. Required when the run has a token limit.

    Examples
    --------
    >>> from scikitplot.agent import Decision
    >>> Decision.act("search", {"query": "spans"}).kind
    'act'
    >>> Decision.finish("42", rationale="found it").kind
    'finish'
    """

    kind: str
    action: Action | None = None
    answer: Any = None
    rationale: str = ""
    usage: Usage | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("act", "finish"):
            raise ValueError(
                f'Decision.kind must be "act" or "finish", got {self.kind!r}.'
            )
        if self.kind == "finish":
            if self.action is not None:
                raise ValueError("a finish Decision carries no action.")
            object.__setattr__(self, "answer", plain(self.answer, what="answer"))
        else:
            if not isinstance(self.action, Action):
                raise TypeError("an act Decision needs an Action.")
            if self.answer is not None:
                raise ValueError("an act Decision carries no answer.")
        _check_text(self.rationale, "Decision.rationale")
        if self.usage is not None and not isinstance(self.usage, Usage):
            raise TypeError("Decision.usage must be a Usage or None.")

    @classmethod
    def act(
        cls,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        rationale: str = "",
        usage: Usage | None = None,
    ) -> Decision:
        """Decide to call ``tool`` with ``arguments``."""
        return cls(
            "act",
            action=Action(tool, {} if arguments is None else arguments),
            rationale=rationale,
            usage=usage,
        )

    @classmethod
    def finish(
        cls, answer: Any, *, rationale: str = "", usage: Usage | None = None
    ) -> Decision:
        """Decide to finish with ``answer`` (a JSON value)."""
        return cls("finish", answer=answer, rationale=rationale, usage=usage)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {
            "kind": self.kind,
            "action": None if self.action is None else self.action.to_dict(),
            "answer": plain(self.answer),
            "rationale": self.rationale,
            "usage": None if self.usage is None else self.usage.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Decision:
        """Rebuild a decision written by :meth:`to_dict`."""
        _keys(data, {"kind", "action", "answer", "rationale", "usage"}, "decision")
        usage = None if data["usage"] is None else Usage(**_usage(data["usage"]))
        if data["kind"] == "finish":
            return cls.finish(data["answer"], rationale=data["rationale"], usage=usage)
        if data["kind"] != "act" or not isinstance(data["action"], Mapping):
            raise StateError(f"a decision of unknown kind: {data!r}")
        _keys(data["action"], {"tool", "arguments"}, "action")
        return cls.act(
            data["action"]["tool"],
            data["action"]["arguments"],
            rationale=data["rationale"],
            usage=usage,
        )


def _usage(data: Any) -> dict[str, Any]:
    _keys(data, {"tokens", "cost"}, "usage")
    return dict(data)


def _keys(data: Any, expected: set[str], what: str) -> None:
    if not isinstance(data, Mapping):
        raise StateError(f"{what}: expected an object, got {type(data).__name__}")
    if set(data) != expected:
        missing = sorted(expected - set(data))
        unknown = sorted(set(data) - expected)
        raise StateError(f"{what}: missing {missing}, unknown {unknown}")


def approval_digest(
    run_id: str,
    step: int,
    action: Action,
    *,
    effect: str,
    scopes: Any,
    head: str,
) -> str:
    """
    Return the digest an approval is bound to (invariant I4).

    It covers the run, the step, the tool, its canonical arguments, the
    tool's effect and scopes *as registered when the digest is computed*, and
    ``head``, the digest of the run's last audit record before the request.
    An approval given for one action authorises no other; none at all once
    the tool's effect or scopes change; and none in another run that reuses
    the run id, whose history (its first record names its goal and start
    time) differs.
    """
    return digest(
        {
            "run_id": run_id,
            "head": head,
            "step": step,
            "tool": action.tool,
            "arguments": action.arguments,
            "effect": str(effect),
            "scopes": sorted(scopes),
        }
    )


def call_key(run_id: str, step: int, action: Action) -> str:
    """Return the idempotency key of one call: its run, step, tool, arguments."""
    return digest(
        {
            "run_id": run_id,
            "step": step,
            "tool": action.tool,
            "arguments": action.arguments,
        }
    )


@dataclasses.dataclass(frozen=True, eq=False)
class ApprovalRequest:
    """
    An action waiting at the Approval gate.

    Parameters
    ----------
    run_id : str
    step : int
        The step the action belongs to.
    decision : Decision
        The planner's decision that proposed it.
    effect : str
        The tool's effect when the request was made.
    scopes : tuple of str
        The tool's scopes when the request was made, sorted.
    reason : str
        Why approval is needed (the effect or the tool the policy names).
    head : str
        The digest of the run's last audit record before the request.
    digest : str
        :func:`approval_digest` of the above; an :class:`Approval` must name it.
    gates : tuple of dict
        The gates the action passed before this one.
    attempt : int, default=1
        Which start of the call this is (an idempotent call retried after a
        crash counts on), so a pause for approval does not reset the count.

    Raises
    ------
    TypeError, ValueError
        If a field is malformed (ids and digests are 64 hexadecimal digits).
    """

    run_id: str
    step: int
    decision: Decision
    effect: str
    scopes: tuple[str, ...]
    reason: str
    head: str
    digest: str
    gates: tuple[Mapping[str, Any], ...] = ()
    attempt: int = 1

    def __post_init__(self) -> None:
        if _check_count(self.attempt, "ApprovalRequest.attempt") < 1:
            raise ValueError("ApprovalRequest.attempt counts from 1.")
        for name in ("run_id", "head", "digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or not RUN_ID.fullmatch(value):
                raise ValueError(
                    f"ApprovalRequest.{name} must be 64 hexadecimal digits."
                )
        _check_count(self.step, "ApprovalRequest.step")
        if not isinstance(self.decision, Decision) or self.decision.action is None:
            raise TypeError("ApprovalRequest.decision must be an act Decision.")
        _check_text(self.effect, "ApprovalRequest.effect", empty=False)
        _check_text(self.reason, "ApprovalRequest.reason")
        scopes = tuple(self.scopes)
        if not all(isinstance(x, str) for x in scopes) or list(scopes) != sorted(
            set(scopes)
        ):
            raise ValueError("ApprovalRequest.scopes must be distinct strings, sorted.")
        object.__setattr__(self, "scopes", scopes)
        object.__setattr__(self, "gates", tuple(plain(list(self.gates))))

    @property
    def action(self) -> Action:
        """The action waiting for approval."""
        assert self.decision.action is not None  # noqa: S101 - an act decision
        return self.decision.action

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {
            "run_id": self.run_id,
            "step": self.step,
            "decision": self.decision.to_dict(),
            "effect": self.effect,
            "scopes": list(self.scopes),
            "reason": self.reason,
            "head": self.head,
            "digest": self.digest,
            "gates": plain(list(self.gates)),
            "attempt": self.attempt,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ApprovalRequest:
        """Rebuild a request written by :meth:`to_dict`."""
        _keys(
            data,
            {
                "run_id",
                "step",
                "decision",
                "effect",
                "scopes",
                "reason",
                "head",
                "digest",
                "gates",
                "attempt",
            },
            "pending",
        )
        decision = Decision.from_dict(data["decision"])
        if decision.action is None:
            raise StateError("pending: the decision is not an act decision")
        return cls(
            run_id=data["run_id"],
            step=_check_count(data["step"], "pending.step"),
            decision=decision,
            effect=data["effect"],
            scopes=tuple(data["scopes"]),
            reason=data["reason"],
            head=data["head"],
            digest=data["digest"],
            gates=tuple(_list(data["gates"], "pending.gates")),
            attempt=data["attempt"],
        )


@dataclasses.dataclass(frozen=True)
class Approval:
    """
    A person's answer to an :class:`ApprovalRequest`.

    Parameters
    ----------
    digest : str
        The request's :attr:`ApprovalRequest.digest`: what is approved.
    granted : bool
        ``True`` approves the action; ``False`` refuses it.
    approver : str
        Who answered, recorded in the state and the audit trail.
    note : str, default=""
        Recorded with the answer.

    Examples
    --------
    >>> from scikitplot.agent import Approval
    >>> Approval("0" * 64, granted=True, approver="allen").granted
    True
    """

    digest: str
    granted: bool
    approver: str
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.digest, str) or not RUN_ID.fullmatch(self.digest):
            raise ValueError("Approval.digest must be 64 hexadecimal digits.")
        if self.granted is not True and self.granted is not False:
            raise TypeError("Approval.granted must be a bool.")
        _check_text(self.approver, "Approval.approver", empty=False)
        _check_text(self.note, "Approval.note")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, eq=False)
class Step:
    """
    One completed step: a decision and what came of it.

    Parameters
    ----------
    index : int
        0-based; the order of the planner's decisions.
    decision : Decision
    outcome : StepOutcome
    gates : tuple of dict
        Each gate's result, in order (``{"gate": ..., "passed": ...}``).
    result : object, default=None
        The tool's JSON result, for an executed step.
    error : AgentError or None, default=None
        Why the step did not execute, failed, or was rejected.
    key : str or None, default=None
        The call's idempotency key (:func:`call_key`), for an act step.
    seconds : float, default=0.0
        Time the tool call took.
    """

    index: int
    decision: Decision
    outcome: StepOutcome
    gates: tuple[Mapping[str, Any], ...] = ()
    result: Any = None
    error: AgentError | None = None
    key: str | None = None
    seconds: float = 0.0

    def __post_init__(self) -> None:  # noqa: PLR0912
        _check_count(self.index, "Step.index")
        if not isinstance(self.decision, Decision):
            raise TypeError("Step.decision must be a Decision.")
        outcome = StepOutcome(self.outcome)
        object.__setattr__(self, "outcome", outcome)
        if outcome not in _OUTCOMES[self.decision.kind]:
            raise StateError(
                f"step {self.index}: a {self.decision.kind} decision cannot have "
                f"outcome {outcome.value!r}"
            )
        if self.error is not None and not isinstance(self.error, AgentError):
            raise TypeError("Step.error must be an AgentError or None.")
        if (self.error is None) != (outcome in _WITHOUT_ERROR):
            raise StateError(
                f"step {self.index}: outcome {outcome.value!r} "
                + ("needs an error" if self.error is None else "carries no error")
            )
        if outcome is not StepOutcome.EXECUTED and self.result is not None:
            raise StateError(f"step {self.index}: only an executed step has a result")
        object.__setattr__(self, "result", plain(self.result, what="Step.result"))
        if outcome in _CALLED:
            if not isinstance(self.key, str) or not RUN_ID.fullmatch(self.key):
                raise StateError(
                    f"step {self.index}: a called step needs its 64-hex key"
                )
        elif self.key is not None:
            raise StateError(f"step {self.index}: only a called step has a key")
        object.__setattr__(self, "gates", tuple(plain(list(self.gates))))
        object.__setattr__(self, "seconds", _check_amount(self.seconds, "Step.seconds"))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {
            "index": self.index,
            "decision": self.decision.to_dict(),
            "outcome": StepOutcome(self.outcome).value,
            "gates": plain(list(self.gates)),
            "result": plain(self.result),
            "error": None if self.error is None else self.error.to_dict(),
            "key": self.key,
            "seconds": self.seconds,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Step:
        """Rebuild a step written by :meth:`to_dict`."""
        _keys(
            data,
            {
                "index",
                "decision",
                "outcome",
                "gates",
                "result",
                "error",
                "key",
                "seconds",
            },
            "step",
        )
        return cls(
            index=_check_count(data["index"], "step.index"),
            decision=Decision.from_dict(data["decision"]),
            outcome=StepOutcome(data["outcome"]),
            gates=tuple(_list(data["gates"], "step.gates")),
            result=plain(data["result"]),
            error=(
                None if data["error"] is None else AgentError.from_dict(data["error"])
            ),
            key=data["key"],
            seconds=_check_amount(data["seconds"], "step.seconds"),
        )


#: Outcomes each kind of decision can have.
_OUTCOMES = {
    "act": frozenset(
        {
            StepOutcome.EXECUTED,
            StepOutcome.FAILED,
            StepOutcome.DENIED,
            StepOutcome.INTERRUPTED,
        }
    ),
    "finish": frozenset({StepOutcome.FINISHED, StepOutcome.REJECTED}),
}
#: Outcomes that carry no error; every other one carries one.
_WITHOUT_ERROR = frozenset({StepOutcome.EXECUTED, StepOutcome.FINISHED})
#: Outcomes of a tool call that started, which therefore have a call key.
_CALLED = frozenset({StepOutcome.EXECUTED, StepOutcome.FAILED, StepOutcome.INTERRUPTED})
#: Stop reasons each ended status can have.
_STOPS = {
    RunStatus.COMPLETED: frozenset({StopReason.DONE}),
    RunStatus.AWAITING_APPROVAL: frozenset({StopReason.APPROVAL_REQUIRED}),
    RunStatus.STOPPED: frozenset(
        {
            StopReason.MAX_STEPS,
            StopReason.MAX_COST,
            StopReason.MAX_TOKENS,
            StopReason.TIMEOUT,
        }
    ),
    RunStatus.FAILED: frozenset(
        {
            StopReason.PLANNER_FAILED,
            StopReason.INVALID_DECISION,
            StopReason.USAGE_NOT_REPORTED,
            StopReason.DONE_CHECK_FAILED,
        }
    ),
}


@dataclasses.dataclass
class Spent:
    """What a run has used so far, against its :class:`Limits`."""

    steps: int = 0
    tool_calls: int = 0
    cost: float = 0.0
    tokens: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Spent:
        """Rebuild what :meth:`to_dict` wrote."""
        _keys(data, {"steps", "tool_calls", "cost", "tokens", "seconds"}, "spent")
        return cls(
            steps=_check_count(data["steps"], "spent.steps"),
            tool_calls=_check_count(data["tool_calls"], "spent.tool_calls"),
            cost=_check_amount(data["cost"], "spent.cost"),
            tokens=_check_count(data["tokens"], "spent.tokens"),
            seconds=_check_amount(data["seconds"], "spent.seconds"),
        )


_STATE_KEYS = {
    "schema_version",
    "run_id",
    "goal",
    "status",
    "stop_reason",
    "steps",
    "spent",
    "answer",
    "pending",
    "in_flight",
    "error",
    "audit_head",
}


@dataclasses.dataclass
class RunState:
    """
    The whole of one run, saved after every step.

    Parameters
    ----------
    run_id : str
        64 hexadecimal digits.
    goal : str
        What the run is for, as given to :meth:`Agent.run`.
    status : RunStatus, default="running"
    stop_reason : StopReason or None, default=None
        Set whenever ``status`` is not ``running``.
    steps : list of Step
    spent : Spent
    answer : object, default=None
        The accepted answer, for a completed run.
    pending : ApprovalRequest or None, default=None
        The action waiting for approval, for a paused run.
    in_flight : dict or None, default=None
        ``{"step", "key", "idempotent", "attempt", "decision", "gates"}``
        while a tool call runs; a state saved with it set is a run that
        stopped during the call.
    error : AgentError or None, default=None
        Why the run failed or stopped.
    audit_head : dict or None, default=None
        ``{"seq", "digest"}`` of the run's last audit record, so a resumed
        run continues the same hash chain.

    Notes
    -----
    **User.** A planner receives a copy of the state rebuilt from its JSON
    form, so it reads exactly what was saved and cannot change the run.
    """

    run_id: str
    goal: str
    status: RunStatus = RunStatus.RUNNING
    stop_reason: StopReason | None = None
    steps: list[Step] = dataclasses.field(default_factory=list)
    spent: Spent = dataclasses.field(default_factory=Spent)
    answer: Any = None
    pending: ApprovalRequest | None = None
    in_flight: dict[str, Any] | None = None
    error: AgentError | None = None
    audit_head: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not RUN_ID.fullmatch(self.run_id):
            raise ValueError("RunState.run_id must be 64 hexadecimal digits.")
        _check_text(self.goal, "RunState.goal", empty=False)
        self.status = RunStatus(self.status)
        if self.stop_reason is not None:
            self.stop_reason = StopReason(self.stop_reason)
        self.check()

    def check(self) -> None:  # noqa: PLR0912
        """
        Check that the state is one a run can be in.

        Raises
        ------
        StateError
            If the status and the stop reason disagree; the steps are not
            numbered 0, 1, ...; a pending request or an in-flight call is not
            for the next step of this run; a completed run does not end with
            its accepted answer; or ``spent.steps`` is not the number of
            decisions the steps account for.
        """
        status, reason, steps = self.status, self.stop_reason, self.steps
        if status is RunStatus.RUNNING:
            if reason is not None:
                raise StateError("a running run has no stop reason")
        elif reason not in _STOPS[status]:
            raise StateError(
                f"status {status.value!r} cannot have stop reason {reason!r}; "
                f"expected one of {sorted(r.value for r in _STOPS[status])}"
            )
        if not isinstance(steps, list) or not all(isinstance(x, Step) for x in steps):
            raise StateError("RunState.steps must be a list of Step")
        if [x.index for x in steps] != list(range(len(steps))):
            raise StateError(
                f"step indices {[x.index for x in steps]} are not 0, 1, ..."
            )
        if not isinstance(self.spent, Spent):
            raise StateError("RunState.spent must be a Spent")
        waiting = self.pending is not None or self.in_flight is not None
        if (status is RunStatus.AWAITING_APPROVAL) != (self.pending is not None):
            raise StateError("pending is set exactly when the run awaits approval")
        if self.pending is not None:
            if not isinstance(self.pending, ApprovalRequest):
                raise StateError("RunState.pending must be an ApprovalRequest")
            if self.pending.run_id != self.run_id or self.pending.step != len(steps):
                raise StateError("pending is not the next step of this run")
        if self.in_flight is not None:
            if status is not RunStatus.RUNNING or self.pending is not None:
                raise StateError("a call is in flight only while the run is running")
            if self.in_flight.get("step") != len(steps):
                raise StateError("the call in flight is not the next step of this run")
        low = len(steps) + (1 if waiting else 0)
        if not low <= self.spent.steps <= len(steps) + 1:
            raise StateError(
                f"spent.steps {self.spent.steps} for {len(steps)} steps"
                + (" and one waiting" if waiting else "")
            )
        finished = [x.index for x in steps if x.outcome is StepOutcome.FINISHED]
        if status is RunStatus.COMPLETED:
            if finished != [len(steps) - 1]:
                raise StateError("a completed run ends with its one accepted answer")
            if digest(plain(self.answer)) != digest(steps[-1].decision.answer):
                raise StateError("the answer is not the accepted step's answer")
        elif finished or self.answer is not None:
            raise StateError("only a completed run has an accepted answer")
        if self.error is not None and not isinstance(self.error, AgentError):
            raise StateError("RunState.error must be an AgentError or None")
        if (self.error is not None) != (
            status in (RunStatus.STOPPED, RunStatus.FAILED)
        ):
            raise StateError("an error is set exactly when the run stopped or failed")

    @property
    def done(self) -> bool:
        """Whether the run has ended (completed, stopped or failed)."""
        return self.status in (RunStatus.COMPLETED, RunStatus.STOPPED, RunStatus.FAILED)

    def observations(self) -> list[dict[str, Any]]:
        """
        Return each step as a planner reads it, in order.

        Returns
        -------
        list of dict
            ``{"index", "kind", "tool", "arguments", "outcome", "result",
            "error"}`` per step; ``error`` is ``{"code", "message"}`` or
            ``None``.
        """
        out = []
        for step in self.steps:
            action = step.decision.action
            out.append(
                {
                    "index": step.index,
                    "kind": step.decision.kind,
                    "tool": None if action is None else action.tool,
                    "arguments": None if action is None else plain(action.arguments),
                    "outcome": step.outcome.value,
                    "result": plain(step.result),
                    "error": (
                        None
                        if step.error is None
                        else {
                            "code": step.error.code.value,
                            "message": step.error.message,
                        }
                    ),
                }
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping (schema version :data:`SCHEMA_VERSION`)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "goal": self.goal,
            "status": self.status.value,
            "stop_reason": None if self.stop_reason is None else self.stop_reason.value,
            "steps": [s.to_dict() for s in self.steps],
            "spent": self.spent.to_dict(),
            "answer": plain(self.answer),
            "pending": None if self.pending is None else self.pending.to_dict(),
            "in_flight": plain(self.in_flight),
            "error": None if self.error is None else self.error.to_dict(),
            "audit_head": plain(self.audit_head),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunState:
        """
        Rebuild a state written by :meth:`to_dict`.

        Raises
        ------
        StateError
            If a key is missing or unknown, the schema version is not
            :data:`SCHEMA_VERSION`, or a field is malformed.
        """
        _keys(data, _STATE_KEYS, "state")
        if data["schema_version"] != SCHEMA_VERSION:
            raise StateError(
                f"state schema version {data['schema_version']!r}; this version "
                f"of scikitplot.agent reads {SCHEMA_VERSION}"
            )
        try:
            return cls(
                run_id=data["run_id"],
                goal=data["goal"],
                status=RunStatus(data["status"]),
                stop_reason=(
                    None
                    if data["stop_reason"] is None
                    else StopReason(data["stop_reason"])
                ),
                steps=[Step.from_dict(s) for s in _list(data["steps"], "steps")],
                spent=Spent.from_dict(data["spent"]),
                answer=plain(data["answer"]),
                pending=(
                    None
                    if data["pending"] is None
                    else ApprovalRequest.from_dict(data["pending"])
                ),
                in_flight=_in_flight(data["in_flight"]),
                error=(
                    None
                    if data["error"] is None
                    else AgentError.from_dict(data["error"])
                ),
                audit_head=_audit_head(data["audit_head"]),
            )
        except StateError:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise StateError(f"a malformed state: {exc}") from None

    def copy(self) -> RunState:
        """Return an independent copy, rebuilt from :meth:`to_dict`."""
        return RunState.from_dict(self.to_dict())


def _list(value: Any, what: str) -> list[Any]:
    if not isinstance(value, list):
        raise StateError(f"{what}: expected a list")
    return value


def _in_flight(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    _keys(
        value,
        {"step", "key", "idempotent", "attempt", "decision", "gates"},
        "in_flight",
    )
    _check_count(value["step"], "in_flight.step")
    if not isinstance(value["key"], str) or not RUN_ID.fullmatch(value["key"]):
        raise StateError("in_flight.key must be 64 hexadecimal digits")
    if value["idempotent"] is not True and value["idempotent"] is not False:
        raise StateError("in_flight.idempotent must be a bool")
    if _check_count(value["attempt"], "in_flight.attempt") < 1:
        raise StateError("in_flight.attempt counts from 1")
    if Decision.from_dict(value["decision"]).action is None:
        raise StateError("in_flight.decision is not an act decision")
    _list(value["gates"], "in_flight.gates")
    return dict(plain(value))


def _audit_head(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    _keys(value, {"seq", "digest"}, "audit_head")
    _check_count(value["seq"], "audit_head.seq")
    if not isinstance(value["digest"], str) or not RUN_ID.fullmatch(value["digest"]):
        raise StateError("audit_head.digest must be 64 hexadecimal digits")
    return dict(plain(value))
