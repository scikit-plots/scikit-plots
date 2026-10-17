# scikitplot/agent/_loop.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The agent loop: Perceive, Reason/Plan, Act, Observe, Reflect, within limits.

Each iteration:

1. the run's limits are checked; one reached ends the run (``stopped``);
2. **Perceive, Reason/Plan.** The planner reads a copy of the state and the
   tool descriptions, and returns a :class:`~scikitplot.agent.Decision`;
3. **Act.** An ``act`` decision passes the gate chain; a denied action
   becomes the step's outcome, an approved one runs its tool;
4. **Observe.** The result, the error or the denial is appended to the state
   as a :class:`~scikitplot.agent.Step`, audited, and the state is saved;
5. **Reflect.** The next planner call reads that step. A ``finish`` decision
   ends the run when the done criterion accepts the answer.

Notes
-----
**Developer.** The planner proposes; only the runtime disposes. A planner
cannot run a tool, change the state or raise a limit: it returns a value the
runtime checks. Every planner call is counted against ``max_steps`` before it
is made, so a run ends after at most ``max_steps`` decisions whatever the
planner does (invariant I1).
"""

from __future__ import annotations

import secrets
import time
from typing import Any, Callable, Iterable

from ._audit import AuditSink, AuditTrail, MemoryAuditSink
from ._canonical import digest, plain
from ._errors import AgentError, ErrorCode, StateError
from ._policy import (
    GateKind,
    GateResult,
    Limits,
    Policy,
    approval_reason,
    check_action,
)
from ._state import (
    Approval,
    ApprovalRequest,
    Decision,
    RunState,
    RunStatus,
    Step,
    StepOutcome,
    StopReason,
    approval_digest,
    call_key,
)
from ._store import StateStore
from ._tools import Tool, Toolset

__all__ = ["Agent"]

#: Characters of an exception message kept in a step's error.
_MESSAGE_CHARS = 1000
#: Times one idempotent call is started across crashes before it is given up.
MAX_ATTEMPTS = 3

Planner = Callable[[RunState, list], Decision]


def _short(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= _MESSAGE_CHARS else text[: _MESSAGE_CHARS - 3] + "..."


class Agent:
    """
    Run a goal with a planner and tools, within a policy and limits.

    Parameters
    ----------
    tools : Toolset or iterable of Tool
        What the agent may call.
    planner : callable
        ``planner(state, tools) -> Decision``: ``state`` is a copy of the
        :class:`~scikitplot.agent.RunState` and ``tools`` the toolset's
        :meth:`~scikitplot.agent.Toolset.describe`. A model adapter or a rule.
    policy : Policy or None, default=None
        ``None`` is ``Policy()``: ``read`` tools without scopes only.
    limits : Limits or None, default=None
        ``None`` is ``Limits()``: at most 16 decisions.
    approver : callable or None, default=None
        ``approver(request) -> Approval or None``, asked when an action needs
        approval. ``None`` (or an answer of ``None``) pauses the run with
        status ``awaiting_approval``; :meth:`resume` it with an
        :class:`~scikitplot.agent.Approval`.
    done : callable or None, default=None
        ``done(answer, state) -> bool or str``: the task's completion
        criterion. ``True`` accepts the planner's answer; ``False`` or a
        reason rejects it, and the run goes on. ``None`` accepts any answer.
    audit : AuditSink or None, default=None
        Where audit records go; ``None`` is a new
        :class:`~scikitplot.agent.MemoryAuditSink`, kept as :attr:`audit`.
    store : StateStore or None, default=None
        Where the state is saved after each step; ``None`` keeps it only in
        the returned state.
    clock : callable or None, default=None
        Seconds, for limits and step times; ``None`` is
        :func:`time.monotonic`.
    now : callable or None, default=None
        Wall-clock time written in audit records; ``None`` is :func:`time.time`.

    Raises
    ------
    TypeError
        If an argument has the wrong type.

    Examples
    --------
    >>> from scikitplot.agent import Agent, Decision, Tool
    >>> def planner(state, tools):
    ...     if not state.steps:
    ...         return Decision.act("ping")
    ...     return Decision.finish(state.steps[-1].result)
    >>> agent = Agent([Tool("ping", lambda: "pong")], planner)
    >>> state = agent.run("check the line")
    >>> state.status.value, state.answer, len(agent.audit.records) > 0
    ('completed', 'pong', True)
    """

    def __init__(
        self,
        tools: Toolset | Iterable[Tool],
        planner: Planner,
        *,
        policy: Policy | None = None,
        limits: Limits | None = None,
        approver: Callable[[ApprovalRequest], Approval | None] | None = None,
        done: Callable[[Any, RunState], Any] | None = None,
        audit: AuditSink | None = None,
        store: StateStore | None = None,
        clock: Callable[[], float] | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.tools = tools if isinstance(tools, Toolset) else Toolset(tools)
        if not callable(planner):
            raise TypeError("Agent.planner must be callable.")
        self.planner = planner
        self.policy = Policy() if policy is None else policy
        if not isinstance(self.policy, Policy):
            raise TypeError("Agent.policy must be a Policy or None.")
        self.limits = Limits() if limits is None else limits
        if not isinstance(self.limits, Limits):
            raise TypeError("Agent.limits must be a Limits or None.")
        for name, value in (("approver", approver), ("done", done)):
            if value is not None and not callable(value):
                raise TypeError(f"Agent.{name} must be callable or None.")
        self.approver = approver
        self.done = done
        self.audit = MemoryAuditSink() if audit is None else audit
        if not isinstance(self.audit, AuditSink):
            raise TypeError("Agent.audit must be an AuditSink or None.")
        if store is not None and not isinstance(store, StateStore):
            raise TypeError("Agent.store must be a StateStore or None.")
        self.store = store
        self.clock = time.monotonic if clock is None else clock
        self.now = time.time if now is None else now

    # ------------------------------------------------------------------ public

    def run(self, goal: str, *, run_id: str | None = None) -> RunState:
        """
        Start a run toward ``goal`` and drive it until it ends or pauses.

        Parameters
        ----------
        goal : str
            What the run is for; the planner reads it from the state.
        run_id : str or None, default=None
            64 hexadecimal digits; ``None`` draws a random one.

        Returns
        -------
        RunState
            The last state: ``completed``, ``stopped``, ``failed``, or
            ``awaiting_approval``.

        Raises
        ------
        ValueError
            If ``run_id`` names a run the store or the audit sink already
            holds: a run id is used once, so an approval or a trail of one run
            can never be taken for another's.
        """
        state = RunState(
            run_id=secrets.token_hex(32) if run_id is None else run_id, goal=goal
        )
        if self.store is not None and self.store.has(state.run_id):
            raise ValueError(f"run id {state.run_id} is already in the store.")
        if self.audit.head(state.run_id) is not None:
            raise ValueError(f"run id {state.run_id} is already in the audit trail.")
        session = _Session(self, state, sink_head=None)
        session.emit(
            "run_started",
            {
                # A random nonce makes this run's history, and so every
                # approval digest bound to it, unique even when its id, goal
                # and clock repeat another run's.
                "nonce": secrets.token_hex(32),
                "goal": goal,
                "policy": self.policy.to_dict(),
                "limits": self.limits.to_dict(),
                "tools": self.tools.describe(),
            },
        )
        return session.drive()

    def resume(
        self, state: RunState | str, *, approvals: Iterable[Approval] = ()
    ) -> RunState:
        """
        Continue a paused or interrupted run.

        Parameters
        ----------
        state : RunState or str
            The saved state, or its run id (read from :attr:`store`).
        approvals : iterable of Approval, default=()
            Answers to the pending :class:`~scikitplot.agent.ApprovalRequest`.
            The one whose digest names the request is used; at resume the
            action's digest is recomputed from the tools as they are now and
            every gate is run again, so an approval of an action that changed
            authorises nothing (``approval_stale``).

        Returns
        -------
        RunState
            The last state. A run that has ended is returned unchanged; a
            paused run with no matching approval stays paused.

        Raises
        ------
        StateError
            If a store is set and holds a newer state of the run than the one
            given: resuming an old copy again would repeat what the newer
            state already records (an approved call, for one).
        ValueError
            If two approvals name the request and disagree, or a run id is
            given without a store.

        Notes
        -----
        **User.** With a store, give the run id: the latest state is read
        from it, and a second resume of a finished run is a no-op. Without a
        store, the caller holds the states, and resuming the same copy twice
        is the caller's own replay. Every resume writes audit records, so the
        state it returns replaces the one given, even when a paused run stays
        paused: resume that one next.
        """
        if isinstance(state, str):
            if self.store is None:
                raise ValueError("Agent.resume(run_id) needs a store.")
            state = self.store.load(state)
        elif isinstance(state, RunState):
            state = state.copy()
            if self.store is not None and self.store.has(state.run_id):
                stored = self.store.load(state.run_id)
                if stored.audit_head != state.audit_head:
                    raise StateError(
                        f"the store holds a newer state of run {state.run_id}; "
                        "resume by its run id"
                    )
        else:
            raise TypeError("Agent.resume needs a RunState or a run id.")
        approvals = tuple(approvals)
        if not all(isinstance(a, Approval) for a in approvals):
            raise TypeError("Agent.resume: every approval must be an Approval.")
        if state.done:
            return state
        session = _Session(self, state, sink_head=self.audit.head(state.run_id))
        session.emit(
            "run_resumed",
            {
                "status": state.status.value,
                "approvals": [a.digest for a in approvals],
                "in_flight": state.in_flight is not None,
            },
        )
        if state.in_flight is not None and session.recover_in_flight() == "paused":
            return session.save()
        if state.status is RunStatus.AWAITING_APPROVAL and not session.answer_pending(
            approvals
        ):
            return session.save()
        return session.drive()


def _behind(sink: AuditSink, state: RunState, sink_head: dict[str, Any] | None) -> bool:
    """Whether the trail is ahead of ``state``; raise if the two disagree.

    ``False`` when the sink holds nothing of the run (a new sink: the chain
    goes on from the state) or holds exactly the state's last record.
    ``True`` when the state's last record is an earlier record of the trail.

    Raises
    ------
    StateError
        If the state is newer than the trail, or not part of its history
        (records were lost, or the trail is another run's); or if the state
        awaits approval while the trail goes on past it: that state has been
        resumed already, and resuming it again would use its approval twice.
    """
    head = state.audit_head
    if sink_head is None or sink_head == head:
        return False
    if (
        head is not None
        and sink_head["seq"] > head["seq"]
        and sink.digest_at(state.run_id, head["seq"]) == head["digest"]
    ):
        if state.status is RunStatus.AWAITING_APPROVAL:
            raise StateError(
                f"run {state.run_id}: the audit trail goes on past this paused "
                "state, so it was resumed already; resume its latest state"
            )
        return True
    raise StateError(
        f"run {state.run_id}: the audit trail (last record {sink_head}) does not "
        f"hold this state's last record ({head}); records were lost, or the "
        "trail belongs to another history"
    )


class _Session:
    """One stretch of a run in this process: its trail, clock and steps."""

    def __init__(
        self, agent: Agent, state: RunState, *, sink_head: dict[str, Any] | None
    ) -> None:
        self.agent = agent
        self.state = state
        head = state.audit_head
        behind = _behind(agent.audit, state, sink_head)
        self.trail = AuditTrail(
            agent.audit,
            state.run_id,
            sink_head if behind else head,
            agent.now,
        )
        if behind:
            # Records were written after the state was last saved (a save
            # failed, or the process stopped between the two): the chain goes
            # on from the trail, and says where the state was.
            self.emit("state_behind_audit", {"state_head": head})
        self.last = agent.clock()
        self.described = agent.tools.describe()

    # ------------------------------------------------------------ bookkeeping

    def tick(self) -> None:
        """Add the time since the last tick to ``spent.seconds``.

        Each forward move between two readings counts and a backward one
        counts nothing, so a clock that steps back neither returns time to
        the run nor hides the time spent after it.
        """
        now = self.agent.clock()
        if now > self.last:
            self.state.spent.seconds += now - self.last
        self.last = now

    def skip_time(self) -> None:
        """Do not count the time since the last tick (an approver's wait)."""
        self.last = self.agent.clock()

    def emit(self, kind: str, data: dict[str, Any]) -> None:
        self.trail.emit(kind, data)
        self.state.audit_head = self.trail.head

    def save(self) -> RunState:
        self.tick()
        if self.agent.store is not None:
            self.agent.store.save(self.state)
        return self.state

    def end(
        self, status: RunStatus, reason: StopReason, error: AgentError | None = None
    ) -> RunState:
        state = self.state
        state.status, state.stop_reason, state.error = status, reason, error
        self.tick()
        self.emit(
            "run_ended",
            {
                "status": status.value,
                "stop_reason": reason.value,
                "error": None if error is None else error.to_dict(),
                "spent": state.spent.to_dict(),
            },
        )
        return self.save()

    def add_step(self, step: Step, *, save: bool = True) -> None:
        self.state.steps.append(step)
        self.emit(
            "step",
            {
                "index": step.index,
                "kind": step.decision.kind,
                "outcome": step.outcome.value,
                "gates": plain(list(step.gates)),
                "key": step.key,
                "result_digest": (
                    None
                    if step.outcome is not StepOutcome.EXECUTED
                    else digest(step.result)
                ),
                "error": None if step.error is None else step.error.to_dict(),
            },
        )
        if save:
            self.save()

    # ------------------------------------------------------------------- loop

    def limit_reached(self) -> tuple[StopReason, AgentError] | None:
        limits, spent = self.agent.limits, self.state.spent
        checks = (
            (StopReason.MAX_STEPS, "max_steps", limits.max_steps, spent.steps),
            (StopReason.TIMEOUT, "max_seconds", limits.max_seconds, spent.seconds),
            (StopReason.MAX_TOKENS, "max_tokens", limits.max_tokens, spent.tokens),
            (StopReason.MAX_COST, "max_cost", limits.max_cost, spent.cost),
        )
        for reason, name, limit, value in checks:
            if limit is not None and value >= limit:
                return reason, AgentError(
                    ErrorCode.LIMIT_REACHED,
                    f"the run reached {name}={limit!r} ({value!r} spent)",
                    {"limit": name, "value": limit, "spent": value},
                )
        return None

    def decide(self, index: int) -> Decision | RunState:
        """Ask the planner; return its decision, or the ended state."""
        agent, state = self.agent, self.state
        try:
            produced = agent.planner(state.copy(), plain(self.described))
        except Exception as exc:  # noqa: BLE001 - recorded as the run's failure
            return self.end(
                RunStatus.FAILED,
                StopReason.PLANNER_FAILED,
                AgentError(
                    ErrorCode.PLANNER_FAILED,
                    f"the planner raised {_short(exc)}",
                    {"step": index, "exception_type": type(exc).__name__},
                ),
            )
        problem = None
        if not isinstance(produced, Decision):
            problem = (
                f"the planner must return a Decision, got {type(produced).__name__}"
            )
        else:
            try:
                # A plain copy: what is gated, approved, recorded and run is
                # this one value, whatever the planner's object does later.
                return Decision.from_dict(plain(produced.to_dict()))
            except Exception as exc:  # noqa: BLE001 - a malformed decision
                problem = f"the planner's Decision is malformed: {_short(exc)}"
        return self.end(
            RunStatus.FAILED,
            StopReason.INVALID_DECISION,
            AgentError(ErrorCode.INVALID_DECISION, problem, {"step": index}),
        )

    def drive(self) -> RunState:
        agent, state = self.agent, self.state
        while True:
            self.tick()
            reached = self.limit_reached()
            if reached is not None:
                return self.end(RunStatus.STOPPED, *reached)
            index = len(state.steps)
            state.spent.steps += 1
            decision = self.decide(index)
            if isinstance(decision, RunState):
                return decision
            if decision.usage is None and agent.limits.max_tokens is not None:
                return self.end(
                    RunStatus.FAILED,
                    StopReason.USAGE_NOT_REPORTED,
                    AgentError(
                        ErrorCode.USAGE_NOT_REPORTED,
                        "max_tokens is set, and the planner's decision reports no "
                        "usage: the limit could not be enforced",
                        {"step": index},
                    ),
                )
            if decision.usage is not None:
                state.spent.tokens += decision.usage.tokens
                state.spent.cost += decision.usage.cost
            self.emit("decision", {"step": index, "decision": decision.to_dict()})
            if decision.kind == "finish":
                ended = self.finish(index, decision)
                if ended is not None:
                    return ended
                continue
            if self.act(index, decision) == "paused":
                return self.save()

    def finish(self, index: int, decision: Decision) -> RunState | None:
        done = self.agent.done
        verdict: Any = True
        if done is not None:
            try:
                verdict = done(decision.answer, self.state.copy())
            except Exception as exc:  # noqa: BLE001 - recorded as the run's failure
                return self.end(
                    RunStatus.FAILED,
                    StopReason.DONE_CHECK_FAILED,
                    AgentError(
                        ErrorCode.DONE_CHECK_FAILED,
                        f"the done criterion raised {_short(exc)}",
                        {"step": index},
                    ),
                )
            if (
                verdict is not True
                and verdict is not False
                and not isinstance(verdict, str)
            ):
                return self.end(
                    RunStatus.FAILED,
                    StopReason.DONE_CHECK_FAILED,
                    AgentError(
                        ErrorCode.DONE_CHECK_FAILED,
                        "the done criterion must return True, False or a reason, "
                        f"got {type(verdict).__name__}",
                        {"step": index},
                    ),
                )
        if verdict is True:
            # Saved once, by ``end``: a state with the accepted answer and a
            # running status is never written.
            self.add_step(Step(index, decision, StepOutcome.FINISHED), save=False)
            self.state.answer = decision.answer
            return self.end(RunStatus.COMPLETED, StopReason.DONE)
        reason = verdict if isinstance(verdict, str) and verdict else None
        self.add_step(
            Step(
                index,
                decision,
                StepOutcome.REJECTED,
                error=AgentError(
                    ErrorCode.ANSWER_REJECTED,
                    reason or "the done criterion refused the answer",
                    {"step": index},
                ),
            )
        )
        return None

    def act(
        self,
        index: int,
        decision: Decision,
        *,
        approval: Approval | None = None,
        request: ApprovalRequest | None = None,
        attempt: int = 1,
    ) -> str:
        """Gate and run one action: ``"done"`` or ``"paused"``.

        ``request`` and ``approval`` are given together at resume: the
        approval digest is then recomputed with the request's audit head and
        the tools as they are now.
        """
        agent, state = self.agent, self.state
        action = decision.action
        assert action is not None  # noqa: S101 - an act decision
        self.tick()
        tool, gates = check_action(
            action, agent.tools, agent.policy, agent.limits, state.spent
        )
        if not gates[-1].passed:
            self.deny(index, decision, gates)
            return "done"
        assert tool is not None  # noqa: S101 - validation passed
        reason = approval_reason(tool, agent.policy)
        if reason is not None:
            head = self.trail.prev if request is None else request.head
            bound = approval_digest(
                state.run_id,
                index,
                action,
                effect=tool.effect,
                scopes=tool.scopes,
                head=head,
            )
            if approval is None:
                request = ApprovalRequest(
                    run_id=state.run_id,
                    step=index,
                    decision=decision,
                    effect=tool.effect.value,
                    scopes=tuple(sorted(tool.scopes)),
                    reason=reason,
                    head=head,
                    digest=bound,
                    gates=tuple(g.to_dict() for g in gates),
                    attempt=attempt,
                )
                approval = self.ask(request)
                if approval is None:
                    return self.pause(request)
            verdict = self.check_approval(approval, bound)
            gates.append(verdict)
            if not verdict.passed:
                self.deny(index, decision, gates)
                return "done"
        self.execute(index, decision, tool, gates, attempt=attempt)
        return "done"

    def ask(self, request: ApprovalRequest) -> Approval | None:
        approver = self.agent.approver
        if approver is None:
            return None
        self.tick()
        try:
            answer = approver(request)
        except Exception as exc:  # noqa: BLE001 - nothing runs unapproved: pause
            self.skip_time()
            self.emit("approval_error", {"step": request.step, "error": _short(exc)})
            return None
        self.skip_time()  # waiting for an approval is not running time
        if answer is None:
            return None
        if not isinstance(answer, Approval):
            self.emit(
                "approval_error",
                {
                    "step": request.step,
                    "error": (
                        f"the approver returned {type(answer).__name__}, "
                        "not an Approval"
                    ),
                },
            )
            return None
        return answer

    def check_approval(self, approval: Approval, bound: str) -> GateResult:
        self.emit(
            "approval",
            {
                "digest": approval.digest,
                "granted": approval.granted,
                "approver": approval.approver,
                "note": approval.note,
                "matches": approval.digest == bound,
            },
        )
        if approval.digest != bound:
            return GateResult(
                GateKind.APPROVAL,
                False,
                AgentError(
                    ErrorCode.APPROVAL_STALE,
                    "the approval names another action: the tool, its arguments, "
                    "effect or scopes changed since it was approved",
                    {"gate": "approval", "approved": approval.digest, "action": bound},
                ),
            )
        if not approval.granted:
            return GateResult(
                GateKind.APPROVAL,
                False,
                AgentError(
                    ErrorCode.APPROVAL_DENIED,
                    f"{approval.approver} refused the action"
                    + (f": {approval.note}" if approval.note else ""),
                    {"gate": "approval", "approver": approval.approver},
                ),
            )
        return GateResult(GateKind.APPROVAL, True)

    def pause(self, request: ApprovalRequest) -> str:
        state = self.state
        state.status = RunStatus.AWAITING_APPROVAL
        state.stop_reason = StopReason.APPROVAL_REQUIRED
        state.pending = request
        self.emit(
            "approval_requested",
            {
                "step": request.step,
                "digest": request.digest,
                "reason": request.reason,
                "action": request.action.to_dict(),
                "effect": request.effect,
                "scopes": list(request.scopes),
            },
        )
        return "paused"

    def deny(self, index: int, decision: Decision, gates: list[GateResult]) -> None:
        self.add_step(
            Step(
                index,
                decision,
                StepOutcome.DENIED,
                gates=tuple(g.to_dict() for g in gates),
                error=gates[-1].error,
            )
        )

    def execute(
        self,
        index: int,
        decision: Decision,
        tool: Tool,
        gates: list[GateResult],
        *,
        attempt: int,
    ) -> None:
        agent, state = self.agent, self.state
        action = decision.action
        assert action is not None  # noqa: S101 - an act decision
        key = call_key(state.run_id, index, action)
        state.in_flight = {
            "step": index,
            "key": key,
            "idempotent": tool.idempotent,
            "attempt": attempt,
            "decision": decision.to_dict(),
            "gates": [g.to_dict() for g in gates],
        }
        # Charged before the call: a call that starts is spent, whether it
        # returns, raises, or the process stops during it (invariant I6).
        state.spent.tool_calls += 1
        state.spent.cost += tool.cost
        self.emit(
            "tool_call",
            {"step": index, "key": key, "attempt": attempt, "action": action.to_dict()},
        )
        self.save()  # write-ahead: a crash from here on is seen at resume
        started = agent.clock()
        error: AgentError | None = None
        result: Any = None
        try:
            # A fresh copy: the tool cannot change the recorded arguments.
            raw = tool.function(**plain(action.arguments))
        except Exception as exc:  # noqa: BLE001 - the tool's failure is an outcome
            error = AgentError(
                ErrorCode.TOOL_FAILED,
                f"{tool.name!r} raised {_short(exc)}",
                {"tool": tool.name, "exception_type": type(exc).__name__},
            )
        else:
            try:
                result = plain(raw, what=f"the result of {tool.name!r}")
            except (TypeError, ValueError) as exc:
                error = AgentError(
                    ErrorCode.TOOL_RESULT_NOT_JSON,
                    str(exc),
                    {"tool": tool.name},
                )
        seconds = max(0.0, agent.clock() - started)
        state.in_flight = None
        self.add_step(
            Step(
                index,
                decision,
                StepOutcome.EXECUTED if error is None else StepOutcome.FAILED,
                gates=tuple(g.to_dict() for g in gates),
                result=result if error is None else None,
                error=error,
                key=key,
                seconds=seconds,
            )
        )

    # ----------------------------------------------------------------- resume

    def recover_in_flight(self) -> str:
        """Settle a call in flight when the run stopped: ``"done"`` or ``"paused"``.

        The call was charged when it started. It runs again only if the tool
        is idempotent *as registered now* (the saved flag alone is not
        trusted) and it has run fewer than :data:`MAX_ATTEMPTS` times; the
        new attempt passes every gate again, the Budget gate included.
        """
        state = self.state
        flight = state.in_flight
        assert flight is not None  # noqa: S101 - checked by the caller
        decision = Decision.from_dict(flight["decision"])
        assert decision.action is not None  # noqa: S101 - checked by the state
        index, attempt = flight["step"], flight["attempt"]
        state.in_flight = None
        agent = self.agent
        tool = agent.tools.get(decision.action.tool)
        if tool is None:
            why = "its tool is no longer registered"
        elif flight["idempotent"] is not True or not tool.idempotent:
            why = "it is not idempotent"
        elif attempt >= MAX_ATTEMPTS:
            why = f"it was started {attempt} times, the most allowed"
        else:
            self.tick()
            _, gates = check_action(
                decision.action, agent.tools, agent.policy, agent.limits, state.spent
            )
            if gates[-1].passed:
                self.emit(
                    "retry_in_flight",
                    {"step": index, "key": flight["key"], "attempt": attempt + 1},
                )
                return self.act(index, decision, attempt=attempt + 1)
            error = gates[-1].error
            why = f"a retry was denied: {error.message if error else 'a gate refused'}"
        self.add_step(
            Step(
                index,
                decision,
                StepOutcome.INTERRUPTED,
                gates=tuple(flight["gates"]),
                error=AgentError(
                    ErrorCode.TOOL_INTERRUPTED,
                    "the call was in flight when the run stopped and is not "
                    f"repeated, because {why}; whether it took effect is unknown",
                    {"key": flight["key"], "attempt": attempt},
                ),
                key=flight["key"],
            )
        )
        return "done"

    def answer_pending(self, approvals: tuple[Approval, ...]) -> bool:
        """Apply the approval that names the pending request; ``False`` if none."""
        state = self.state
        request = state.pending
        assert request is not None  # noqa: S101 - awaiting approval
        matching = [a for a in approvals if a.digest == request.digest]
        if len({(a.granted, a.approver, a.note) for a in matching}) > 1:
            raise ValueError("two approvals name the pending action and disagree.")
        if not matching:
            if approvals:
                self.emit(
                    "approval_mismatch",
                    {
                        "pending": request.digest,
                        "given": [a.digest for a in approvals],
                    },
                )
            return False
        state.status, state.stop_reason, state.pending = RunStatus.RUNNING, None, None
        self.act(
            request.step,
            request.decision,
            approval=matching[0],
            request=request,
            attempt=request.attempt,
        )
        return True
