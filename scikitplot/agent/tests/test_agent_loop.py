# scikitplot/agent/tests/test_agent_loop.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The agent loop: stopping conditions, denials, approvals, resume, audit (R5-1).
"""

from __future__ import annotations

import pytest

from .. import (
    Agent,
    Approval,
    Decision,
    ErrorCode,
    JsonFileStore,
    Limits,
    MemoryAuditSink,
    MemoryStore,
    Policy,
    RunState,
    RunStatus,
    StateError,
    StepOutcome,
    StopReason,
    Tool,
    Usage,
    verify_audit,
)

RUN = "c" * 64
QUERY = {
    "type": "object",
    "properties": {"query": {"type": "string", "minLength": 1}},
    "required": ["query"],
    "additionalProperties": False,
}
PATH = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}
#: Steps, calls and seconds the tests set and assert.
STEPS = 3
STEP = 5.0
CALLS = 2


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Script:
    """A planner that returns the next decision of a list (or calls it)."""

    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.seen = []

    def __call__(self, state, tools):
        self.seen.append((state, tools))
        item = (
            self.decisions[len(self.seen) - 1]
            if len(self.seen) <= len(self.decisions)
            else self.decisions[-1]
        )
        return item(state) if callable(item) else item


def search_tool(log=None, **kwargs):
    def search(query):
        if log is not None:
            log.append(query)
        return {"hits": [f"doc:{query}"]}

    return Tool("search", search, parameters=QUERY, **kwargs)


def save_tool(log, **kwargs):
    def save(path):
        log.append(path)
        return {"saved": path}

    return Tool("save", save, parameters=PATH, effect="write", **kwargs)


def _kinds(agent):
    return [r["kind"] for r in agent.audit.records]


# ------------------------------------------------------------- the happy path


class TestRun:
    def test_act_observe_finish(self):
        planner = Script(
            Decision.act("search", {"query": "spans"}),
            lambda s: Decision.finish(s.steps[-1].result["hits"][0]),
        )
        agent = Agent([search_tool()], planner)
        state = agent.run("find spans", run_id=RUN)
        assert state.status is RunStatus.COMPLETED
        assert state.stop_reason is StopReason.DONE
        assert state.answer == "doc:spans"
        assert [s.outcome for s in state.steps] == [
            StepOutcome.EXECUTED,
            StepOutcome.FINISHED,
        ]
        assert state.spent.steps == 2  # noqa: PLR2004
        assert state.spent.tool_calls == 1
        assert verify_audit(agent.audit.records) == []
        assert _kinds(agent) == [
            "run_started",
            "decision",
            "tool_call",
            "step",
            "decision",
            "step",
            "run_ended",
        ]
        assert state.audit_head["seq"] == len(agent.audit.records) - 1

    def test_the_planner_reads_a_copy_and_the_tools(self):
        def meddle(state):
            state.spent.steps = -100  # a copy: the run does not see it
            state.steps.clear()
            return Decision.finish("ok")

        planner = Script(meddle)
        agent = Agent([search_tool()], planner)
        state = agent.run("g")
        assert state.spent.steps == 1
        assert planner.seen[0][1][0]["name"] == "search"

    def test_a_tool_failure_is_an_observation(self):
        def broken(query):
            raise ConnectionError("index down")

        planner = Script(
            Decision.act("search", {"query": "x"}),
            lambda s: Decision.finish(s.observations()[-1]["error"]["code"]),
        )
        state = Agent([Tool("search", broken, parameters=QUERY)], planner).run("g")
        assert state.steps[0].outcome is StepOutcome.FAILED
        assert "ConnectionError: index down" in state.steps[0].error.message
        assert state.answer == "tool_failed"

    @pytest.mark.parametrize("result", [{1: "x"}, float("nan"), object()])
    def test_a_result_that_is_not_json_fails_the_step(self, result):
        planner = Script(Decision.act("bad"), Decision.finish(None))
        state = Agent([Tool("bad", lambda: result)], planner).run("g")
        assert state.steps[0].outcome is StepOutcome.FAILED
        assert state.steps[0].error.code is ErrorCode.TOOL_RESULT_NOT_JSON
        RunState.from_dict(state.to_dict())  # the state stays strict JSON

    def test_the_done_criterion_rejects_then_accepts(self):
        planner = Script(Decision.finish("short"), Decision.finish("a long answer"))
        agent = Agent(
            [],
            planner,
            done=lambda answer, state: len(answer) > 5 or "too short",  # noqa: PLR2004
        )
        state = agent.run("g")
        assert [s.outcome for s in state.steps] == [
            StepOutcome.REJECTED,
            StepOutcome.FINISHED,
        ]
        assert state.steps[0].error.message == "too short"
        assert state.answer == "a long answer"

    @pytest.mark.parametrize(
        ("done", "reason"),
        [
            (lambda a, s: 1 / 0, StopReason.DONE_CHECK_FAILED),
            (lambda a, s: 1, StopReason.DONE_CHECK_FAILED),
        ],
    )
    def test_a_broken_done_criterion_fails_the_run(self, done, reason):
        state = Agent([], Script(Decision.finish("x")), done=done).run("g")
        assert state.status is RunStatus.FAILED
        assert state.stop_reason is reason


# ------------------------------------------------------------- stopping


class TestStopping:
    def test_max_steps_ends_every_run(self):
        planner = Script(Decision.act("search", {"query": "x"}))  # forever
        state = Agent([search_tool()], planner, limits=Limits(max_steps=STEPS)).run("g")
        assert state.status is RunStatus.STOPPED
        assert state.stop_reason is StopReason.MAX_STEPS
        assert len(state.steps) == STEPS == state.spent.steps
        assert state.error.code is ErrorCode.LIMIT_REACHED

    def test_max_tool_calls_denies_then_the_planner_finishes(self):
        def planner(state, tools):
            if state.steps and state.steps[-1].outcome is StepOutcome.DENIED:
                return Decision.finish("enough")
            return Decision.act("search", {"query": "x"})

        state = Agent([search_tool()], planner, limits=Limits(max_tool_calls=2)).run(
            "g"
        )
        outcomes = [s.outcome.value for s in state.steps]
        assert outcomes == ["executed", "executed", "denied", "finished"]
        assert state.steps[2].error.code is ErrorCode.BUDGET_EXCEEDED

    def test_timeout_is_checked_before_each_decision(self):
        clock = Clock()

        def slow(query):
            clock.now += STEP
            return []

        planner = Script(Decision.act("slow", {"query": "x"}))
        agent = Agent(
            [Tool("slow", slow, parameters=QUERY)],
            planner,
            limits=Limits(max_seconds=STEP * 2),
            clock=clock,
        )
        state = agent.run("g")
        assert state.stop_reason is StopReason.TIMEOUT
        # Calls start at 0 s and 5 s; at 10 s nothing starts.
        assert state.spent.tool_calls == CALLS
        assert state.spent.seconds == STEP * 2

    def test_tokens_need_reported_usage(self):
        planner = Script(Decision.act("search", {"query": "x"}))
        state = Agent([search_tool()], planner, limits=Limits(max_tokens=100)).run("g")
        assert state.status is RunStatus.FAILED
        assert state.stop_reason is StopReason.USAGE_NOT_REPORTED

    def test_max_tokens_and_max_cost(self):
        usage = Usage(tokens=40, cost=0.5)
        planner = Script(Decision.act("search", {"query": "x"}, usage=usage))
        state = Agent([search_tool()], planner, limits=Limits(max_tokens=100)).run("g")
        assert state.stop_reason is StopReason.MAX_TOKENS
        # Three decisions of 40 tokens: the third's were spent when made.
        assert state.spent.tokens == 3 * 40  # noqa: PLR2004
        state = Agent(
            [search_tool(cost=1.0)], planner, limits=Limits(max_cost=3.0)
        ).run("g")
        assert state.stop_reason is StopReason.MAX_COST
        assert state.spent.cost == 3.0  # noqa: PLR2004

    @pytest.mark.parametrize(
        ("planner", "reason"),
        [
            (lambda s, t: 1 / 0, StopReason.PLANNER_FAILED),
            (lambda s, t: {"tool": "search"}, StopReason.INVALID_DECISION),
        ],
    )
    def test_a_broken_planner_fails_the_run(self, planner, reason):
        agent = Agent([search_tool()], planner)
        state = agent.run("g")
        assert state.status is RunStatus.FAILED
        assert state.stop_reason is reason
        assert _kinds(agent)[-1] == "run_ended"


# ------------------------------------------------------------- denials


class TestDenials:
    @pytest.mark.parametrize(
        ("decision", "tool", "policy", "code"),
        [
            (Decision.act("missing"), {}, None, "unknown_tool"),
            (Decision.act("search", {"query": 3}), {}, None, "invalid_arguments"),
            (
                Decision.act("search", {"query": "x"}),
                {"scopes": {"corpus:read"}},
                None,
                "scope_denied",
            ),
            (
                Decision.act("search", {"query": "x"}),
                {},
                Policy(deny_tools={"search"}),
                "permission_denied",
            ),
            (
                Decision.act("search", {"query": "x"}),
                {"effect": "external"},
                None,
                "permission_denied",
            ),
        ],
    )
    def test_a_denied_action_is_an_audited_outcome(self, decision, tool, policy, code):
        log = []
        planner = Script(decision, Decision.finish("gave up"))
        agent = Agent([search_tool(log, **tool)], planner, policy=policy)
        state = agent.run("g")
        assert log == []  # the tool never ran
        assert state.steps[0].outcome is StepOutcome.DENIED
        assert state.steps[0].error.code.value == code
        assert state.status is RunStatus.COMPLETED
        step_records = [r for r in agent.audit.records if r["kind"] == "step"]
        assert step_records[0]["data"]["outcome"] == "denied"
        assert step_records[0]["data"]["error"]["code"] == code
        assert "tool_call" not in _kinds(agent)


# ------------------------------------------------------------- approvals


def _writer_agent(log, **kwargs):
    planner = Script(
        Decision.act("save", {"path": "report.md"}),
        lambda s: Decision.finish(s.steps[-1].outcome.value),
    )
    return Agent(
        [save_tool(log)],
        planner,
        policy=Policy(allow_effects={"read", "write"}),
        **kwargs,
    )


class TestApproval:
    def test_a_write_pauses_then_resumes_with_approval(self):
        log, store = [], MemoryStore()
        agent = _writer_agent(log, store=store)
        paused = agent.run("write the report", run_id=RUN)
        assert paused.status is RunStatus.AWAITING_APPROVAL
        assert paused.stop_reason is StopReason.APPROVAL_REQUIRED
        assert log == []
        assert paused.steps == []
        request = paused.pending
        assert request.action.arguments == {"path": "report.md"}
        assert store.load(RUN).pending.digest == request.digest
        done = agent.resume(RUN, approvals=[Approval(request.digest, True, "allen")])
        assert log == ["report.md"]
        assert done.status is RunStatus.COMPLETED
        assert done.answer == "executed"
        gates = [g["gate"] for g in done.steps[0].gates]
        assert gates == ["validation", "permissions", "scope", "budget", "approval"]
        assert verify_audit(agent.audit.records) == []
        kinds = _kinds(agent)
        assert kinds.index("approval_requested") < kinds.index("run_resumed")
        assert kinds.index("approval") < kinds.index("tool_call")

    def test_a_refusal_is_an_outcome(self):
        log = []
        agent = _writer_agent(log)
        paused = agent.run("g")
        refused = Approval(paused.pending.digest, False, "allen", note="not today")
        done = agent.resume(paused, approvals=[refused])
        assert log == []
        assert done.steps[0].error.code is ErrorCode.APPROVAL_DENIED
        assert "not today" in done.steps[0].error.message
        assert done.answer == "denied"

    def test_an_approval_of_another_action_leaves_the_run_paused(self):
        agent = _writer_agent([])
        paused = agent.run("g")
        again = agent.resume(paused, approvals=[Approval("f" * 64, True, "allen")])
        assert again.status is RunStatus.AWAITING_APPROVAL
        assert _kinds(agent)[-1] == "approval_mismatch"

    def test_a_stale_approval_authorises_nothing(self):
        # Invariant I4: the tool's scopes changed after the request was made.
        log = []
        paused = _writer_agent(log).run("g")
        approval = Approval(paused.pending.digest, True, "allen")
        changed = Agent(
            [save_tool(log, scopes={"files:write"})],
            Script(lambda s: Decision.finish(s.steps[-1].error.code.value)),
            policy=Policy(
                allow_effects={"read", "write"}, granted_scopes={"files:write"}
            ),
        )
        done = changed.resume(paused, approvals=[approval])
        assert log == []
        assert done.steps[0].error.code is ErrorCode.APPROVAL_STALE
        assert done.answer == "approval_stale"

    def test_gates_run_again_at_resume(self):
        log = []
        paused = _writer_agent(log).run("g")
        approval = Approval(paused.pending.digest, True, "allen")
        stricter = Agent(
            [save_tool(log)],
            Script(Decision.finish("x")),
            policy=Policy(allow_effects={"read", "write"}, deny_tools={"save"}),
        )
        done = stricter.resume(paused, approvals=[approval])
        assert log == []
        assert done.steps[0].error.code is ErrorCode.PERMISSION_DENIED

    def test_an_approver_answers_at_once(self):
        log, asked = [], []

        def approver(request):
            asked.append(request.reason)
            return Approval(request.digest, True, "policy-bot")

        state = _writer_agent(log, approver=approver).run("g")
        assert log == ["report.md"]
        assert state.status is RunStatus.COMPLETED
        assert "'write' effects" in asked[0]

    @pytest.mark.parametrize(
        "approver", [lambda r: 1 / 0, lambda r: True, lambda r: None]
    )
    def test_an_approver_without_an_answer_pauses(self, approver):
        agent = _writer_agent([], approver=approver)
        assert agent.run("g").status is RunStatus.AWAITING_APPROVAL

    def test_conflicting_approvals_are_refused(self):
        agent = _writer_agent([])
        paused = agent.run("g")
        digest = paused.pending.digest
        with pytest.raises(ValueError, match="disagree"):
            agent.resume(
                paused,
                approvals=[Approval(digest, True, "a"), Approval(digest, False, "a")],
            )

    def test_time_waiting_for_approval_does_not_count(self):
        clock = Clock()
        agent = _writer_agent([], clock=clock, limits=Limits(max_seconds=1.0))
        paused = agent.run("g")
        clock.now += 3600.0
        done = agent.resume(
            paused, approvals=[Approval(paused.pending.digest, True, "a")]
        )
        assert done.status is RunStatus.COMPLETED
        assert done.spent.seconds == 0.0


# ------------------------------------------------------------- resume


class Crash(BaseException):
    """Stops the process mid-call, as a kill would (not an Exception)."""


class TestResume:
    def _crashing(self, log, store, idempotent):
        calls = {"n": 0}

        def flaky(query):
            calls["n"] += 1
            log.append(query)
            if calls["n"] == 1:
                raise Crash
            return "ok"

        tool = Tool("flaky", flaky, parameters=QUERY, idempotent=idempotent)
        planner = Script(
            Decision.act("flaky", {"query": "x"}),
            lambda s: Decision.finish(s.steps[-1].outcome.value),
        )
        return Agent([tool], planner, store=store)

    def test_an_idempotent_call_in_flight_runs_again(self, tmp_path):
        log, store = [], JsonFileStore(tmp_path)
        agent = self._crashing(log, store, idempotent=True)
        with pytest.raises(Crash):
            agent.run("g", run_id=RUN)
        saved = store.load(RUN)
        assert saved.in_flight["step"] == 0
        assert saved.steps == []
        done = agent.resume(RUN)
        assert log == ["x", "x"]
        assert done.answer == "executed"
        assert done.spent.tool_calls == 2  # both starts are charged  # noqa: PLR2004

    def test_a_non_idempotent_call_in_flight_is_not_repeated(self, tmp_path):
        # Invariant I5: whether it took effect is unknown, so it is not redone.
        log, store = [], JsonFileStore(tmp_path)
        agent = self._crashing(log, store, idempotent=False)
        with pytest.raises(Crash):
            agent.run("g", run_id=RUN)
        done = agent.resume(RUN)
        assert log == ["x"]
        assert done.steps[0].outcome is StepOutcome.INTERRUPTED
        assert done.steps[0].error.code is ErrorCode.TOOL_INTERRUPTED
        assert done.spent.tool_calls == 1  # charged: it may have run
        assert done.answer == "interrupted"

    def test_an_ended_run_is_returned_unchanged(self):
        agent = Agent([], Script(Decision.finish("x")))
        state = agent.run("g")
        before = len(agent.audit.records)
        assert agent.resume(state).to_dict() == state.to_dict()
        assert len(agent.audit.records) == before

    def test_the_audit_chain_spans_sessions(self, tmp_path):
        store, sink = JsonFileStore(tmp_path), MemoryAuditSink()
        log = []
        paused = _writer_agent(log, store=store, audit=sink).run("g", run_id=RUN)
        other = _writer_agent(log, store=store, audit=sink)  # a new process, same sink
        other.resume(RUN, approvals=[Approval(paused.pending.digest, True, "allen")])
        assert verify_audit(sink.records) == []
        assert store.load(RUN).audit_head["seq"] == len(sink.records) - 1

    def test_a_pending_step_that_does_not_fit_is_refused(self):
        agent = _writer_agent([])
        paused = agent.run("g")
        data = paused.to_dict()
        data["pending"]["step"] = 4
        with pytest.raises(StateError, match="not the next step"):
            agent.resume(
                RunState.from_dict(data),
                approvals=[Approval(paused.pending.digest, True, "a")],
            )

    def test_resume_by_id_needs_a_store(self):
        with pytest.raises(ValueError, match="needs a store"):
            Agent([], Script()).resume(RUN)


# ------------------------------------------------------------- construction


class TestAgentArguments:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"planner": 3},
            {"policy": {}},
            {"limits": 3},
            {"approver": 3},
            {"done": "yes"},
            {"audit": []},
            {"store": {}},
        ],
    )
    def test_refused(self, kwargs):
        base = {"tools": [], "planner": Script()}
        with pytest.raises(TypeError):
            Agent(**{**base, **kwargs})
