# scikitplot/agent/tests/test_agent_hardening.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
What an adversarial planner, tool, store or clock must not be able to do.

Each test names the review finding (5a review 1) it holds fixed.
"""

from __future__ import annotations

import json

import pytest

from .. import (
    Action,
    Agent,
    Approval,
    Decision,
    ErrorCode,
    JsonFileStore,
    JsonlAuditSink,
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
    read_audit,
    verify_audit,
)
from .._canonical import MAX_INT_DIGITS
from .._loop import MAX_ATTEMPTS
from .._state import call_key

RUN = "a" * 64
TEXT = {
    "type": "object",
    "properties": {"text": {"type": "string", "maxLength": 3}},
    "required": ["text"],
    "additionalProperties": False,
}
WRITE = Policy(allow_effects={"read", "write"})


class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


def writer(log, **kwargs):
    def save(text):
        log.append(text)
        return "saved"

    return Tool("save", save, parameters=TEXT, effect="write", **kwargs)


def save_then_finish(state, tools):
    if not state.steps:
        return Decision.act("save", {"text": "hi"})
    return Decision.finish(state.steps[-1].outcome.value)


# ----------------------------------------------------------- 1: approvals


class TestApprovalIsUsedOnce:
    def test_a_resumed_copy_is_refused_once_the_store_moved_on(self):
        log, store = [], MemoryStore()
        agent = Agent([writer(log)], save_then_finish, policy=WRITE, store=store)
        paused = agent.run("g", run_id=RUN)
        approval = Approval(paused.pending.digest, True, "allen")
        agent.resume(paused, approvals=[approval])
        with pytest.raises(StateError, match="newer state"):
            agent.resume(paused, approvals=[approval])
        assert agent.resume(RUN, approvals=[approval]).status is RunStatus.COMPLETED
        assert log == ["hi"]

    def test_a_run_id_is_used_once(self):
        store, sink = MemoryStore(), MemoryAuditSink()
        agent = Agent([], lambda s, t: Decision.finish(1), store=store, audit=sink)
        agent.run("g", run_id=RUN)
        with pytest.raises(ValueError, match="already in the store"):
            agent.run("other", run_id=RUN)
        no_store = Agent([], lambda s, t: Decision.finish(1), audit=sink)
        with pytest.raises(ValueError, match="already in the audit trail"):
            no_store.run("other", run_id=RUN)

    def test_an_approval_of_one_run_does_not_open_another_with_its_id(self):
        log = []
        first = Agent([writer(log)], save_then_finish, policy=WRITE, now=lambda: 1.0)
        old = first.run("pay the invoice", run_id=RUN)
        approval = Approval(old.pending.digest, True, "allen")
        second = Agent([writer(log)], save_then_finish, policy=WRITE, now=lambda: 2.0)
        new = second.run("pay the invoice", run_id=RUN)
        assert new.pending.digest != old.pending.digest
        assert (
            second.resume(new, approvals=[approval]).status
            is RunStatus.AWAITING_APPROVAL
        )
        assert log == []


# ----------------------------------------------------------- 2, 3: strict JSON


class TestValuesAModelCanProduce:
    SURROGATE = json.loads('"\\ud800"')

    def test_a_lone_surrogate_in_arguments_fails_the_planner(self):
        def planner(state, tools):
            return Decision.act("save", {"text": self.SURROGATE})

        state = Agent([writer([])], planner, policy=WRITE).run("g")
        assert state.stop_reason is StopReason.PLANNER_FAILED
        assert "surrogate" in state.error.message

    @pytest.mark.parametrize("bad", ["surrogate", "huge"])
    def test_a_result_strict_json_cannot_hold_fails_the_step(self, bad):
        value = self.SURROGATE if bad == "surrogate" else 10 ** (MAX_INT_DIGITS + 1)
        def planner(s, t):
            return Decision.finish(None) if s.steps else Decision.act("t")

        state = Agent([Tool("t", lambda: value)], planner).run("g")
        assert state.steps[0].error.code is ErrorCode.TOOL_RESULT_NOT_JSON
        assert state.status is RunStatus.COMPLETED
        RunState.from_dict(json.loads(json.dumps(state.to_dict())))


# ----------------------------------------------------------- 4, 11: one value


class TestWhatRunsIsWhatWasChecked:
    def test_an_action_that_changes_after_validation_runs_as_validated(self):
        class Shifty(Action):
            reads = 0

            def __getattribute__(self, name):
                if name == "arguments":
                    type(self).reads += 1
                    if type(self).reads > 1:
                        return {"text": "rm -rf / --no-preserve-root"}
                return object.__getattribute__(self, name)

        log = []

        def planner(state, tools):
            if state.steps:
                return Decision.finish(None)
            return Decision("act", action=Shifty("save", {"text": "hi"}))

        def approver(request):
            return Approval(request.digest, True, "allen")

        Agent([writer(log)], planner, policy=WRITE, approver=approver).run("g")
        assert log in ([], ["hi"])  # never the swapped arguments

    def test_a_decision_whose_to_dict_raises_is_invalid(self):
        class Broken(Decision):
            def to_dict(self):
                raise RuntimeError("no")

        state = Agent([], lambda s, t: Broken.finish(1)).run("g")
        assert state.stop_reason is StopReason.INVALID_DECISION

    def test_a_tool_cannot_change_the_recorded_arguments(self):
        schema = {
            "type": "object",
            "properties": {"xs": {"type": "array"}},
            "required": ["xs"],
            "additionalProperties": False,
        }

        def grow(xs):
            xs.append(999)
            return len(xs)

        def planner(s, t):
            return (
                Decision.finish(None)
                if s.steps
                else Decision.act("grow", {"xs": [1]})
            )
        state = Agent([Tool("grow", grow, parameters=schema)], planner).run(
            "g", run_id=RUN
        )
        step = state.steps[0]
        assert step.decision.action.arguments == {"xs": [1]}
        assert step.key == call_key(RUN, 0, step.decision.action)


# ----------------------------------------------------------- 5, 6: in flight


class Crash(BaseException):
    """Stops the process mid-call."""


class TestInFlight:
    def _crashing(self, store, runs, **kwargs):
        def flaky(text):
            runs.append(text)
            raise Crash

        tool = Tool("flaky", flaky, parameters=TEXT, idempotent=True, **kwargs)
        def planner(s, t):
            return (
                Decision.finish(s.steps[-1].outcome.value)
                if s.steps
                else Decision.act("flaky", {"text": "x"})
            )
        return Agent([tool], planner, store=store)

    def test_every_start_is_charged_and_attempts_are_capped(self, tmp_path):
        store, runs = JsonFileStore(tmp_path), []
        agent = self._crashing(store, runs, cost=10.0)
        with pytest.raises(Crash):
            agent.run("g", run_id=RUN)
        for _ in range(MAX_ATTEMPTS - 1):
            with pytest.raises(Crash):
                agent.resume(RUN)
        done = agent.resume(RUN)
        assert len(runs) == MAX_ATTEMPTS
        assert done.spent.tool_calls == MAX_ATTEMPTS
        assert done.spent.cost == 10.0 * MAX_ATTEMPTS
        assert done.steps[0].outcome is StepOutcome.INTERRUPTED

    def test_a_retry_passes_the_budget_gate(self, tmp_path):
        store, runs = JsonFileStore(tmp_path), []
        agent = self._crashing(store, runs)
        agent.limits = Limits(max_tool_calls=1)
        with pytest.raises(Crash):
            agent.run("g", run_id=RUN)
        done = agent.resume(RUN)
        assert runs == ["x"]
        # The first start may have run: recorded as interrupted, with its key.
        assert done.steps[0].outcome is StepOutcome.INTERRUPTED
        assert "a retry was denied" in done.steps[0].error.message
        assert done.steps[0].key is not None

    def test_a_saved_flag_does_not_make_a_write_idempotent(self, tmp_path):
        log, store = [], JsonFileStore(tmp_path)

        def save(text):
            log.append(text)
            raise Crash

        tool = Tool("save", save, parameters=TEXT, effect="write")
        agent = Agent(
            [tool],
            save_then_finish,
            policy=WRITE,
            store=store,
            approver=lambda r: Approval(r.digest, True, "a"),
        )
        with pytest.raises(Crash):
            agent.run("g", run_id=RUN)
        data = json.loads(store.path(RUN).read_text(encoding="utf-8"))
        data["in_flight"]["idempotent"] = True  # forged
        store.path(RUN).write_text(json.dumps(data), encoding="utf-8")
        done = agent.resume(RUN)
        assert log == ["hi"]
        assert done.steps[0].outcome is StepOutcome.INTERRUPTED


# ----------------------------------------------------------- 7: audit vs state


class TestAuditAndState:
    def test_a_failed_save_does_not_fork_the_chain(self):
        class Flaky(MemoryStore):
            saves = 0

            def save(self, state):
                type(self).saves += 1
                if type(self).saves == 3:  # noqa: PLR2004
                    raise OSError("disk full")
                super().save(state)

        sink, store = MemoryAuditSink(), Flaky()
        def planner(s, t):
            return Decision.finish("ok") if len(s.steps) > 1 else Decision.act("t")

        agent = Agent([Tool("t", lambda: 1)], planner, audit=sink, store=store)
        with pytest.raises(OSError, match="disk full"):
            agent.run("g", run_id=RUN)
        done = agent.resume(RUN)
        assert done.status is RunStatus.COMPLETED
        assert verify_audit(sink.records) == []
        assert "state_behind_audit" in [r["kind"] for r in sink.records]

    @pytest.mark.parametrize(
        "record",
        [
            [1, 2],
            {
                "seq": 0,
                "run_id": ["x"],
                "kind": "k",
                "time": 0.0,
                "data": {},
                "prev": "0",
                "digest": "x",
            },
        ],
    )
    def test_verify_lists_malformed_records(self, record):
        assert verify_audit([record])


# ----------------------------------------------------------- 8: state


def _completed():
    agent = Agent(
        [Tool("t", lambda: 1)],
        lambda s, t: Decision.finish(2) if s.steps else Decision.act("t"),
    )
    return agent.run("g", run_id=RUN).to_dict()


class TestStateIsConsistent:
    @pytest.mark.parametrize(
        ("mutate", "match"),
        [
            (lambda d: d["steps"][1].update(index=7), "indices"),
            (lambda d: d.update(answer=3), "answer"),
            (lambda d: d.update(stop_reason="max_steps"), "cannot have stop reason"),
            (lambda d: d["steps"].pop(), "accepted answer"),
            (lambda d: d["steps"][0].update(key=5), "key"),
            (lambda d: d["steps"][0].update(outcome="finished"), "cannot have outcome"),
            (lambda d: d["spent"].update(steps=0), "spent.steps"),
            (lambda d: d.update(audit_head={"seq": "x", "digest": "0" * 64}), "seq"),
            (
                lambda d: d.update(
                    error={"code": "tool_failed", "message": "m", "details": {}}
                ),
                "error is set",
            ),
        ],
    )
    def test_from_dict_refuses(self, mutate, match):
        data = _completed()
        mutate(data)
        with pytest.raises(StateError, match=match):
            RunState.from_dict(data)

    def test_a_pending_request_of_another_run_is_refused(self):
        agent = Agent([writer([])], save_then_finish, policy=WRITE)
        data = agent.run("g", run_id=RUN).to_dict()
        data["pending"]["run_id"] = "b" * 64
        with pytest.raises(StateError, match="not the next step of this run"):
            RunState.from_dict(data)
        data["pending"]["run_id"] = RUN
        data["pending"]["digest"] = "not hex"
        with pytest.raises(StateError):
            RunState.from_dict(data)


# ----------------------------------------------------------- 9: tokens


def test_a_decision_past_the_token_limit_runs_no_tool():
    log = []

    def planner(state, tools):
        if state.steps:
            return Decision.finish(None, usage=Usage(0))
        return Decision.act("save", {"text": "hi"}, usage=Usage(10**6))

    state = Agent(
        [writer(log)], planner, policy=WRITE, limits=Limits(max_tokens=100)
    ).run("g")
    assert log == []
    assert state.steps[0].error.code is ErrorCode.BUDGET_EXCEEDED
    assert state.stop_reason is StopReason.MAX_TOKENS


# ----------------------------------------------------------- 10: store


def test_a_store_file_of_another_run_is_refused(tmp_path):
    store = JsonFileStore(tmp_path)
    Agent([], lambda s, t: Decision.finish(1), store=store).run("g", run_id=RUN)
    other = "b" * 64
    store.path(other).write_bytes(store.path(RUN).read_bytes())
    with pytest.raises(StateError, match="holds run"):
        store.load(other)


# ----------------------------------------------------------- 12, 13: time


class TestTime:
    def test_a_clock_that_steps_back_hides_no_time(self):
        clock = Clock(100.0)

        def planner(state, tools):
            clock.now += 10.0  # thinking
            return Decision.act("back")

        def back():
            clock.now -= 50.0

        agent = Agent(
            [Tool("back", back)], planner, limits=Limits(max_seconds=25.0), clock=clock
        )
        state = agent.run("g")
        assert state.stop_reason is StopReason.TIMEOUT
        assert state.spent.seconds >= 25.0  # noqa: PLR2004

    def test_an_approvers_wait_is_not_running_time(self):
        clock = Clock()

        def approver(request):
            clock.now += 3600.0
            return Approval(request.digest, True, "slow")

        state = Agent(
            [writer([])],
            save_then_finish,
            policy=WRITE,
            approver=approver,
            clock=clock,
            limits=Limits(max_seconds=10.0),
        ).run("g")
        assert state.status is RunStatus.COMPLETED
        assert state.spent.seconds == 0.0


# ----------------------------------------------------------- re-review 5a


class TestReReview:
    def test_a_paused_copy_cannot_be_resumed_twice_without_a_store(self):
        # 1a: the shared trail shows the copy was resumed already.
        log, sink = [], MemoryAuditSink()
        agent = Agent([writer(log)], save_then_finish, policy=WRITE, audit=sink)
        paused = agent.run("g")
        approval = Approval(paused.pending.digest, True, "allen")
        agent.resume(paused, approvals=[approval])
        with pytest.raises(StateError, match="resumed already"):
            agent.resume(paused, approvals=[approval])
        assert log == ["hi"]

    def test_runs_with_one_id_goal_and_clock_still_differ(self):
        # 1b: the nonce in run_started.
        log = []
        made = [
            Agent([writer(log)], save_then_finish, policy=WRITE, now=lambda: 0.0).run(
                "g", run_id=RUN
            )
            for _ in range(2)
        ]
        assert made[0].pending.digest != made[1].pending.digest

    def test_a_trail_that_lost_records_is_refused(self, tmp_path):
        # N1: the state is ahead of the trail.
        path = tmp_path / "audit.jsonl"
        store = MemoryStore()
        agent = Agent(
            [writer([])],
            save_then_finish,
            policy=WRITE,
            store=store,
            audit=JsonlAuditSink(path),
        )
        paused = agent.run("g", run_id=RUN)
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("".join(lines[:-2]), encoding="utf-8")  # truncated
        fresh = Agent(
            [writer([])],
            save_then_finish,
            policy=WRITE,
            store=store,
            audit=JsonlAuditSink(path),
        )
        with pytest.raises(StateError, match="does not hold this state"):
            fresh.resume(RUN, approvals=[Approval(paused.pending.digest, True, "a")])
        assert len(read_audit(path)) == len(lines) - 2

    def test_a_torn_line_does_not_stop_other_runs(self, tmp_path):
        # N2: head() skips what does not parse; verify_audit still reports it.
        path = tmp_path / "audit.jsonl"
        Agent([], lambda s, t: Decision.finish(1), audit=JsonlAuditSink(path)).run("g")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write('{"run_id": "x", "seq"')  # a crash mid-write
        state = Agent(
            [], lambda s, t: Decision.finish(2), audit=JsonlAuditSink(path)
        ).run("h")
        assert state.status is RunStatus.COMPLETED
        with pytest.raises(ValueError, match="not JSON"):
            read_audit(path)

    def test_the_head_cache_sees_another_writer(self, tmp_path):
        # N3: one file, two sinks; each sees the other's records.
        path = tmp_path / "audit.jsonl"
        first, second = JsonlAuditSink(path), JsonlAuditSink(path)
        Agent([], lambda s, t: Decision.finish(1), audit=first).run("g", run_id=RUN)
        assert first.head(RUN) is not None
        other = "b" * 64
        Agent([], lambda s, t: Decision.finish(1), audit=second).run("g", run_id=other)
        assert first.head(other) == second.head(other)
        with pytest.raises(ValueError, match="already in the audit trail"):
            Agent([], lambda s, t: Decision.finish(1), audit=first).run(
                "g", run_id=other
            )

    def test_a_surrogate_in_a_key_is_refused(self):
        # 2: keys are text too.
        surrogate = json.loads('"\\ud800"')
        schema = {
            "type": "object",
            "properties": {"m": {"type": "object"}},
            "required": ["m"],
            "additionalProperties": False,
        }

        def return_mapping(m):
            return m

        def invalid_argument_planner(s, t):
            return Decision.act("t", {"m": {surrogate: 1}})

        state = Agent(
            [Tool("t", return_mapping, parameters=schema)],
            invalid_argument_planner,
        ).run("g")
        assert state.stop_reason is StopReason.PLANNER_FAILED

        def invalid_result_tool():
            return {surrogate: 1}

        def invalid_result_planner(s, t):
            return Decision.finish(None) if s.steps else Decision.act("u")

        result = Agent(
            [Tool("u", invalid_result_tool)],
            invalid_result_planner,
        ).run("g")
        assert result.steps[0].error.code is ErrorCode.TOOL_RESULT_NOT_JSON

    def test_a_pause_does_not_reset_the_attempt_count(self, tmp_path):
        # 5: an idempotent write that crashes, approved at every start.
        store, starts = JsonFileStore(tmp_path), []

        def flaky(text):
            starts.append(text)
            raise Crash

        tool = Tool("flaky", flaky, parameters=TEXT, effect="write", idempotent=True)
        def planner(s, t):
            return (
                Decision.finish(s.steps[-1].outcome.value)
                if s.steps
                else Decision.act("flaky", {"text": "x"})
            )
        agent = Agent([tool], planner, policy=WRITE, store=store)
        state = agent.run("g", run_id=RUN)
        for _ in range(10):
            if state.status is RunStatus.AWAITING_APPROVAL:
                approval = Approval(state.pending.digest, True, "a")
                try:
                    state = agent.resume(RUN, approvals=[approval])
                except Crash:
                    state = store.load(RUN)
                    try:
                        state = agent.resume(RUN)
                    except Crash:
                        state = store.load(RUN)
            elif state.done:
                break
        assert len(starts) == MAX_ATTEMPTS
        assert state.steps[0].outcome is StepOutcome.INTERRUPTED

    def test_verify_reports_a_seq_that_is_not_an_integer(self):
        sink = MemoryAuditSink()
        Agent([], lambda s, t: Decision.finish(1), audit=sink).run("g")
        sink.records[0]["seq"] = "a"
        assert any("seq is not an integer" in p for p in verify_audit(sink.records))
