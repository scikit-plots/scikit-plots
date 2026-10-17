# scikitplot/agent/tests/test_agent_parts.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The parts of an agent run: tools, policy and gates, state, store, audit.
"""

from __future__ import annotations

import json
import math
import os

import pytest

from .. import (
    Action,
    AgentError,
    Approval,
    Decision,
    ErrorCode,
    GateKind,
    JsonFileStore,
    JsonlAuditSink,
    Limits,
    MemoryAuditSink,
    MemoryStore,
    Policy,
    RunState,
    RunStatus,
    SchemaError,
    Spent,
    StateError,
    Step,
    StepOutcome,
    StopReason,
    Tool,
    ToolEffect,
    Toolset,
    Usage,
    read_audit,
    verify_audit,
)
from .._audit import GENESIS, AuditTrail
from .._policy import approval_reason, check_action
from .._state import approval_digest, call_key

RUN = "a" * 64
QUERY = {
    "type": "object",
    "properties": {"query": {"type": "string", "minLength": 1}},
    "required": ["query"],
    "additionalProperties": False,
}


def search(query):
    return [query]


# ---------------------------------------------------------------- tools


class TestTool:
    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"name": "1bad"}, ValueError, "Tool.name"),
            ({"name": "a" * 65}, ValueError, "Tool.name"),
            ({"function": 3}, TypeError, "callable"),
            ({"description": 3}, TypeError, "description"),
            ({"effect": "delete"}, ValueError, "effect"),
            ({"scopes": "corpus:read"}, TypeError, "iterable"),
            ({"scopes": ["has space"]}, ValueError, "scopes"),
            ({"cost": -1}, ValueError, "cost"),
            ({"cost": math.nan}, ValueError, "cost"),
            ({"cost": True}, TypeError, "cost"),
            ({"idempotent": 1}, TypeError, "idempotent"),
        ],
    )
    def test_fields(self, kwargs, error, match):
        base = {"name": "search", "function": search, "parameters": QUERY}
        with pytest.raises(error, match=match):
            Tool(**{**base, **kwargs})

    def test_schema_must_fit_the_signature(self):
        with pytest.raises(SchemaError, match="no keyword argument"):
            Tool("t", lambda q: q, parameters=QUERY)
        with pytest.raises(SchemaError, match="does not require"):
            Tool("t", lambda query, k: query, parameters=QUERY)
        exec_ns = {}
        exec("def f(x, /, query): return x", exec_ns)  # noqa: S102 - 3.8+ syntax
        with pytest.raises(SchemaError, match="positional-only"):
            Tool("t", exec_ns["f"], parameters=QUERY)
        Tool("t", lambda **kwargs: kwargs, parameters=QUERY)  # **kwargs takes any

    def test_defaults(self):
        tool = Tool("ping", lambda: "pong")
        assert tool.parameters == {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        assert tool.effect is ToolEffect.READ
        assert tool.idempotent is True
        writer = Tool("save", lambda: None, effect="write")
        assert writer.idempotent is False
        assert writer.describe()["scopes"] == []

    def test_toolset(self):
        tools = Toolset([Tool("b", search, parameters=QUERY), Tool("a", lambda: 1)])
        assert tools.names() == ("a", "b")
        assert [t["name"] for t in tools.describe()] == ["a", "b"]
        with pytest.raises(ValueError, match="two tools"):
            Toolset([Tool("a", lambda: 1), Tool("a", lambda: 2)])
        with pytest.raises(TypeError, match="not a Tool"):
            Toolset([search])


# ---------------------------------------------------------------- policy


def _check(action, tools, policy=None, limits=None, spent=None):
    return check_action(
        action, Toolset(tools), policy or Policy(), limits or Limits(), spent or Spent()
    )


class TestGates:
    def test_least_privilege_by_default(self):
        policy = Policy()
        assert policy.allow_effects == {ToolEffect.READ}
        assert policy.granted_scopes == frozenset()
        assert policy.needs_approval_for("write")
        assert policy.needs_approval_for("external")

    def test_every_gate_passes(self):
        tool, results = _check(
            Action("search", {"query": "x"}), [Tool("search", search, parameters=QUERY)]
        )
        assert tool.name == "search"
        assert [(r.gate, r.passed) for r in results] == [
            (GateKind.VALIDATION, True),
            (GateKind.PERMISSIONS, True),
            (GateKind.SCOPE, True),
            (GateKind.BUDGET, True),
        ]

    @pytest.mark.parametrize(
        ("action", "tool", "policy", "limits", "spent", "gate", "code"),
        [
            (Action("nope"), {}, None, None, None, "validation", "unknown_tool"),
            (
                Action("search", {"query": ""}),
                {},
                None,
                None,
                None,
                "validation",
                "invalid_arguments",
            ),
            (
                Action("search", {"q": "x"}),
                {},
                None,
                None,
                None,
                "validation",
                "invalid_arguments",
            ),
            (
                Action("search", {"query": "x"}),
                {},
                Policy(deny_tools={"search"}),
                None,
                None,
                "permissions",
                "permission_denied",
            ),
            (
                Action("search", {"query": "x"}),
                {},
                Policy(allow_tools={"other"}),
                None,
                None,
                "permissions",
                "permission_denied",
            ),
            (
                Action("search", {"query": "x"}),
                {"effect": "write"},
                None,
                None,
                None,
                "permissions",
                "permission_denied",
            ),
            (
                Action("search", {"query": "x"}),
                {"scopes": {"corpus:read"}},
                None,
                None,
                None,
                "scope",
                "scope_denied",
            ),
            (
                Action("search", {"query": "x"}),
                {},
                None,
                Limits(max_tool_calls=1),
                Spent(tool_calls=1),
                "budget",
                "budget_exceeded",
            ),
            (
                Action("search", {"query": "x"}),
                {"cost": 2.0},
                None,
                Limits(max_cost=3.0),
                Spent(cost=1.5),
                "budget",
                "budget_exceeded",
            ),
            (
                Action("search", {"query": "x"}),
                {},
                None,
                Limits(max_seconds=1.0),
                Spent(seconds=1.0),
                "budget",
                "budget_exceeded",
            ),
        ],
    )
    def test_each_gate_denies(  # noqa: PLR0917
        self, action, tool, policy, limits, spent, gate, code
    ):
        tools = [Tool("search", search, parameters=QUERY, **tool)]
        _, results = _check(action, tools, policy, limits, spent)
        last = results[-1]
        assert not last.passed
        assert last.gate.value == gate
        assert last.error.code.value == code
        assert last.error.details["gate"] == gate
        assert all(r.passed for r in results[:-1])

    def test_a_scope_granted_passes(self):
        tools = [Tool("search", search, parameters=QUERY, scopes={"corpus:read"})]
        _, results = _check(
            Action("search", {"query": "x"}),
            tools,
            Policy(granted_scopes={"corpus:read"}),
        )
        assert results[-1].passed

    def test_the_budget_is_exact_at_the_limit(self):
        tools = [Tool("search", search, parameters=QUERY, cost=1.5)]
        _, results = _check(
            Action("search", {"query": "x"}),
            tools,
            None,
            Limits(max_cost=3.0),
            Spent(cost=1.5),
        )
        assert results[-1].passed  # 1.5 + 1.5 == 3.0 stays within

    def test_approval_reason(self):
        writer = Tool("save", lambda: None, effect="write")
        reader = Tool("read", lambda: None)
        assert approval_reason(writer, Policy()) is not None
        assert approval_reason(reader, Policy()) is None
        assert "tool 'read'" in approval_reason(reader, Policy(approval_tools={"read"}))

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"max_steps": None}, ValueError),
            ({"max_steps": 0}, ValueError),
            ({"max_steps": 1.5}, TypeError),
            ({"max_steps": True}, TypeError),
            ({"max_cost": math.inf}, ValueError),
            ({"max_tokens": 2.0}, TypeError),
            ({"max_seconds": -1}, ValueError),
        ],
    )
    def test_limits(self, kwargs, error):
        with pytest.raises(error):
            Limits(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"allow_effects": "read"},
            {"allow_effects": ["delete"]},
            {"granted_scopes": "corpus:read"},
            {"deny_tools": [1]},
        ],
    )
    def test_policy_fields(self, kwargs):
        with pytest.raises((TypeError, ValueError)):
            Policy(**kwargs)


# ---------------------------------------------------------------- state


def _state(**kwargs):
    return RunState(run_id=RUN, goal="g", **kwargs)


class TestState:
    def test_decision_constructors(self):
        act = Decision.act("search", {"query": ("a",)}, usage=Usage(3, 0.5))
        assert act.kind == "act"
        assert act.action.arguments == {"query": ["a"]}
        assert Decision.from_dict(act.to_dict()).to_dict() == act.to_dict()
        fin = Decision.finish({"a": 1})
        assert Decision.from_dict(fin.to_dict()).answer == {"a": 1}

    @pytest.mark.parametrize(
        ("build", "error"),
        [
            (lambda: Decision("maybe"), ValueError),
            (lambda: Decision("act"), TypeError),
            (lambda: Decision("finish", action=Action("x")), ValueError),
            (lambda: Decision.act("x", {"a": float("nan")}), ValueError),
            (lambda: Decision.finish({1: "a"}), TypeError),
            (lambda: Decision.act(""), ValueError),
            (lambda: Usage(tokens=-1), TypeError),
            (lambda: Usage(cost=math.inf), ValueError),
            (lambda: Approval("x", True, "me"), ValueError),
            (lambda: Approval("0" * 64, 1, "me"), TypeError),
            (lambda: Approval("0" * 64, True, " "), ValueError),
            (lambda: RunState(run_id="../x", goal="g"), ValueError),
            (lambda: RunState(run_id=RUN, goal=""), ValueError),
            (lambda: RunState(run_id=RUN, goal="g", status="stopped"), StateError),
        ],
    )
    def test_refused(self, build, error):
        with pytest.raises(error):
            build()

    def test_round_trip(self):
        state = _state()
        state.steps.append(
            Step(
                0,
                Decision.act("search", {"query": "x"}),
                StepOutcome.EXECUTED,
                gates=({"gate": "validation", "passed": True},),
                result=["x"],
                key="b" * 64,
                seconds=0.25,
            )
        )
        state.steps.append(
            Step(
                1,
                Decision.finish("no"),
                StepOutcome.REJECTED,
                error=AgentError(ErrorCode.ANSWER_REJECTED, "too short"),
            )
        )
        state.spent = Spent(steps=2, tool_calls=1, cost=1.0, tokens=0, seconds=0.5)
        text = json.dumps(state.to_dict(), allow_nan=False)
        again = RunState.from_dict(json.loads(text))
        assert again.to_dict() == state.to_dict()
        assert again.observations()[1]["error"] == {
            "code": "answer_rejected",
            "message": "too short",
        }

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda d: d.update(schema_version=2),
            lambda d: d.update(extra=1),
            lambda d: d.pop("goal"),
            lambda d: d.update(status="unknown"),
            lambda d: d.update(steps={}),
            lambda d: d["spent"].update(cost=-1),
            lambda d: d.update(in_flight={"step": 0}),
            lambda d: d.update(error={"code": "nope", "message": "m", "details": {}}),
            lambda d: d.update(status="stopped"),  # without a stop reason
        ],
    )
    def test_from_dict_is_strict(self, mutate):
        data = _state().to_dict()
        mutate(data)
        with pytest.raises(StateError):
            RunState.from_dict(data)

    def test_digests_bind_the_action(self):
        action = Action("save", {"path": "a"})
        head = "d" * 64

        def bound(  # noqa: PLR0917
            run=RUN,
            step=0,
            act=action,
            effect="write",
            scopes=("files:write",),
            at=head,
        ):
            return approval_digest(
                run, step, act, effect=effect, scopes=scopes, head=at
            )

        assert bound() == bound(
            act=Action("save", {"path": "a"}), scopes=["files:write"]
        )
        others = [
            bound(run="e" * 64),
            bound(step=1),
            bound(act=Action("save", {"path": "b"})),
            bound(effect="external"),
            bound(scopes=()),
            bound(at="f" * 64),
        ]
        assert bound() not in others
        assert len(set(others)) == len(others)
        assert call_key(RUN, 0, action) != call_key(RUN, 1, action)


# ---------------------------------------------------------------- store


class TestStore:
    @pytest.mark.parametrize("make", ["memory", "file"])
    def test_round_trip(self, make, tmp_path):
        store = MemoryStore() if make == "memory" else JsonFileStore(tmp_path / "runs")
        state = _state()
        store.save(state)
        assert store.load(RUN).to_dict() == state.to_dict()
        assert store.run_ids() == (RUN,)
        with pytest.raises(KeyError):
            store.load("b" * 64)

    def test_a_run_id_cannot_name_another_file(self, tmp_path):
        store = JsonFileStore(tmp_path)
        for bad in ("../" + "a" * 61, "A" * 64, "a" * 63):
            with pytest.raises(ValueError, match="64 hexadecimal"):
                store.path(bad)

    def test_a_failed_save_keeps_the_old_state(self, tmp_path, monkeypatch):
        store = JsonFileStore(tmp_path)
        store.save(_state())
        before = store.path(RUN).read_bytes()

        def broken(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", broken)
        with pytest.raises(OSError, match="disk full"):
            store.save(_state(steps=[]))
        assert store.path(RUN).read_bytes() == before
        assert [p.name for p in tmp_path.iterdir()] == [f"{RUN}.json"]  # no temp left

    def test_a_corrupt_file_is_refused(self, tmp_path):
        store = JsonFileStore(tmp_path)
        store.path(RUN).write_text("{not json", encoding="utf-8")
        with pytest.raises(StateError, match="not JSON"):
            store.load(RUN)


# ---------------------------------------------------------------- audit


def _trail(sink, n=3, head=None):
    clock = iter(range(100))
    trail = AuditTrail(sink, RUN, head, lambda: float(next(clock)))
    for i in range(n):
        trail.emit("event", {"i": i})
    return trail


class TestAudit:
    def test_chain(self):
        sink = MemoryAuditSink()
        trail = _trail(sink)
        records = sink.records
        assert [r["seq"] for r in records] == [0, 1, 2]
        assert records[0]["prev"] == GENESIS
        assert records[1]["prev"] == records[0]["digest"]
        assert trail.head == {"seq": 2, "digest": records[2]["digest"]}
        assert verify_audit(records) == []

    def test_a_chain_continues_from_its_head(self):
        sink = MemoryAuditSink()
        head = _trail(sink, 2).head
        _trail(sink, 2, head=head)
        assert [r["seq"] for r in sink.records] == [0, 1, 2, 3]
        assert verify_audit(sink.records) == []

    @pytest.mark.parametrize(
        ("tamper", "found"),
        [
            (lambda rs: rs[1]["data"].update(i=9), "digest does not match"),
            (lambda rs: rs.pop(1), "seq 2, expected 1"),
            (lambda rs: rs.reverse(), "expected 0"),
            (lambda rs: rs[2].pop("prev"), "missing"),
        ],
    )
    def test_tampering_is_found(self, tamper, found):
        sink = MemoryAuditSink()
        _trail(sink)
        records = sink.records
        tamper(records)
        assert any(found in p for p in verify_audit(records))

    def test_jsonl_round_trip(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        _trail(JsonlAuditSink(path, fsync=True))
        _trail(JsonlAuditSink(path), n=1)  # a second run id-less chain: same run
        records = read_audit(path)
        assert len(records) == 4  # noqa: PLR2004
        problems = verify_audit(records)
        assert problems  # a second genesis
        assert "expected 3" in problems[0]
        path.write_text("{}\n[]\n", encoding="utf-8")
        with pytest.raises(ValueError, match=":2: not a JSON object"):
            read_audit(path)

    def test_a_sink_that_fails_raises(self):
        class Broken(MemoryAuditSink):
            def write(self, record):
                raise OSError("read-only")

        trail = AuditTrail(Broken(), RUN, None, lambda: 0.0)
        with pytest.raises(OSError, match="read-only"):
            trail.emit("event", {})
        assert trail.head is None  # nothing counted that was not written

    def test_status_enum_values(self):
        assert RunStatus("awaiting_approval") is RunStatus.AWAITING_APPROVAL
        assert StopReason("max_steps") is StopReason.MAX_STEPS
        assert str(StepOutcome.DENIED) == "denied"
