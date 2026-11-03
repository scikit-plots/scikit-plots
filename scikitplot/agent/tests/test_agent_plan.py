"""
Plans: carried by decisions, checked, audited, resumed; progress cannot be hidden.
"""

from __future__ import annotations

import json

import pytest

from .. import (
    Agent,
    Approval,
    Decision,
    ErrorCode,
    MemoryStore,
    PlanItem,
    PlanStatus,
    Policy,
    RunState,
    RunStatus,
    StateError,
    StopReason,
    verify_audit,
)
from .test_agent_loop import RUN, Script, save_tool, search_tool

P = PlanItem


def _plan(*items):
    return [P(i, t, s) for i, t, s in items]


def test_a_plan_is_recorded_read_by_the_planner_and_audited():
    seen = []

    def finish(state):
        seen.append(state.plan)
        return Decision.finish(
            "ok", plan=_plan(("find", "find the docs", "done"), ("answer", "answer", "done"))
        )

    planner = Script(
        Decision.act(
            "search",
            {"query": "spans"},
            plan=_plan(("find", "find the docs", "in_progress"), ("answer", "answer", "pending")),
        ),
        finish,
    )
    agent = Agent([search_tool()], planner)
    state = agent.run("g", run_id=RUN)
    assert state.status is RunStatus.COMPLETED
    assert [i.status for i in seen[0]] == [PlanStatus.IN_PROGRESS, PlanStatus.PENDING]
    assert [i.status for i in state.plan] == [PlanStatus.DONE, PlanStatus.DONE]
    decisions = [r for r in agent.audit.records if r["kind"] == "decision"]
    assert decisions[0]["data"]["decision"]["plan"][0]["status"] == "in_progress"
    assert verify_audit(agent.audit.records) == []
    again = RunState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert again.plan == state.plan


def test_a_decision_without_a_plan_keeps_the_plan_and_the_old_mapping():
    first = _plan(("a", "step a", "in_progress"))
    planner = Script(
        Decision.act("search", {"query": "x"}, plan=first),
        Decision.finish("done"),
    )
    state = Agent([search_tool()], planner).run("g")
    assert state.plan == tuple(first)
    assert set(Decision.finish("x").to_dict()) == {
        "kind",
        "action",
        "answer",
        "rationale",
        "usage",
    }


@pytest.mark.parametrize(
    "second, message",
    [
        (_plan(("b", "other", "pending")), "missing"),
        (_plan(("a", "read", "pending")), "final"),
        (_plan(("a", "read, rewritten", "done")), "final"),
    ],
)
def test_hiding_or_rewriting_progress_fails_the_run(second, message):
    planner = Script(
        Decision.act("search", {"query": "x"}, plan=_plan(("a", "read", "done"))),
        Decision.finish("ok", plan=second),
    )
    state = Agent([search_tool()], planner).run("g")
    assert state.status is RunStatus.FAILED
    assert state.stop_reason is StopReason.INVALID_DECISION
    assert state.error.code is ErrorCode.INVALID_DECISION
    assert message in state.error.message
    assert state.plan == tuple(_plan(("a", "read", "done")))  # unchanged


def test_dropping_an_item_is_allowed_and_final():
    planner = Script(
        Decision.act("search", {"query": "x"}, plan=_plan(("a", "read", "pending"))),
        Decision.act("search", {"query": "y"}, plan=_plan(("a", "read", "dropped"))),
        Decision.finish("ok", plan=_plan(("a", "read", "dropped"), ("b", "new", "done"))),
    )
    state = Agent([search_tool()], planner).run("g")
    assert state.status is RunStatus.COMPLETED
    assert [(i.id, i.status.value) for i in state.plan] == [("a", "dropped"), ("b", "done")]


def test_the_plan_survives_an_approval_pause():
    log, store = [], MemoryStore()
    planner = Script(
        Decision.act("save", {"path": "r.md"}, plan=_plan(("save", "save it", "in_progress"))),
        lambda s: Decision.finish("ok", plan=_plan(("save", "save it", "done"))),
    )
    agent = Agent(
        [save_tool(log)], planner, policy=Policy(allow_effects={"read", "write"}), store=store
    )
    paused = agent.run("g", run_id=RUN)
    assert paused.status is RunStatus.AWAITING_APPROVAL
    assert paused.plan[0].status is PlanStatus.IN_PROGRESS
    assert store.load(RUN).plan == paused.plan
    done = agent.resume(RUN, approvals=[Approval(paused.pending.digest, True, "allen")])
    assert done.status is RunStatus.COMPLETED
    assert done.plan[0].status is PlanStatus.DONE


def test_a_tampered_plan_history_is_refused_at_load():
    planner = Script(
        Decision.act("search", {"query": "x"}, plan=_plan(("a", "read", "done"))),
        Decision.act("search", {"query": "y"}, plan=_plan(("a", "read", "done"))),
        Decision.finish("ok"),
    )
    state = Agent([search_tool()], planner).run("g")
    data = state.to_dict()
    data["steps"][1]["decision"]["plan"] = []  # silently remove the done item
    with pytest.raises(StateError, match="step 1: plan items"):
        RunState.from_dict(data)


@pytest.mark.parametrize(
    "items, error",
    [
        ([P("a", "x", "in_progress"), P("b", "y", "in_progress")], ValueError),
        ([P("a", "x"), P("a", "y")], ValueError),
        ([P(f"i{n}", "x") for n in range(65)], ValueError),
        ("abc", TypeError),
    ],
)
def test_invalid_plans_cannot_be_built(items, error):
    with pytest.raises(error):
        Decision.finish("x", plan=items)


@pytest.mark.parametrize(
    "args",
    [("", "x"), ("-a", "x"), ("a" * 65, "x"), ("a", " "), ("a", "x" * 501), ("a", "x", "later")],
)
def test_invalid_items(args):
    with pytest.raises(ValueError):
        PlanItem(*args)


def test_a_planner_building_an_invalid_plan_fails_the_run():
    def planner(state, tools):
        return Decision.finish("x", plan=[P("a", "x", "in_progress"), P("b", "y", "in_progress")])

    state = Agent([search_tool()], planner).run("g")
    assert state.status is RunStatus.FAILED
    assert state.stop_reason is StopReason.PLANNER_FAILED
