"""
Sub-runs: bounded by the parent's budget, no extra privilege, one audit trail.
"""

from __future__ import annotations

import pytest

from .. import (
    Agent,
    Decision,
    ErrorCode,
    Limits,
    Policy,
    RunStatus,
    StepOutcome,
    Tool,
    delegate,
    verify_audit,
)
from .._delegate import child_run_id
from .test_agent_loop import QUERY, RUN, Script, save_tool, search_tool


def _searching_child(calls=3, **tool):
    """A child that searches ``calls`` times, then answers."""

    def planner(state, tools):
        if len(state.steps) < calls:
            return Decision.act("search", {"query": f"q{len(state.steps)}"})
        return Decision.finish(f"found {len(state.steps)}")

    return Agent([search_tool(**tool)], planner)


def _parent(child_tool, limits=None, policy=None, then=None):
    planner = Script(
        Decision.act(child_tool.name, {"goal": "look it up"}),
        then or (lambda s: Decision.finish(s.steps[-1].result)),
    )
    return Agent([child_tool], planner, limits=limits, policy=policy)


def test_a_child_runs_and_its_answer_is_the_result():
    parent = _parent(delegate("research", _searching_child(2)))
    state = parent.run("g", run_id=RUN)
    assert state.status is RunStatus.COMPLETED
    result = state.answer
    assert result["status"] == "completed" and result["answer"] == "found 2"
    assert result["run_id"] == child_run_id(RUN, state.steps[0].key)


def test_child_spend_is_charged_to_the_parent():
    parent = _parent(delegate("research", _searching_child(3, cost=2.0)))
    state = parent.run("g")
    assert state.spent.tool_calls == 1 + 3  # the delegation and the child's calls
    assert state.spent.cost == 6.0


@pytest.mark.parametrize("calls, cost", [(4, 1.0), (10, 3.0)])
def test_the_parent_limits_are_never_exceeded(calls, cost):
    limits = Limits(max_steps=4, max_tool_calls=4, max_cost=7.0)
    parent = _parent(
        delegate("research", _searching_child(calls, cost=cost), max_steps=20), limits=limits
    )
    state = parent.run("g")
    assert state.spent.tool_calls <= 4
    assert state.spent.cost <= 7.0
    child = state.steps[0].result["spent"]
    # The delegation itself used one call, so the child had three; a call
    # over its cap is a denied step of the child, not an overspend.
    assert child["tool_calls"] <= 3 and child["cost"] <= 7.0


def test_a_child_cap_below_the_parent_budget_applies():
    parent = _parent(
        delegate("research", _searching_child(9), max_tool_calls=2),
        limits=Limits(max_tool_calls=50),
    )
    state = parent.run("g")
    assert state.steps[0].result["spent"]["tool_calls"] == 2


def test_no_budget_left_fails_the_step_without_starting_a_child():
    # Charging the delegation's own call uses the last tool call, so the
    # child would have none.
    first = Tool("search", lambda query: "x", parameters=QUERY)
    child_tool = delegate("research", _searching_child(1))
    planner = Script(
        Decision.act("search", {"query": "a"}),
        Decision.act("research", {"goal": "more"}),
        Decision.finish("end"),
    )
    parent = Agent([first, child_tool], planner, limits=Limits(max_tool_calls=2))
    state = parent.run("g")
    step = state.steps[1]
    assert step.outcome is StepOutcome.FAILED
    assert "no budget left" in step.error.message
    assert not [r for r in parent.audit.records if r["kind"] == "subrun"]


def test_a_delegate_has_no_privilege_the_parent_lacks():
    scoped = delegate("research", _searching_child(1, scopes={"corpus:read"}))
    assert scoped.scopes == frozenset({"corpus:read"})
    state = _parent(scoped, then=Decision.finish("gave up")).run("g")
    assert state.steps[0].outcome is StepOutcome.DENIED
    assert state.steps[0].error.code is ErrorCode.SCOPE_DENIED


def test_a_child_that_can_write_needs_the_parents_approval():
    log = []
    writer = Agent(
        [save_tool(log)],
        Script(Decision.act("save", {"path": "x"}), Decision.finish("saved")),
        policy=Policy(allow_effects={"read", "write"}),
    )
    tool = delegate("writer", writer)
    assert tool.effect.value == "write"
    paused = _parent(tool, policy=Policy(allow_effects={"read", "write"})).run("g")
    assert paused.status is RunStatus.AWAITING_APPROVAL
    assert log == []  # nothing ran before a person approved the delegation


def test_one_trail_links_parent_and_child():
    parent = _parent(delegate("research", _searching_child(2)))
    state = parent.run("g", run_id=RUN)
    records = parent.audit.records
    assert verify_audit(records) == []
    child_id = state.answer["run_id"]
    (link,) = [r for r in records if r["kind"] == "subrun"]
    assert link["run_id"] == RUN and link["data"]["run_id"] == child_id
    child_records = [r for r in records if r["run_id"] == child_id]
    assert child_records and child_records[-1]["kind"] == "run_ended"
    assert link["data"]["audit_head"]["seq"] == child_records[-1]["seq"]


def test_nested_delegation_stays_inside_the_outer_budget():
    inner = delegate("inner", _searching_child(5))
    middle = Agent(
        [inner],
        Script(Decision.act("inner", {"goal": "deeper"}), lambda s: Decision.finish("ok")),
    )
    outer = _parent(delegate("middle", middle), limits=Limits(max_tool_calls=4))
    state = outer.run("g")
    assert state.spent.tool_calls <= 4
    assert verify_audit(outer.audit.records) == []


def test_delegate_refusals():
    with pytest.raises(TypeError, match="must be an Agent"):
        delegate("x", object())
    with pytest.raises(ValueError, match="no tools"):
        delegate("x", Agent([], Script(Decision.finish("x"))))
    with pytest.raises(ValueError):
        delegate("x", _searching_child(1), max_steps=0)
