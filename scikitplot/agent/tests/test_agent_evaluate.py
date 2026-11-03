"""
evaluate_audit: numbers from the trail, refused on a broken trail.
"""

from __future__ import annotations

import pytest

from .. import (
    Agent,
    Approval,
    Decision,
    MemoryAuditSink,
    MemoryStore,
    PlanItem,
    Policy,
    StateError,
    delegate,
    evaluate_audit,
)
from .test_agent_loop import Script, save_tool, search_tool


def _mixed_trail():
    sink, store, log = MemoryAuditSink(), MemoryStore(), []
    # run 1: search, denied save (no write effect), finish
    Agent(
        [search_tool(), save_tool(log)],
        Script(
            Decision.act("search", {"query": "a"}, plan=[PlanItem("a", "find", "done")]),
            Decision.act("save", {"path": "x"}),
            Decision.finish("ok"),
        ),
        audit=sink,
    ).run("one", run_id="1" * 64)
    # run 2: a write approved after a pause
    agent = Agent(
        [save_tool(log)],
        Script(Decision.act("save", {"path": "y"}), lambda s: Decision.finish("saved")),
        policy=Policy(allow_effects={"read", "write"}),
        audit=sink,
        store=store,
    )
    paused = agent.run("two", run_id="2" * 64)
    agent.resume("2" * 64, approvals=[Approval(paused.pending.digest, True, "allen")])
    # run 3: a delegation (one child)
    child = Agent([search_tool()], Script(Decision.act("search", {"query": "c"}), Decision.finish("c")))
    Agent(
        [delegate("research", child)],
        Script(Decision.act("research", {"goal": "dig"}), Decision.finish("done")),
        audit=sink,
    ).run("three", run_id="3" * 64)
    return sink.records


def test_per_run_numbers():
    evaluation = evaluate_audit(_mixed_trail())
    runs = {r.goal: r for r in evaluation.runs}
    one = runs["one"]
    assert (one.status, one.decisions, one.tool_calls) == ("completed", 3, 1)
    assert one.outcomes == {"denied": 1, "executed": 1, "finished": 1}
    assert one.denials == {"permission_denied": 1}
    assert one.plan_changes == 1
    two = runs["two"]
    assert (two.approvals_requested, two.approvals_granted, two.status) == (1, 1, "completed")
    three = runs["three"]
    assert three.subruns == 1
    (child,) = [r for r in evaluation.runs if r.parent == three.run_id]
    assert child.status == "completed" and child.tool_calls == 1


def test_summary_excludes_sub_runs():
    summary = evaluate_audit(_mixed_trail()).summary
    assert summary["n_runs"] == 3 and summary["n_ended"] == 3
    assert summary["completion_rate"] == 1.0
    assert summary["approval_grant_rate"] == 1.0
    assert summary["denial_rate"] == pytest.approx(1 / 7)  # 3 + 2 + 2 steps
    assert summary["tool_failure_rate"] == 0.0


def test_a_paused_run_has_not_ended():
    sink, log = MemoryAuditSink(), []
    Agent(
        [save_tool(log)],
        Script(Decision.act("save", {"path": "y"})),
        policy=Policy(allow_effects={"read", "write"}),
        audit=sink,
    ).run("paused")
    (run,) = evaluate_audit(sink.records).runs
    assert run.status is None and run.cost is None
    assert evaluate_audit(sink.records).summary["completion_rate"] is None


def test_a_broken_trail_is_refused():
    records = [dict(r) for r in _mixed_trail()]
    records[3] = {**records[3], "time": records[3]["time"] + 1}
    with pytest.raises(StateError, match="does not verify"):
        evaluate_audit(records)


def test_an_empty_trail():
    evaluation = evaluate_audit([])
    assert evaluation.runs == () and evaluation.summary["n_runs"] == 0
    assert evaluation.to_dict()["summary"]["mean_decisions"] is None
