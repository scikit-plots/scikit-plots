# scikitplot/agent/tests/test_agent_workflow.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Workflow graphs (W1): static checks, the planner, fan-out, replay and resume.
"""

from __future__ import annotations

import threading

import pytest

from .. import (
    Agent,
    Approval,
    Branch,
    Decision,
    Edge,
    ErrorCode,
    FanOut,
    Finish,
    Limits,
    MemoryArtifactStore,
    MemoryStore,
    Policy,
    RunState,
    RunStatus,
    StepOutcome,
    StopReason,
    Task,
    Tool,
    WorkflowError,
    WorkflowGraph,
    verify_audit,
)
from .test_agent_loop import PATH, QUERY, RUN, save_tool, search_tool

#: Loop bound the tests set and assert.
BOUND = 3
#: Count at which the conditional-edge test branches.
EVEN = 2


#: One tool object: a graph refuses two different tools with one name.
COUNT = Tool("count", lambda: 1)


def _count_tool():
    return COUNT


def _counting_graph(bound=BOUND, **finish):
    return WorkflowGraph(
        "tick",
        [
            Task(
                "tick", _count_tool(), into="n", max_visits=bound, on_exhausted="done"
            ),
            Finish("done", answer=lambda view: view.channels["n"], **finish),
        ],
        [Edge("tick", "tick")],
        channels={"n": "add"},
    )


def _run(graph, goal="g", **agent):
    agent = Agent(graph.tools(), graph.planner(), **agent)
    return agent, agent.run(goal, run_id=RUN)


# ------------------------------------------------------------ simple: a sequence


class TestSequence:
    def test_a_sequence_runs_its_nodes_in_order_and_answers(self):
        graph = WorkflowGraph.sequence(
            Task(
                "find",
                search_tool(),
                arguments=lambda v: {"query": v.goal},
                into="hits",
            ),
            Finish("answer", answer=lambda v: v.channels["hits"]),
            channels={"hits": "append"},
        )
        agent, state = _run(graph, goal="spans")
        assert state.status is RunStatus.COMPLETED
        assert state.answer == [{"hits": ["doc:spans"]}]
        assert [s.outcome for s in state.steps] == [
            StepOutcome.EXECUTED,
            StepOutcome.FINISHED,
        ]
        assert verify_audit(agent.audit.records) == []

    def test_every_call_still_passes_the_gates(self):
        log = []
        graph = WorkflowGraph.sequence(
            Task("write", save_tool(log), arguments={"path": "r.md"}),
            Finish("done", answer=lambda v: v.last.outcome.value),
        )
        _agent, state = _run(graph)  # Policy() allows read effects only
        assert log == []
        assert state.steps[0].error.code is ErrorCode.PERMISSION_DENIED
        assert state.answer == "denied"

    def test_a_finish_without_answer_returns_every_channel(self):
        graph = WorkflowGraph.sequence(
            Finish("done"), channels={"a": "replace"}, initial={"a": 1}
        )
        assert _run(graph)[1].answer == {"a": 1}

    def test_a_sequence_must_end_at_a_finish(self):
        with pytest.raises(WorkflowError, match="last node"):
            WorkflowGraph.sequence(Task("t", _count_tool()))
        with pytest.raises(WorkflowError, match="at least one"):
            WorkflowGraph.sequence()


# ------------------------------------------------------ complex: edges and loops


class TestEdgesAndLoops:
    def test_a_bounded_loop_ends_at_its_exhaustion_node(self):
        _agent, state = _run(_counting_graph())
        assert state.status is RunStatus.COMPLETED
        assert state.answer == BOUND
        assert len(state.steps) == BOUND + 1

    def test_conditional_edges_are_tried_in_order(self):
        graph = WorkflowGraph(
            "tick",
            [
                Task(
                    "tick", _count_tool(), into="n", max_visits=10, on_exhausted="more"
                ),
                Finish("even", answer=lambda v: ["even", v.channels["n"]]),
                Finish("more"),
            ],
            [
                Edge("tick", "even", when=lambda v: v.channels["n"] == EVEN),
                Edge("tick", "tick"),
            ],
            channels={"n": "add"},
        )
        assert _run(graph)[1].answer == ["even", EVEN]

    def test_a_when_that_is_not_bool_is_refused(self):
        graph = WorkflowGraph(
            "t",
            [Task("t", _count_tool()), Finish("f")],
            [Edge("t", "f", when=lambda v: 1), Edge("t", "f")],
        )
        _agent, state = _run(graph)
        assert state.stop_reason is StopReason.PLANNER_FAILED
        assert "not bool" in state.error.message

    def test_a_failed_step_can_be_routed_on(self):
        boom = Tool("boom", lambda: 1 / 0)
        graph = WorkflowGraph(
            "try",
            [
                Task("try", boom),
                Finish("ok", answer=lambda v: "ok"),
                Finish("bad", answer=lambda v: v.last.error_code),
            ],
            [Edge("try", "ok", when=lambda v: v.last.ok), Edge("try", "bad")],
        )
        assert _run(graph)[1].answer == "tool_failed"

    def test_reaching_a_bound_without_exhaustion_node_fails_the_run(self):
        graph = WorkflowGraph(
            "tick",
            [Task("tick", _count_tool(), max_visits=2), Finish("never")],
            [Edge("tick", "never", when=lambda v: False), Edge("tick", "tick")],
        )
        _agent, state = _run(graph)
        assert state.stop_reason is StopReason.PLANNER_FAILED
        assert "max_visits=2" in state.error.message

    def test_a_rejected_answer_follows_on_rejected(self):
        graph = WorkflowGraph(
            "tick",
            [
                Task(
                    "tick",
                    _count_tool(),
                    into="n",
                    max_visits=5,
                    on_exhausted="give_up",
                ),
                Finish("done", answer=lambda v: v.channels["n"], on_rejected="tick"),
                Finish("give_up", answer=lambda v: "gave up"),
            ],
            [Edge("tick", "done")],
            channels={"n": "add"},
        )
        _agent, state = _run(graph, done=lambda answer, state: answer >= BOUND)
        assert state.answer == BOUND
        outcomes = [s.outcome for s in state.steps]
        assert outcomes.count(StepOutcome.REJECTED) == BOUND - 1

    def test_a_rejected_answer_without_on_rejected_fails(self):
        graph = WorkflowGraph.sequence(Finish("done", answer=lambda v: 0))
        _agent, state = _run(graph, done=lambda answer, state: False)
        assert state.stop_reason is StopReason.PLANNER_FAILED
        assert "rejected" in state.error.message


# ------------------------------------------------------------- static checks


class TestStaticChecks:
    @pytest.mark.parametrize(
        ("nodes", "edges", "match"),
        [
            ([Finish("f")], [], "start node"),
            ([Task("s", _count_tool()), Finish("f")], [], "no outgoing edge"),
            (
                [Task("s", _count_tool()), Finish("f"), Finish("g")],
                [Edge("s", "f")],
                "cannot be reached",
            ),
            (
                [
                    Task("s", _count_tool()),
                    Task("t", _count_tool(), max_visits=2),
                    Finish("f"),
                ],
                [Edge("s", "f", when=lambda v: False), Edge("s", "t"), Edge("t", "t")],
                "no Finish",
            ),
            (
                [Task("s", _count_tool()), Finish("f")],
                [Edge("s", "s", when=lambda v: True), Edge("s", "f")],
                "has no node with max_visits",
            ),
            (
                [Task("s", _count_tool()), Finish("f")],
                [Edge("s", "f"), Edge("s", "s")],
                "could never be taken",
            ),
            (
                [Task("s", _count_tool()), Finish("f")],
                [Edge("s", "x")],
                "'x' is not a node",
            ),
            (
                [Task("s", _count_tool(), into="nope"), Finish("f")],
                [Edge("s", "f")],
                "is not a channel",
            ),
        ],
    )
    def test_malformed_graphs_are_refused(self, nodes, edges, match):
        start = nodes[0].name if nodes[0].name != "f" else "s"
        with pytest.raises(WorkflowError, match=match):
            WorkflowGraph(start, nodes, edges)

    def test_an_exhaustion_that_leads_back_is_refused(self):
        with pytest.raises(WorkflowError, match="leads back"):
            WorkflowGraph(
                "a",
                [
                    Task("a", _count_tool(), max_visits=2, on_exhausted="b"),
                    Task("b", _count_tool()),
                    Finish("f"),
                ],
                [Edge("a", "b"), Edge("b", "f", when=lambda v: False), Edge("b", "a")],
            )

    def test_a_cycle_around_a_bounded_node_is_refused(self):
        """Removing bounded nodes must leave no cycle, not just one per SCC."""
        with pytest.raises(WorkflowError, match=r"\['b', 'c'\]"):
            WorkflowGraph(
                "a",
                [
                    Task("a", _count_tool(), max_visits=2, on_exhausted="f"),
                    Task("b", _count_tool()),
                    Task("c", _count_tool()),
                    Finish("f"),
                ],
                [
                    Edge("a", "b"),
                    Edge("b", "c"),
                    Edge("c", "b", when=lambda v: True),
                    Edge("c", "a"),
                ],
            )

    def test_two_tools_with_one_name_are_refused(self):
        with pytest.raises(WorkflowError, match="two different tools"):
            WorkflowGraph.sequence(
                Task("a", _count_tool()),
                Task("b", Tool("count", lambda: 2)),
                Finish("f"),
            )

    def test_a_concurrent_fanout_name_must_not_be_a_tool_name(self):
        count = _count_tool()
        with pytest.raises(WorkflowError, match="already used"):
            WorkflowGraph.sequence(
                Task("first", count),
                FanOut(
                    "count",
                    [Branch("x", search_tool()), Branch("y", Tool("other", lambda: 2))],
                    mode="concurrent",
                ),
                Finish("f"),
            )

    @pytest.mark.parametrize(
        ("build", "match"),
        [
            (lambda: Task("bad name", _count_tool()), "must match"),
            (lambda: Task("t", "not a tool"), "must be a Tool"),
            (lambda: Task("t", _count_tool(), max_visits=0), "max_visits"),
            (lambda: Task("t", _count_tool(), on_exhausted="f"), "needs max_visits"),
            (
                lambda: Task("t", _count_tool(), into="a", update=lambda v, r: {}),
                "not both",
            ),
            (lambda: FanOut("f", [Branch("a", _count_tool())]), "two or more"),
            (
                lambda: FanOut("f", [Branch("a", _count_tool())] * 2),
                "unique",
            ),
            (
                lambda: FanOut(
                    "f",
                    [Branch("a", _count_tool()), Branch("b", _count_tool())],
                    mode="parallel",
                ),
                "mode",
            ),
            (lambda: Task("t", _count_tool(), arguments=[1]), "mapping"),
        ],
    )
    def test_malformed_nodes_are_refused(self, build, match):
        with pytest.raises(WorkflowError, match=match):
            build()

    def test_unknown_reducers_and_initial_values_are_refused(self):
        with pytest.raises(WorkflowError, match="reducer must be one of"):
            WorkflowGraph.sequence(Finish("f"), channels={"a": "sum"})
        with pytest.raises(WorkflowError, match="unknown channel"):
            WorkflowGraph.sequence(Finish("f"), initial={"a": 1})

    def test_the_fingerprint_covers_structure(self):
        assert _counting_graph().fingerprint == _counting_graph().fingerprint
        assert _counting_graph(2).fingerprint != _counting_graph(3).fingerprint


# ------------------------------------------------------------------- fan-out


def _fanout_graph(mode, log=None, **fanout):
    def tool(name, value):
        def run(query):
            if log is not None:
                log.append(name)
            return {name: f"{value}:{query}"}

        return Tool(name, run, parameters=QUERY)

    return WorkflowGraph.sequence(
        FanOut(
            "both",
            [
                Branch(
                    "lexical", tool("lex", "L"), arguments=lambda v: {"query": v.goal}
                ),
                Branch(
                    "dense", tool("den", "D"), arguments=lambda v: {"query": v.goal}
                ),
            ],
            mode=mode,
            into="joined",
            **fanout,
        ),
        Finish("answer", answer=lambda v: v.channels["joined"]),
        channels={"joined": "replace"},
    )


class TestFanOut:
    @pytest.mark.parametrize("mode", ["ordered", "concurrent"])
    def test_both_modes_join_in_declared_order(self, mode):
        _agent, state = _run(_fanout_graph(mode), goal="q")
        assert state.status is RunStatus.COMPLETED
        assert list(state.answer) == ["lexical", "dense"]
        assert state.answer == {"lexical": {"lex": "L:q"}, "dense": {"den": "D:q"}}

    def test_ordered_is_one_audited_step_per_branch(self):
        _agent, state = _run(_fanout_graph("ordered"))
        assert [s.decision.action.tool for s in state.steps[:2]] == ["lex", "den"]

    def test_concurrent_is_one_step_judged_as_its_branches_together(self):
        graph = _fanout_graph("concurrent")
        (tool,) = graph.tools()
        assert tool.name == "both"
        assert tool.cost == 2.0  # noqa: PLR2004
        _agent, state = _run(graph)
        assert len(state.steps) == 2  # noqa: PLR2004
        assert state.spent.cost == 2.0  # noqa: PLR2004

    def test_concurrent_branches_really_overlap(self):
        barrier = threading.Barrier(2, timeout=5)

        def wait(query):
            barrier.wait()  # deadlocks (then times out) unless both run at once
            return query

        graph = WorkflowGraph.sequence(
            FanOut(
                "pair",
                [
                    Branch("a", Tool("wa", wait, parameters=QUERY), {"query": "a"}),
                    Branch("b", Tool("wb", wait, parameters=QUERY), {"query": "b"}),
                ],
                mode="concurrent",
                into="out",
            ),
            Finish("f", answer=lambda v: v.channels["out"]),
            channels={"out": "replace"},
        )
        assert _run(graph)[1].answer == {"a": "a", "b": "b"}

    def test_a_failing_concurrent_branch_fails_the_whole_step(self):
        boom = Tool("boom", lambda: 1 / 0)
        graph = WorkflowGraph.sequence(
            FanOut(
                "pair",
                [Branch("ok", _count_tool()), Branch("bad", boom)],
                mode="concurrent",
                into="out",
            ),
            Finish("f", answer=lambda v: [v.last.outcome.value, v.channels["out"]]),
            channels={"out": "replace"},
        )
        state = _run(graph)[1]
        assert state.answer == ["failed", None]
        assert "'bad'" in state.steps[0].error.message

    def test_a_concurrent_write_branch_needs_the_write_policy(self):
        log = []
        graph = WorkflowGraph.sequence(
            FanOut(
                "pair",
                [
                    Branch("w", save_tool(log), {"path": "p"}),
                    Branch("r", _count_tool()),
                ],
                mode="concurrent",
            ),
            Finish("f", answer=lambda v: v.last.error_code),
        )
        assert _run(graph)[1].answer == "permission_denied"
        assert log == []

    def test_an_ordered_branch_denial_keeps_the_others(self):
        log = []
        graph = WorkflowGraph.sequence(
            FanOut(
                "pair",
                [
                    Branch("w", save_tool(log), {"path": "p"}),
                    Branch("r", _count_tool()),
                ],
                into="out",
            ),
            Finish(
                "f",
                answer=lambda v: [v.channels["out"], v.branches["w"].outcome.value],
            ),
            channels={"out": "replace"},
        )
        assert _run(graph)[1].answer == [{"r": 1}, "denied"]


# ---------------------------------------------- replay, resume and determinism


class TestReplay:
    def test_replay_reproduces_the_position_digest(self):
        graph = _counting_graph()
        _agent, state = _run(graph)
        first = graph.replay(state)
        again = graph.replay(RunState.from_dict(state.to_dict()))
        assert first.finished
        assert again.finished
        assert first.digest() == again.digest()
        assert first.visits == {"tick": BOUND, "done": 1}

    def test_a_state_from_another_planner_is_refused(self):
        other = Agent(
            [_count_tool()],
            lambda s, t: Decision.act("count") if not s.steps else Decision.finish(0),
        )
        state = other.run("g", run_id=RUN)
        graph = WorkflowGraph.sequence(
            Task("t", search_tool(), {"query": "x"}), Finish("f")
        )
        with pytest.raises(WorkflowError, match="not produced by this graph"):
            graph.replay(state)

    def test_a_paused_write_resumes_where_it_was(self):
        log, store = [], MemoryStore()
        graph = WorkflowGraph.sequence(
            Task("look", search_tool(), {"query": "x"}, into="seen"),
            Task("write", save_tool(log), {"path": "out.md"}),
            Finish(
                "f", answer=lambda v: [len(v.channels["seen"]), v.last.outcome.value]
            ),
            channels={"seen": "append"},
        )
        agent = Agent(
            graph.tools(),
            graph.planner(),
            policy=Policy(allow_effects={"read", "write"}),
            store=store,
        )
        paused = agent.run("g", run_id=RUN)
        assert paused.status is RunStatus.AWAITING_APPROVAL
        before = graph.replay(paused)
        assert before.node == "write"
        done = agent.resume(
            RUN, approvals=[Approval(paused.pending.digest, True, "allen")]
        )
        assert done.answer == [1, "executed"]
        assert log == ["out.md"]
        assert verify_audit(agent.audit.records) == []

    def test_offloaded_results_need_the_artifact_store(self):
        artifacts = MemoryArtifactStore()
        graph = WorkflowGraph.sequence(
            Task("look", search_tool(), {"query": "x"}, into="seen"),
            Finish("f", answer=lambda v: v.channels["seen"]),
            channels={"seen": "replace"},
        )
        agent = Agent(
            graph.tools(),
            graph.planner(artifacts=artifacts),
            artifacts=artifacts,
            offload_results_over=1,
        )
        state = agent.run("g", run_id=RUN)
        assert state.answer == {"hits": ["doc:x"]}
        assert state.steps[0].result_ref is not None
        with pytest.raises(WorkflowError, match="offloaded"):
            graph.replay(state)
        assert graph.replay(state, artifacts=artifacts).finished

    def test_the_agent_step_limit_still_applies(self):
        graph = _counting_graph(bound=50)
        _agent, state = _run(graph, limits=Limits(max_steps=4))
        assert state.status is RunStatus.STOPPED
        assert len(state.steps) == 4  # noqa: PLR2004

    def test_a_graph_drives_a_token_limited_agent(self):
        """Graph decisions report zero usage, so max_tokens can be enforced."""
        _agent, state = _run(
            _counting_graph(), limits=Limits(max_steps=10, max_tokens=5)
        )
        assert state.status is RunStatus.COMPLETED


def test_path_schema_is_shared_with_the_loop_tests():
    assert PATH["required"] == ["path"]
