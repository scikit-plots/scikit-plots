"""
Offloaded tool results: content-addressed, linked to the audit, verified on read.
"""

from __future__ import annotations

import json

import pytest

from .. import (
    Agent,
    Approval,
    Decision,
    DirectoryArtifactStore,
    JsonFileStore,
    MemoryArtifactStore,
    MemoryStore,
    Policy,
    RunState,
    RunStatus,
    StateError,
    Tool,
    verify_audit,
)
from .._canonical import canonical_json, digest
from .test_agent_loop import QUERY, RUN, Script, save_tool, search_tool

BIG = {"rows": [{"id": i, "text": "x" * 40} for i in range(200)]}


def big_tool():
    return Tool("search", lambda query: BIG, parameters=QUERY)


def _finish(state):
    ref = state.observations()[-1]["result_ref"]
    return Decision.finish("inline" if ref is None else ref["digest"])


def _agent(artifacts=None, **kwargs):
    planner = Script(
        Decision.act("search", {"query": "a"}),
        Decision.act("search", {"query": "b"}),
        _finish,
    )
    return Agent([big_tool()], planner, artifacts=artifacts, **kwargs)


@pytest.mark.parametrize("make", [MemoryArtifactStore, "dir"])
def test_a_large_result_is_offloaded_and_linked_to_the_audit(make, tmp_path):
    store = DirectoryArtifactStore(tmp_path / "a") if make == "dir" else make()
    agent = _agent(store, offload_results_over=1000)
    state = agent.run("g", run_id=RUN)
    assert state.status is RunStatus.COMPLETED
    step = state.steps[0]
    assert step.result is None
    assert step.result_ref == {
        "digest": digest(BIG),
        "bytes": len(canonical_json(BIG).encode("utf-8")),
    }
    assert store.get(step.result_ref["digest"]) == BIG
    audited = [r["data"]["result_digest"] for r in agent.audit.records if r["kind"] == "step"]
    assert audited[0] == step.result_ref["digest"]
    assert verify_audit(agent.audit.records) == []
    # Same content twice: one artifact (content addressed, idempotent put).
    assert state.steps[1].result_ref == step.result_ref
    assert state.answer == digest(BIG)


def test_the_audit_digest_is_the_same_with_and_without_offloading():
    plain_agent = _agent(None)
    offloading = _agent(MemoryArtifactStore(), offload_results_over=0)
    a = [r["data"]["result_digest"] for r in plain_agent.audit.records]
    plain_agent.run("g", run_id=RUN)
    offloading.run("g", run_id=RUN)
    digests = [
        [r["data"]["result_digest"] for r in agent.audit.records if r["kind"] == "step"]
        for agent in (plain_agent, offloading)
    ]
    assert a == [] and digests[0][:2] == digests[1][:2]


def test_small_results_stay_inline_and_the_state_is_unchanged():
    planner = Script(Decision.act("search", {"query": "a"}), Decision.finish("ok"))
    state = Agent(
        [search_tool()], planner, artifacts=MemoryArtifactStore(), offload_results_over=1000
    ).run("g")
    assert state.steps[0].result == {"hits": ["doc:a"]}
    assert "result_ref" not in state.to_dict()["steps"][0]


def test_offloading_keeps_saved_states_small(tmp_path):
    inline = JsonFileStore(tmp_path / "inline")
    offload = JsonFileStore(tmp_path / "offload")
    _agent(None, store=inline).run("g", run_id=RUN)
    _agent(DirectoryArtifactStore(tmp_path / "art"), store=offload, offload_results_over=1000).run(
        "g", run_id=RUN
    )
    small = offload.path(RUN).stat().st_size
    large = inline.path(RUN).stat().st_size
    assert small * 10 < large


def test_resume_after_approval_keeps_references(tmp_path):
    log = []
    store = MemoryStore()
    artifacts = DirectoryArtifactStore(tmp_path)

    def save(path):
        log.append(path)
        return BIG

    planner = Script(
        Decision.act("save", {"path": "r.md"}),
        lambda s: Decision.finish("ok"),
    )
    tool = Tool(
        "save",
        save,
        parameters={"type": "object", "properties": {"path": {"type": "string"}},
                    "required": ["path"], "additionalProperties": False},
        effect="write",
    )
    agent = Agent(
        [tool], planner, policy=Policy(allow_effects={"read", "write"}), store=store,
        artifacts=artifacts, offload_results_over=100,
    )
    paused = agent.run("g", run_id=RUN)
    done = agent.resume(RUN, approvals=[Approval(paused.pending.digest, True, "allen")])
    assert done.status is RunStatus.COMPLETED
    ref = store.load(RUN).steps[0].result_ref
    assert artifacts.get(ref["digest"]) == BIG


def test_a_changed_artifact_is_refused(tmp_path):
    store = DirectoryArtifactStore(tmp_path)
    key = store.put({"a": 1})
    store.path(key).write_text('{"a":2}', encoding="utf-8")
    with pytest.raises(StateError, match="does not have this digest"):
        store.get(key)
    store.path(key).write_text("not json", encoding="utf-8")
    with pytest.raises(StateError, match="not JSON"):
        store.get(key)


def test_store_contracts(tmp_path):
    for store in (MemoryArtifactStore(), DirectoryArtifactStore(tmp_path)):
        assert store.put([1, 2]) == store.put([1, 2]) == digest([1, 2])
        assert store.has(digest([1, 2]))
        with pytest.raises(KeyError):
            store.get("0" * 64)
        for bad in ("../" + "0" * 61, "x", 3):
            with pytest.raises(ValueError):
                store.get(bad)
        with pytest.raises(ValueError):
            store.put(float("nan"))


def test_agent_parameter_refusals():
    with pytest.raises(ValueError, match="needs an artifact store"):
        Agent([search_tool()], Script(Decision.finish("x")), offload_results_over=10)
    with pytest.raises(ValueError, match="int >= 0"):
        Agent([search_tool()], Script(Decision.finish("x")), artifacts=MemoryArtifactStore(),
              offload_results_over=-1)
    with pytest.raises(TypeError, match="ArtifactStore"):
        Agent([search_tool()], Script(Decision.finish("x")), artifacts={})


def test_malformed_result_refs_are_refused_at_load():
    state = _agent(MemoryArtifactStore(), offload_results_over=0).run("g", run_id=RUN)
    good = state.to_dict()
    for change in (
        lambda s: s["steps"][0].__setitem__("result", {"x": 1}),  # both inline and ref
        lambda s: s["steps"][0]["result_ref"].__setitem__("digest", "z"),
        lambda s: s["steps"][0]["result_ref"].pop("bytes"),
        lambda s: s["steps"][0]["result_ref"].__setitem__("bytes", -1),
    ):
        data = json.loads(json.dumps(good))
        change(data)
        with pytest.raises(StateError):
            RunState.from_dict(data)
