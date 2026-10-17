"""
R5-1 gate: a single agent with tools, on a fixed task set, under its gates.

Run: ``python bench_r5a.py <repository> faa7665 > bench_r5a.out``.

Corpus and queries are those of the R4-3 benchmark: the corpus user guide at
a pinned commit (``argv[2]``), chunked by ``recursive`` (500), BM25 search
(``RetrievalIndex``, keyword). A task is the first half of every 4th sentence;
it succeeds when the agent's answer is the id of a chunk whose context covers
the whole sentence.

Tools (``read``, scope ``corpus:read``): ``search(query, top_k)`` returns ids
(cost 1); ``read(doc_id)`` returns a chunk's text (cost 0.25). A ``save``
tool (``write``) is used by the last check only. Planners, deterministic rules:

* ``one_shot``: search once, answer the first id;
* ``verify``: search, read the hits in order, answer the first whose text
  holds the query verbatim (the first id if none does);
* ``verify_k3``: as ``verify``, reading at most 3 hits.

Reported per planner: success, mean decisions, tool calls and cost, runs whose
audit chain verifies, and runs where every executed call has its ``tool_call``
record. Then: without the scope every action is denied and no tool runs; a
``write`` tool without an approver pauses the run and writes nothing.
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import shutil
import subprocess  # noqa: S404 - git, with fixed arguments
import sys
import tempfile

from scikitplot.agent import (
    Agent,
    Decision,
    Limits,
    MemoryAuditSink,
    Policy,
    RunStatus,
    StepOutcome,
    Tool,
    verify_audit,
)
from scikitplot.corpus import (
    BuilderConfig,
    CorpusBuilder,
    RetrievalConfig,
    RetrievalIndex,
)

GUIDE = "docs/source/user_guide/corpus"
GIT = shutil.which("git") or "git"
SENTENCE = re.compile(r"(?<=[.!?])\s+")
PARAGRAPH = re.compile(r"\n\s*\n")
MIN_CHARS = 40
EVERY = 4
CONFIG = RetrievalConfig(match_mode="keyword", top_k=10)
READ_POLICY = Policy(granted_scopes={"corpus:read"})


def pinned_guide(
    root: pathlib.Path, commit: str, guide: str = GUIDE
) -> list[pathlib.Path]:
    """Write the ``.rst`` pages of ``guide`` at ``commit`` to a temporary folder."""
    names = subprocess.run(  # noqa: S603
        [
            GIT,
            "-C",
            str(root),
            "ls-tree",
            "--name-only",
            f"{commit}:{guide}",
        ],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    folder = pathlib.Path(tempfile.mkdtemp())
    for name in sorted(n for n in names if n.endswith(".rst")):
        blob = subprocess.run(  # noqa: S603
            [GIT, "-C", str(root), "show", f"{commit}:{guide}/{name}"],  # noqa: S607
            check=True,
            capture_output=True,
        ).stdout
        (folder / name).write_bytes(blob)
    return sorted(folder.glob("*.rst"))


def sentences(text: str) -> list[tuple[int, int]]:
    """Sentence spans of at least MIN_CHARS characters inside one paragraph."""
    out, pos = [], 0
    for match in SENTENCE.finditer(text):
        out.append((pos, match.start()))
        pos = match.end()
    out.append((pos, len(text)))
    return [
        (s, e)
        for s, e in out
        if e - s >= MIN_CHARS and not PARAGRAPH.search(text, s, e)
    ]


def build(files):
    """Chunk the pages, index them, and make the task set."""
    docs = []
    for path in files:
        config = BuilderConfig(
            chunker="recursive",
            chunker_kwargs={"chunk_size": 500},
            normalize=False,
            filter_kwargs={"min_words": 0, "min_chars": 0},
        )
        docs += CorpusBuilder(config).build(path).documents
    index = RetrievalIndex(CONFIG)
    index.build(docs)
    tasks = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for k, (s, e) in enumerate(sentences(text)):
            if k % EVERY:
                continue
            words = text[s:e].split()
            query = " ".join(words[: max(1, len(words) // 2)])
            relevant = {
                d.doc_id
                for d in docs
                if pathlib.Path(str(d.input_path)).name == path.name
                and d.char_start is not None
                and d.char_start <= s
                and e <= d.char_end
            }
            tasks.append((query, relevant))
    return docs, index, tasks


def object_schema(**properties):
    """Return a closed object schema whose properties are all required but ``top_k``."""
    return {
        "type": "object",
        "properties": properties,
        "required": [p for p in properties if p != "top_k"],
        "additionalProperties": False,
    }


def make_tools(index, by_id, executed, saved):
    """Return the tools; ``executed`` and ``saved`` record their calls."""

    def search(query, top_k=10):
        executed.append("search")
        config = RetrievalConfig(match_mode="keyword", top_k=top_k)
        return [h.doc.doc_id for h in index.search(query, config=config)]

    def read(doc_id):
        executed.append("read")
        return by_id[doc_id].text

    def save(text):
        saved.append(text)
        return "saved"

    return [
        Tool(
            "search",
            search,
            parameters=object_schema(
                query={"type": "string", "minLength": 1},
                top_k={"type": "integer", "minimum": 1, "maximum": 10},
            ),
            scopes={"corpus:read"},
            cost=1.0,
        ),
        Tool(
            "read",
            read,
            parameters=object_schema(doc_id={"type": "string"}),
            scopes={"corpus:read"},
            cost=0.25,
        ),
        Tool(
            "save",
            save,
            parameters=object_schema(text={"type": "string"}),
            effect="write",
        ),
    ]


def _tool(step):
    return None if step.decision.action is None else step.decision.action.tool


def planner_for(max_reads, *, then_save=False):
    """Return a rule planner: search, read up to ``max_reads`` hits, answer."""

    def planner(state, tools):
        query, steps = state.goal, state.steps
        if not steps:
            return Decision.act("search", {"query": query})
        ids = steps[0].result or []
        if not ids:
            return Decision.finish(None)
        reads = [s for s in steps[1:] if _tool(s) == "read"]
        answer = next(
            (
                s.decision.action.arguments["doc_id"]
                for s in reads
                if s.outcome is StepOutcome.EXECUTED and query in s.result
            ),
            None,
        )
        if answer is None:
            if len(reads) < min(max_reads, len(ids)):
                return Decision.act("read", {"doc_id": ids[len(reads)]})
            answer = ids[0]
        if then_save and not any(_tool(s) == "save" for s in steps):
            return Decision.act("save", {"text": answer})
        return Decision.finish(answer)

    return planner


def main(root: pathlib.Path, commit: str) -> dict:
    """Run every check and return the report."""
    docs, index, tasks = build(pinned_guide(root, commit))
    executed, saved = [], []
    tools = make_tools(index, {d.doc_id: d for d in docs}, executed, saved)
    report = {"tasks": len(tasks)}
    for label, planner in (
        ("one_shot", planner_for(0)),
        ("verify", planner_for(10)),
        ("verify_k3", planner_for(3)),
    ):
        ok = steps = calls = cost = chains = matched = 0
        for query, relevant in tasks:
            sink = MemoryAuditSink()
            executed.clear()
            agent = Agent(
                tools,
                planner,
                policy=READ_POLICY,
                limits=Limits(max_steps=16),
                audit=sink,
            )
            state = agent.run(query)
            if state.status is not RunStatus.COMPLETED:
                raise AssertionError(state.to_dict())
            ok += state.answer in relevant
            steps += state.spent.steps
            calls += state.spent.tool_calls
            cost += state.spent.cost
            chains += verify_audit(sink.records) == []
            audited = sum(r["kind"] == "tool_call" for r in sink.records)
            matched += audited == len(executed) == state.spent.tool_calls
        n = len(tasks)
        report[label] = {
            "success": round(ok / n, 4),
            "decisions": round(steps / n, 2),
            "tool_calls": round(calls / n, 2),
            "cost": round(cost / n, 2),
            "audit_chains_whole": chains,
            "calls_all_audited": matched,
        }
    denied = executions = 0
    for query, _ in tasks:
        executed.clear()
        state = Agent(tools, planner_for(10), limits=Limits(max_steps=4)).run(query)
        denied += all(
            s.outcome is StepOutcome.DENIED
            for s in state.steps
            if s.decision.kind == "act"
        )
        executions += len(executed)
    report["no_scope"] = {"runs_all_denied": denied, "tool_executions": executions}
    paused = 0
    policy = Policy(granted_scopes={"corpus:read"}, allow_effects={"read", "write"})
    for query, _ in tasks:
        state = Agent(tools, planner_for(10, then_save=True), policy=policy).run(query)
        paused += (
            state.status is RunStatus.AWAITING_APPROVAL
            and state.pending.action.tool == "save"
        )
    report["write_without_approver"] = {"paused": paused, "written": len(saved)}
    return report


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    result = main(pathlib.Path(sys.argv[1]), sys.argv[2])
    sys.stdout.write(json.dumps(result, indent=1) + "\n")
