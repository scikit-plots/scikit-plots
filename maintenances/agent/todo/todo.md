# `scikitplot.agent` maintenance: active tasks

The agent layer of round 33 (Run 5). The campaign's status and its open
decisions are kept in [`../../corpus/todo/todo.md`](../../corpus/todo/todo.md);
this file plans and records the increments `scikitplot.agent` owns.
Completed tasks are archived verbatim under
[`../_maintenance/history/`](../_maintenance/history/).

Rules that apply to every increment: [`skills/_engineering/`](../../../skills/_engineering/SKILL.md).

## Task: round 33 / Run 5 - agent layer (R5-1, R5-2, R5-3)

### Context
- Plan item (`01_verdict_and_run_plan.md` Run 5; design proposal in
  `31_rag_agent_roadmap.md` section 6): typed state; a tool registry with
  Validation, Permissions, Scope, Budget and Approval gates; stopping
  conditions (done criteria, max iterations, token and cost budget,
  timeout, human checkpoint); persisted, resumable state; an audit trail.
- Exists: `corpus/_agentic.py` (A1 routing, A2 bounded investigation,
  "Autonomy stops at A2"). The agent package is the A3/A4 layer above it:
  it calls retrieval as one tool and never re-implements it.
- Base: 4c (`2c283f3`). PR 865 fragments.

### Increments
- [x] 5a. `scikitplot.agent` MVP (R5-1)
  - Big picture. A caller-supplied **planner** (a model adapter or a rule)
    proposes the next step; the **runtime** alone decides whether it runs.
    Perceive (the planner reads the state) -> Reason/Plan (it returns a
    decision) -> Act (the gate chain, then the tool) -> Observe (the result
    or the denial is appended to the state) -> Reflect (the next planner
    call sees it). Every step is audited, and the state is saved after it.
  - Invariants:
    - I1 Termination is provable: every planner call and every tool call is
      counted against a finite limit (`max_steps` is required and finite).
    - I2 A denied action is an outcome in the state and the audit trail,
      never an exception lost in the loop.
    - I3 Least privilege by default: a policy that names nothing allows
      `read` tools with no scope, and nothing else.
    - I4 An approval is bound to the action's digest (run, step, tool,
      canonical arguments, effect, scopes, and the audit head before the
      request); at resume the digest is recomputed and every earlier gate is
      re-run, so a stale approval authorises nothing. A run id is used once,
      and with a store a resumed copy older than the stored state is refused.
    - I5 A recorded step is never redone. A non-idempotent call in flight at
      a crash is not repeated (`interrupted`); an idempotent one is started
      at most `MAX_ATTEMPTS` (3) times, each start gated and charged.
    - I6 Budgets are arithmetic and checked before each call (Rule 127): a
      call is charged when it starts; a running call is not interrupted; a
      tool's own timeout bounds it.
    - I7 A limit that cannot be enforced is refused: `max_tokens` with a
      planner that reports no usage fails the run (`USAGE_NOT_REPORTED`).
    - I8 State and audit are strict JSON (`allow_nan=False`); a tool result
      that is not is a tool failure, not a corrupt state file.
    - I9 Independence: standard library only; no `scikitplot` sibling at
      module level (joins TIER1 in `test_submodule_independence.py`).
  - Modules: `_errors.py` (`AgentError` record, error codes), `_canonical.py`
    (canonical JSON and digests), `_schema.py` (a declared JSON Schema
    subset; any other keyword is refused at registration), `_tools.py`
    (`ToolEffect`, `Tool`, `Toolset`), `_policy.py` (`Policy`, `Limits`,
    `GateKind`, `GateResult`, the chain), `_state.py` (`RunStatus`,
    `StopReason`, `Action`, `Decision`, `Step`, `RunState` with a schema
    version), `_audit.py` (hash-chained append-only records, memory and
    JSONL sinks, `verify_audit`), `_store.py` (`MemoryStore`,
    `JsonFileStore`: atomic replace, run ids `[0-9a-f]{64}`), `_loop.py`
    (`Agent`, `run`, `resume`, the Approval gate). `ApprovalRequest` and
    `Approval` live in `_state.py`, with the digests they are bound to.
  - Gate (no model needed, deterministic): a fixed task set over the corpus
    guide index with a rule planner, reporting success, steps, tool calls
    and cost per task; every tool call and every denial in the audit trail;
    tests for each gate's denial, each stop reason, resume after a pause,
    a stale approval, idempotent replay, audit tampering and state
    round-trip on Python 3.9.
  - [x] package (9 modules, standard library only), tests (270: schema
        subset against `jsonschema` on a case table and 15,000 random
        pairs; every gate's denial; every stop reason; approval, pause,
        resume, stale and replayed approvals; in-flight recovery; audit
        tampering; state consistency), guide (runs as a test), API
        reference, fragment, maintenance plane and skill, labeler, towncrier
        section, TIER1 independence
  - [x] benchmark committed (`../_maintenance/evidence/bench_r5a.py`, `.out`)

### Results review - 5a (2026-10-17)
- Review 1 (`r34/review5a/`), 16 findings, each fixed with a test: [High]
  an approval could be reused (one snapshot resumed twice, a new run with
  the old id); lone surrogates and integers past 4300 digits crashed the
  loop; [Med] an `Action` subclass could swap arguments after validation;
  an idempotent call in flight was retried uncharged and trusted the saved
  flag; a failed save forked the audit chain; `from_dict` took inconsistent
  states; the Budget gate ignored tokens; [Low] a store file of another run,
  tools mutating recorded arguments, a clock stepping back, an approver's
  wait counted as running time, two plan inaccuracies, the benchmark not in
  the repository, `verify_audit` raising on malformed records.
- Re-review (`r34/review5a2/`): 10 fixed, 5 partial, 2 new: a trail cut
  after the state continued silently; a torn JSONL line stopped every run;
  plus `head()` reading the whole file per run, the no-store replay, an
  equal-clock run id, surrogate keys, the attempt count reset by an
  approval pause, a denied retry recorded without its key, a non-integer
  `seq`. All fixed with tests: ancestry check (`digest_at`), a random
  nonce in `run_started`, a cached head, attempts carried in the request.
- Negative proofs (`r34/neg5a/`, final tests): reviewed version, the new
  hardening module does not import and the others fail 3; first fixes, 8
  failed; final, 270 passed.
- Python 3.9 and 3.13 (`uv` venvs with pytest and jsonschema): 270 tests,
  16 doctests and the guide test pass on both.
- Suites: 9758 passed + the baseline 15 (same ids as 3b).
- Gates against `r34/base4c` (from `2c283f3`): towncrier exit 0, a new
  `scikitplot.agent` section (105); names unique; ruff and black nothing new
  (Rule 128); non-ASCII 0; Python 3.9 grammar 0; labeler PASS.
- Harness on `snap5a` (untouched; identical to the commit before this
  line): 203 passed, 4 failed, 2 skipped; the 4 failures are the corpus
  legs, each 5107 passed and the same 14 DNS failures as 3b.
- Lessons: Rules 128, 129.
- Not tested by the CI harness: `scikitplot.agent` ships in the full
  distribution only (like `levenshtein`); a partial distribution for it is
  decision D9.
- [x] 5b. Bounded CRAG and agentic retrieval (R5-2): `corpus` owns it;
  planned and recorded in [`../../corpus/todo/todo.md`](../../corpus/todo/todo.md)
- [x] 5c. MCP tools on the Run 2 contracts (R5-3): `mcp` owns it; planned
  and recorded in [`../../mcp/todo/todo.md`](../../mcp/todo/todo.md)

### Run 5 gate (2026-10-17)
- Single agent plus tools on a fixed task set (`bench_r5a.py`): success,
  decisions, calls and cost per planner; every tool call audited and every
  chain verified in all 246 runs; denied actions tested (no scope: every
  action denied, no tool ran; a write without an approver: every run paused,
  nothing written).
- Multi-agent coordination: not built. The roadmap admits it only if it
  beats single agent plus tools on the task set; nothing here calls for it.
