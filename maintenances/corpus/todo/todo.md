# `scikitplot.corpus` maintenance: active tasks

The active campaign is round 33, the AI-plane review (corpus, with mcp,
cleanprompt and `_cli` where a contract crosses into them). Each increment
is planned here, then delivered as a drop-in with a results document.
Completed runs are archived verbatim under
[`../_maintenance/history/`](../_maintenance/history/):

| Run | Archive |
|---|---|
| 1, correctness floor | [`r33_run1_correctness.md`](../_maintenance/history/r33_run1_correctness.md) |
| 2, retrieval contract (2d-2g) | [`r33_run2_retrieval.md`](../_maintenance/history/r33_run2_retrieval.md) |
| 4-A, evaluation (2a-2c, pulled forward) | [`r33_run4a_evaluation.md`](../_maintenance/history/r33_run4a_evaluation.md) |
| 3, ingestion contract (3a-3e) | [`r33_run3_ingestion.md`](../_maintenance/history/r33_run3_ingestion.md) |
| 4, structured chunking (4a-4c) | [`r33_run4_structured_chunking.md`](../_maintenance/history/r33_run4_structured_chunking.md) |

Rules that apply to every increment: [`plugins/_shared/skills/scikitplot-engineering-rules/`](../../../plugins/_shared/skills/scikitplot-engineering-rules/SKILL.md).

## Campaign status (2026-10-17)

- Run 4: 4a, 4b, 4R (records by plane, owned by
  [`_repo`](../../_repo/todo/todo.md)) and 4c delivered; Run 4 complete.
- Run 5: complete (5a agent, 5b corrective retrieval, 5c MCP scope), planned
  in [`../../agent/todo/todo.md`](../../agent/todo/todo.md). Run 6: 6a
  (R6-1, R6-2; [`../../random/todo/todo.md`](../../random/todo/todo.md))
  and 6c (R6-4; [`../../annoy/todo/todo.md`](../../annoy/todo/todo.md))
  delivered.
  Their scope, from the round 33 review plan
  (`01_verdict_and_run_plan.md` of the review bundle), with the plane that
  owns each item:
  - **Run 5, agent layer (corpus, mcp).** R5-1 `scikitplot.agent` MVP:
    typed state; a tool registry with Validation, Permissions, Scope,
    Budget and Approval gates; stopping conditions (done criteria, max
    iterations, token and cost budget, timeout, human checkpoint);
    persisted, resumable state; an audit trail. R5-2 bounded CRAG and
    agentic retrieval under a call budget, not a new index class. R5-3 MCP
    tools call the Run 2 contracts, with no MCP-specific fusion or filter.
    Gate: against single-agent plus tools on a fixed task set, success,
    cost and calls; every tool action audited; denied actions tested.
  - **Run 6, native, platform and structure.** R6-1 `random.index` bias
    (`random`, D5); R6-2 `KissSeedSequence.spawn` lock (`random`); R6-3
    free-threading contracts for `MemoryMap` and `spawn` (`memmap`,
    `random`); R6-4 annoy tests write into the package (`annoy`, Rule 30);
    R6-5 cleanprompt grouped by import layer with a graph gate
    (`cleanprompt`); R6-6 GraphRAG G1 over `_graph.py`, gated by multi-hop
    queries (corpus).

## Open decisions (settled at the end of the campaign, as the maintainer asked)

| ID | Question | Options (recommended first) |
|---|---|---|
| D1 | Instruction-set policy for distributed wheels | (a) Portable baseline by default, plus a meson option `-Dnative-arch=true` for local builds. (b) An `x86-64-v2` floor. (c) An `x86-64-v3` (AVX2) floor. (d) Runtime dispatch. |
| D4 | `SemanticBackend.EMBEDDING` (R3-6 waits on it) | (a) Implement sentence-level breakpoint detection. (b) Remove the member. |
| D5 | Unbiased bounded integers | (a) In `scikitplot.random` only; annoy keeps its draw. (b) In both, with a release note that seeded annoy builds change. |
| D6 | The corpus and mcp trackers predate round 33 (`check_trackers.py`: stale fingerprints and inventories; `MAINTAINING.md` still says the runtime is empty) | (a) Refresh them with `--update` after Run 6, when the runtime stops moving. (b) Refresh now. |
| D7 | Embedding cache default for a model not pinned to a commit (3c) | (a) `unpinned="bypass"`, as implemented. (b) `"allow"`. |
| D8 | `regex` as a test dependency (3d: the script tests skip without it) | (a) Add it to the corpus test extra. (b) Keep it optional. |
| D9 | `scikitplot.agent` as a partial distribution (5a ships it in the full distribution only, untested by the partial-distribution harness) | (a) Add `scikit-plots-agent` (standard library only; test extra `jsonschema`) after Run 5. (b) Keep it in the full distribution. |
| D10 | Echo the server's retrieval scope in `search_docs` responses (5c: the scope is server configuration and responses do not report it; the corpus `filter_capability`, every operator EMULATED, is dropped too) | (a) Add a `scope` object to `structuredContent` (a wire-contract change, with its tests). (b) Keep it in the server's configuration and logs only. |
| D11 | Durable agent memory (`_agentic.py` defers it until seven prerequisites exist: tenant scope, provenance, trust, TTL, delete/forget, poisoning defence, promotion review) | (a) A governed memory collection in corpus: records are `CorpusDocument`s with those seven as required fields and checks, recalled through the existing fusion; `agent` only calls it. (b) A store inside `agent`. (c) Keep deferring. |
| D13 | Markdown sources and the default chunker (`ParagraphChunker` splits a fenced block at its blank lines and strips the first line's indentation; `StructureChunker` keeps both) | (a) `.md`/`.rst` use `StructureChunker` when no chunker is chosen. (b) Keep the chunker explicit and say so in the guide. |

Settled: D2 (fusion semantics, as recommended) and D3 ((a) 64-hex ids
everywhere, with opt-in UUIDv7). D14: (a), PR 926. D12: (a), PR 928.

## Task: round 33 / Run 5 / 5b - bounded corrective retrieval (R5-2)

### Context
- Plan item (`01_verdict_and_run_plan.md` Run 5, R5-2; roadmap section 5,
  "CRAG"): evaluate the retrieved evidence, then retry or re-route under a
  call budget; a stage around the retrieval contract, not a new index class.
- Exists: `_agentic.py` A2 `run_investigation` (its refine step re-runs the
  same query: it never rewrites); `RetrievalExecutionPlan` (4c).
- Base: 5a (`45d96ce`).

### Design
- `corpus/_correction.py`: `CorrectiveRetrieval(plan, grader, rewrite,
  max_rounds, fusion, deadline_seconds, on_error, clock)`; a round is one
  `index.search` (or `plan.run`); each round is graded against the original
  query by `grader(query, hits) -> GradeResult` (`correct`, `ambiguous`,
  `incorrect`, score in [0, 1], the grader's identity); `correct` is
  accepted (`accepted`, or `corrected` for a later round); otherwise the
  caller's `rewrite(query, rounds)` proposes the next query while rounds
  remain and the deadline (checked before the rewrite and before the round)
  has not passed; at the end the ambiguous rounds are fused by
  `FusionPolicy` (`partial`) or the retrieval `abstained`. A failed or
  cancelled round is not graded. `CorrectionResult` reads like a response
  (iteration, status, `error_records`), so `evaluate_retrieval(bind(index))`
  measures it.
- `TermCoverageGrader(top_k, accept, reject)`: the deterministic grader
  (share of query words in the top hits); a model grader is any callable.

### Results review - 5b (2026-10-17)
- Benchmark (`../_maintenance/evidence/bench_r5b.py`, `.out`; pinned
  `faa7665`): answerable 82, noisy 82, off-corpus 14 (Levenshtein guide, not
  indexed). The default grader never abstains (off-corpus 0.00); a strict
  top-1 grader (0.9 / 0.6) abstains on 0.71 of off-corpus queries with
  answerable recall@1 unchanged at 0.91, and answers 0.89 of noisy queries
  (recall@1 0.91 -> 0.82). Rewriting by halves did not help (noisy 0.91 ->
  0.88 default; 0.82 -> 0.80 strict). No grader or rewrite becomes a
  default; reported in the guide.
- Review 1 (`r34/review5b/`), 11 findings, each fixed with a test: [High]
  identity-less hits raised out of `run()` when fusing; a cancelled plan
  read as `empty` (the harness's guard bypassed); [Med] a degraded response
  without a record; the rewrite called after the deadline; a `GradeResult`
  subclass skipping its checks; one ambiguous round not fused (top_k,
  weights); [Low] proposals not stripped; the grader's identity overwritten;
  numpy reals refused; signature errors undocumented; guide claims and
  NumPyDoc sections.
- Re-review (`r34/review5b2/`): 9 fixed, 2 partial, 2 new (the deadline no
  longer guarded the round after a slow rewrite; with no rewrite it marked
  every slow partial result degraded) and 5 Low (cancelled precedence, the
  fusion catch hiding a policy's own bug, one identity-less round degraded,
  the default-grader sentence, thresholds kept as numpy floats). All fixed
  with tests.
- Negative proofs (`r34/neg5b/`, final tests): reviewed 13 failed; first
  fixes 5 failed; final 63 passed.
- Suites: 9822 passed + the baseline 15 (same ids as 3b).
- Gates against `r34/base5a` (from `45d96ce`): towncrier exit 0 (one bullet,
  Rule 126); ruff and black nothing new; non-ASCII 0; Python 3.9 grammar 0.
- Harness on `snap5b` (untouched; identical to the commit before this
  line): 203 passed, 4 failed, 2 skipped; the 4 failures are the corpus
  legs, each 5170 passed and the same 14 DNS failures as 3b.
- Lessons: Rule 127 extended (the check before every call includes a
  component's helper calls).

### PR 909: routing extension (query shapes)
- [x] `RoutingRule`, `RoutingPolicy`, `QueryShape`, `DEFAULT_ROUTING_POLICY`,
  `query_tokens` (`_routing.py`); `route(..., has_metadata=, policy=)` and
  `RoutePlan.shapes`/`unavailable` (`_agentic.py`). Design reviewed
  2026-11-03: additive (a plain query routes exactly as before), rules as
  data (custom policies, other languages, `matcher` callables), no model.
- Shapes map to legs we have: `multi_hop`, `hierarchy` -> graph;
  `aggregation`, `negation`, `temporal` -> metadata. A leg is added only
  when the index has it; a missing one is reported in `unavailable`.
- Whole-word matching only (no caller regex): linear in the query length,
  checked with a 20 000-word query.
- Acceptance: the labelled routing set (`tests/data/routing_set.json`, 45
  queries: every shape, combinations, unavailable legs, near misses such as
  `notebook`, `knot`, `sectional`) passes; 73 tests in `test__routing.py`.

### PRs 919 to 926: research self-test fixes (CX-05 to CX-12)
- [x] Done (`../_maintenance/HISTORY.md`, 2026-11-09 and 2026-11-10); the
  round's ledger is empty. CX-06 took option (c); D14 took (a).
- Next: decisions D11 to D13; a public text-step base once the usage skills
  can be regenerated.

### PRs 928 to 931: D12, CX-14, CX-15 (a), CX-16 (a), CX-17, CX-18, CX-19
- [x] Done (`../_maintenance/HISTORY.md`, 2026-11-12).

### PR 932: CX-20; CX-21 and CX-22 as policies; reader snapshots
- [x] Done (`../_maintenance/HISTORY.md`, 2026-11-13).
- Next: choices in ledger notes CX-16, CX-23; D11, D13.

### PR 933: CX-24 (empty audio file)
- [x] Done (`../_maintenance/HISTORY.md`, 2026-11-13).
- Next: ZipReader member warnings name a temporary path (CX-25).

### PR 934: CX-25 (ZIP member warnings name the archive)
- [x] Done (`../_maintenance/HISTORY.md`, 2026-11-13).
- Next: choices in ledger notes CX-16, CX-23; D11, D13.

### PR 935: CX-16 (c), PDF page text the same on every read
- [x] Done (`../_maintenance/HISTORY.md`, 2026-11-13).
- Next: CX-23 (XMLReader error type); D11, D13.
