# `scikitplot.mcp` maintenance: active tasks

Round 33 work that changes `scikitplot.mcp`. The campaign's status and open
decisions are in [`../../corpus/todo/todo.md`](../../corpus/todo/todo.md);
Run 5 is planned in [`../../agent/todo/todo.md`](../../agent/todo/todo.md).
Completed tasks are archived under [`../_maintenance/history/`](../_maintenance/history/).

Rules that apply to every increment: [`plugins/scikitplot-dev/skills/scikitplot-engineering-rules/`](../../../plugins/scikitplot-dev/skills/scikitplot-engineering-rules/SKILL.md).

## Task: round 33 / Run 5 / 5c - MCP on the Run 2 contracts (R5-3)

### Context
- Plan item: MCP tools call the Run 2 contracts; there is no MCP-specific
  fusion or filter. Run 5 needs Run 2's filters "because a tool that
  retrieves must respect the caller's scope".
- Found: fusion is already the shared contract (R2-1: `_hybrid.py`'s
  `reciprocal_rank_fusion` is a thin adapter over `FusionPolicy`, the
  byte-identical copy `test_submodule_independence.py` guards); the corpus
  hybrid retriever (`CorpusIndexRetriever`) calls `RetrievalIndex.search`
  and re-implements nothing. No MCP retriever can be scoped: none passes the
  R2-2 `filter=`.
- Base: 5b (`29bf510`).

### Design
- `CorpusIndexRetriever(..., filter=None)`: a corpus filter (expression or
  JSON form) checked by `validate_filter` at construction (imported lazily:
  MCP stays importable without corpus) and passed to every
  `index.search(filter=...)`; `filter` property.
- `CorpusAnnoyRetriever.from_corpus_annoy(..., filter=None)`: checked before
  the build; with a filter every mode returns a `CorpusIndexRetriever`
  (the vector seam `RetrievalIndex.query` takes no filter).
- CLI `--corpus-filter PATH` (JSON file), only with `--corpus-annoy`; no
  environment variable (a scope is set where the server starts); a missing,
  unreadable or malformed filter stops the server before the build.

### Results review - 5c (2026-10-17)
- Review 1 (`r34/review5c/`): [High] the server could not start with
  `--corpus-filter`, nor (already before 5c) with `--corpus-mode hybrid` or
  `keyword`: `create_server(document_reader=retriever.get)` met a
  `CorpusIndexRetriever` without `get`; [Med] responses do not report the
  scope (the corpus `filter_capability` is dropped too); [Low] the claim "no
  fusion or filter of its own" was broader than the profile; an empty `and`
  scope accepted; `TypeError` missing from Raises. Fixed with tests:
  `CorpusIndexRetriever.get` serves only documents in scope (one outside
  reads as missing); a serve-path test through `main()`; the claim narrowed;
  an empty `and` or `or` refused. The scope report is decision D10.
- Re-review (`r34/review5c2/`): every fix holds, end to end with the real SDK
  in-memory client (a hidden document's resource fails like an unknown id);
  4 Low (the README's example filter matched nothing in the default build;
  duplicate ids served differently by `get` and `search`; `documents` not
  checked against the index; the empty-`or` wording), fixed: `get` returns
  the first copy in scope, the example uses `input_path`, `documents=`
  documented as the index's own.
- Negative proofs (`r34/neg5c/`, final tests): base 21 failed; reviewed 9
  failed; first fixes 1 failed; final 22 passed (the real-Annoy test skips
  where Annoy cannot load; it passes in the full environment).
- Suites: 9845 passed + the baseline 15 (same ids as 3b).
- Gates against `r34/base5b` (from `75aa061`): towncrier exit 0; ruff and
  black nothing new; non-ASCII 0; Python 3.9 grammar 0; labeler PASS;
  submodule independence 17 passed.
- Harness on `snap5c` (untouched; identical to the commit before this
  line): 203 passed, 4 failed, 2 skipped; the 4 failures are the corpus
  legs, each 5170 passed and the same 14 DNS failures as 3b; the mcp part on
  Python 3.13 passed 324 (alone) and 351 (with corpus).
