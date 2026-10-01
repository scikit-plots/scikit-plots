# Corpus maintenance history

## 2026-09-28 — lexical retrieval review (cross-subsystem round)

Reviewed with MCP after a reading of current hybrid-retrieval practice (BM25 +
dense fused by rank, rerank, context budget, caching).

- **CX-01 — free text reached FTS5 as syntax.** `SQLiteStorage.query` passed
  `StorageQuery.full_text` to `MATCH` verbatim, so FTS5 read it as an
  expression: `roc_auc_score()`, `sklearn.metrics:auc`, `C++`, `random-state`,
  `area under the curve?` and any pasted error message were syntax errors (9 of
  13 realistic queries), and a plain question required every word. `full_text`
  is now one quoted phrase — its documented meaning, and what the in-memory and
  JSONL backends emulate — and never an error. New
  `SQLiteStorage.search_text(text, limit, collection_id=None)` is ranked BM25
  retrieval over free text: any word, each compound word also as its parts,
  score `-bm25()` (higher is better). Tests: `TestFreeTextSearch` (34 fail
  with the query passed verbatim).
- **CX-03 — `_BM25Index` scanned every document for every query term.** It now
  walks a postings list per term. Scores and order are bit-identical to the
  scan (1 500 randomised comparisons; `TestBM25PostingsEqualTheDefinition`
  checks against the definition). 50 000 documents: query 62 ms -> 5 ms; build
  1.17 s -> 1.85 s, paid once.
- **Recorded, not changed:** four BM25 implementations exist (`_BM25Index`, the
  MCP demo backend, `scikitplot.rank_bm25`, SQLite FTS5 `bm25()`), with two
  tokenisations (`\w+` keeps `_`; FTS5 `unicode61` splits on it). Consolidating
  them is a design decision with compatibility cost, not a fix.
- Evidence: `evidence/round-2026-09-28-lexical.log`, `evidence/probe_lexical.py`.
- Consumer note (CX-04, owned by MCP): the MCP `--corpus-mode hybrid|keyword`
  profile now calls `RetrievalIndex.search` directly instead of re-fusing legs
  itself. That makes `search`'s keyword-only `config` and the `LegOutcome`
  fields (`leg`, `status`, `hit_count`, `error.message`) a cross-package
  contract: MCP's `test_corpus_index_retriever.py` fails if they change. No
  corpus code changed.
- Gate state on arrival: `check_trackers.py` stops at `missing required path:
  skills/corpus` (absent from the supplied archive). Unchanged by this round.
  Suite: 3 490 passed, 10 skipped; the 61 failures are the same set as before
  the round (NLTK data packages absent from the review environment).

The earlier Corpus campaign records describe R00–R16 review and IMPL-01–18 implementation work, including a historical `3206 passed / 27 skipped / 4 xfailed` suite. Those records are retained under `history/` and `checkpoints/` for rationale only.

On 2026-09-12 the supplied archive was found to contain no Corpus runtime files. The maintenance model was therefore rebased to fail closed: historical completion is not current evidence, and the next action is source restoration/retrieval before semantic continuation.
