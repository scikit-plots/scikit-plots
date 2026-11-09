# Corpus maintenance history

## 2026-11-09 — research self-test round (PR 919)

64 text files of collected research (3.35 M characters, 22,034 paragraph
chunks) were run through corpus and MCP: every file through
`CorpusPipeline`, `CorpusBuilder`, `SQLiteStorage` and both MCP lexical
retrievers, with timing, peak memory, repeat-run determinism and edge-case
queries. Every file processed; known-item recall@5 1.00 on the FTS5 leg;
no non-deterministic result.

- **CX-05: NUL and lone surrogates broke FTS5 free text.** CX-01 quoted
  every query as FTS5 strings, but no string can carry a NUL (SQLite:
  `OperationalError: unterminated string`) or a lone surrogate, U+D800 to
  U+DFFF (not encodable: `UnicodeEncodeError`, which CPython 3.11 and newer
  report as the connection's previous error when an earlier statement
  failed). `search_text` and `StorageQuery.full_text` raised for 7 of the 13
  probe queries, and the MCP lexical leg reported `failed`, so hybrid search
  lost its keyword leg; a JSON request can carry either character. Both now
  separate words like a space (`_FTS5_UNREPRESENTABLE`, applied in
  `_fts5_phrase` and before `_fts5_any` splits); FTS5 already splits stored
  text at a NUL. Every one of the 1,114,112 code points was checked on the
  three SQLite builds below: only these are refused. Tests: 81 in
  `TestFreeTextSearch` (66 fail on `main`; the other 15 pin the class and
  the characters kept, so a wider or shifted class fails); 10 mutations each
  caught. Verified on CPython 3.8 to 3.15.0b4, free-threaded 3.13t and
  3.14t, SQLite 3.46.0, 3.50.4 and 3.53.1 (corpus suites from its 3.9 floor;
  MCP from 3.8): identical failure sets before and after.
- **Recorded, not changed** (each a ledger note under
  `upcoming_changes/scikitplot/corpus/`): CX-06 a document holding a lone
  surrogate is accepted, then refused by SQLite and JSONL storage; CX-07
  image reader tests fail instead of skipping without Pillow; CX-08
  `detect_script` tests Latin last, 30% of `CorpusBuilder.build` on English
  text; CX-09 the README `CorpusBuilder` example raises (hybrid search,
  no embeddings); CX-10 on CPython 3.15 the atomic-publication test forks a
  threaded process, and the fork warning fails it.
- Known, not new: on Python 3.8 `scikitplot.corpus` does not import
  (`dict[str, Any]` at run time in `_types.py`); the partial distribution
  declares 3.9 and the root `>=3.8` was recorded in
  `maintenances/_repo/_maintenance/history/partial_distributions.md`.
- Evidence: `evidence/round-2026-11-09-fts5-text.log`,
  `evidence/probe_fts5_text.py`, `evidence/probe_fts5_sqlite.py`.

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
  plugins/scikitplot-dev/skills/corpus` (absent from the supplied archive). Unchanged by this round.
  Suite: 3 490 passed, 10 skipped; the 61 failures are the same set as before
  the round (NLTK data packages absent from the review environment).

The earlier Corpus campaign records describe R00–R16 review and IMPL-01–18 implementation work, including a historical `3206 passed / 27 skipped / 4 xfailed` suite. Those records are retained under `history/` and `checkpoints/` for rationale only.

On 2026-09-12 the supplied archive was found to contain no Corpus runtime files. The maintenance model was therefore rebased to fail closed: historical completion is not current evidence, and the next action is source restoration/retrieval before semantic continuation.
