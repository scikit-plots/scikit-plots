# History

## 2026-09-28 — hybrid retrieval through the server (CX-04)

- **CX-04 (gap).** corpus `RetrievalIndex` already fused BM25 and Annoy, but
  the `--corpus-annoy` profile could only serve the dense leg: exact
  identifiers and error strings depended on a hash or model embedding finding
  them. `from_corpus_annoy(..., mode=...)` now accepts `semantic` (unchanged
  default), `hybrid` or `keyword`. The two new modes return a
  `CorpusIndexRetriever`, which asks the corpus index with its own config
  (`dataclasses.replace(config, match_mode=..., top_k=k)`) and maps corpus legs
  to `LegRecord` (corpus `skipped` legs are dropped, they did not run).
  Retrieval semantics stay in corpus; MCP only chooses the mode and reports it.
- CLI: `--corpus-mode` / `SCIKITPLOT_MCP_CORPUS_MODE`, validated before any
  build. The CLI keeps `strict=True`, so a broken embedder fails the query
  there; the degrade-to-lexical behaviour is for library callers
  (`strict=False`): semantic → `failed`, 0 hits; hybrid → `degraded`, lexical
  hits kept (probe output in the log).
- Defect found while wiring: `RetrievalIndex.search` takes `config`
  keyword-only; passing it positionally made every hybrid query `failed`.
  `test_hybrid_asks_the_corpus_with_its_own_config` pins it.
- Test fix: `pytest.importorskip` only skips `ModuleNotFoundError`; the Annoy
  extension raises a plain `ImportError` when root attributes are missing, so
  the integration test now skips on `ImportError` explicitly.
- Annoy is verifiable with the supplied `scikitplot/externals` build, using a
  scratch root shim for `__version__`/`get_config` (not shipped). annoy +
  cexternals: 650 passed; `test_repr_html` fails only under the shim's
  `display="text"` (passes with `"diagram"`), and `test_fd_sentinel` needs
  `scikitplot._testing` (absent from the archive).
- No retrieval-quality claim: the probe uses `HashEmbedder`, which is
  near-lexical, so modes cannot be ranked from it.
- Evidence: `evidence/round-2026-09-28-hybrid.log`, `evidence/probe_hybrid.py`.
  Gate state unchanged (FAIL on arrival, not hand-refreshed).

## 2026-09-28 — lexical leg and token cost (cross-subsystem round)

- **CX-01 (lexical leg).** `Bm25Retriever.from_corpus_sqlite` used
  `StorageQuery.full_text`, which FTS5 read as syntax: 9 of 13 realistic
  technical queries failed the leg and hybrid search silently ran dense-only,
  and it reported `1/rank` instead of BM25. It now calls
  `SQLiteStorage.search_text` and carries the BM25 score (0 of 13 fail).
- **CX-02 (token cost).** `build_search_docs_result` sent identical passages
  once per source. The first copy is kept, every other source is listed under
  its citation's `also_in`, `duplicates_merged` counts them, and merged copies
  do not use up `max_results`. `CitationOutput.also_in` and
  `SearchDocsOutput.duplicates_merged` were added to the closed wire models:
  without them the typed path raised `ValidationError` on the first duplicate
  (shown by removing the field).
- **Considered and kept:** the untrusted-content notice is repeated in every
  passage block (~30 tokens each). It keeps each block self-labelled wherever a
  client places it; removing it trades a security property for tokens.
- **Caching, analysed:** a KV cache lives in the inference engine and prompt
  caching in the provider; neither is this package's to implement. What this
  package controls for them is *determinism* — identical inputs give
  byte-identical tool output (fixed tie-breaks), which is what lets a
  provider's prefix cache hit. A response or retrieval cache here would need an
  invalidation signal a generic `DocsRetriever` does not expose; without one it
  returns stale answers, so none was added. A similarity ("semantic") cache is
  a guess about equivalence and would be the wrong default for a tool that
  cites sources.
- Evidence: `evidence/round-2026-09-28-lexical.log`, `evidence/probe_lexical.py`.
- Gate state on arrival: `maintenance_status=FAIL` from a stale runtime
  fingerprint and inventory (22 test modules recorded as 16). Not refreshed:
  the lanes' test partitions are not recorded, and `VERIFICATION.md` forbids
  editing a fingerprint without regenerating evidence. Suite: 257 passed,
  2 skipped; `test_protocol_in_memory.py` needs `mcp>=2` (1.27 installed).

## 2026-09-12 — maintenance-plane reset

The inherited MCP maintenance campaign contained useful M00–M14 findings but
its active entry points had drifted away from the repository layout: the checker
resolved `maintenances/mcp` as though it were `scikitplot/mcp`, read-order files
were absent, and docs still instructed maintainers to run a runtime-local
`_maintenance` path that does not exist in this snapshot.

The old live surface was preserved under
`history/legacy_live_2026-09-12/`. The active plane now follows the same
runtime/maintenance/skill separation used by the newer submodule campaigns.
