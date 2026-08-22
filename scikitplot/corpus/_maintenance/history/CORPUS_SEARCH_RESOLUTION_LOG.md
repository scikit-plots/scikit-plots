# Corpus search-path hardening — resolution log

> Merge into `scikitplot_corpus_DEEP_SEMANTIC_REVIEW_GUIDE.md` (Part VI register
> and §29 positive controls). Per the handbook maintenance rule, findings are
> marked resolved with evidence, not deleted.

## Scope of this change set

Centralised the dense vector backend for semantic/hybrid search, made
**Annoy the default** backend (compatible with both shipped index classes),
unified the score contract to cosine similarity in `[-1, 1]`, added a
vector-level `query` seam, and restructured the MCP retriever to consume corpus
rather than re-implement vector search.

Files: `corpus/_similarity/_backends.py` (new), `corpus/_similarity/_similarity.py`,
`corpus/_corpus_builder.py`, `corpus/_similarity/tests/test__backends.py` (new),
`mcp/_corpus_annoy.py`.

## Register updates

### CORPUS-ALG-001 — Similarity/semantic numeric, determinism, and quality contracts
- **Status:** PARTIALLY RESOLVED (score/numeric/determinism half).
- **Resolved now:** All backends return one score scale — cosine in `[-1, 1]`
  (`_backends.py`). Query vectors are dimension- and finiteness-validated
  (`_validate_query_vector`); build embeddings are validated for shape,
  emptiness, and finiteness (`_validate_embeddings`). Brute-force ties are
  deterministic (stable argsort, index-ascending). Broad `except Exception:
  pass` fallbacks in `_search_semantic` were removed; the method now delegates
  to a single validated backend.
- **Still open (P2):** exact-vs-ANN differential *quality* gate and index-
  generation/provenance stamping on `SearchResult`. Tracks to the future
  quality-differential CI job.
- **Evidence/tests:** `corpus/_similarity/tests/test__backends.py`
  (`TestBruteForce`, `TestAnnoyBackend`, `TestSelectBackend`).

### CORPUS-ALG-002 (new) — Builder auto-embed used ndarray truthiness
- **Classification:** VERIFIED DEFECT. **Priority:** P1. **Status:** RESOLVED.
- **Evidence:** `_corpus_builder.py` used `if embs and len(embs) > 0` and
  `embs[0] if embs else None` where `embs = engine.embed([...])` returns an
  ndarray of shape `(1, dim)`; `bool()` on a `(1, dim>1)` array raises
  `ValueError: truth value ... is ambiguous`, so semantic/hybrid search raised
  for any real multi-dimensional embedding model.
- **Fix:** `embs is not None and len(embs) > 0` at both sites (`:902`, `:1841`).
  Note: `_similarity.py:320` was left unchanged — that `embs` is a Python list,
  where `if embs` is a correct emptiness check.
- **Tests:** regression idiom asserted in `test__backends.py` fixtures and the
  end-to-end MCP integration test.

### CORPUS-MCP-001 (new) — MCP retriever wiring defects and duplication
- **Classification:** VERIFIED DEFECT + ARCHITECTURAL DEBT. **Priority:** P1.
  **Status:** RESOLVED.
- **Evidence (pre-fix `mcp/_corpus_annoy.py`):**
  1. `_EngineWrap.embed` called `engine.encode(...)` (no such method on the
     corpus `EmbeddingEngine`) then `engine.embed(text)` with a bare `str`
     (contract requires `list[str]`), so the real query path never produced a
     usable vector.
  2. `_AnnWrap` converted Annoy angular distance with `1 - d/2`, which is not
     the inverse of `d = sqrt(2(1 - cos))` (orthogonal → `0.293` instead of
     `0`; opposite → `0` instead of `-1`).
  3. The builder built a `SimilarityIndex` (`build_index=True`) that was then
     discarded while a second, ad-hoc Annoy index was rebuilt from
     `result.documents` (double work), and `SimilarityIndex` did not satisfy
     the declared `VectorIndex` protocol.
- **Fix:** `SimilarityIndex` gained `query(vector, k) -> [(doc_id, cosine)]`
  and `backend_name`. `from_corpus_annoy` now builds one Annoy-backed
  `SimilarityIndex` via `index_kwargs` and consumes its `query` seam; the query
  embedder uses `engine.embed([text])[0]` with the same `embedding_model`.
  `_AnnWrap`/`_EngineWrap`/`_infer_dim` and the direct `scikitplot.annoy` import
  were removed. Angular→cosine now uses `1 - d**2/2` inside the corpus backend.
- **Tests:** `test__backends.py::TestAnnoyBackend::test_highlevel_cosine_recovery`
  (formula), plus MCP integration (embed-with-list, provenance mapping, unified
  cosine score).

## §29 positive controls to preserve (add)

| Control | Evidence/intent | Required regression test |
| --- | --- | --- |
| Unified cosine score contract | Every ANN backend returns cosine in `[-1, 1]`, descending, index-ascending ties. | `TestBruteForce`, `TestAnnoyBackend` exact-cosine agreement. |
| Centralised backend selection, Annoy default | `select_backend("auto")` order `annoy→faiss→voyager→bruteforce`; explicit-unavailable fails fast; brute-force is the numpy floor. | `TestSelectBackend`. |
| Dual Annoy-impl compatibility | Works against `scikitplot.annoy.Index` and `scikitplot.annoy._annoy.Index`; dtype kwargs forwarded only where accepted. | `TestAnnoyBackend` (highlevel/cython/auto-fallback/dtype). |
| Corpus-owned vector seam for MCP | `SimilarityIndex.query` satisfies the MCP `VectorIndex` protocol; MCP no longer re-implements vector search. | MCP integration test in the mcp suite. |
| Graceful dense degradation | Non-finite embeddings disable the dense index (observable warning) while sparse keyword search continues. | `TestSimilarityIndexSeam::test_degrades_to_sparse_on_non_finite_embeddings`. |
