# Maintaining `scikitplot.corpus`

This is the **durable memory** for the `scikitplot.corpus` hardening campaign:
the contracts, invariants, verification gates, and the exact finding-by-finding
state, written so **any new chat/session can resume without losing context**.

It is the human entry point. Its companions live in `_maintenance/`:

- `_maintenance/scikitplot_corpus_DEEP_SEMANTIC_REVIEW_GUIDE.md` — the
  **authoritative finding register** (every finding's evidence, fix, tests,
  status). This file summarises; the guide is the source of truth.
- `_maintenance/METHODOLOGY.md` — the reusable deep-review process (how a
  finding is grounded → fixed → tested → recorded).
- `_maintenance/SESSION_LOG.md` — chronological log of what each session closed.
- `_maintenance/CORPUS_SEARCH_RESOLUTION_LOG.md` — the search-path sub-log.

**Rule:** findings are marked resolved *with evidence*, never deleted; when code,
the guide, or this file changes, update them together.

---

## How to resume this work in a fresh chat

1. **Read**, in order: this file → `_maintenance/METHODOLOGY.md` → the register
   guide's summary table (§27-area) → `_maintenance/SESSION_LOG.md`.
2. **Load the code** from the latest `scikit-plots.zip`. The full package does
   **not import** in a bare environment (compiled C extensions absent), so tests
   run against modules in isolation — see *Environment & harness patterns* below.
3. **Pick a finding** that is still OPEN/PARTIAL and in-sandbox verifiable (see
   *Campaign status*). Skip the ones flagged as needing a live network or a
   maintainer design decision unless you are given that direction.
4. **Follow the per-finding workflow** in `METHODOLOGY.md`: ground against the
   real source → minimal root-cause fix → standalone harness + permanent in-repo
   pytest → update the register (summary row + detail Status + §29 positive
   controls) → re-stage → present.
5. **Keep every change Python 3.8 → 3.15+ safe** (see the compatibility rule
   below) and **never `==`-pin** a dependency.
6. **Append to `SESSION_LOG.md`** what you closed.

---

## Campaign status (grounded in the register guide)

**17 findings RESOLVED, 3 PARTIAL, the rest OPEN.** Each resolved/partial finding
has a permanent guard test (see *Positive controls*). Authoritative detail +
evidence is in the review guide; this is the index.

### Resolved (17)

| ID | What it closed |
| --- | --- |
| CORPUS-ALG-001 | Similarity numeric/score/determinism contract + quality gate + result provenance |
| CORPUS-ALG-002 | Builder auto-embed ndarray-truthiness `ValueError` |
| CORPUS-MCP-001 | MCP retriever wiring (bad embed call, wrong angular formula, double index) onto the corpus `query` seam |
| CORPUS-API-001 | Facade `__all__`/alias identity (PipelineResult rebind, de-duped `__all__`); `NormalizerConfig` binding left OPEN by design |
| CORPUS-ID-001 | `make_doc_id` full-content hash (v2) — was hashing only `text[:64]` (collisions) |
| CORPUS-CACHE-001 | Content-addressed embedding cache key (was a spoofable `(model,path,mtime,n)` proxy) |
| CORPUS-TMP-001 | Shared `_atomic` publish primitive across cache/export/storage/download/multimodal (was predictable `.tmp`) |
| CORPUS-STO-001 | SQLite transactional `save`/`save_batch` (was autocommit + partial persist) |
| CORPUS-STO-002 | JSONL durability (disk-before-memory, atomic rewrite) |
| CORPUS-RES-001 | NLTK downloads off by default; opt-in via arg/env, actionable errors |
| CORPUS-NET-001 | SSRF per-hop redirect validation across all request paths |
| CORPUS-SEC-001 | Pickle/joblib load: integrity gate (pre-deserialize hash) + post-load type validation |
| CORPUS-XML-001 | XML/ALTO hardened (XXE / billion-laughs / external-DTD) on both lxml and stdlib |
| CORPUS-ARC-001 | Streamed archive extraction with actual-byte budget (bounded memory) |
| CORPUS-ARC-002 | Shared nested-archive depth cap (`ArchiveNestingError`; zip-quine bounded) |
| CORPUS-ARC-003 | Transactional archive publish (private staging + atomic `os.replace`) |
| CORPUS-DOC-002 | `load_documents` example no longer contradicts `trusted=False` default |

### Partial — advanced with a tested foundation, remainder scoped

| ID | Done | Remaining (why deferred) |
| --- | --- | --- |
| CORPUS-NET-002 | Fail-closed DNS + all-A/AAAA-record validation (`_resolve_and_validate`) | Peer-IP **pinning** transport (TOCTOU rebinding) — needs live network + TLS to build/test safely |
| CORPUS-NET-003 | SSRF policy single-sourced (WebReader delegates to the shared gate) | Full Fetcher/Transport extraction — needs live network |
| CORPUS-PKG-001 | Read-only `capability_snapshot()` consolidating scattered discovery | Wiring into a run/build provenance manifest (overlaps OBS-001) + locked-env reproduction |

### Open — design-decision-heavy or environment-bound (do not implement unilaterally)

`PIPE-001` (result contract), `CHK-001` (restartable-commit), `PLG-001` (plugin
contract), `CON-001` (concurrency matrix — classified *OPEN QUESTION*),
`WASM-001`, `PRV-001` (privacy/governance model), `SCH-001` (universal artifact
envelope), `OBS-001` (provenance manifest result type), `TYP-001` (typing surface
— **verified sub-finding: no PEP 561 `py.typed` marker exists, so the shipped
`.pyi` stubs are inert; the fix is package-wide, hence a maintainer call**),
`PERF-001` (benchmarks), `DOC-001` (mechanically-executable examples — blocked by
the package not importing without C extensions). See the guide for each.

---

## Reusable primitives created (use these; do not re-implement)

| Primitive | Location | Guarantee |
| --- | --- | --- |
| `atomic_write_path` / `atomic_write_bytes` | `_atomic.py` | mkstemp + fsync + `os.replace` + dir-fsync; no predictable temp, no partial file |
| `ensure_nltk_resource` / `downloads_allowed` | `_resources.py` | opt-in resource downloads; actionable offline errors |
| `_get_with_validated_redirects` | `_url_handler.py` | disables auto-redirect; SSRF-validates every hop before connect |
| `_resolve_and_validate` / `_is_blocked_ip` | `_url_handler.py` | fail-closed DNS; validates all A/AAAA records (v4+v6) |
| `hardened_lxml_parser` / `parse_stdlib_secure` | `_readers/_xml_safety.py` | XXE/entity/DTD-safe XML on both backends |
| `stream_copy_bounded` | `_archive_handler.py` | block-streamed extraction on an actual-decompressed-byte budget |
| `_publish_extracted` + private staging | `_archive_handler.py` | transactional archive publish (atomic, cleaned up on failure) |
| `_archive_ctx` + `ArchiveNestingError` | `_readers/_zip.py` | shared nested-archive depth cap |
| `capability_snapshot` | `_capabilities.py` | read-only reproducibility snapshot of backends + versions |
| `_verify_artifact_integrity` / `_validate_loaded_documents` | `_export/_export.py` | pre-deserialize hash gate + post-load type check |

---

## Python 3.8 → 3.15+ compatibility rule (non-negotiable)

Every module carries `from __future__ import annotations`, so all annotations are
strings and PEP 585/604 syntax (`list[str]`, `str | None`) is safe **in
annotations**. New code must NOT use, in *evaluated* positions:

- subscripted builtin generics — `cast(dict[...], x)`, `isinstance(x, tuple[...])`,
  module-level `X = list[int]`;
- `X | Y` unions in `isinstance` / `except` / module-level aliases;
- version-gated APIs — `match`/`case`, `str.removeprefix`/`removesuffix` (3.9),
  `functools.cache` (3.9), `graphlib`/`zoneinfo` (3.9),
  `hashlib(..., usedforsecurity=)` (3.9), dataclass `kw_only`/`slots=` (3.10),
  subscripted `contextvars.ContextVar[...]` (3.9).

Audited across the campaign (2026-07). The one breaker found and removed was a
`usedforsecurity=False` in `make_doc_id`.

---

## Positive controls — the guard tests that must stay green

The review guide's §29 lists the invariants and their tests (21 controls). The
permanent guard suites added by the campaign:

```
_similarity/tests/test__backends.py            _similarity/tests/test__quality_differential.py
tests/test__api_manifest.py                    tests/test__atomic.py
tests/test__resources.py                       tests/test__archive_budget.py   (+TestTransactionalPublish)
tests/test__capabilities.py                    tests/test__url_handler_ssrf.py
tests/test__url_handler_redirects.py           _readers/tests/test__xml_safety.py
_readers/tests/test__zip_depth.py              _readers/tests/test__web.py    (TestWebReaderSsrfConsolidation)
_embeddings/tests/test__embedding.py           _storage/tests/test__storage.py
_export/tests/test__export_security.py         tests/test__schema_extended.py
```

These plus the module's existing suites are the always-green gate; run them
before packaging, and rewrite (never delete) a test if a contract legitimately
changes.

---

## Original scope note (search subsystem)

The sections below document the durable contracts of the dense/hybrid search
path (`_similarity/`) and its consumption by `scikitplot.mcp`, which was the
first area hardened (ALG-001/ALG-002/MCP-001).

## Architecture at a glance

```
CorpusBuilder --embed--> CorpusDocument.embedding
                              |
                    SimilarityIndex.build
                              |
        +---------------------+-----------------------+
     sparse (BM25)                              dense (ANN backend)
        |                                              |
   _search_keyword                         _similarity/_backends.py
        |                              annoy > faiss > voyager > bruteforce
        +----------------- hybrid (RRF) ---------------+
                              |
              SimilarityIndex.search(str)  ->  list[SearchResult]
              SimilarityIndex.query(vec)   ->  list[(doc_id, cosine)]   <-- MCP seam
                              |
                 mcp/_corpus_annoy.CorpusAnnoyRetriever
```

The dense vector backend is **centralised** in `_similarity/_backends.py`.
`SimilarityIndex` never branches on the concrete backend; it delegates through
one contract. `scikitplot.mcp` consumes the corpus via `SimilarityIndex.query`
and does not re-implement embedding, vector search, or score maths.

## Backend contract (`ANNBackend`)

Every backend implements three methods and honours one score contract.

- `is_available() -> bool` — runtime deps importable.
- `build(embeddings)` — accepts an `(n_docs, dim)` `float32` matrix. Rows need
  not be unit-normalised; the backend normalises internally for cosine.
- `query(vector, k) -> list[(row_index, score)]`.

Invariants — enforce these for any new backend, and cover them with a test:

1. **Unified score = cosine similarity in `[-1, 1]`**, higher is better. FAISS
   `IndexFlatIP` on normalised vectors gives cosine directly; Annoy angular
   distance `d` is converted with `cos = 1 - d**2/2` (the exact inverse of
   `d = sqrt(2*(1 - cos))`; a plain `1 - d/2` is wrong); Voyager cosine
   distance is `1 - cos`.
2. **Descending score order with deterministic, index-ascending tie breaks**
   (brute force uses a stable argsort).
3. **Validation:** `build` rejects non-2-D, empty, or non-finite embeddings;
   `query` rejects dimension mismatch and non-finite vectors; a zero-norm query
   returns `[]` (cosine undefined).

**Result provenance.** `SearchResult` carries `backend` (dense backend name, or `None` for strict/keyword) and `index_generation` (the build that produced it). Both use `compare=False`, so provenance never affects equality/hashing. `SimilarityIndex.index_generation` increments on every successful `build`, enabling stale-result detection. Embedding-model identity is *not* stamped here — it travels with the document embeddings (CORPUS-CACHE-001).

### Selection policy

- `select_backend("auto")` resolves the first available backend in
  `DEFAULT_BACKEND_ORDER = ("annoy", "faiss", "voyager", "bruteforce")` — Annoy
  is the default; brute-force (numpy) is the always-available floor.
- An **explicitly named** backend that is unavailable raises `RuntimeError`
  (fail fast). Only `"auto"` degrades.
- `SimilarityIndex.build` lets *config* errors (unknown/unavailable explicit
  backend) propagate, but treats *data* errors from `backend.build` (e.g.
  non-finite embeddings) as "dense disabled" with an observable warning, so a
  single bad vector never fails a whole corpus build while sparse search
  continues.

### Annoy dual-implementation compatibility

`AnnoyBackend` works against **both** shipped index classes, which share the
`add_item` / `build(n_trees)` / `get_nns_by_vector(vector, n, search_k,
include_distances)` contract:

- `scikitplot.annoy.Index` (high-level, mixin-composed) — has the validated
  bulk `add_items(..., ensure_all_finite=True)` path.
- `scikitplot.annoy._annoy.Index` (Cython) — additionally accepts `dtype`
  (embedding precision) and `index_dtype` (id width for very large corpora).

`SearchConfig.annoy_impl` selects `"auto"` (high-level first, else Cython),
`"highlevel"`, or `"cython"`. `annoy_dtype` / `annoy_index_dtype` are forwarded
only to constructors that accept them (a `TypeError` on unexpected kwargs falls
back to the plain `(f, metric)` constructor).

## Customisable search knobs (`SearchConfig`)

`match_mode` (`strict`/`keyword`/`semantic`/`hybrid`), `top_k`,
`semantic_threshold`, `keyword_threshold`, `hybrid_alpha`, `rrf_k`,
`use_normalized_text`, `case_sensitive`, and the dense-backend controls
`backend`, `annoy_impl`, `annoy_metric`, `annoy_n_trees`, `annoy_search_k`,
`annoy_dtype`, `annoy_index_dtype`. All are validated in `__post_init__`.

## MCP integration rule

`mcp/_corpus_annoy.from_corpus_annoy` builds **one** Annoy-backed
`SimilarityIndex` (via `BuilderConfig.index_kwargs`) and consumes its `query`
seam. The query embedder uses the **same** `embedding_model` as the corpus and
calls `EmbeddingEngine.embed([text])[0]` (the engine takes `list[str]` and
returns `(n, dim)`). Do not: (a) re-embed with a different model, (b) build a
second vector index, or (c) reimplement distance->score — all three were prior
defects (CORPUS-MCP-001).

## Verification gates

```bash
# Unit + contract regressions (Annoy paths use in-process fakes; native libs
# optional). Green in a minimal env.
pytest scikitplot/corpus/_similarity/tests/test__backends.py -q
pytest scikitplot/corpus/_similarity/tests/test__similarity.py -q

# Quality-differential recall gate. Brute-force/FAISS are exact (recall 1.0);
# Annoy must clear the recall gate and not regress with more trees. Native ANN
# sub-tests skip when the library is absent and become real gates when present.
pytest scikitplot/corpus/_similarity/tests/test__quality_differential.py -q

# MCP wiring (in scikitplot/mcp/.../tests): from_corpus_annoy over corpus.
pytest scikitplot/mcp -q -k corpus_annoy
```

Always-green rule: run all four before packaging. When adding a backend, add it
to `DEFAULT_BACKEND_ORDER`, register it in `_BACKENDS`, and extend both
`test__backends.py` (contract) and `test__quality_differential.py` (recall).

## Change checklist

- [ ] New/changed dense backend keeps the unified cosine score contract and all
      three build/query invariants; covered by a contract test.
- [ ] `select_backend` policy preserved (auto order, explicit fail-fast).
- [ ] `SimilarityIndex.query` seam unchanged in shape `list[(doc_id, float)]`
      (the MCP `VectorIndex` protocol depends on it).
- [ ] Quality gate thresholds reviewed if the synthetic corpus or `k` changes.
- [ ] Review guide finding register + this file updated together.

## Resolved-finding index (search path)

| ID | Title | Status |
| --- | --- | --- |
| CORPUS-ALG-001 | Similarity numeric/determinism/quality contracts | RESOLVED — numeric/score/determinism + quality gate (`test__quality_differential.py`) + result provenance (`SearchResult.backend`/`.index_generation`) |
| CORPUS-ALG-002 | Builder auto-embed ndarray truthiness (`ValueError`) | RESOLVED (`_corpus_builder.py:902,1841`) |
| CORPUS-MCP-001 | MCP retriever wiring: bad embed call, wrong angular formula, double index, missing seam | RESOLVED (`SimilarityIndex.query`; `mcp/_corpus_annoy.py` restructured) |

---

## Environment & harness patterns (how tests actually run here)

The full `scikitplot` package cannot be imported in a bare environment because
the compiled C extensions are absent. Findings are therefore verified against
modules **in isolation**, using two patterns under a scratch `harness/` dir:

1. **Standalone module load** — for stdlib-only modules with no relative imports
   (`_atomic`, `_resources`, `_readers/_xml_safety`, `_archive_handler`,
   `_capabilities`): load via `importlib.util.spec_from_file_location` and test
   the functions directly.

2. **Real-graph package harness** — for modules with relative imports
   (`_storage`, `_url_handler`, `_export`, `_readers/_web`, `_readers/_zip`,
   `_similarity`): copy the whole `corpus/` tree into a temp package, **neutralise
   the facade** by overwriting `corpus/__init__.py` (and the relevant
   sub-package `__init__.py`) with an empty file, then import the target module
   by its real path so its dependency graph loads *without* the heavy facade.
   Monkeypatch seams at the module boundary (e.g. `U.socket.getaddrinfo`).

Every fix ships **both** a fast standalone harness (for the tight loop) and a
permanent in-repo `pytest` guard (for CI). The in-repo guards are the durable
artifact; the harnesses are disposable.
