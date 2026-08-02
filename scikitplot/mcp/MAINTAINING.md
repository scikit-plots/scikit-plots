# Maintaining `scikitplot.mcp`

Durable memory for the MCP server module: what it is, the contracts that must
not break, how to verify it, and how to run a deep review here — written so **any
new chat/session resumes without losing context**.

This is the human entry point. Companions in `_maintenance/`:

- `_maintenance/DESIGN.md` — the module design & requirements review (why it
  exists, architecture, security posture, the **gated decisions D1–D4**, roadmap,
  and the step-by-step corpus-wiring upgrade). Read this first for *intent*.
- `_maintenance/METHODOLOGY.md` — the reusable deep-review process (shared
  discipline with `scikitplot.corpus`).
- `_maintenance/MCP_REVIEW_GUIDE.md` — the **finding register** for the mcp review
  campaign, seeded with grounded candidate findings. Fill it the way the corpus
  guide was filled.
- `_maintenance/SESSION_LOG.md` — chronological log of what each session did.

**Rule:** findings are marked resolved with evidence, never deleted; code +
`MCP_REVIEW_GUIDE.md` + this file change together.

---

## How to resume in a fresh chat

1. Read: this file → `_maintenance/DESIGN.md` (intent + gated decisions) →
   `_maintenance/METHODOLOGY.md` → `_maintenance/MCP_REVIEW_GUIDE.md`.
2. Load code from the latest `scikit-plots.zip`. The SDK-agnostic core imports
   with **stdlib only** — `_core.py` and `_hybrid.py` are directly testable
   without the MCP SDK, corpus, or annoy. `_corpus_annoy.from_corpus_annoy` /
   `Bm25Retriever.from_corpus_sqlite` need corpus/annoy to *execute*, but their
   pure logic is testable via injected seams / test doubles.
3. Pick a candidate finding from `MCP_REVIEW_GUIDE.md` (or the DESIGN's gated
   items) that is in-sandbox verifiable. Skip the ones needing a design decision
   (D1–D4) or a live MCP client unless given direction.
4. Follow the per-finding workflow in `METHODOLOGY.md`.
5. Keep every change **Python 3.8 → 3.15+ safe** and **never `==`-pin**.
6. Append to `SESSION_LOG.md`.

---

## What this module is (one screen)

`scikitplot.mcp` is the **MCP server that exposes scikit-plots documentation
retrieval** (flagship tool `search_docs`) to MCP clients (Claude, Cursor,
Copilot) and to the AI documentation panel — from one retrieval core. It **does
not** re-implement retrieval: `scikitplot.corpus` owns ingest/chunk/embed/store
and `scikitplot.annoy` owns the vector index. mcp *composes* them.

```
corpus (ingest+embed+provenance) ─┐
                                  ├─► DocsRetriever ─► build_search_docs_result() ─► MCP server
annoy  (ANN vector index) ────────┘   (CorpusAnnoyRetriever)  (cited, injection-safe)   (stdio | HTTP-SSE, GATED)
```

Three layers, decreasing stability top→bottom:

1. **Retrieval core — delivered, tested (24 tests).** `_core.py`
   (`DocsRetriever` protocol, `RetrievedChunk`, `build_search_docs_result`),
   `_hybrid.py` (`HybridRetriever` RRF fusion + `Bm25Retriever`), `_corpus_annoy.py`
   (`CorpusAnnoyRetriever`, the flagship corpus+annoy composition).
2. **Server / transport layer — GATED (D1).** stdio + HTTP-SSE on the official
   `mcp` SDK (FastMCP recommended). Not implemented; the core is SDK-agnostic so
   this stays a thin, swappable shell.
3. **Consumers** — external MCP clients + the AI-panel proxy.

---

## Durable contracts (do not break without a test + register update)

### `DocsRetriever` protocol (`_core.py`)
`search(query: str, k: int = 5) -> list[RetrievedChunk]`, best first. A
`typing.Protocol` (structural) so backends need not subclass or import. Every
retriever (`CorpusAnnoyRetriever`, `Bm25Retriever`, `HybridRetriever`, test
doubles) satisfies it.

### `RetrievedChunk` (`_core.py`)
Frozen dataclass — the boundary type between retrieval and the tool layer:
`text, source_uri, score, doc_id, title, anchor, extra`. Concrete retrievers map
their native result (corpus `SearchResult`/`CorpusDocument`, annoy neighbour)
onto this shape.

### The safety chokepoint — `build_search_docs_result` (`_core.py`)
The **single** place that enforces citation shape + text safety. Invariants:

- **Untrusted text is sanitised**: control chars stripped (`_CONTROL_RE`),
  length-capped (`MAX_CHUNK_CHARS = 4000`; titles/anchors/ids capped at 200).
- **Result count capped** at `MAX_RESULTS = 20`.
- **Citation URLs validated** by `_safe_uri` to http(s)/relative — no
  `javascript:` / `data:`. (See candidate finding MCP-SEC-001 about
  protocol-relative `//host` URIs.)
- **Output shape** is the MCP `tools/call` response:
  `{"content": [...], "structuredContent": {"query", "citations"}, "isError": False}`.

Any new sanitisation or citation logic goes **here**, not in individual
retrievers.

### RRF fusion (`_hybrid.py`)
`score(d) = Σ_legs weight / (rrf_k + rank_leg(d))`, `rrf_k = 60`. Fuse by *rank*,
never by raw score (dense cosine and BM25 are not comparable). A leg that raises
is skipped (resilient) — but see candidate finding MCP-HYB-001 about *which*
exceptions should be swallowed.

### Composition rule (DRY guardrail)
If `scikitplot.corpus` grows a capability, mcp **calls** it; it does not fork it.
`CorpusAnnoyRetriever` consumes the corpus `SimilarityIndex.query(vector, k)`
seam and the *same* `EmbeddingEngine` model as the corpus — do not (a) re-embed
with a different model, (b) build a second vector index, or (c) reimplement
distance→score (all three were prior corpus-side defects, CORPUS-MCP-001).

### No import-time side effects
Importing `scikitplot.mcp` must never load a model, open a network connection,
or build an index — matching `scikitplot.annoy`. The core has **no hard runtime
deps** (stdlib only); corpus/annoy/`mcp` are extras behind import guards with
actionable errors.

---

## Verification gates

```bash
# SDK-agnostic core + hybrid — green with NO mcp SDK / corpus / annoy present.
pytest scikitplot/mcp/tests/test_mcp_core.py -q      # 13 tests (citation + injection safety)
pytest scikitplot/mcp/tests/test_hybrid.py    -q      # 11 tests (RRF fusion + BM25 leg)
```

These 24 tests are the **invariant that must stay green** through any change.
When the server layer (D1) or the real corpus wiring (D2) lands, add integration
tests behind import guards — never at the cost of the SDK-agnostic core's
independence from those deps.

---

## Python 3.8 → 3.15+ compatibility rule

Same rule as corpus (see `corpus/MAINTAINING.md`). The mcp source currently
passes: all modules carry `from __future__ import annotations`, and there are no
evaluated-position subscripted generics, `|`-unions, or version-gated APIs. Keep
it that way; **never `==`-pin** — extras use ranges.

---

## Change checklist

- [ ] Retrieval logic changed? `DocsRetriever` shape unchanged; `RetrievedChunk`
      fields stable; the 24 core/hybrid tests still green.
- [ ] Sanitisation/citation change lives in `build_search_docs_result` only.
- [ ] No import-time side effects introduced; core still stdlib-only.
- [ ] Corpus/annoy consumed via their public seams (no forking, no second index,
      no re-embed with a different model).
- [ ] 3.8→3.15+ clean; no `==` pins.
- [ ] `MCP_REVIEW_GUIDE.md` finding register + this file updated together;
      `SESSION_LOG.md` appended.
