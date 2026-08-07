# `scikitplot.mcp` — module design & requirements review


> **2026-08-05 implementation update:** D1 is resolved. The package now contains
> an official MCP Python SDK v2 `MCPServer`, stdio and Streamable HTTP CLI
> execution, a Docker profile, and an optional `/healthz` route. HTTP+SSE wording
> below is historical design context; Streamable HTTP is the active remote
> transport. Retrieved text is sanitized and bounded but remains semantically
> untrusted—it is not made “prompt-injection safe.”

**Status:** runnable server + verified SDK-agnostic core. The real corpus/annoy
field mapping and production integration remain gated on the decisions in §9. Nothing
here reimplements retrieval, embedding, or MCP formatting that other modules
already provide.

---

## 1. Why this module exists

scikit-plots already owns the hard parts of documentation RAG:

- **`scikitplot.corpus`** — ingest (RST/MyST/Markdown/HTML/PDF/…), chunk,
  normalise, enrich, **embed** (`EmbeddingEngine`), search (`SimilarityIndex`,
  `SQLiteStorage` FTS5), and MCP-shaped adapters (`to_mcp_tool_result`,
  `to_mcp_resources`, `MCPCorpusServer`, `LangChainCorpusRetriever`).
- **`scikitplot.annoy`** — a persistent approximate-nearest-neighbour **vector
  index** (`Index.add_item` / `build` / `get_nns_by_vector`, memory-mapped I/O).

This module provides the **server that exposes this over the Model Context Protocol**
to external clients (Claude, Cursor, Copilot) *and* to the scikit-plots AI
documentation panel — from one retrieval core. That is `scikitplot.mcp`.

This turns the "no first-party RAG with citations" gap from the Biel benchmark
into an **owned, self-hostable** capability, and directly counters Biel's hosted
"Docs MCP Server" with an in-repo, BSD-licensed equivalent the site owner runs.

---

## 2. Scope — and what it must NOT duplicate

| Concern | Owner | `scikitplot.mcp` role |
|---|---|---|
| Ingest / chunk / embed / store | `scikitplot.corpus` | consume via a retriever |
| Vector ANN index | `scikitplot.annoy` | consume via a retriever |
| Format docs as MCP responses | `corpus.to_mcp_tool_result` etc. | prefer it; the core's builder is the transport-neutral fallback + safety chokepoint |
| **MCP protocol / transport / server** | — (new) | **owned here** |
| Tool/resource/prompt registration, auth, limits, confirmation | — (new) | **owned here** |

DRY guardrail: if corpus grows a capability, `scikitplot.mcp` calls it; it does
not fork it.

---

## 3. Architecture

```
  scikitplot.corpus  ─┐  (ingest + embed + provenance)
                      ├──►  DocsRetriever  ──►  build_search_docs_result()  ──►  MCP server
  scikitplot.annoy   ─┘   (CorpusAnnoyRetriever)   (cited, bounded, untrusted)      (stdio | Streamable HTTP)
                                                                                    │
                              ┌─────────────────────────────────────────────┬──────┘
                              ▼                                             ▼
                   External MCP clients                        scikit-plots AI panel proxy
                   (Claude, Cursor, Copilot)                   (inject cited context into chat)
```

Three layers, decreasing stability top-to-bottom:

1. **Retrieval core (delivered, tested).** `DocsRetriever` protocol +
   `RetrievedChunk` + `CorpusAnnoyRetriever` (composes corpus+annoy by
   injection; `from_corpus_annoy()` does the real wiring behind an import
   guard). `build_search_docs_result()` renders results as an MCP `tools/call`
   response with **source citations** (deep-linked to the page/section anchor)
   and **bounded untrusted handling** (control-strip, length-cap,
   http(s)/relative-only citation URLs). SDK-agnostic → testable with no MCP SDK, corpus, or annoy
   present. **25 core regression tests green.**
2. **Server layer (delivered).** `_server.py` registers tools/resources on the
   official MCP Python SDK v2 `MCPServer`; `__main__.py` runs stdio or
   Streamable HTTP. The core remains SDK-agnostic, keeping the shell thin and
   swappable.
3. **Consumers.** External MCP clients connect directly; the AI panel proxy
   calls the same retrieval core to inject cited context into its chat path
   (closing the benchmark's A1 "citations" aim with owned components).

---

## 3b. Hybrid retrieval (dense + BM25 + graph)

A single retriever underperforms on documentation: dense embeddings miss exact
API symbols / flags / error strings; lexical search misses paraphrase. So the
default `DocsRetriever` is a **`HybridRetriever`** fusing legs, each itself a
plain `DocsRetriever`:

| Leg | Backend (owned) | Strength |
|---|---|---|
| **Dense** | `CorpusAnnoyRetriever` (corpus embeddings + annoy ANN) | paraphrase, synonymy |
| **Sparse / BM25** | `Bm25Retriever` over `corpus.SQLiteStorage` **FTS5 (BM25 by default)** | exact tokens, symbols, errors |
| **Graph** *(designed)* | expand seed hits along cross-refs / "see also" / section tree, re-rank | relationships neither leg sees |

**Fusion = Reciprocal Rank Fusion.** Dense (cosine) and BM25 scores are not
comparable and normalising them is brittle, so we fuse by *rank*:
`score(d) = Σ_legs weight/(rrf_k + rank_leg(d))` (`rrf_k≈60`). A doc found by
several legs is boosted; one miscalibrated leg can't dominate; a leg that raises
is skipped (resilient). **Delivered and tested** (`_hybrid.py`,
`reciprocal_rank_fusion` + `HybridRetriever` + `Bm25Retriever`; 22 tests).

The **graph** leg is specified but not bound: it needs to know what relationship
metadata corpus exposes on `CorpusDocument` (cross-references, section
hierarchy, provenance links). This is one of the things to confirm from the
corpus source (see §Step-by-step upgrade).



**Tools**
- `search_docs(query, k=5)` — flagship; read-only; returns cited passages.
  *(delivered)*
- `get_document(doc_id)` resource — fetch one bounded chunk for follow-up. *(delivered when a document reader is configured)*
- *(future, separate)* scikit-plots capability tools (e.g. render a plot) — only
  if we want the MCP server to expose more than docs; each sensitive/side-effect
  tool gates behind explicit confirmation (§6).

**Resources**
- Documentation chunks as `resources/read` targets, via `corpus.to_mcp_resources`
  — lets clients pull exact sources behind a citation.

**Prompts**
- *(optional)* a "grounded answer" prompt template that instructs the client to
  answer only from `search_docs` results and cite them.

---

## 5. Transports & deployment

- **stdio** — local clients (Claude Desktop, Cursor, Windsurf) spawn the server
  as a subprocess. Zero network exposure; the natural default.
- **Streamable HTTP** — hosted/self-hosted for the AI panel proxy and remote clients.
  Reuses the deployment story already built for the proxy (`_docker_proxy`,
  HF Space, cloud) and the same `ALLOWED_ORIGINS` / secret-in-env policy.

Same principle as the rest of the project: **self-hostable, secrets server-side
only.** Unlike Biel's hosted MCP, the index and the data never leave the owner's
infrastructure.

---

## 6. Security (grounded in the security review's MCP guidance)

- **Read-only tools need no confirmation; sensitive/write tools require explicit
  user confirmation.** `search_docs` / `get_document` are read-only. Any future
  tool with side effects must surface a confirmation and be off by default.
- **Retrieved text is untrusted data.** The core control-strips and length-caps
  it; the server layer must present it to the model as *context, not
  instructions* (prompt-injection defence-in-depth). Citation URLs are validated
  to http(s)/relative — no `javascript:`/`data:` (tested).
- **Auth & limits on HTTP transport** — per-principal quota, concurrency, model/
  tool allow-lists — reuse the proxy's controls rather than inventing new ones.
- **No secret in the client or the index artifacts.** Consistent with
  `CONFIG_ARCHITECTURE.md`.

---

## 7. Dependency strategy (per the versioning policy)

- Core: **no hard runtime deps** (stdlib only) — always importable.
- Extras (ranges, no `==` pins): `mcp` (transport), `scikitplot[corpus]`,
  `scikitplot[annoy]`, `sentence-transformers` (via corpus).
- Import guards give **actionable errors** when an extra is missing
  (`from_corpus_annoy` already does this).
- No import-time side effects (no model load, no network), matching
  `scikitplot.annoy`.

---

## 8. Delivered now vs gated

| Item | Status |
|---|---|
| `DocsRetriever` protocol + `RetrievedChunk` | **Done** |
| `build_search_docs_result` (citations + bounded untrusted-content handling) | **Done**, covered by regression tests |
| `HybridRetriever` (RRF) + `Bm25Retriever` + `reciprocal_rank_fusion` | **Done**, 22 tests |
| `CorpusAnnoyRetriever` (DI composition + `from_corpus_annoy` guard) | **Done** (real wiring path documented; needs corpus/annoy to execute) |
| MCP server/transport (stdio + Streamable HTTP) on MCP Python SDK v2 | **Done** |
| Real corpus→annoy field mapping (embedding/provenance) verified against installed versions | **Gated** — D2 |
| Panel proxy consuming the retriever for cited chat | **Gated** — D3 |
| Bounded `docs://chunk/{doc_id}` resource | **Done** when a document reader is configured |
| Grounded-answer prompts / broader non-doc tools | **Gated** — D4 |

---

## 9. Decisions needed

- **D1 — Transport SDK: RESOLVED.** Use official MCP Python SDK v2
  `MCPServer`; do not hand-roll JSON-RPC.
- **D2 — Index backend default.** `scikitplot.annoy` (persistent, memory-mapped)
  as the default vector index, with `corpus.SimilarityIndex` as an in-process
  alternative and `SQLiteStorage` FTS5 for hybrid keyword+vector? Confirm the
  default.
- **D3 — Panel integration.** Should the AI-panel proxy call this retriever to
  inject cited context now (closes benchmark A1), or ship the MCP server first
  and integrate the panel after?
- **D4 — Surface breadth.** Docs-only (`search_docs`/`get_document`) for v0, or
  also expose other scikit-plots tools via MCP from the start?

---

## 10. Roadmap

1. **v0 delivered:** stdio and Streamable HTTP MCP server exposing
   `search_docs`, optional bounded document resources, Docker startup, and
   `/healthz`; keep the real SDK client round-trip in CI.
2. **v0.1 (on D2/D3):** verify corpus→annoy mapping against installed versions,
   make the hybrid retriever the production default, and let the panel proxy
   inject cited context.
3. **v0.2 (on D4):** grounded-answer prompts and optional non-doc tools, each
   behind the appropriate authorization/confirmation boundary.
4. Fold the "content-gaps" view (benchmark A4) on top — questions with weak
   `search_docs` scores are exactly the gap signal.

---

## 11. Step-by-step upgrade (once the corpus submodule is uploaded)

The tolerant field-access in `_corpus_annoy.py` / `_hybrid.py` (embedding,
provenance, FTS5 rows) is written to be *replaced* by exact calls verified
against the real source. When the corpus submodule lands, the verified steps:

1. **Read the real schema.** Confirm `CorpusDocument` field names for embedding,
   text, source URI, title, section anchor, and any cross-reference / "see also"
   metadata (decides whether the **graph** leg is feasible now).
2. **Pin the dense wiring.** Replace `_get_embedding` / `_doc_to_record` /
   `EmbeddingEngine` calls with the exact corpus API; add a small integration
   test that builds a tiny corpus and round-trips one query through annoy.
3. **Pin the BM25 wiring.** Replace `from_corpus_sqlite`'s `SQLiteStorage` /
   `StorageQuery` access with the exact FTS5 query + `bm25()` rank; test on a
   3-doc store that an exact-symbol query beats the dense-only result.
4. **Bind or defer the graph leg** based on step 1's metadata.
5. **Tune hybrid defaults** (weights, `rrf_k`, `fanout`) on a handful of real
   doc queries; keep the RRF fusion, adjust only the knobs.
6. **Wire the production backend** into the delivered MCP server and verify with
   a real MCP v2 client end to end.
7. **Upgrade the AI panel** (D3): the proxy calls the same `HybridRetriever` to
   inject cited context; verify citations deep-link to the right sections.

Every step is a reversible commit with a green gate; the SDK-agnostic core and
its SDK-independent tests stay the invariant that must remain green throughout.
