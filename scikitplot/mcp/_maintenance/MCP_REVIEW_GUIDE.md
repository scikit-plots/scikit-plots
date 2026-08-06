# `scikitplot.mcp` — deep review guide (finding register)

Companion to `MAINTAINING.md` and `METHODOLOGY.md`. This is the **register to
fill** during the mcp review campaign — the analogue of the corpus
`scikitplot_corpus_DEEP_SEMANTIC_REVIEW_GUIDE.md`.


> **2026-08-05 update:** the original register predates the runnable server.
> MCP-SRV-001 is now **RESOLVED** with `_server.py`, `__main__.py`, Docker-aware
> Streamable HTTP startup, `/healthz`, and CLI/server tests. The core URI,
> finite-score, fragment, duplicate-RRF, strict-mode, and deterministic-ordering
> controls are also implemented in the current source. Historical finding text
> is retained as audit history rather than deleted.

The findings below are **seeded candidates** from a first grounded read of the
source (2026-08). Each is either CONFIRMED (behaviour verified) or CANDIDATE
(needs the campaign to confirm severity/validity). Several are now resolved; historical
problem statements remain below as audit evidence. A future session should
verify, prioritise, fix, and gate them via the per-finding workflow, updating the
`Status` rows.

Scope reviewed so far: `_core.py`, `_hybrid.py`, `_corpus_annoy.py`, `__init__.py`
(69 tests green; Python 3.10+ server target). The current CLI and server tests
are reviewed; the corpus-integration paths against a real corpus source.

---

## Summary

| ID | Pri | Status | Title |
| --- | --- | --- | --- |
| MCP-SEC-001 | P2 | RESOLVED | `_safe_uri` admits protocol-relative `//host` citation URLs |
| MCP-CORE-002 | P2 | RESOLVED | Non-finite scores (`NaN`/`inf`) flow into the JSON-RPC payload |
| MCP-HYB-001 | P2 | RESOLVED | Blind `except Exception` in retriever legs swallows bugs, not just unavailability |
| MCP-COUP-001 | P2 | CANDIDATE / OPEN | Tolerant duck-typed coupling to corpus internals not pinned to a verified API |
| MCP-CORE-003 | P3 | RESOLVED | Citation `anchor` appended to link without fragment validation |
| MCP-HYB-002 | P3 | RESOLVED | RRF tie-break determinism depends on leg iteration order (needs a test) |
| MCP-SRV-001 | P1 | RESOLVED | Server / transport layer delivered with stdio + Streamable HTTP |
| MCP-INT-001 | P1 | GATED (D2) | Real corpus→annoy field mapping unverified against installed source |

Priorities are provisional. GATED items require a maintainer decision (DESIGN §9)
and/or a live environment; do not implement unilaterally.

---

## MCP-SEC-001 — protocol-relative citation URLs

| | |
| --- | --- |
| Classification | SECURITY (CONFIRMED behaviour) |
| Priority | P2 |
| Source | `_core.py::_safe_uri`, used by `build_search_docs_result` |
| Evidence | `_safe_uri` returns the URI when its scheme is in `{"http","https",""}`. `urlparse("//evil.com/x").scheme == ""` with `netloc == "evil.com"`, so a **protocol-relative** URL passes the filter. A poisoned corpus `source_uri` of `//evil.com/…` then becomes a citation link that a browser/Markdown renderer resolves to an external host — defeating the "http(s)/relative-only" intent. |
| Expected invariant | A citation URL is either an absolute http(s) URL or a *same-document* relative path; protocol-relative (`//host`) and scheme-relative host forms are rejected (empty scheme **with** a non-empty `netloc`). |
| Suggested fix | In `_safe_uri`, reject when `scheme == "" and parsed.netloc` (protocol-relative), and consider rejecting a leading `//`. Add a test asserting `//evil.com`, `\\evil.com`, and `https://ok` behave as expected. |
| Exit criteria | Test covers protocol-relative + backslash + valid http(s)/relative; `build_search_docs_result` never emits a `//host` citation link. |
| Status | RESOLVED — protocol-relative, backslash/UNC, credentialed, and malformed absolute URLs are rejected and tested. |

## MCP-CORE-002 — non-finite scores enter the payload

| | |
| --- | --- |
| Classification | ROBUSTNESS / PROTOCOL (CONFIRMED behaviour) |
| Priority | P2 |
| Source | `_core.py::build_search_docs_result` (`float(c.score) if isinstance(c.score,(int,float)) else 0.0`) |
| Evidence | `isinstance(float("nan"), float)` is `True`, so a `NaN`/`inf` score passes straight through into `structuredContent.citations[].score`. `json.dumps(float("nan"))` emits `NaN`, which is **not valid JSON** per the spec and breaks strict JSON-RPC parsers on the client side. |
| Expected invariant | Emitted scores are finite floats; non-finite values are coerced (e.g. to `0.0`) or dropped. |
| Suggested fix | Replace the score guard with a finiteness check (`math.isfinite`), coercing non-finite to `0.0`. Test with `NaN`/`inf` chunk scores. |
| Exit criteria | No non-finite value can appear in a built result; test proves coercion. |
| Status | RESOLVED — `_coerce_finite_score` guarantees finite JSON-safe scores and regression tests cover NaN/infinities. |

## MCP-HYB-001 — blind exception swallowing hides bugs

| | |
| --- | --- |
| Classification | ROBUSTNESS / OBSERVABILITY |
| Priority | P2 |
| Source | `_hybrid.py` (`HybridRetriever.search`, `Bm25Retriever.search`), `_corpus_annoy.py` (`CorpusAnnoyRetriever.search`, doc-lookup) — all `except Exception: continue/return []` |
| Evidence | Resilience ("a down backend must not kill retrieval") is intended, but catching *all* exceptions also silently swallows genuine bugs (`TypeError`, `AttributeError`, `KeyError` from a coding error), which the corpus methodology explicitly forbids (no silent failures). A miscalibrated leg and a broken leg are indistinguishable, with no signal. |
| Expected invariant | Backend *unavailability* (import/connection) is tolerated silently; a *bug* is surfaced (logged at warning with the leg identity, or re-raised in a strict mode). |
| Suggested fix | Narrow the caught set and/or log the exception with the leg name; consider a `strict` flag. Keep the resilient default. |
| Exit criteria | Test: a leg raising a "bug" exception is logged/surfaced; a leg raising an "unavailable" exception is skipped; retrieval still returns other legs' results. |
| Status | RESOLVED — resilient mode logs and skips; strict mode re-raises and is tested. |

## MCP-COUP-001 — unpinned coupling to corpus internals

| | |
| --- | --- |
| Classification | ARCHITECTURAL DEBT (matches DESIGN §11) |
| Priority | P2 |
| Source | `_corpus_annoy.py::_doc_to_record` / `from_corpus_annoy`; `_hybrid.py::Bm25Retriever.from_corpus_sqlite` |
| Evidence | These use tolerant `getattr(...)`/dict fallbacks over many candidate field names (`normalized_text`/`text`/`content`; `source_uri`/`source`/`url`/`path`; `StorageQuery`/`.rows`) because the exact corpus API was not yet pinned. That is fragile: a corpus field rename silently degrades retrieval (empty text/URI) instead of failing loudly. |
| Expected invariant | The corpus is consumed via its **verified public API** (exact `CorpusDocument`/`SearchResult`/`SQLiteStorage`/`StorageQuery` fields), with a small integration test round-tripping one query. |
| Suggested fix | Follow DESIGN §11 steps 1–3 against the installed corpus source: pin the dense wiring, pin the BM25 wiring, decide the graph leg. Replace tolerant access with exact calls; keep a thin adapter. |
| Exit criteria | Integration test builds a tiny corpus and round-trips a query through the dense and BM25 legs; no silent empty-field degradation. |
| Status | OPEN (needs the installed corpus source to pin; verify the pure logic via seams meanwhile). |

## MCP-CORE-003 — citation anchor not fragment-validated

| | |
| --- | --- |
| Classification | ROBUSTNESS (minor) |
| Priority | P3 |
| Source | `_core.py::build_search_docs_result` (`link = link + ("#" + s["anchor"] ...)`) |
| Evidence | The `anchor` is control-stripped and 200-capped but not validated as a URL fragment; it is concatenated onto the citation link. Low risk (control chars already gone), but a malformed anchor can produce an odd link. |
| Expected invariant | The appended fragment is a well-formed, percent-encoded fragment. |
| Suggested fix | Percent-encode the anchor fragment; add a test. |
| Exit criteria | Anchor with spaces/`#`/reserved chars yields a valid fragment. |
| Status | RESOLVED — fragments are percent-encoded and replace existing fragments; tested. |

## MCP-HYB-002 — RRF fusion tie-break determinism

| | |
| --- | --- |
| Classification | CORRECTNESS (verify) |
| Priority | P3 |
| Source | `_hybrid.py::HybridRetriever.search` (`sorted(fused.items(), key=…, reverse=True)`) |
| Evidence | Equal fused scores are broken by dict insertion order (leg iteration order). `sorted` is stable, so the result is deterministic *given* input order, but this is not asserted; a change in leg order would reorder ties. |
| Expected invariant | Fusion output is deterministic and documented (e.g. stable by first-seen order, or by a secondary key). |
| Suggested fix | Add a determinism test; optionally a documented secondary tie-break (e.g. `doc_id`). |
| Exit criteria | Test pins tie-break behaviour. |
| Status | RESOLVED — sorting uses fused score, first-seen order, then identity; deterministic behavior is tested. |

## MCP-SRV-001 — server / transport layer (RESOLVED)

| | |
| --- | --- |
| Classification | ARCHITECTURE (gated) |
| Priority | P1 |
| Source | `_server.py`, `__main__.py`, `tests/test_cli.py`, `tests/test_server.py`. |
| Evidence | The SDK v2 server registers `search_docs`, optional resources, and `/healthz`; the CLI runs stdio or Streamable HTTP and has an explicit Docker profile. |
| Decision required | Resolved in favor of the official MCP Python SDK v2 `MCPServer`. |
| Exit criteria | Thin SDK shell delivered; local suite green. Keep a real MCP v2 client round-trip in dependency-enabled CI. |
| Status | RESOLVED in code; real-SDK client round-trip remains a CI gate. |

## MCP-INT-001 — real corpus→annoy mapping (GATED, D2)

| | |
| --- | --- |
| Classification | INTEGRATION (gated) |
| Priority | P1 |
| Source | `_corpus_annoy.from_corpus_annoy`, DESIGN §8/§9 (D2), §11. |
| Evidence | The end-to-end corpus build → annoy index → query path is documented but executes only with corpus/annoy installed and its field mapping verified against the installed versions. |
| Decision required | D2 — confirm the default vector index (annoy persistent vs corpus `SimilarityIndex` in-process vs SQLite FTS5 for hybrid). |
| Exit criteria | Integration test builds a tiny real corpus and round-trips a query; field mapping pinned (subsumes MCP-COUP-001). |
| Status | GATED — needs the installed corpus source + D2. |

---

## Positive controls to preserve (mcp)

| Control | Intent | Test |
| --- | --- | --- |
| Bounded untrusted citations | control-strip + length-cap + scheme-validated citation URLs in one chokepoint; semantic instructions remain untrusted | `tests/test_mcp_core.py` |
| RRF fusion by rank | dense + BM25 fused by rank, resilient to a failing leg | `tests/test_hybrid.py` |
| SDK-agnostic core | core imports & tests pass with no mcp SDK / corpus / annoy | both suites (run in a bare env) |
| No import-time side effects | importing the package loads no model/network/index | (add a guard test) |
