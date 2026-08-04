# Session log — `scikitplot.corpus` hardening

Newest at top. One entry per session; keep it terse. The authoritative detail is
the review guide; this is the "what happened when" trail so a fresh chat can see
the arc without re-reading everything.

Status legend: RESOLVED = fixed + gated; PARTIAL = advanced + remainder scoped.

---

## 2026-08 — handoff / maintenance infrastructure

Built the permanent memory so any session resumes cleanly:
- Rewrote `corpus/MAINTAINING.md` into a full resume doc (campaign status,
  reusable primitives, compat rule, positive-control index, harness patterns).
- Added `_maintenance/METHODOLOGY.md` (this process) and this log.
- Deferred further corpus fixes by request; shifted focus to seed the
  `scikitplot.mcp` review (see `mcp/MAINTAINING.md` + `mcp/_maintenance/`).

Grounded campaign state at handoff: **17 RESOLVED, 3 PARTIAL** (NET-002 IP-pinning,
NET-003 Fetcher/Transport, PKG-001 run-manifest), rest OPEN.

## 2026-07 — archive + capability + typing sweep

- CORPUS-ARC-001 RESOLVED — `stream_copy_bounded`; streamed extraction, actual-byte budget, bounded memory (`tests/test__archive_budget.py`).
- CORPUS-ARC-003 RESOLVED — private staging + atomic publish (`TestTransactionalPublish`).
- CORPUS-ARC-002 RESOLVED — shared depth cap via `_archive_ctx`/`ArchiveNestingError` (`_readers/tests/test__zip_depth.py`).
- CORPUS-SEC-001 RESOLVED — pickle integrity gate + post-load type validation (`_export/tests/test__export_security.py`).
- CORPUS-DOC-002 RESOLVED — `load_documents` example fixed; guard asserts the shown message.
- CORPUS-XML-001 RESOLVED — `_readers/_xml_safety.py`; XXE/billion-laughs on both backends.
- CORPUS-PKG-001 PARTIAL — `capability_snapshot()` (`tests/test__capabilities.py`); manifest wiring remains.
- Python 3.8→3.15+ audit across all touched files; removed the sole `usedforsecurity=` breaker in `make_doc_id`; recorded the standing rule.
- TYP-001: verified sub-finding — no PEP 561 `py.typed` marker, so shipped stubs are inert (fix is package-wide → maintainer call).

## 2026-07 (earlier) — SSRF + storage + identity + caching

- CORPUS-NET-001 RESOLVED — per-hop redirect validation (`_get_with_validated_redirects`).
- CORPUS-NET-002 PARTIAL — fail-closed DNS + all-record validation (`_resolve_and_validate`); IP-pinning remains.
- CORPUS-NET-003 PARTIAL — SSRF policy single-sourced (WebReader delegates); transport extraction remains.
- CORPUS-STO-001/002 RESOLVED — SQLite transactional writes; JSONL durability.
- CORPUS-ID-001 RESOLVED — full-content doc-id v2 (was `text[:64]`).
- CORPUS-CACHE-001 RESOLVED — content-addressed embedding cache key.
- CORPUS-TMP-001 RESOLVED — `_atomic` shared across cache/export/storage/download/multimodal.
- CORPUS-RES-001 RESOLVED — NLTK downloads opt-in.
- CORPUS-API-001 RESOLVED — facade identity/`__all__` (NormalizerConfig binding left OPEN).

## 2026-06/07 — search path (first area)

- CORPUS-ALG-001/ALG-002/MCP-001 RESOLVED — `_similarity/_backends.py` unified cosine contract + quality gate + provenance; builder ndarray-truthiness; MCP retriever onto the corpus `query` seam. See `CORPUS_SEARCH_RESOLUTION_LOG.md`.
