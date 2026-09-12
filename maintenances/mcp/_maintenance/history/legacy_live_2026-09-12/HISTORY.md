# History — compressed

Read only when the *why* behind a current rule is unclear. Nothing here is
current source truth.

---

## Prior campaign (pre-unified-kit)

A deep review of `scikitplot.mcp` produced three artifacts —
`MCP_DEEP_REVIEW_REPORT.md`, `MCP_REDESIGN_PLAN.md`, `MCP_VERIFICATION_MATRIX.md`
— plus closure responses to two audit rounds. Its durable outcomes:

- **The single-wire-protocol rule**, and the Python/SDK tier model (Tier L/S/N).
  Recorded in `MCP_COMPATIBILITY_POLICY.md`, which remains authoritative.
- **The `_server.py`-only SDK import rule**, which is why importing
  `scikitplot.mcp` needs no SDK.
- **The `DocsRetriever` Protocol seam**, which is why MCP has zero runtime
  imports of Corpus.
- **Strict wire validation** and the unknown-argument policy.
- **The stale-file lifecycle** (ACTIVE → `history/` → removed after two
  releases), implemented by `stale_lifecycle.py`.

Those artifacts are superseded as *plans* by the unified review kit's
`MCP_UNIFIED_DEFERRED_REVIEW_GUIDE.md` (runs M00–M12) and are archived under
`history/`. They remain the audit register for how the rules above were reached.

---

## Corpus campaign — what it changed for MCP

The Corpus review (R00–R16) and implementation (IMPL-01–IMPL-18) completed
before MCP's campaign began. Two consequences matter here.

**Renames.** `SearchResult` → `RetrievalHit`, `SearchConfig` →
`RetrievalConfig`, `SimilarityIndex` → `RetrievalIndex`, `ANNBackend` →
`VectorIndexBackend`, `MCPToolInput`/`MCPToolResult` → `ToolCallInput`/
`ToolCallResult`; `Document`, `LegacyPipelineResult` and `ChunkStrategy` deleted.

**MCP was unaffected in code.** Because retrieval arrives by injection, MCP has
no runtime import of Corpus, so the renames produced *six documentation
references* and zero breakage. That is the seam working as designed, and it is
worth knowing before someone proposes tightening it further.

**New contracts, built and tested rather than designed:** `RetrievalResponse`
with derived status and per-leg outcomes, `ErrorRecord`, `CapabilityStatus`
(seven states), `ComponentCatalog`, `EmbeddingManifest`, `ANNIndexArtifact` with
its ordinal→doc_id sidecar. Runs M03 and M04 map against these.

One Corpus decision constrains MCP directly: `ToolCallInput` and
`ToolCallResult` are **protocol-neutral payload shapes that live in Corpus**, and
a Corpus test asserts its `_types` module imports neither `pydantic` nor `mcp`.
They are named for MCP's shape, not owned by MCP's wire format.

---

## The maintenance set

`LIVE_V1` mirrors `scikitplot/corpus/_maintenance/` file-for-file. The principle
carried over: *a document that describes what a script can check should be
replaced by the script; one that records why cannot be.* Hence
`check_trackers.py`, which turned two documented boundary conventions into
enforced rules.
---

## M14 — Corpus + Annoy Docker/CI showcase (2026-08-21)

Added an explicit local `--corpus-annoy PATH` MCP CLI profile. The profile
forces the Corpus Annoy vector backend, defaults to the deterministic public
`HashEmbedder`, and can opt into a named model via
`--corpus-embedding-model`. `CorpusAnnoyRetriever` now supports an explicit
batch embedder and implements `get(doc_id)` so the MCP resource surface works
for the same indexed documents returned by search.

The gallery showcase builds a temporary Hamlet corpus from public
`scikitplot.corpus.HAMLET_TEXT`, verifies direct retrieval and CLI self-test,
and can opt into a real `scikitplot mcp --docker` HTTP subprocess plus official
`mcp.Client` query via `SCIKITPLOT_GALLERY_RUN_MCP_DOCKER=1`. Normal docs builds
do not bind a server.

Verification in the review harness: `164 passed, 2 skipped` for the full offline
MCP suite; local HashEmbedder/Corpus retrieval is also exercised end-to-end with
the exact `RetrievalIndex` seam using the dependency-free brute-force backend.
The native Annoy extension and MCP SDK are unavailable in this harness, so the
live Annoy+HTTP round trip remains a provisioned-CI gate rather than a claimed
PASS.
