# Registry — open findings, boundaries, run sequence

The authoritative register. Findings are marked resolved **with evidence**,
never deleted. Machine-readable mirror: `STATE.json`.

---

## 1. Open findings (M00 input — revalidate, do not accept)

| ID | Sev | Finding |
|---|---|---|
| `MCP-M00-01` | P2 | **The suite fails collection without `[mcp]`.** 4 of 6 test files import `pydantic` unguarded → `2 skipped, 4 errors, 0 passed`. `test_protocol_in_memory.py` and `test_mcp_http_live.py` use `pytest.importorskip` and behave correctly. Same class as Corpus's O-7, where a clean install could not run the suite. |
| `MCP-M00-02` | P3 | **Corpus-derived documentation drift.** Six references to renamed Corpus symbols across `_core.py` and `_corpus_annoy.py` — **all docstrings/comments; zero runtime imports of Corpus exist.** Map in `TRACKER_LOGICAL.md` §3. |
| `MCP-M00-03` | P3 | **Three boundary invariants hold but nothing checks them.** `check_trackers.py` now covers two statically; the third (no SDK on import) wants a test. |
| `MCP-M00-04` | P3 | **31 markdown files for 16 source modules** (ratio 1.9). Disposition in `SUBMODULE_STRUCTURE.md` §3: archive 4, fold 2, drop 1. |

Evidence for each is in `STATE.json` → `open_findings`.

---

## 2. Boundary registry

| Direction | Rule | Status |
|---|---|---|
| MCP → SDK | only `_server.py` imports `mcp`/`pydantic` | **HOLDS**, gated |
| MCP → Corpus | never at module scope; injected via `DocsRetriever` | **HOLDS**, gated |
| MCP → Annoy | same; reached only through `CorpusAnnoyRetriever` | **HOLDS**, gated |
| Corpus → MCP | none. Corpus must not import MCP | **HOLDS** |
| Corpus ← MCP naming | `ToolCallInput`/`ToolCallResult` live in Corpus and are protocol-**neutral**; a Corpus test asserts `_types` imports neither `pydantic` nor `mcp` | **HOLDS** |

**The seam.** `DocsRetriever` is a Protocol. MCP declares the shape it needs;
implementations are injected. `CorpusAnnoyRetriever` is the only place Corpus and
Annoy meet MCP, behind that Protocol, at call time.

---

## 3. Corpus contracts to map (runs M03–M04)

Built and tested, not merely designed — this is what changed while MCP waited.

| Contract | The question it poses |
|---|---|
| `RetrievalResponse` / `RetrievalStatus` | **M04's core question.** `DEGRADED` is neither success nor error. A wire response that flattens it to either is wrong — and flattening is the default thing an adapter does. |
| `LegOutcome` / `LegStatus` | Is a per-leg diagnostics resource wanted, so a client can see *which* evidence path degraded? |
| `ErrorRecord` | Already JSON-serialisable. Reuse it as the wire error payload rather than inventing a second shape. |
| `CapabilityStatus` (7 states) | `server_capabilities()` should **consume** this vocabulary. `BROKEN` vs `ABSENT` matters to a client: reinstall versus install. |
| `ComponentCatalog` | Answers "what backends exist" **without importing them** — exactly what a capability resource needs. |
| `EmbeddingManifest`, `ANNIndexArtifact` | Relevant to ANNOY's campaign; MCP only needs to not contradict them. |

---

## 4. Run sequence

Guide: `MCP_UNIFIED_DEFERRED_REVIEW_GUIDE.md` in the review kit.

```text
M00  snapshot + deferred-issue revalidation     <- NEXT
M01  packaging and Python tier contract
M02  import / optionality contract
M03  runtime capability truth
M04  Corpus neutral-result integration
M05  SearchService ownership
M06  tool/resource schema and strict validation
M07  transport lifecycle
M08  plugins and integrations
M09  security / trust boundaries
M10  agentic capability boundary
M11  compatibility / migration
M12  final real-SDK closure matrix
```

**Do not begin implementation before the runs close.** The Corpus campaign's
value came from establishing the big picture first: 55 findings and 23 *disproofs*
before a line of implementation, which is why the implementation went 18
increments without a red suite.

Two runs are worth flagging in advance:

- **M02** can now be *proven* rather than argued: importing `scikitplot.mcp`
  loads neither the SDK nor a retrieval backend, and the gate enforces both
  import rules statically.
- **M04** is the run with real content, because the Corpus contracts it maps
  against now exist. Before, it would have been design against a design.

---

## 5. Project sequence

```text
Corpus   review COMPLETE (R00–R16) | implementation COMPLETE (IMPL-01–18)
MCP      NEXT — run M00
ANNOY    after MCP
CLI      after ANNOY
         ↓
   cross-module consolidation → final implementation DAG → code
```
