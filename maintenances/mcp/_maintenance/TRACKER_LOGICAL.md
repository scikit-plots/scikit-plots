# Logical Tracker — MCP contracts, invariants, and Corpus-derived drift

What the code **promises**. Not re-derivable from the tree; maintained by hand.

Machine-readable mirror: `TRACKER.json` → `logical.contracts`.

---

## 1. The contracts

| Contract | Module | Invariant that must not break |
|---|---|---|
| `DocsRetriever` | `_core.py` | **A Protocol, not a base class.** Implementations are *injected*; MCP never imports a retrieval backend. Widening it to fit one backend turns it into a capability lottery. |
| `SearchService` | `_core.py` | **Protocol-neutral.** Returns neutral result shapes, never MCP wire types. This is what lets the same service back a CLI or a library caller. |
| `HybridRetriever` | `_hybrid.py` | **Reciprocal Rank Fusion only.** No score-space fusion across metrics — adding a BM25 score to a cosine similarity yields a meaningless number. |
| `CorpusAnnoyRetriever` | `_corpus_annoy.py` | The **only** place Corpus and Annoy meet MCP, behind the Protocol, imported at **call time**. Module-scope import of either breaks the optionality contract. |
| `server_capabilities()` | `_capabilities.py` | Reports what **this build** can do. Never claims an unprobed capability; absence of the SDK is a reported state, not a crash. |
| transport | `_server.py` | **The only module permitted to import the MCP SDK.** A second SDK importer makes `[mcp]` non-optional for everyone. |
| version guard | `_version.py` | One wire protocol. SDK absence **degrades**; it must not fail `import scikitplot.mcp`. |

---

## 2. Cross-cutting invariants

| Invariant | Status | How it is (or should be) enforced |
|---|---|---|
| `import scikitplot.mcp` needs no SDK | **holds** — verified: only `numpy` is pulled in | should be a test, like Corpus's import gate |
| Exactly one module imports `mcp`/`pydantic` | **holds** — `_server.py` | should be a test |
| Corpus is not imported at module scope | **holds** — no runtime corpus import exists | should be a test |
| MCP tests skip cleanly without `[mcp]` | **BROKEN** — see §4 | `pytest.importorskip` |
| Capability entries are probed, not assumed | assumed to hold | unverified |

Three of these are true today and **nothing checks them**. That is the same
condition that let Corpus's import hygiene decay until a warnings-as-errors run
made 49 test files uncollectable. Converting them to tests is the cheapest,
highest-value work available in this module.

---

## 3. Corpus-derived drift — read before M00

The Corpus review + implementation campaign renamed public symbols and added
contracts. **The good news is structural:** MCP has *zero runtime imports of
Corpus*, so none of it breaks code here. The drift is confined to **documentation
and comments**.

Verified by grep across `scikitplot/mcp/**/*.py`:

| Old Corpus name | New name | Where it appears in MCP | Kind |
|---|---|---|---|
| `SearchResult` | `RetrievalHit` | `_core.py:86` | docstring |
| `SearchConfig` | `RetrievalConfig` | `_corpus_annoy.py:297` | comment |
| `SimilarityIndex` | `RetrievalIndex` | `_core.py:19`, `_corpus_annoy.py:24,29,67` | docstring/comment |
| `Document` | *(deleted; use `CorpusDocument`)* | `_corpus_annoy.py:334` | comment — prose use of the word, verify before editing |
| `ANNBackend` | `VectorIndexBackend` | none | — |
| `MCPToolInput` / `MCPToolResult` | `ToolCallInput` / `ToolCallResult` | none in MCP | — |

**This is a documentation fix, not a migration.** Six references across two
files. Do it in M00 so the rest of the campaign reads accurate prose.

### New Corpus contracts MCP will map against

Built and tested, not merely designed — this is what changed since the MCP guide
was written:

| Corpus contract | What MCP does with it |
|---|---|
| `RetrievalResponse` + `RetrievalStatus` | **Map carefully.** `DEGRADED` is not an error and not a success; a wire response that flattens it to either is wrong. This is run **M04**'s core question. |
| `LegOutcome` / `LegStatus` | Per-leg detail available for a diagnostics resource, if one is wanted |
| `ErrorRecord` | Already JSON-serialisable — a wire error payload should reuse it rather than inventing a second shape |
| `CapabilityStatus` (7 states) | `server_capabilities()` should **consume** this vocabulary, not define a parallel one |
| `ComponentCatalog` | Answers "what backends exist" without importing them |
| `ToolCallInput` / `ToolCallResult` | **Protocol-neutral payload shapes living in Corpus.** Adding a `pydantic` model or an `mcp` import to them moves wire concerns into Corpus and breaks the boundary. A Corpus test asserts `_types` imports neither — do not defeat it from this side. |

---

## 4. Known logical debt

| Item | Consequence today |
|---|---|
| **4 of 6 test files import `pydantic` unguarded** | Without `[mcp]` installed the MCP suite **fails collection** — 4 errors, 2 skipped, 0 passed. Only `test_protocol_in_memory.py` and `test_mcp_http_live.py` use `importorskip`. This is the same class as Corpus's O-7. |
| Three boundary invariants unverified | §2 — true but unchecked |
| `server_capabilities()` may define its own vocabulary | Should consume Corpus's `CapabilityStatus`; check in M03 |
| 31 markdown files for 16 source files | See `SUBMODULE_STRUCTURE.md` §3 |

---

## 5. Contracts deliberately absent

Recorded so nobody "discovers" a gap that was a decision.

| Absent | Why |
|---|---|
| A retrieval implementation inside MCP | Retrieval belongs to Corpus; MCP declares a Protocol |
| A second wire protocol | Single-wire-protocol rule; see `MCP_COMPATIBILITY_POLICY.md` |
| Agentic orchestration | Corpus owns A1/A2; MCP exposes it, does not reimplement it |
| Its own capability enum | Corpus's `CapabilityStatus` is the vocabulary |
| Durable session state | Corpus's agentic session is ephemeral by design; MCP must not persist it behind Corpus's back |
