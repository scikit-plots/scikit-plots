# Submodule Structure — where things go, and what to drop

MCP is 16 source files and 31 markdown files. This file answers *where does a new
thing go*, and records what should be removed.

---

## 1. The shape today

```text
scikitplot/mcp/
│
├── boundary ──────── _server.py        THE ONLY SDK IMPORTER — wire protocol,
│                                       tool/resource registration
│
├── contracts ─────── _core.py          DocsRetriever Protocol, SearchService
│                     _capabilities.py  server_capabilities()
│                     _version.py       SDK tier + single-wire-protocol rule
│
├── retrieval ─────── _hybrid.py        HybridRetriever (RRF)
│   (behind the       _corpus_annoy.py  the real Corpus + Annoy wiring
│    Protocol)        _demo.py          in-memory demo retriever
│
├── entry ─────────── __main__.py       CLI, arg parsing, transport selection
│
├── integrations/ ─── framework adapters (agno, openclaw)
├── plugins/ ──────── client plugin bundles
└── _maintenance/ ─── this folder (no runtime code)
```

The layout is intentionally **flat**: nine root modules, no deep packages. At
3 182 source LOC that is correct — packaging this would add navigation cost for
no isolation gain.

---

## 2. Where does a new thing go?

| You are adding | Put it | Also do |
|---|---|---|
| A new MCP tool or resource | `_server.py` | schema + strict-validation test + `server_capabilities()` entry |
| A new retrieval source | a new module implementing `DocsRetriever` | import its backend at **call time**; never at module scope |
| A new fusion strategy | `_hybrid.py` | rank fusion only — no score-space fusion across metrics |
| A new framework adapter | `integrations/<name>/` | depend on the public API only, never on `_server` |
| A new client plugin bundle | `plugins/` | no Python; manifest + prompts only |
| Anything importing `mcp` or `pydantic` | `_server.py`, or nowhere | the gate will reject a second importer |
| Retrieval scoring or ranking logic | **not here** | that belongs to Corpus; MCP consumes contracts |

**When in doubt:** a new root-level module is fine. A new subpackage needs a
reason beyond tidiness.

---

## 3. Documentation debt — the module's largest

**31 markdown files for 16 source modules**, sixteen of them live in
`_maintenance/`. Several predate the unified review kit and overlap with it.

### Disposition

| File | Verdict | Why |
|---|---|---|
| `MAINTAINING.md` | **KEEP** — updated | Entry point |
| `README.md` | **KEEP** | User-facing |
| `_maintenance/RULESET.md` | **KEEP** | Durable rules |
| `_maintenance/MCP_COMPATIBILITY_POLICY.md` | **KEEP** | The 14-point Python/SDK tier policy; nothing supersedes it |
| `_maintenance/DESIGN.md` | **KEEP** | Records *intent* — not derivable from code |
| `_maintenance/MCP_VERIFICATION_MATRIX.md` | **FOLD** into `VERIFICATION.md` | Same role, two files |
| `_maintenance/MCP_DEEP_REVIEW_REPORT.md` | **ARCHIVE** → `history/` | Superseded by the unified review kit's `MCP_UNIFIED_DEFERRED_REVIEW_GUIDE.md` |
| `_maintenance/MCP_REDESIGN_PLAN.md` | **ARCHIVE** → `history/` | Its waves are now the M00–M12 run sequence |
| `_maintenance/MCP_CLOSURE_AUDIT_RESPONSE.md` | **ARCHIVE** → `history/` | Point-in-time response |
| `_maintenance/MCP_CLOSURE_R1_R6_RESPONSE.md` | **ARCHIVE** → `history/` | Point-in-time response |
| `_maintenance/METHODOLOGY.md` | **DROP** | A copy of the shared review discipline; the kit is the source |
| `_maintenance/CHANGELOG_IDEMPOTENT.md` | **FOLD** into `HISTORY.md` | A changelog fragment |
| `_maintenance/CI_OUTPUT_ROUTING.md` | **KEEP** | Operational, still true |
| `_maintenance/DOCKER.md` | **KEEP** | Operational |
| `_maintenance/IDEMPOTENT_TESTING.md` | **KEEP** | Test-design contract |
| `_maintenance/STRICT_WIRE_VALIDATION.md` | **KEEP** | Wire contract |
| `_maintenance/UNKNOWN_ARGUMENTS_AND_MANIFESTS.md` | **KEEP** | Wire contract |
| `_maintenance/STALE_FILES.md` | **KEEP** | The lifecycle it describes is sound and already applied |
| `_maintenance/history/*` (6 files) | **KEEP as-is** | Correctly archived already |

**Net: 4 archived, 2 folded, 1 dropped** — `_maintenance/` goes from 16 live
markdown files to 9, without losing a single decision.

The existing `stale_lifecycle.py` already implements ACTIVE → `history/` →
removed. **Use it** rather than deleting by hand; it never deletes without
`--apply`.

---

## 4. Rules for expanding MCP

### The SDK import rule is absolute

`_server.py` is the only module permitted to `import mcp` or `import pydantic`.
A second importer makes `[mcp]` a hard dependency for everyone who imports
`scikitplot.mcp`. **`check_trackers.py` enforces this.**

### Retrieval backends are injected, never imported

Corpus and Annoy reach MCP through the `DocsRetriever` Protocol at call time.
Verified today: importing `scikitplot.mcp` pulls in only `numpy`. The gate
enforces it.

### Do not create a parallel vocabulary

| You want | It already exists as |
|---|---|
| "is this capability usable?" | Corpus's `CapabilityStatus` (7 states) |
| "how did retrieval go?" | Corpus's `RetrievalStatus` / `LegStatus` |
| "what went wrong?" | Corpus's `ErrorRecord` — already JSON-serialisable |
| "which backends exist?" | Corpus's `ComponentCatalog` |

MCP **maps** these to the wire. It does not redefine them. A second vocabulary
for the same question is worse than none, because consumers must learn which one
applies where.

### Every test guards its optional dependency

`pytest.importorskip` at the top, as `test_protocol_in_memory.py` does. Without
it the whole suite fails to collect on a machine without `[mcp]` — which is the
state the tree is in today.

---

## 5. Submodule review checklist

```text
[ ] Does the change import mcp/pydantic outside _server.py?        -> reject
[ ] Does it import corpus or annoy at module scope?                -> reject
[ ] Does it define a status/error vocabulary Corpus already has?   -> reuse
[ ] Does it reimplement scoring, ranking or fusion?                -> belongs in Corpus
[ ] Does every new test guard its optional dependency?
[ ] Does every new public type have a row in TRACKER_LOGICAL.md?
[ ] Does every new tool have a schema + strict-validation test?
[ ] Does every new capability get *probed*, not assumed?
[ ] python _maintenance/check_trackers.py                          -> exit 0
[ ] pytest scikitplot/mcp -q                                       -> green
```

---

## 6. Innovative directions, with prerequisites

| Direction | Needs first | Value |
|---|---|---|
| **Three boundary tests** (no-SDK import, no-corpus import, single SDK importer) | nothing — all three properties hold today | Converts convention into contract; cheapest high-value work in the module |
| **Guard the 4 unguarded test files** | nothing | The suite becomes collectable without `[mcp]` |
| **Map `RetrievalStatus` to the wire** | Corpus contracts (built) | `DEGRADED` stops being flattened into success or error — the M04 question |
| **Reuse `ErrorRecord` as the wire error payload** | Corpus contracts (built) | One error shape end to end, already serialisable |
| **Consume `CapabilityStatus` in `server_capabilities()`** | Corpus contracts (built) | `BROKEN` vs `ABSENT` becomes visible to clients |
| **Expose `ComponentCatalog` as a resource** | Corpus contracts (built) | Clients can ask what exists without the server importing backends |
| **Per-leg diagnostics resource** | `LegOutcome` (built) | Clients can see *which* evidence path degraded |

The first two are hours of work and remove real defects. The middle three are the
substance of runs M03–M04 and are now unblocked, because the Corpus contracts they
map against are **built and tested**, not merely designed.

---

## 7. What *not* to do

| Tempting | Why not |
|---|---|
| Import Corpus at module scope "for typing" | Use `TYPE_CHECKING`; a runtime import collapses the seam |
| Add `pydantic` models to Corpus's `ToolCallInput`/`ToolCallResult` | Those are protocol-**neutral** payload shapes; a Corpus test asserts `_types` imports neither `pydantic` nor `mcp` |
| Reimplement RRF differently here | Two implementations that disagree is worse than one that is imperfect |
| Widen `DocsRetriever` to fit one backend | Turns a Protocol into a capability lottery |
| Add a second wire protocol | Single-wire-protocol rule; see `MCP_COMPATIBILITY_POLICY.md` |
| Persist agentic session state | Corpus's session is ephemeral **by design**; persisting it here reintroduces the seven prerequisites it was designed to avoid |
| Write another markdown file | Ratio is already 1.9. Write the test instead |
