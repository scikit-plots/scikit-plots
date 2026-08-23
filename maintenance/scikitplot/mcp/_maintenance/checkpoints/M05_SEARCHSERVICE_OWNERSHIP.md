# Run M05 — SearchService ownership

```text
run_id            M05
date              2026-08-21
source_sha256     ee7593f81b35bd90a1dd2ba03691aadd2fa491599bce5a2b85dd70f88b14bb2c
scope             SearchService ownership; integrations; CLI self-test; contract records
guide             MCP_UNIFIED_DEFERRED_REVIEW_GUIDE.md §10
mode              REVIEW + IMPLEMENTATION INCREMENT
exit gate         MET — ownership decided and implemented; MCP-D02 and MCP-M00-07 closed
```

---

## 1. Pre-run sweep (recommended at the end of M04)

M04 found that `or []` silently destroys a status-carrying result. Sweeping the
tree before touching anything:

| Site | Verdict |
|---|---|
| `_core.py:287` `islice(chunks or (), limit)` | **safe** — does not rebind `chunks`, so `status_of(chunks)` still sees the outcome |
| `_hybrid.py:126` `items or ()` | **safe** — `_deduplicate` returns a plain `list` |
| `_corpus_annoy.py:195` `self._index.query(...) or []` | **latent** — ANN index returns tuples today; would flatten if ever handed a status-carrying value |
| `_hybrid.py:344` `self._fts(query, k) or []` | **safe** — callable returns tuples |
| `_hybrid.py:449` `getattr(result, "documents", result) or []` | **latent** — sits on the Corpus `SQLiteStorage` seam |

No live defects. Two latent risks recorded rather than preemptively changed.

---

## 2. The decisive finding: Option B is already done

The guide asks to choose between keeping `SearchService` server-tier (A) or
**extracting a neutral search coordinator to Corpus** (B).

Corpus already has one:

```text
scikitplot.corpus.RetrievalIndex.search(
    query: str, *, config=None, query_embedding=None
) -> RetrievalResponse
```

Its docstring: *"Hits sorted by descending score, plus a per-leg account of how
the search went."* Corpus also ships `Retriever`, `LexicalRetriever`,
`DenseRetriever`, `GraphRetriever`, `RetrievalConfig`, `RetrievalQuery` and
`AgenticRetrievalSession` — a complete neutral retrieval layer, all publicly
reachable.

**There is nothing to extract.** Option B's work item is complete on the Corpus
side, which makes **Option A** the answer: MCP's `SearchService` stays server-tier
and framework integrations use a neutral service.

---

## 3. But Option A did not work as written — and why

`SearchService` was not a wire adapter. Reading it (`_server.py:110-175`), its
responsibilities were:

| Responsibility | Tier |
|---|---|
| query type/emptiness/length validation | **neutral** |
| `k` type and range validation | **neutral** |
| bounded concurrency (`BoundedSemaphore`) | **neutral** |
| retriever invocation + failure wrapping | **neutral** |
| `build_search_docs_result` shaping | **neutral** (`_core.py`, Tier-L) |
| `CitationOutput` / `SearchDocsOutput` construction | wire |

Its own docstring said *"Validated, bounded orchestration independent from the
MCP SDK itself"* — self-described as orchestration, not adaptation. Roughly one
line in six was wire-specific, and the other five sat behind a module-scope
`pydantic` import.

That is precisely why `MCP-D02` existed. The agno toolkit wanted the
orchestration and had no way to get it without the wire models. Its code made
this visible:

```python
out = self._service.search(...)          # build pydantic models
return {..., "citations": [c.model_dump() for c in out.citations], ...}
```

It constructed pydantic models and immediately tore them down. The round trip was
pure cost — and the sole reason three user-facing "Legacy Retrieval tier" claims
were false.

Meanwhile `build_search_docs_result` already emitted the toolkit's exact shape,
pydantic-free:

```console
Tier-L structuredContent keys: ['citations','count','message','passages','query','security']
toolkit needs               : ['citations','count','message','passages','query']
all present?: True
```

**Decision — Option A, with the tier boundary corrected:** the neutral
orchestration moves *down* to Tier-L; `SearchService` becomes the thin wire
adapter Option A assumed it already was. The neutral coordinator is **not**
duplicated into Corpus — Corpus's `RetrievalIndex` remains the neutral
coordinator for corpus-backed retrieval, while MCP's Tier-L coordinator serves
the base install, where `scikitplot.corpus` may be absent. Responsibilities
differ clearly, which is the guide's condition for having both.

---

## 4. Implementation

```text
NEW class scikitplot.mcp.SearchCoordinator          (_core.py, Tier-L, public)
MOD scikitplot.mcp._server.SearchService            -> thin adapter, delegates entirely
MOD integrations/agno/docs_toolkit.py               -> uses SearchCoordinator
MOD __main__.py                                     -> --self-test uses SearchCoordinator
MOD integrations/__init__.py, integrations/README.md, docs_toolkit docstring
MOD _maintenance/TRACKER.json                       -> MCP-M00-07 correction
```

`SearchService` now reimplements **nothing**: validation and concurrency bounds
are inherited from the coordinator, so the two cannot drift.

### `--self-test` was contradicting its own tests

`__main__.py:544` ran `SearchService(...).search(...)` then
`result.model_dump(mode="json")` — the same build-and-tear-down. The tests are
named `test_backend_self_test_is_repeatable_and_avoids_server_creation`, so the
intended contract was already "no server tier"; the implementation used it
anyway. Repointed to the coordinator, which is what the names assert.

---

## 5. Verification

```console
$ pytest scikitplot/mcp -q --maxfail=100 --ignore=<the 4 known-erroring files>
1 failed, 114 passed, 2 skipped
```

Progression across the campaign, same command:

```text
M00-R baseline    6 failed,  96 passed
after M04         6 failed, 109 passed
after M05         1 failed, 114 passed
```

Five failures closed **by fixing code, not by guarding tests**:

| Test | Now |
|---|---|
| `test_docs_toolkit_is_sdk_free_and_read_only` | **passes** — the toolkit genuinely is SDK- and pydantic-free |
| `test_backend_self_test_is_repeatable_and_avoids_server_creation` | **passes** |
| `test_backend_self_test_can_require_exact_canary` | **passes** |
| `test_backend_self_test_required_match_fails_closed` | **passes** |
| `test_backend_self_test_expected_doc_id_fails_closed` | **passes** |

The one remaining failure is
`test_searchservice_lazily_requires_pydantic_but_is_reachable`, which requires
`pydantic` **by name and by design**. It is a legitimately server-tier test and
needs an `importorskip` guard, not a code change — the "guard" half of
`MCP-M00-12`.

Base-install behaviour, measured with `pydantic`/`mcp`/`starlette` blocked:

```console
toolkit constructed and searched WITHOUT pydantic
keys: ['citations','count','message','passages','query'] | count: 1

$ python -m scikitplot.mcp --self-test          exit=0   (previously a traceback)
$ python -m scikitplot.mcp --help               exit=0
$ --list-capabilities / --print-effective-config exit=0
```

Gate and boundary after the change:

```console
$ check_trackers.py
physical tracker matches the tree (18 source / 16 test files, 4235 / 2279 LOC)
EXIT=0

import scikitplot.mcp -> corpus: False | pydantic: False
```

---

## 6. Findings closed

| ID | Status |
|---|---|
| `MCP-D02` | **CLOSED** — the integration no longer imports `_server`; guide §7 satisfied |
| `MCP-M00-07` | **CLOSED** — a protocol-neutral service layer now exists, and `TRACKER.json` records reality: `SearchService` is `_server.py`/wire, `SearchCoordinator` is `_core.py`/neutral |
| `MCP-D08` (partial) | The three false "Legacy Retrieval tier" claims are now **true**, and name `SearchCoordinator` rather than `SearchService`. The broader SDK-free/protocol-neutral terminology audit remains open. |
| `MCP-M00-12` (partial) | 5 of 6 failures closed by fixing code; 1 remains, needing a guard |

---

## 7. What this does not fix

- `test_searchservice_lazily_requires_pydantic_but_is_reachable` still needs an
  `importorskip` guard (deliberately not added here: guarding is a test change,
  and the remaining four collection errors of `MCP-M00-01` belong to the same
  decision).
- `MCP-M00-01`'s four collection errors are untouched.
- `MCP-D01` / `M03-01..03` (capability truth) untouched — `sdk_version` and
  `sdk_compatible` are still absent.
- `MCP-D03`'s Protocol question stays open: `DocsRetriever.search` is still
  annotated `-> list[RetrievedChunk]`. `SearchCoordinator` passes the outcome
  through to `build_search_docs_result` unchanged, so M04's status survives, but
  the Protocol still cannot *require* it.
- `MCP-M00-10` (packaging marker), `MCP-M00-11` (stale anchor), `M01-01`
  (undeclared `starlette`) untouched.
- `RULESET.md`, `MCP_COMPATIBILITY_POLICY.md`, `DESIGN.md` still absent;
  `DESIGN.md` is still cited from `_hybrid.py:22,346`.

---

## 8. Run record

```text
run_id                  M05
scope                   SearchService ownership
decision                Option A, with the tier boundary corrected: neutral orchestration
                        moves DOWN to Tier-L (SearchCoordinator); SearchService becomes the
                        wire adapter it was assumed to be. Corpus's RetrievalIndex remains
                        the neutral coordinator for corpus-backed retrieval; MCP's serves
                        the base install. Responsibilities differ clearly, satisfying the
                        guide's "do not create both unless" condition.
evidence                Corpus RetrievalIndex.search -> RetrievalResponse already exists
                        SearchService was ~5/6 neutral by responsibility
                        build_search_docs_result already emitted the toolkit's exact shape
closed                  MCP-D02, MCP-M00-07, 5 of 6 MCP-M00-12 failures, 3 false doc claims
regressions             NONE (96 -> 114 passing across M04+M05)
next exact action       M06 (tool/resource schema and strict validation). Now well-placed:
                        SearchCoordinator.validate() is the single Tier-L validation point,
                        so M06 can reconcile it against the JSON Schema the SDK derives and
                        against _forbid_unknown_tool_arguments, rather than auditing
                        validation scattered across two tiers.
                        Carry into M06: map Corpus ErrorRecord.category onto the wire
                        (deferred from M04), and decide the guard for the one remaining
                        server-tier test together with MCP-M00-01's four collection errors.
```
