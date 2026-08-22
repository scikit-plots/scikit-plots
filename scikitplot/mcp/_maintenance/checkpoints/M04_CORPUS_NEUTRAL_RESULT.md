# Run M04 — Corpus neutral-result integration

```text
run_id            M04
date              2026-08-21
source_sha256     ee7593f81b35bd90a1dd2ba03691aadd2fa491599bce5a2b85dd70f88b14bb2c
scope             MCP retrieval tier <-> Corpus neutral-result contracts
guide             MCP_UNIFIED_DEFERRED_REVIEW_GUIDE.md §9
mode              REVIEW + IMPLEMENTATION INCREMENT (first implementation of the campaign)
exit gate         MET — required invariant now holds and is guarded by tests
```

---

## 0. Scope note

Runs M00–M03 were review-only. This run was authorised to produce changed files,
restricted to the Corpus-facing retrieval sections. Unrelated logic was
deliberately left untouched: no change to fusion ranking, RRF, deduplication,
scoring, sanitisation, URI handling, the server tier, packaging, or capabilities.

---

## 1. Corpus contracts — read from the tree, not assumed

`scikitplot/corpus/_retrieval.py`, all publicly reachable from
`scikitplot.corpus`:

```text
RetrievalStatus   SUCCESS | EMPTY | DEGRADED | FAILED | CANCELLED
LegStatus         SUCCESS | EMPTY | DEGRADED | FAILED | SKIPPED
LegOutcome        leg, status, hit_count, generation, backend, error
RetrievalResponse hits, legs, query, status(derived)
ErrorRecord       code, category(SOURCE|PARSE|VALIDATION|CAPABILITY), message
```

`RetrievalStatus` matches the guide's five states **exactly**. Two Corpus design
decisions shaped this run:

- `RetrievalResponse` implements `__iter__`, `__len__`, `__getitem__` and
  `__bool__`, and its docstring records that it *"stands in for the plain list it
  replaced"*. Sequence compatibility was a deliberate migration strategy.
- `LegOutcome.__post_init__` **enforces** that a `DEGRADED`/`FAILED` leg carries
  an `error`, because *"an unexplained degradation is exactly as unhelpful as no
  degradation signal at all."*

Both were adopted rather than reinvented.

---

## 2. The required invariant was violated — confirmed

Guide §9: `FAILED retrieval != EMPTY retrieval`, and MCP must not say
*"No matching documentation"* when every retrieval path failed.

Before this run, `_core.py:307` did exactly that, unconditionally:

```python
if not safe:
    message = "No matching documentation was found for this query."
    ...
    "isError": False,
```

with every retriever returning a bare `list[RetrievedChunk]` and `strict=False`
by default — so a total backend failure reached the wire as a confident,
factually wrong "no such documentation", flagged **not** an error.

---

## 3. Design decision

**The Protocol does not change.** `RetrievalOutcome` subclasses `list`, so
`DocsRetriever.search(...) -> list[RetrievedChunk]` remains *true* and every
existing caller keeps working: `len()`, iteration, indexing, slicing and
`if results:` are unaffected. This is the minimal-impact route to the invariant,
and it is the same route Corpus took.

**The vocabulary is consumed, not redefined.** The status strings are exactly
`RetrievalStatus`'s values. `to_corpus_status()` resolves them to the real enum
**lazily**, so the module-scope boundary (`check_trackers.py`) still holds, and
`assert_matches_corpus()` fails the suite if the two ever diverge.

Rejected alternatives:

| Option | Why not |
|---|---|
| Return `corpus.RetrievalResponse` directly | Would make `scikitplot.corpus` a hard dependency of the MCP retrieval tier and break the base install. |
| Wrap rather than subclass `list` | Every existing caller and the Protocol annotation would need changing — large blast radius for no gain. |
| Raise on failure instead | Changes `strict=False` semantics, which callers rely on for resilience. |
| Add an out-of-band error channel | Two sources of truth; the guide asks for one status on the result. |

---

## 4. Changed files

```text
NEW  scikitplot/mcp/_outcome.py                        (+~330)
NEW  scikitplot/mcp/tests/test_mcp_retrieval_outcome.py (+~190, 13 tests)
MOD  scikitplot/mcp/_core.py                            (~40 lines, one branch + one import)
MOD  scikitplot/mcp/_corpus_annoy.py                    (~35 lines, dense leg)
MOD  scikitplot/mcp/_hybrid.py                          (~70 lines, fusion + lexical legs)
MOD  scikitplot/mcp/_maintenance/TRACKER.json           (regenerated via --update)
```

`_core.py` is touched in exactly one behavioural place — the `if not safe:`
branch — plus one import. The success path, sanitisation, citation building,
scoring and URI handling are untouched.

---

## 5. A real bug found *while* implementing

The first implementation still reported `EMPTY` for a fully-failed fusion. Cause,
in `HybridRetriever.search`:

```python
raw_hits = retriever.search(query, per_leg) or []
```

An empty `RetrievalOutcome` is **falsy**, so `or []` replaced a `FAILED` outcome
with a plain empty list and discarded the status — *the exact flattening this run
exists to remove*, reintroduced by a defensive idiom that predates it.

Fixed by testing `is None` instead, with a comment naming the trap, and pinned by
`test_fusion_does_not_discard_a_falsy_failed_leg`.

**This generalises:** any `or []`, `or ()` or truthiness guard applied to a
status-carrying sequence silently destroys the status. Worth a grep before M05.

A second correction followed: propagating an inner `FAILED` as `DEGRADED` made
"every leg failed → `FAILED`" unreachable. Inner status is now carried through
verbatim so Corpus's derivation rule works as written.

---

## 6. Verification

Status matrix, measured end to end:

| Case | status | hits | `isError` | message |
|---|---|---|---|---|
| all legs fail | `failed` | 0 | **True** | "Documentation retrieval failed; this is not a statement that no documentation matches…" |
| fail + ok | `degraded` | 1 | False | passages returned |
| fail + empty | `degraded` | 0 | False | "…but at least one retrieval path did not run, so this result may be incomplete." |
| all empty | `empty` | 0 | False | "No matching documentation was found for this query." |
| all ok | `success` | 1 | False | passages returned |
| **legacy bare `[]`** | `empty` | 0 | False | unchanged from before |

```console
$ python scikitplot/mcp/_maintenance/check_trackers.py
physical tracker matches the tree (18 source / 16 test files, 4074 / 2279 LOC)
EXIT=0

$ pytest scikitplot/mcp/tests/test_mcp_retrieval_outcome.py -q
13 passed

$ pytest scikitplot/mcp -q --maxfail=100 --ignore=<the 4 known-erroring files>
6 failed, 109 passed, 2 skipped
```

**Baseline was `6 failed, 96 passed, 2 skipped`.** Same six failures — all
pre-existing `pydantic` failures recorded as `MCP-M00-12`, none touched by this
change. **Zero regressions; +13 tests.**

Boundary re-verified after the change:

```text
import scikitplot.mcp  ->  corpus: False | pydantic: False
```

The drift gate *caught* the new file before `--update` (`DRIFT: totals.source_loc
+16%`, exit 1) — the gate did its job, and the reconciliation is recorded here
rather than performed silently.

---

## 7. What this does *not* fix

- **`MCP-D03` is only half closed.** Retrieval now reports its status truthfully,
  but the `DocsRetriever` Protocol still advertises `-> list[RetrievedChunk]`.
  That was deliberate (§3), and it means a third-party retriever returning a bare
  list is treated as before — correct, but statusless. Whether the Protocol
  should be widened is an M05/M06 decision, not a local one.
- `CANCELLED` is defined and rejected-if-unknown, but nothing in MCP produces it;
  there is no cancellation path in the retrieval tier yet.
- Corpus's richer `LegOutcome` fields (`generation`, `backend`, `ErrorRecord`
  with `code`/`category`) are **not** yet carried through. MCP's `LegRecord` uses
  a plain string `error`. Mapping `ErrorCategory` onto the wire is the natural
  next increment and belongs with M06 (tool result contract).
- `MCP-M00-12`'s six failures, `MCP-M00-07`, `MCP-D01`, `MCP-D02`, `MCP-D05` and
  `MCP-M00-10` are all untouched — out of the authorised scope.

---

## 8. Run record

```text
run_id                  M04
source_sha256           ee7593f81b35bd90a1dd2ba03691aadd2fa491599bce5a2b85dd70f88b14bb2c
                        (fourth distinct archive; anchor still stale -> MCP-M00-11)
scope                   Corpus-facing retrieval sections only
prior findings          re-verified as still live in this tree before proceeding:
                        _present() still collapses (M03-03); sdk_version still absent
                        (D01/M03-01); Protocol still list-typed and strict=False (D03).
                        MCP source diffs vs the M03 tree were formatting-only reflow.
confirmed               MCP-D03 violated exactly as predicted
implemented             neutral-result envelope; FAILED/DEGRADED/EMPTY distinguished
                        at the wire; 13 guard tests
found while fixing      `or []` flattening of a falsy failed outcome (§5)
regressions             NONE (96 -> 109 passing, same 6 pre-existing failures)
next exact action       M05 (SearchService ownership) — still the campaign pivot, now
                        blocking MCP-D02, MCP-D08 and the remaining half of MCP-D03.
                        Before M05, grep the tree for `or []` / `or ()` applied to
                        retrieval results: §5 shows the idiom silently destroys status,
                        and M05 will move result-shaping code between tiers.
                        RULESET.md, MCP_COMPATIBILITY_POLICY.md and DESIGN.md remain
                        absent (DESIGN.md is still cited from _hybrid.py:22,346).
```
