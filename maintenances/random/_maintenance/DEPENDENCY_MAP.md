# Dependency Map — antecessors, successors, and review order

**Identical in every `_maintenance/` folder.** If you edit it, edit all of them.

Derived from the source tree by AST, not asserted. Regenerate with
`_maintenance/dependency_map.py`.

---

## 1. The graph

```text
                    cexternals/_annoy          (no outgoing edges — standalone)
                         ▲   ▲   ▲
        cython-extern ×6 │   │   │ ×1 cython-extern
       module-scope   ×2 │   │   └──────────────── random
        deferred      ×1 │   └── ×2 cython-extern ─ memmap
                       annoy
                         ▲
                         │ ×4 deferred
                      corpus
                         ▲
                         │ ×5 deferred
                        mcp
```

Plus one edge outside the family: `_compat → cexternals` (module-scope ×1).

**There are no cycles.** Every edge points one way, so the review order below is
a topological sort rather than a preference.

---

## 2. Edge table

| From | To | Kind | Sites | Meaning |
|---|---|---|---:|---|
| `annoy` | `cexternals` | cython-extern | 6 | ABI mirrored from `annoylib.h`, `kissrandom.h`, `annoy_type_support.h` |
| `annoy` | `cexternals` | module-scope | 2 | imports the `Annoy` C type at import time |
| `annoy` | `cexternals` | deferred | 1 | call-time |
| `memmap` | `cexternals` | cython-extern | 2 | `mman.h` |
| `random` | `cexternals` | cython-extern | 1 | `kissrandom.h` |
| `_compat` | `cexternals` | module-scope | 1 | outside the family |
| `corpus` | `annoy` | **deferred** | 4 | **all call-time** — this is why importing Corpus does not load Annoy |
| `mcp` | `corpus` | **deferred** | 5 | **all call-time** — this is why importing MCP loads neither Corpus nor an SDK |
| `annoy`, `_cli` | `config` | deferred | 2 | configuration |
| `utils` | `_testing` | module-scope | 5 (tests only) | test helpers |

**`cexternals` has zero outgoing edges.** Its standalone claim is verified, not
assumed.

### The two edges that carry the architecture

`corpus → annoy` and `mcp → corpus` are **entirely deferred**. Every one of the
nine sites is a call-time import. That is not an accident — it is what makes
`[corpus]` and `[mcp]` genuinely optional, and it is verified: importing
`scikitplot.mcp` pulls in only `numpy`.

If either edge ever becomes module-scope, three optionality contracts break at
once. Corpus's and MCP's `check_trackers.py` both gate this.

---

## 3. Antecessor / successor by submodule

| Submodule | Antecessors (must be stable first) | Successors (break if this changes) |
|---|---|---|
| `cexternals/_annoy` | — | `annoy`, `memmap`, `random`, `_compat` |
| `random` | `cexternals` | — |
| `memmap` | `cexternals` | — |
| `annoy` | `cexternals` | `corpus` (deferred) |
| `corpus` | `annoy` (deferred) | `mcp` (deferred) |
| `mcp` | `corpus` (deferred) | — |
| `_cli` | `config`, `utils` | — |

**Partial dependencies** — where a successor uses only part of an antecessor:

| Edge | Uses only |
|---|---|
| `random` → `cexternals` | `kissrandom.h` |
| `memmap` → `cexternals` | `mman.h` |
| `annoy` → `cexternals` | `annoylib.h`, `kissrandom.h`, `annoy_type_support.h`, the `Annoy` type |
| `corpus` → `annoy` | the ANN backend seam only — one of four `VectorIndexBackend` implementations |
| `mcp` → `corpus` | retrieval contracts only, through the `DocsRetriever` Protocol |

`kissrandom.h` is the one header with **two** consumers. A change to it touches
index construction *and* user-facing randomness together.

---

## 4. Review order, and why

```text
1. corpus        ✅ COMPLETE          most successors depend on its contracts
2. mcp           ▶ M00 pending        thin adapter; needs corpus contracts stable
3. annoy family  ▶ A00 next           cexternals first within the family
4. _cli          ⏳ after annoy        consumes everything
```

Within the annoy family the order is forced by the graph:

```text
cexternals  ──►  random   (kissrandom.h)
            ──►  memmap   (mman.h)
            ──►  annoy    (3 headers + the C type)
```

`cexternals` is reviewed **first** because it has three successors and no
antecessors. `random` and `memmap` are independent of each other and can be
reviewed in either order. `annoy` is last in the family: it consumes the most,
and `A15` depends on Corpus contracts that are already complete.

**Why corpus went first, in retrospect.** It has the most successors
(`mcp` directly, `_cli` transitively), so its contract churn is the most
expensive. Reviewing it last would have invalidated work done above it — and in
fact the Corpus renames did produce drift in MCP, though only in documentation,
because the edge is deferred.

---

## 5. Cross-campaign findings

Facts one campaign establishes that another must honour.

| From | To | Fact |
|---|---|---|
| Corpus IMPL-13 | `annoy` | `ANNIndexArtifact` exists — index dir + versioned ordinal→doc_id sidecar + manifest. **Unblocks A15.** |
| Corpus IMPL-12 | `annoy` | `VectorIndexBackend` declares 11 capability members; Annoy's backend must answer all truthfully |
| Corpus IMPL-13 | `annoy` | `supports_persistence=True` is declared **now**. If Annoy's native save/load does not round-trip, that declaration is wrong |
| Corpus IMPL-02 | `mcp` | Renames produced 6 documentation references in MCP and **zero breakage** — the deferred edge working as designed |
| Corpus | `mcp` | `ToolCallInput`/`ToolCallResult` are protocol-**neutral** and live in Corpus; a Corpus test asserts `_types` imports neither `pydantic` nor `mcp` |
| MCP M02 | family | MCP's optionality proof depends on `corpus → annoy` staying deferred |

---

## 6. When a change crosses a boundary

```text
[ ] Which successors does this submodule have?            (§3)
[ ] Is the edge deferred or module-scope?                 (§2)
[ ] If deferred, does the change make it module-scope?    -> breaks optionality
[ ] Does it touch cexternals/_annoy/src/?                 -> 3 consumers rebuild
[ ] Does it change a contract a completed campaign relies on?   (§5)
[ ] Run check_trackers.py in this submodule AND each successor
```

**The asymmetry to remember:** a change in an antecessor is never local, and a
change in a successor almost always is. That is the whole reason for the order.
