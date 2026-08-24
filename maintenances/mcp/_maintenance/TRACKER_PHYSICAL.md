# Physical Tracker — what is on disk

Re-derived from the live tree. **Do not hand-edit the numbers**; run the gate:

```console
$ python scikitplot/mcp/_maintenance/check_trackers.py
```

Machine-readable mirror: `TRACKER.json` → `physical`.

---

## 1. Totals

```text
source files    18      source LOC    4 686
test files      16      test LOC      2 656
markdown files  31
```

Test-to-source LOC ratio **0.57** — above the 0.50 tripwire.

---

## 2. Inventory

| Area | src | src LOC | tests | test LOC |
|---|---:|---:|---:|---:|
| `(root)` | 10 | 3 839 | — | — |
| `integrations/` | 5 | 189 | — | — |
| `_maintenance/` | 3 | 658 | — | — |
| `tests/` | — | — | 16 | 2 656 |

## 3. Largest source modules

| LOC | Module |
|---:|---|
| 765 | `__main__.py` |
| 543 | `_core.py` |
| 482 | `_corpus_annoy.py` |
| 466 | `_hybrid.py` |
| 427 | `_server.py` |
| 359 | `_outcome.py` |
| 356 | `_capabilities.py` |
| 354 | `_demo.py` |
| 344 | `_maintenance/check_trackers.py` |

`__main__.py` remains below the 1 200 LOC tripwire. M14 deliberately adds only
the backend-selection/configuration seam there; Corpus and Annoy imports remain
lazy inside functions.

---

## 4. Markdown distribution

```text
markdown files  31
source files    18
ratio           1.72
```

This remains below the 2.5 markdown:source tripwire.

---

## 5. Tripwires

| Metric | Now | Tripwire | Status |
|---|---:|---:|---|
| modules importing the MCP SDK | **1** | any second | PASS |
| modules importing `corpus` at module scope | **0** | any | PASS |
| test : source LOC | **0.57** | < 0.50 | PASS |
| markdown : source files | **1.72** | > 2.5 | PASS |
| largest module | **765** | > 1 200 | PASS |

---

## 6. Load-bearing physical properties

Properties of the *layout* that nothing currently checks.

| Property | Held by | Protected by |
|---|---|---|
| `import scikitplot.mcp` needs no SDK | `_server.py` isolating the import | **nothing — should be a test** |
| Importing MCP does not import Corpus | injection through `DocsRetriever` | **nothing — should be a test** |
| Only one module imports `mcp`/`pydantic` | convention | **nothing — should be a test** |

Verified true at the time of writing: importing `scikitplot.mcp` pulls in only
`numpy`, and no `mcp`, `pydantic`, `scikitplot.corpus` or `annoy`.

**All three are true and unchecked.** That is exactly the condition that let
Corpus's import hygiene decay silently until a strict run made 49 test files
uncollectable. Three small tests would convert convention into contract, and it
is the cheapest high-value work in this module.

---

## 7. Known physical debt

**The MCP suite does not collect without `[mcp]` installed.**

```text
$ python -m pytest scikitplot/mcp -q
2 skipped, 4 errors in 0.72s
E   ModuleNotFoundError: No module named 'pydantic'
```

Four of six test files import `pydantic` transitively without a guard;
`test_protocol_in_memory.py` and `test_mcp_http_live.py` use
`pytest.importorskip` and behave correctly. The fix is to make the other four
match — the same class of defect as Corpus's missing `pytest-doctestplus`, where
a clean install could not run the suite at all.

**31 markdown files.** Sixteen live in `_maintenance/`, six are archived in
`history/`. Several predate the review kit and overlap with it.
