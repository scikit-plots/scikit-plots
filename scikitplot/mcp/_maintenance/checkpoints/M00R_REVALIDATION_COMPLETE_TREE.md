# Run M00-R — Revalidation against the complete tree

```text
run_id            M00-R  (re-execution of M00 §5 against a new archive)
date              2026-08-18
scope             scikitplot/mcp/** (review only)
production code   NOT MODIFIED
exit gate         NOT MET — one blocker cleared, one remains (§6)
```

---

## 0. What changed

The archive is complete and **imports**. Every finding previously marked
`UNKNOWN (runtime)` is now settled by execution rather than inference.

```text
blocker 1  archive incomplete / cannot import   -> CLEARED
blocker 2  three KEEP documents absent          -> STILL OPEN
```

**No MCP source changed.** Diff against the M00 tree shows only the M00
deliverables were added:

```console
$ diff -rq work/scikitplot/mcp work2/scikitplot/mcp
Files …/_maintenance/STATE.json … differ
Only in …/_maintenance: checkpoints
```

Both checkpoint files and `STATE.json` were integrated **byte-identical**
(`cmp` clean). So every static finding from M00 carries forward unchanged; this
run adds runtime evidence and one new finding.

---

## 1. Source anchor — third distinct hash

```text
recorded anchor (MAINTAINING.md)   611bdbf3b0a366276b9538e510e974d9400491c84209fc0d35cb9bb058cb8f38
M00 reviewed                       43f5060c971f8be0272efaef9cc092b1bb7edc22614df9fa0c23ad26beb42c61
M00-R reviewed (this run)          119855928dd052165c71efa61fa505aa206726265c7c375943084e64793c4018
```

Still not the recorded anchor. `MAINTAINING.md` line 10 and
`TRACKER.json`/`STATE.json` should be updated to `1198559…` once this tree is
accepted as canonical, or the mismatch will re-trigger `REVERIFY` on every future
run. Recorded as **`MCP-M00-11`**.

---

## 2. Mandated commands — both now run

### Gate — exit 0

```console
$ python scikitplot/mcp/_maintenance/check_trackers.py
physical tracker matches the tree (17 source / 15 test files, 3515 / 2096 LOC)
EXIT=0
```

### Suite — the documented expectation is CONFIRMED

```console
$ python -m pytest scikitplot/mcp -q -p no:cacheprovider --maxfail=100
SKIPPED [1] …/integration/test_mcp_http_live.py:62: set SCIKITPLOT_MCP_RUN_LIVE=1 …
SKIPPED [1] …/test_protocol_in_memory.py:7: could not import 'mcp': No module named 'mcp'
ERROR …/test_mcp_hardening.py
ERROR …/test_mcp_runtime_status.py
ERROR …/test_mcp_version_guard.py
ERROR …/test_server.py
======================= 2 skipped, 4 errors in 0.48s =======================
```

`2 skipped, 4 errors` — **exactly as documented**, with exactly the four files
named in `MCP-M00-01`. Environment: `pydantic`, `mcp`, `starlette`, `anyio`,
`httpx` all absent, i.e. the intended "without `[mcp]`" condition.

`MCP-M00-01` is therefore promoted from `VERIFIED (static) / UNKNOWN (runtime)`
to **`VERIFIED`**. Its corrected denominator (4 of 13 test modules, not 4 of 6)
stands.

**`--maxfail=1` truncated this to `1 skipped, 1 error`** on the literal
documented command. `MCP-M00-09` is not cosmetic: the documented command
understates its own expected output.

---

## 3. NEW — `MCP-M00-12` (P2): fixing the four guards will not green the suite

Collection errors abort the run before any test executes, so nobody has seen
what happens after collection succeeds. Ignoring the four erroring files:

```console
$ python -m pytest scikitplot/mcp -q --maxfail=100 \
    --ignore=…/test_mcp_hardening.py --ignore=…/test_mcp_runtime_status.py \
    --ignore=…/test_mcp_version_guard.py --ignore=…/test_server.py
FAILED …/test_cli.py::test_backend_self_test_is_repeatable_and_avoids_server_creation
FAILED …/test_cli.py::test_backend_self_test_can_require_exact_canary
FAILED …/test_cli.py::test_backend_self_test_required_match_fails_closed
FAILED …/test_cli.py::test_backend_self_test_expected_doc_id_fails_closed
FAILED …/test_mcp_import_surface.py::test_searchservice_lazily_requires_pydantic_but_is_reachable
FAILED …/test_mcp_integrations.py::test_docs_toolkit_is_sdk_free_and_read_only
================== 6 failed, 96 passed, 2 skipped in 1.48s ==================
```

**Six further tests fail without `[mcp]`**, every one with
`ModuleNotFoundError: No module named 'pydantic'` reached through
`_server.py:22`.

The true scope of the optional-dependency defect is therefore **4 collection
errors + 6 runtime failures**, not 4 files. Guarding the four named files leaves
a red suite.

**The six are not one problem.** They split three ways, and the distinction
matters for the backlog:

| Tests | Nature | Correct treatment |
|---|---|---|
| 4 × `test_cli.py::test_backend_self_test_*` | `--self-test` deliberately uses the server tier (`__main__.py:40` documents this) | **guard** — legitimately server-tier |
| `test_searchservice_lazily_requires_pydantic_but_is_reachable` | name states the dependency; asserts reachability | **guard** — the test is correct |
| `test_docs_toolkit_is_sdk_free_and_read_only` | **asserts a property the code does not have** | **do NOT guard — fix the code** |

The last one is the important case. See §4.

---

## 4. `MCP-D02` — promoted to a failing test

M00 established D02 statically. It is now a **live red test**, and the test name
states the violated property:

```console
$ pytest …/test_mcp_integrations.py::test_docs_toolkit_is_sdk_free_and_read_only
…/integrations/agno/docs_toolkit.py:35: in __init__
    from scikitplot.mcp._server import SearchService
…/_server.py:22: in <module>
E   ModuleNotFoundError: No module named 'pydantic'
```

A test named `is_sdk_free` fails because the toolkit is not. Guarding it with
`importorskip` would turn a true failure into a silent skip and **erase the
evidence for D02** — the tempting fix is the wrong one.

D02 remains blocked on `MCP-M00-07`: there is no protocol-neutral `SearchService`
to depend on instead.

---

## 5. Runtime confirmations and corrections

### 5a. Boundary invariants — CONFIRMED at runtime

Previously `HOLDS_IN_CODE / UNVERIFIED_AT_RUNTIME`. Measured from outside the
source tree:

```text
import scikitplot.mcp  ->  corpus   NOT loaded
                           annoy    NOT loaded
                           pydantic NOT loaded
                           mcp SDK  NOT loaded
                           numpy    loaded (the only third-party module)
```

`TRACKER_PHYSICAL.md` §6's claim — *"importing `scikitplot.mcp` pulls in only
`numpy`"* — is **VERIFIED**. All three boundary invariants now hold in code and at
runtime.

### 5b. `MCP-D01` — promoted from isolated reproducer to runtime proof

```console
$ mkdir -p /tmp/d01/mcp && touch /tmp/d01/mcp/__init__.py   # any dir named mcp/
$ cd /tmp/d01 && python -c "…server_runtime_status()"
   retrieval_available    True
   server_available       True      <-- no MCP SDK is installed
   sdk_present            True
   reason                 None
```

And the resulting contradiction, in the same interpreter:

```console
status says server_available: True
create_server import FAILED: ModuleNotFoundError No module named 'pydantic'
```

`_capabilities.py:131` states this function exists *"so callers can degrade
gracefully to the SDK-free retrieval layer instead of catching an exception from
`create_server`."* It returns the answer that causes exactly that exception.
**The function fails at its documented purpose.**

Any directory named `mcp/` on `sys.path` triggers this — `find_spec` answers
"is a module named `mcp` importable", which is not the question.

*Correction to M00:* the M00 matrix cited the `cwd == scikitplot/` variant.
That specific path is now unreachable for a different reason (§5c), but the
general defect is broader and confirmed. The finding stands; its example changes.

### 5c. NEW — `MCP-M00-13` (P3, out of MCP scope): `scikitplot/logging.py` shadows stdlib `logging`

```console
$ cd scikitplot && python -c "import scikitplot.mcp"
File "…/scikitplot/logging.py", line 71, in <module>
    from logging import (
ImportError: cannot import name 'CRITICAL' from partially initialized module
'logging' (…/scikitplot/logging.py)
```

With `cwd == scikitplot/`, `scikitplot/logging.py` shadows the stdlib module it
imports from. The package has a guard warning against importing from the source
directory, so this is semi-known, but it fails with a confusing circular-import
error rather than that warning. Out of `scikitplot/mcp/**` scope — route to the
CLI/packaging campaign with `MCP-M00-09`.

### 5d. `MCP-M00-06` — recommendation validated

```console
$ python -c "from scikitplot.mcp._capabilities import server_runtime_status; …"
imported from _capabilities OK; pydantic loaded?: False
status keys: ['python','python_ok','reason','retrieval_available','sdk_present','server_available']
```

`server_runtime_status` is fully reachable pydantic-free with an identical key
set. Re-pointing `test_mcp_runtime_status.py` from `_server` to `_capabilities`
is a safe one-line change that removes one of the four collection errors.

---

## 6. Blocker 2 — still open

The three documents remain absent:

```text
_maintenance/RULESET.md                   MISSING  (marked KEEP; in read order)
_maintenance/MCP_COMPATIBILITY_POLICY.md  MISSING  (marked KEEP; in read order)
_maintenance/DESIGN.md                    MISSING  (marked KEEP; in read order)
```

Still cited from **runtime source**, not merely from docs:

```text
_capabilities.py:149   "Tier model (see ``RULESET.md``)"
_hybrid.py:22          "(see ``DESIGN.md`` §Hybrid)"
_hybrid.py:346         "verify against the installed source (see DESIGN.md §Hybrid)"
```

`MCP-M00-10` is unchanged and live — `pyproject.toml` and `pytest.ini` are
byte-identical to the M00 tree, so the `[mcp]` extra still carries the
`python_version` marker its own comment forbids. **`MCP_COMPATIBILITY_POLICY.md`
is the document that decides whether the comment or the metadata is
authoritative, and it is still missing.**

`MCP-M00-07` is likewise unchanged: `class SearchService` is at `_server.py:110`
while `TRACKER.json` still records `module: _core.py`.

**M01 remains BLOCKED.** It is the packaging and Python-tier run; its central
question is exactly the contradiction in `MCP-M00-10`.

---

## 7. Physical metrics — this tree

| Metric | Recorded | Actual | |
|---|---:|---:|---|
| source files | 17 | **17** | match |
| source LOC | 3 503 | **3 515** | within 10% tolerance |
| test files | 15 | **15** | match |
| test LOC | 2 096 | **2 096** | match |
| markdown files | 31 | **19** | **still stale, still uncompared** |
| test : source LOC | 0.66 | **0.596** | above the 0.50 tripwire |
| markdown : source | 1.9 | **1.12** | below the 2.5 tripwire |

Markdown rose 17 → 19 (the two M00 checkpoints). `TRACKER.json` still records 31
and `compare()` still never reads it — `MCP-M00-05` unchanged.

---

## 8. Status ledger after M00-R

| ID | Before | After |
|---|---|---|
| `MCP-M00-01` | CONFIRMED (static) | **VERIFIED (runtime)** — `2 skipped, 4 errors` reproduced |
| `MCP-M00-02` | CONFIRMED_UNDERSTATED | unchanged |
| `MCP-M00-03` | PARTLY_DISPROVED | **strengthened** — invariants verified at runtime (§5a) |
| `MCP-M00-04` | SUBSTANTIALLY_DISPROVED | unchanged (19 md, not 31) |
| `MCP-M00-05` | NEW | unchanged |
| `MCP-M00-06` | NEW | **recommendation validated** (§5d) |
| `MCP-M00-07` | NEW | unchanged — still live |
| `MCP-M00-08` | NEW | unchanged |
| `MCP-M00-09` | NEW | **strengthened** — `--maxfail=1` truncates the documented run |
| `MCP-M00-10` | NEW | unchanged — still live; still unadjudicable |
| `MCP-M00-11` | — | **NEW** P3: anchor still stale in `MAINTAINING.md` |
| `MCP-M00-12` | — | **NEW** P2: 6 further failures beyond the 4 collection errors |
| `MCP-M00-13` | — | **NEW** P3: `scikitplot/logging.py` shadowing (out of scope) |
| `D01`–`D09` | 9 OPEN | 9 OPEN; **D01 and D02 promoted to runtime proof** |

---

## 9. Run record

```text
run_id                  M00-R
source_sha256           119855928dd052165c71efa61fa505aa206726265c7c375943084e64793c4018
                        (third distinct hash; still != recorded anchor 611bdbf3…)
scope                   scikitplot/mcp/** — review only
commands                check_trackers.py                                  -> exit 0
                        pytest scikitplot/mcp -q                           -> 1 skipped, 1 error (maxfail=1)
                        pytest … --maxfail=100                             -> 2 skipped, 4 errors  [MATCHES DOC]
                        pytest … --maxfail=100 --ignore=<4 erroring files> -> 6 failed, 96 passed, 2 skipped
                        import-surface probe from outside source tree      -> numpy only
                        find_spec shadowing probe                          -> false capability claim
confirmed               MCP-M00-01 (runtime), D01 (runtime), D02 (failing test),
                        boundary invariants (runtime)
new                     MCP-M00-11, MCP-M00-12, MCP-M00-13
corrected               D01 example (cwd==scikitplot/ variant now unreachable; defect broader)
production code changed NO
next exact action       see below
```

### Next exact action

Supply `_maintenance/RULESET.md`, `_maintenance/MCP_COMPATIBILITY_POLICY.md` and
`_maintenance/DESIGN.md`, and update the anchor in `MAINTAINING.md`,
`TRACKER.json` and `STATE.json` to `1198559…`. That closes M00 and unblocks M01.

If those three documents are genuinely lost, that is itself the M01 finding: the
Python/SDK tier contract has no surviving specification, and `MCP-M00-10` must be
resolved by deciding the policy afresh rather than recovering it — a decision
that needs an ADR, not a code edit.

Implementation remains closed until the runs finish. When it opens, the ordered
backlog is unchanged and now fully evidenced:

```text
1. MCP-M00-06   re-point one import      -> removes 1 of 4 collection errors
2. MCP-M00-01   guard 3 remaining files  -> collection succeeds
3. MCP-M00-12   guard 5 of the 6 failures; do NOT guard the 6th
4. MCP-M00-05   extend gate to test files + compare markdown_files
5. MCP-M00-07   correct the contract, then decide M05 -> unblocks D02
```
