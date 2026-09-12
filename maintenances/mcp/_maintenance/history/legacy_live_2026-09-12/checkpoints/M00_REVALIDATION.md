# Run M00 — Snapshot and deferred-issue revalidation

```text
run_id            M00
date              2026-08-18
scope             scikitplot/mcp/** (review only)
guide             MCP_UNIFIED_DEFERRED_REVIEW_GUIDE.md §5
production code   NOT MODIFIED
exit gate         NOT MET — see §1 and §7
```

---

## 0. Headline

**The supplied archive is not the anchored snapshot.** Every status recorded
before this run is therefore `REVERIFY` by the guide's own rule, and M00 was
executed on that basis: nothing was accepted, everything was re-derived.

Revalidation changed the status of **every one of the four recorded findings**.
Not one was accurate as written. Two understate their scope, one is
substantially wrong, one is partly disproved.

---

## 1. Source anchor — MISMATCH (blocking)

```text
recorded anchor   611bdbf3b0a366276b9538e510e974d9400491c84209fc0d35cb9bb058cb8f38
actual archive    43f5060c971f8be0272efaef9cc092b1bb7edc22614df9fa0c23ad26beb42c61
```

Guide, *Multi-run rule*: **"If the source hash changes, all prior issue statuses
become `REVERIFY`."**

`MAINTAINING.md` states: *"If the hash changes, re-verify claims before carrying
them forward."* Both instructions were followed.

This single fact explains most of what follows: the maintenance set describes a
tree that is not the tree supplied.

### 1a. The archive is incomplete

`scikitplot/__init__.py` requires five top-level modules that are absent:

```text
_compat   _lib   _testing   logging   utils
```

Consequence — `import scikitplot` fails, and therefore so does
`import scikitplot.mcp`:

```console
$ python -c "import scikitplot.mcp"
ImportError: cannot import name 'logging' from partially initialized module
'scikitplot' (most likely due to a circular import)
```

**Every runtime claim in the maintenance set is unreproducible from this
archive.** All findings below are therefore established by *static* evidence
(AST, grep, file inventory, isolated reproducers), which is stated per finding.

---

## 2. Mandated commands

### `check_trackers.py` — exit 0, but see §6

```console
$ python scikitplot/mcp/_maintenance/check_trackers.py
physical tracker matches the tree (17 source / 15 test files, 3515 / 2096 LOC)
EXIT=0
```

### `pytest scikitplot/mcp -q` — could not run as documented

```console
$ python -m pytest scikitplot/mcp -q -p no:cacheprovider
ImportError: Error importing plugin "sphinx.testing.fixtures": No module named 'sphinx'
```

Repo-root `pytest.ini` hard-codes `-p sphinx.testing.fixtures` in `addopts`.
After installing `sphinx`, collection still fails at the first module for the
reason in §1a, and `--maxfail=1` (also in `addopts`) halts the run:

```console
collected 0 items / 1 error
ERROR scikitplot/mcp/tests/integration/test_mcp_http_live.py
E   ImportError: cannot import name 'logging' from partially initialized module 'scikitplot'
```

**The documented expectation `2 skipped, 4 errors` was NOT observed and could not
be.** It remains plausible for a built tree — §3 shows the file-level mechanism
is real and the arithmetic works out — but it is `UNKNOWN` here, not `VERIFIED`.

---

## 3. Recorded findings — revalidation

### `MCP-M00-01` → **CONFIRMED (mechanism), COUNT CORRECTED**

Evidence class: `VERIFIED` (static) / `UNKNOWN` (runtime).

The four named files are correct, and the diagnosis is correct. The
**denominator is wrong**:

| Claim | Actual |
|---|---|
| "4 of **6** test files" | 4 of **13** `test_*.py` modules (15 `.py` files incl. `__init__.py`) |

Precise statement: **5 modules import `_server` at module scope; 4 of those 5 are
unguarded.**

| Test module | imports `_server` at module scope | guarded |
|---|---|---|
| `test_protocol_in_memory.py` | yes | **yes** — `importorskip("mcp")` precedes the import |
| `test_mcp_hardening.py` | yes | no |
| `test_mcp_runtime_status.py` | yes | no |
| `test_mcp_version_guard.py` | yes | no |
| `test_server.py` | yes | no — also imports `pydantic` and `starlette` directly |

Guard ordering in `test_protocol_in_memory.py` was checked, not assumed: the
`importorskip` is on line 7, the `_server` import on line 11. It is correct.

`test_server.py` additionally imports `starlette` — an optional dependency **not
declared in the `[mcp]` extra**. Guarding on `pydantic` alone would not make it
collectable.

**Refinement — one of the four is trivially fixable and was missed.** See
`MCP-M00-06`: `test_mcp_runtime_status.py` imports a Tier-L function through the
Tier-S module for no reason.

### `MCP-M00-02` → **CONFIRMED, BUT UNDERSTATED**

Evidence class: `VERIFIED`.

All seven claimed line numbers are accurate. But the list is **not exhaustive**,
and the characterisation *"ALL are docstrings/comments"* is **false**.

Exhaustive scan of `scikitplot/mcp/**` finds **10** occurrences in `.py` source,
not 6:

| Location | Kind | Listed? |
|---|---|---|
| `_core.py:19` | docstring | yes |
| `_core.py:86` | docstring | yes |
| `_corpus_annoy.py:24` | docstring | yes |
| `_corpus_annoy.py:29` | docstring | yes |
| `_corpus_annoy.py:67` | docstring | yes |
| `_corpus_annoy.py:297` | comment | yes |
| `_corpus_annoy.py:334` | prose (`"Document lookup"`) — correctly flagged as not a symbol | yes |
| `_corpus_annoy.py:96` | docstring | **no** |
| `_corpus_annoy.py:98` | docstring | **no** |
| `_corpus_annoy.py:247` | **Sphinx `:class:` cross-reference** | **no** |
| `_corpus_annoy.py:320` | **`RuntimeError` message string** | **no** |

Two of the unlisted occurrences are **not** documentation:

- **`_corpus_annoy.py:247`** — ``:class:`~scikitplot.corpus.SimilarityIndex` ``
  is a Sphinx cross-reference to a **renamed** target. Under nitpick mode this is
  a **docs-build failure**, not a cosmetic comment.
- **`_corpus_annoy.py:320`** — the symbol appears inside a user-facing
  `RuntimeError` message, naming a class that no longer exists.

**Impact revision:** this is not purely "a documentation fix". It includes one
build-breaking reference and one user-facing runtime string.

### `MCP-M00-03` → **PARTLY DISPROVED**

Evidence class: `DISPROVED` (for invariant 3).

The claim — *"Three boundary invariants hold but nothing checks them"*, and
`TRACKER_PHYSICAL.md` §6 *"Protected by: nothing — should be a test"* — is
**wrong for the third invariant.**

`tests/test_mcp_import_surface.py::test_import_and_discovery_without_pydantic`
already tests it, and thoroughly: it spawns a subprocess, purges preloaded
`pydantic` modules, installs an `__import__` blocker, imports `scikitplot.mcp`,
and asserts `'pydantic' not in sys.modules`.

`test_cli_introspection_without_pydantic` extends the same guarantee to
`--help`, `--print-effective-config` and `--list-capabilities`, blocking `mcp`
as well.

Corrected status:

| Invariant | Enforced by |
|---|---|
| only `_server.py` imports `mcp`/`pydantic` | `check_trackers.py` (source only — see `MCP-M00-05`) |
| corpus/annoy never at module scope | `check_trackers.py` (source only) |
| `import scikitplot.mcp` needs no SDK | **`test_mcp_import_surface.py`** — a test exists |

The genuine residual gap is narrower and different: **the gate does not police
test files at all** (`MCP-M00-05`).

Boundary facts independently re-derived by AST (module-scope vs nested):

```text
_server.py         MODULE-SCOPE: pydantic     NESTED: mcp.server, mcp.types, starlette.*
_corpus_annoy.py   MODULE-SCOPE: —            NESTED: scikitplot.corpus
_hybrid.py         MODULE-SCOPE: —            NESTED: scikitplot.corpus
```

All three invariants **do hold** in the code. Only the claim about their
enforcement was wrong.

### `MCP-M00-04` → **SUBSTANTIALLY WRONG**

Evidence class: `DISPROVED` as stated; a real but *different* problem exists.

| Metric | Documented | Actual | |
|---|---:|---:|---|
| markdown files | 31 | **17** | MISMATCH |
| `_maintenance/` markdown | 16 | **8** | MISMATCH |
| `_maintenance/history/` markdown | 6 | **0** | MISMATCH |
| source files | 16 (`TRACKER_PHYSICAL.md`) | **17** | MISMATCH |
| source LOC | 3 182 / 3 503 | **3 515** | MISMATCH |
| markdown : source ratio | 1.9 | **1.0** | MISMATCH |

`_maintenance/history/` contains **no markdown at all** — only
`stale_lifecycle.py` and `update_artifact_manifest.py`. `TRACKER.json`
`largest_source` records those two at `_maintenance/…`; they are actually at
`_maintenance/history/…`.

The stated debt (ratio 1.9, "archive 4, fold 2, drop 1") does not describe this
tree. **The real problem is the inverse — 18 dangling documentation targets:**

```text
CHANGELOG_IDEMPOTENT.md      MCP_CLOSURE_AUDIT_RESPONSE.md   RULESET.md
CI_OUTPUT_ROUTING.md         MCP_CLOSURE_R1_R6_RESPONSE.md   STALE_FILES.md
DESIGN.md                    MCP_COMPATIBILITY_POLICY.md     STRICT_WIRE_VALIDATION.md
DOCKER.md                    MCP_DEEP_REVIEW_REPORT.md       UNKNOWN_ARGUMENTS_AND_MANIFESTS.md
DROP_IN_README.md            MCP_REDESIGN_PLAN.md            MCP_UNIFIED_DEFERRED_REVIEW_GUIDE.md
IDEMPOTENT_TESTING.md        MCP_VERIFICATION_MATRIX.md      METHODOLOGY.md
```

Three of these are **required reading** in `MAINTAINING.md` and
`STATE.json.fresh_chat_read_order`, and are absent:

```text
_maintenance/RULESET.md                    (marked KEEP)
_maintenance/MCP_COMPATIBILITY_POLICY.md   (marked KEEP)
_maintenance/DESIGN.md                     (marked KEEP)
```

`RULESET.md` and `DESIGN.md` are cited from **runtime source**, not just docs —
`_capabilities.py`, `_core.py`, `_hybrid.py`, `integrations/__init__.py`.

**The prescribed read order cannot be completed.** Either the disposition in
`SUBMODULE_STRUCTURE.md` §3 was applied and deleted files marked KEEP, or the
archive is incomplete (§1a makes the latter likely). This archive cannot
distinguish the two — recorded as `UNKNOWN` with a decisive next action in §7.

---

## 4. Deferred queue `MCP-D01`–`D09`

Full detail in `M00_DEFERRED_ISSUE_MATRIX.md`. Summary:

| ID | Status | Basis |
|---|---|---|
| D01 | **OPEN** | presence ≠ compatibility; plus a `find_spec` false positive |
| D02 | **OPEN** | `integrations/agno` imports `_server`, which `SUBMODULE_STRUCTURE.md` §2 forbids |
| D03 | **OPEN** | Protocol return type has no status channel; `strict=False` default |
| D04 | **OPEN** | proven by isolated reproducer |
| D05 | **OPEN** | `except Exception` on an import guard; inconsistent with `_hybrid.py` |
| D06 | **OPEN** | the policy assertions are **commented out**, and the live code contradicts the policy |
| D07 | **OPEN** | a Tier-L test imports through the Tier-S module unnecessarily |
| D08 | **OPEN** | `_core.py` builds `structuredContent`/`isError` yet is called "SDK-free" |
| D09 | **OPEN** | §3 `MCP-M00-04` — 18 dangling targets, wrong counts, wrong module attribution |

**Nine of nine confirmed open.** None was disproved.

---

## 5. New findings raised by this run

### `MCP-M00-05` — P2 — the drift gate is blind to test files

`check_trackers.py` excludes `_maintenance` and `tests` from `check_boundary()`.
Demonstrated empirically:

```console
$ cp probe.py scikitplot/mcp/tests/test__gate_probe.py   # module-scope `import pydantic`
$ python scikitplot/mcp/_maintenance/check_trackers.py
physical tracker matches the tree (17 source / 16 test files, 3515 / 2098 LOC)
EXIT=0                                    <-- violation NOT detected

$ cp probe.py scikitplot/mcp/_probe_src.py               # identical violation, source file
$ python scikitplot/mcp/_maintenance/check_trackers.py
BOUNDARY: _probe_src.py:2 imports 'pydantic' at module scope; only ['_server.py'] may do so
EXIT=1                                    <-- detected
```

**The gate cannot detect a regression of `MCP-M00-01`, the module's highest-severity
open finding.** `MAINTAINING.md` claims both consequences are "now mechanically
enforced by `check_trackers.py`"; that is true only for non-test source.

Two further blind spots:

- `compare()` iterates `recorded["totals"]`, which holds only
  `source_files/source_loc/test_files/test_loc`. **`markdown_files` is recorded
  (31) but never compared** — which is precisely why a 31→17 drift passed at
  exit 0.
- `check_logical()` verifies only that a contract's module **exists**, not that
  the named symbol is defined there — see `MCP-M00-07`.

### `MCP-M00-06` — P2 — Tier-L test imports through the Tier-S module

`test_mcp_runtime_status.py` line 1 declares itself *"SDK-free … (Tier-L, Python
3.8+)"*. Line 9:

```python
from scikitplot.mcp._server import SearchService, server_runtime_status
```

But `server_runtime_status` is **defined in `_capabilities.py:129`** — the
deliberately pydantic-free module — and merely re-exported by `_server.py:35`.

The test imports a Tier-L function through the Tier-S module, contradicting its
own docstring and dragging `pydantic` into collection for no reason. Changing
the import to `_capabilities` removes **one of the four collection errors** with
a one-line change and no behavioural risk. This is the root cause of D07.

### `MCP-M00-07` — P2 — `SearchService` contract is misattributed and its invariant is false

`TRACKER_LOGICAL.md` §1 and `TRACKER.json` both record:

| Contract | Module | Invariant |
|---|---|---|
| `SearchService` | `_core.py` | *"Protocol-neutral. Returns neutral result shapes, never MCP wire types."* |

Both halves are wrong:

- **Module:** `SearchService` is defined at **`_server.py:110`**, not `_core.py`.
  (`grep -rn "class SearchService"` returns exactly one hit.)
- **Invariant:** `SearchService.search()` is annotated `-> SearchDocsOutput`
  (`_server.py:136`); `SearchDocsOutput` subclasses `_ClosedModel(BaseModel)`
  (`_server.py:53,77`). It returns a **pydantic wire model**. "Never MCP wire
  types" is false.

`check_logical()` passes because `_core.py` exists — the gate checks the file,
not the symbol.

**This is the most consequential finding of the run.** The documented
protocol-neutral service layer *does not exist*: the only `SearchService` is
pydantic-backed and lives in the SDK-importing module. That is why D02 happens —
`integrations/agno` must reach into `_server` to obtain it, and the public
`scikitplot.mcp.SearchService` is itself a lazy server-tier export, so "use the
public API" does not fix it.

**Run M05 (`SearchService` ownership) now has concrete content**, and it is a
prerequisite for D02 rather than independent of it.

### `MCP-M00-08` — P3 — malformed user-facing error string

`_corpus_annoy.py:290-292`, implicit concatenation with no separator:

```python
"scikitplot.corpus is required to build the retriever install"
"the corpus/embedding extras (pip install scikit-plots[corpus])."
```

Renders as `…build the retriever installthe corpus/embedding extras…`.

### `MCP-M00-09` — P3 — `pytest.ini` blocks the documented verification command

Repo-root `addopts` hard-codes `-p sphinx.testing.fixtures`, making `sphinx` a
mandatory import for **any** `pytest` invocation, plus `--maxfail=1`, which stops
the suite at the first error and hides the rest. The command in `MAINTAINING.md`
is not runnable on a checkout without `sphinx`.

Out of MCP's declared scope (`scikitplot/mcp/**`), recorded because it blocks
MCP's own verification contract. Route to the CLI/packaging campaign.

### `MCP-M00-10` — P2 — the `[mcp]` extra contradicts its own stated policy, and the test that would catch it is commented out

The `[mcp]` extra in `pyproject.toml` carries an eleven-line comment stating the
rule explicitly:

> *"NO python marker on the SDK line: requesting `[mcp]` on Python 3.8/3.9 must
> FAIL at dependency resolution … A marker would instead silently drop the SDK
> and leave a partial (pydantic-only) extra, which does not mean 'server
> installed'."*

The very next line is:

```toml
'mcp>=2.0.0,<3; python_version >= "3.10"',
```

**The SDK line carries exactly the marker the comment forbids**, producing
exactly the failure mode the comment describes: on Python 3.8/3.9 `pip install
scikit-plots[mcp]` succeeds and installs pydantic only.

The regression gate for this rule exists in
`tests/test_mcp_import_surface.py::test_mcp_extra_declares_server_dependencies`
but its assertions are **commented out**, leaving `sdk_lines` computed and unused
and `import re as _re` unused:

```python
    sdk_lines = [l for l in block.splitlines() if 'mcp>=2.0.0,<3' in l]
    # assert sdk_lines and all('python_version' not in l for l in sdk_lines), (
    #     'the mcp SDK requirement must not carry a python_version marker')
```

This is the concrete substance of **D06** ("packaging regression assertions may be
weaker than policy") — they are not merely weaker, they are disabled, and the
live metadata violates the policy they were written to enforce.

Recorded memory of this decision states the marker must be absent. The in-file
comment agrees. The metadata does not. **`MCP_COMPATIBILITY_POLICY.md` is the
document that would adjudicate, and it is missing (§3)** — hence the M01 hold in
§7.

---

## 6. Tripwire status

| Metric | Recorded | Actual | Tripwire | State |
|---|---:|---:|---|---|
| SDK importers | 1 | **1** | any second | OK |
| module-scope corpus imports | 0 | **0** | any | OK |
| test : source LOC | 0.66 | **0.60** | < 0.50 | OK |
| markdown : source | 1.9 | **1.0** | > 2.5 | OK (recorded value wrong) |
| largest module | 608 | **608** (`__main__.py`) | > 1 200 | OK |
| unguarded test files | 4 of 6 | **4 of 13** | > 0 | **CROSSED** |

The boundary tripwires hold. The recorded *values* for markdown and source
counts do not match the tree, and the gate cannot see the discrepancy.

---

## 7. Exit gate — NOT MET

The guide's gate: *"No deferred issue remains assumed from historical chat."*

Nine deferred issues and four recorded findings were reproduced against the tree,
so nothing is assumed **from chat**. But two conditions block closure:

1. **Anchor mismatch (§1).** Statuses here are `REVERIFY`-grade against the
   intended snapshot, not final.
2. **Three files in the mandated read order are absent (§3).** `RULESET.md`,
   `MCP_COMPATIBILITY_POLICY.md` and `DESIGN.md` are all marked **KEEP** and all
   missing. M01 (packaging and Python tier contract) depends directly on
   `MCP_COMPATIBILITY_POLICY.md`, and D06 cannot be closed without it —
   `MCP-M00-10` shows the live code contradicts the policy as quoted, and the
   policy itself is unavailable to adjudicate.

### Next exact action

Re-supply a complete `scikit-plots.zip` that (a) imports, and (b) contains
`_maintenance/RULESET.md`, `_maintenance/MCP_COMPATIBILITY_POLICY.md` and
`_maintenance/DESIGN.md`; record its sha256 as the new anchor. Then re-run the
two gate commands and confirm `2 skipped, 4 errors`.

**M01 should not start until the anchor is settled** — it is the packaging and
Python-tier run, and `MCP-M00-10` shows a live contradiction in exactly that area
that cannot be adjudicated without the missing policy document.

If the tree is confirmed correct as-is, the cheapest ordered work is:
`MCP-M00-06` (one line, removes a collection error) → `MCP-M00-05` (extend the
gate to test files) → `MCP-M00-07` (correct the contract, then decide M05).

---

## 8. Run record

```text
run_id                  M00
source_sha256           43f5060c971f8be0272efaef9cc092b1bb7edc22614df9fa0c23ad26beb42c61
                        (MISMATCH vs recorded 611bdbf3…)
scope                   scikitplot/mcp/** — review only
files inspected         17 source, 15 test, 17 markdown, TRACKER.json, STATE.json,
                        pyproject.toml [mcp] extra, pytest.ini
commands                check_trackers.py (exit 0)
                        pytest scikitplot/mcp -q (BLOCKED — §1a, §2)
                        AST module-scope import analysis
                        isolated PEP 562 star-import reproducer
                        gate blind-spot probe (test vs source)
                        dangling-doc-reference scan
confirmed               D01–D09 (9 of 9), MCP-M00-01 (count corrected),
                        MCP-M00-02 (scope corrected)
disproved               MCP-M00-03 (partly), MCP-M00-04 (substantially)
new                     MCP-M00-05 … MCP-M00-10
unknown / deferred      runtime suite behaviour; whether missing KEEP docs were
                        deleted or omitted; SDK min/latest matrix (M12)
production code changed NO
next exact action       see §7
```
