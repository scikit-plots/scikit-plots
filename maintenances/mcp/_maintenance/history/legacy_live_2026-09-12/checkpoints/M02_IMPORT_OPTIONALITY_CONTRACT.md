# Run M02 — Import / optionality contract

```text
run_id            M02
date              2026-08-18
source_sha256     119855928dd052165c71efa61fa505aa206726265c7c375943084e64793c4018
scope             import surface, __all__/__getattr__/__dir__, integrations, plugin metadata
guide             MCP_UNIFIED_DEFERRED_REVIEW_GUIDE.md §7
production code   NOT MODIFIED
exit gate         MET — the star-import question is decided (§3); all §7 probes executed
```

Conditions: `pydantic`, `mcp`, `mcp_types`, `starlette`, `anyio`, `httpx` (and
`agno` where relevant) blocked by an `__import__` blocker, on a tree with none of
them installed. This is the strongest form of the "without `[mcp]`" condition.

---

## 1. The seven mandated probes — all pass

| Probe | Result | `pydantic` loaded after? |
|---|---|---|
| `import scikitplot.mcp` | **PASS** | no |
| `server_runtime_status()` | **PASS** | no |
| `server_capabilities()` | **PASS** | no |
| `effective_server_capabilities()` | **PASS** | no |
| `python -m scikitplot.mcp --help` | **PASS** (exit 0) | no |
| `--list-capabilities` | **PASS** (exit 0) | no |
| `--print-effective-config` | **PASS** (exit 0) | no |

The optionality contract **holds for every surface the guide names**. Combined
with M00-R §5a (import pulls only `numpy`), the module's central design claim is
now verified rather than asserted.

This is the module's real strength and it should be recorded as such: the lazy
tier works, and it works under adversarial import blocking, not merely on a
machine that happens to lack the packages.

---

## 2. `__all__` / `__getattr__` / `__dir__`

```text
__all__            24 names, of which 4 are server-tier:
                   CitationOutput, SearchDocsOutput, SearchService, create_server
__getattr__        PEP 562 lazy resolution -> imports _server on first access
__dir__            46 entries; does NOT resolve -> base-safe
plain import       _server NOT loaded
```

`__dir__` is correctly implemented: it unions `globals()` with `__all__` as
strings and never resolves them, so introspection and tab-completion stay
base-safe. `__getattr__` is correct for its purpose.

**The defect is `__all__`, not the lazy machinery.**

---

## 3. `MCP-D04` decided — star import is NOT base-safe, and should be made so

The guide requires a deliberate decision. Measured on the real package:

```text
import scikitplot.mcp        -> PASS,  _server NOT loaded
dir(scikitplot.mcp)          -> PASS,  _server NOT loaded
from scikitplot.mcp import * -> ImportError: pydantic blocked
```

M00 proved this mechanism with an isolated reproducer; it is now confirmed on the
actual package. `from … import *` resolves every name in `__all__`, which invokes
`__getattr__` for the four server-tier names, which imports `_server`, which
imports `pydantic` at module scope.

So a base install has an asymmetry with no stated rationale anywhere in the tree:
**`import scikitplot.mcp` works and `from scikitplot.mcp import *` raises.**

### Decision: remove the four server-tier names from `__all__`

| | keep in `__all__` (today) | remove from `__all__` |
|---|---|---|
| `from … import *` on base install | **ImportError** | succeeds, yields the 20 Tier-L names |
| `scikitplot.mcp.SearchService` | works (lazy) | **still works** — `__getattr__` is unchanged |
| `dir()` / completion | shows all 24 | shows all 24 — `__dir__` unions `globals()` and may keep listing them |
| module's stated design goal | contradicted | satisfied |

Nothing is lost. The four names remain reachable by attribute access, which is
how a server-tier consumer reaches them anyway — and such a consumer has `[mcp]`
installed by definition. What is lost is only the ability to obtain them via star
import on an install that cannot use them.

**Rationale:** `__all__` *is* the star-import surface. Listing optional-tier names
in it defeats the lazy tier at precisely the one operation that resolves every
entry. The alternative — keep and document — preserves a trap in exchange for
nothing.

**Regression gate:** `test_mcp_import_surface.py` currently tests `import
scikitplot.mcp` but never the star import. Whichever way this is decided, the
star-import case must become a test, or the decision is unenforced (the
`MCP-M00-05` pattern again).

---

## 4. `MCP-D02` decided — the integrations violate the guide's criterion

Guide §7: *"No optional framework integration should unexpectedly import
`_server` unless it is **explicitly server-tier**."*

Measured, with `pydantic` and `agno` blocked:

| | import | construct |
|---|---|---|
| `integrations` | **PASS** | — |
| `integrations.openclaw` | **PASS** | `OpenClawMcpConfig()` **PASS** — fully base-safe |
| `integrations.agno` | **PASS** | `ScikitplotDocsToolkit()` **FAIL** — `ImportError: pydantic` |

Importing is lazy and safe. **Constructing** the agno toolkit is not.

### It is not "explicitly server-tier" — it is explicitly the opposite

Three user-facing statements claim Legacy-tier operation. All three are false:

```text
integrations/__init__.py:12   "...backed by the SDK-free retrieval core
                               (SearchService), so it runs on the Legacy
                               Retrieval tier too."

integrations/README.md:8      "...do not require the MCP SDK — ... so it also
                               runs on the Legacy Retrieval tier (Python 3.8+)."

integrations/agno/
  docs_toolkit.py:5           "...it wraps the SDK-free retrieval core
                               (SearchService), so it needs neither the MCP SDK
                               nor any agent framework, and works on the Legacy
                               Retrieval tier (Python 3.8+)..."
```

The Legacy Retrieval tier is the **pydantic-free** base install. The toolkit
requires pydantic. So the guide's criterion is violated in its strongest form:
the integration does not merely fail to declare itself server-tier — it
positively advertises the opposite.

### This is `MCP-D08` causing real damage, not a terminology quibble

Every one of the three statements is built on the same inference:

> `SearchService` is "SDK-free" ⟹ therefore Legacy-tier.

"SDK-free" is *true* (no MCP SDK is imported) and the conclusion is *false*,
because Legacy-tier requires pydantic-free, which is a different property. This
is exactly the distinction guide §2 insists on and `MCP-D08` records. M00 rated
D08 P3 on the grounds that it was terminological; **M02 shows the conflation
produced three incorrect user-facing claims and one failing test.** The rating
holds, but the justification changes: D08 is load-bearing.

The root remains `MCP-M00-07` — `SearchService` is documented as the
protocol-neutral core while actually living in `_server.py` and returning a
pydantic model. Docs and trackers inherited the same wrong model.

**Recommendation:** blocked on M05. Once `SearchService` ownership is decided,
either the toolkit depends on a genuinely neutral service (claims become true),
or the three statements are corrected to declare the toolkit server-tier. Do not
"fix" the docs before M05 decides which is true, and do not guard
`test_docs_toolkit_is_sdk_free_and_read_only` (`MCP-M00-12`) — it is the gate
that keeps this honest.

---

## 5. Plugin metadata

All 17 JSON files across 8 client bundles parse as valid JSON; no Python in
`plugins/` (correct per `SUBMODULE_STRUCTURE.md` §2). Every bundle declares the
same launch:

```json
{"command": "python", "args": ["-m", "scikitplot.mcp"]}
```

### `M02-01` (P2) — the documented actionable error is unreachable on the path every plugin uses

```console
$ python -m scikitplot.mcp          # no [mcp] installed
  File ".../__main__.py", line 51, in create_server
    from ._server import create_server as _create_server
  File ".../_server.py", line 22, in <module>
    from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator
ModuleNotFoundError: No module named 'pydantic'
exit code 1
```

`_server.py:280` contains a carefully written actionable message:

> *"MCP SDK v2 is required for the server layer; on Python >= 3.10 install it
> with: `pip install "mcp>=2.0.0,<3"`."*

**It can never fire when `pydantic` is absent.** `__main__.py:51` imports the
`_server` *module*, so `_server.py:22`'s module-scope `pydantic` import executes
first and raises. The guard at `:274-283` is only reachable when pydantic is
present but the SDK is not — a narrower case than the one users hit.

Compounding it: `__main__.py` never calls `server_runtime_status()` before
`create_server()` at line 564 (grep confirms no reference). The function written
*"so callers can degrade gracefully to the SDK-free retrieval layer instead of
catching an exception from `create_server`"* (`_capabilities.py:131`) is not used
by the module's own entry point.

**Impact:** this is the exact path all 8 plugin bundles invoke. A user wiring up
Claude/Cursor/VS Code/Codex/Cline/Continue/Windsurf/OpenClaw with a base install
gets a raw traceback naming `pydantic` — not the install instruction, and no
mention of the `[mcp]` extra. Exit code 1 is correct; the diagnostics are not.

**Recommendation (M07, transport lifecycle):** have the serve path consult
`server_runtime_status()` and exit with its `reason` before touching `_server`.
This makes the existing degradation contract actually load-bearing and costs one
call. Note it also depends on `MCP-D01` being fixed, or the pre-flight check will
report `server_available: True` in the shadowing case.

### `M02-02` (P3) — bundles hard-code bare `python`

`"command": "python"` is not portable: many Linux distributions ship only
`python3`, and where a virtualenv is not activated by the client, `python`
resolves to a system interpreter without `scikit-plots` installed — producing
`No module named scikitplot`, a different and equally unhelpful failure.

`sys.executable` is not available in static JSON, so this needs either a
documented prerequisite in `plugins/README.md` or per-bundle guidance to use an
absolute interpreter path. Low severity, high user-visibility.

---

## 6. Status changes

| ID | Before | After |
|---|---|---|
| `MCP-D04` | OPEN | **DECIDED** — not base-safe; remove server names from `__all__`; add the missing test (§3) |
| `MCP-D02` | OPEN | **DECIDED** — violates guide §7; three false Legacy-tier claims; blocked on M05 (§4) |
| `MCP-D08` | OPEN P3 | **OPEN P3, justification strengthened** — the conflation produced three false claims and one failing test |
| `MCP-M00-07` | NEW | reinforced — now shown to have propagated into user-facing documentation |
| `M02-01` | — | **NEW P2** — actionable error unreachable on the plugin launch path |
| `M02-02` | — | **NEW P3** — bare `python` in all 8 bundles |

---

## 7. Run record

```text
run_id                  M02
source_sha256           119855928dd052165c71efa61fa505aa206726265c7c375943084e64793c4018
scope                   import / optionality contract
commands                7 guide-mandated probes under an __import__ blocker  -> all PASS
                        star-import / __dir__ / __getattr__ resolution test
                        integration import vs construct matrix (pydantic + agno blocked)
                        plugin metadata JSON validation (17 files, 8 bundles)
                        default plugin launch: python -m scikitplot.mcp
confirmed               optionality contract holds on all named surfaces
decided                 MCP-D04 (star import), MCP-D02 (integration tier)
new                     M02-01 (P2), M02-02 (P3)
production code changed NO
next exact action       M03 (runtime capability truth). It is the natural next run and is
                        already loaded with evidence: MCP-D01 is a confirmed false
                        capability claim (find_spec presence != compatibility, no version
                        range check), and M02-01 shows the entry point does not consult
                        the capability report at all. M03 must also decide whether
                        server_capabilities() adopts Corpus's CapabilityStatus vocabulary
                        rather than defining a parallel one -- the BROKEN vs ABSENT
                        distinction that MCP-D05 needs.
                        M05 remains the pivot: MCP-M00-07 now blocks MCP-D02 and
                        MCP-D08 as well.
```
