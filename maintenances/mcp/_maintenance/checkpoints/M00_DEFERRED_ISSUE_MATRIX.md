# M00 — Deferred issue matrix (`MCP-D01` … `MCP-D09`)

Each issue from `MCP_UNIFIED_DEFERRED_REVIEW_GUIDE.md` §3, reproduced against the
supplied tree. Guide instruction: *"Treat these as `REVERIFY`, not automatically
`OPEN`."*

```text
source_sha256   43f5060c971f8be0272efaef9cc092b1bb7edc22614df9fa0c23ad26beb42c61
                MISMATCH vs recorded anchor 611bdbf3… -> all statuses REVERIFY-grade
runtime         NOT reproducible (import scikitplot fails; see M00_REVALIDATION.md §1a)
basis           static: AST, grep, file inventory, isolated reproducers
```

| ID | Status | Severity | Evidence class |
|---|---|---|---|
| D01 | **OPEN** | P2 | VERIFIED (static) |
| D02 | **OPEN** | P2 | VERIFIED |
| D03 | **OPEN** | P2 | VERIFIED |
| D04 | **OPEN** | P2 | VERIFIED (reproducer) |
| D05 | **OPEN** | P2 | VERIFIED |
| D06 | **OPEN** | P2 | VERIFIED |
| D07 | **OPEN** | P2 | VERIFIED |
| D08 | **OPEN** | P3 | VERIFIED |
| D09 | **OPEN** | P2 | VERIFIED |

**9 of 9 confirmed open. None disproved.**

---

## D01 — runtime status may treat package presence as compatibility

**OPEN.** `_capabilities.py:129-192`.

```python
sdk_present = _present("mcp")          # importlib.util.find_spec("mcp") is not None
...
"server_available": bool(python_ok and sdk_present),
```

Two defects.

**(a) No version check.** The declared contract is `mcp>=2.0.0,<3`. `find_spec`
answers *"is there a module named `mcp`"*, not *"is it in range"*. With SDK 1.x
installed, `server_available` reports `True` while the SDK is outside the
supported range — presence treated as compatibility, exactly as the deferred
issue predicted.

**(b) `find_spec("mcp")` is shadowed by scikit-plots' own package.** Because
`find_spec` resolves against `sys.path`, and `sys.path[0]` is the working
directory:

```console
$ cd scikitplot && python -c "import importlib.util; print(importlib.util.find_spec('mcp').origin)"
/home/claude/work/scikitplot/mcp/__init__.py        <-- scikit-plots' OWN package

$ cd .. && python -c "import importlib.util; print(importlib.util.find_spec('mcp'))"
None
```

Run from inside `scikitplot/`, `sdk_present` becomes `True` and
`server_available` reports `True` **with no MCP SDK installed at all**. The
module then advertises a capability it cannot serve — directly violating
`MAINTENANCE_MODEL.md`'s *"A capability is claimed, not probed"* leak class and
the `server_capabilities()` invariant *"Never claims an unprobed capability."*

**Recommendation (M03):** probe by resolving the installed distribution version
(`importlib.metadata.version("mcp")`) and range-check it, rather than testing for
a module name. Consume Corpus's `CapabilityStatus` so `BROKEN` (present, wrong
version) is distinguishable from `ABSENT`.

---

## D02 — framework-neutral integration may cross the Pydantic/server boundary

**OPEN.** `integrations/agno/docs_toolkit.py:35`.

```python
def __init__(self, retriever=None, *, k_default: int = 5) -> None:
    from scikitplot.mcp._server import SearchService
```

`SUBMODULE_STRUCTURE.md` §2 states adapters must *"depend on the public API only,
**never** on `_server`"*. This one depends on `_server`.

The import is call-time, so the **import surface** stays clean — the boundary is
not breached at import. But constructing the toolkit requires `pydantic`, while
the class docstring says only *"no MCP SDK is imported"* (true, and misleading:
it omits the pydantic requirement).

**This cannot be fixed by "use the public API".** `scikitplot.mcp.SearchService`
is a lazy re-export of the same server-tier object (`__init__.py:_SERVER_EXPORTS`),
so it pulls `pydantic` too. Per `MCP-M00-07`, no protocol-neutral `SearchService`
exists anywhere in the tree.

**Recommendation:** blocked on M05. D02 is a symptom of `MCP-M00-07`, not an
independent defect — sequence it after the `SearchService` ownership decision.

---

## D03 — resilient retrieval may collapse backend failure into empty result

**OPEN.** `_hybrid.py:218,292,312`; `_corpus_annoy.py:188,208`.

The pattern, with `strict: bool = False` as the **default** (`_hybrid.py:163,278`):

```python
except Exception as exc:            # resilience boundary
    logger.warning("… failed and was skipped: %s", exc, exc_info=self._strict)
    if self._strict:
        raise
    continue                        # or: return []
```

**This is a contract-shape problem, not just an implementation choice.** The
Protocol itself has no status channel:

```python
# _core.py:131
def search(self, query: str, k: int = 5) -> list[RetrievedChunk]: ...
```

A bare `list` cannot express *"a leg failed"*. If every leg fails in the default
non-strict mode, the caller receives `[]` — **indistinguishable from "no
documents matched"**. The failure reaches the log and dies there; the wire sees
success-with-zero-results.

This is precisely the M04 question named in `TRACKER_LOGICAL.md` §3: *"`DEGRADED`
is not an error and not a success; a wire response that flattens it to either is
wrong."* The flattening is already present, one layer below the wire.

**Recommendation (M04):** map Corpus's `RetrievalResponse`/`RetrievalStatus`, with
`LegOutcome`/`LegStatus` carrying per-leg detail. Widening `DocsRetriever`'s
return type is a Protocol change and must go through M04, not a local patch —
`MAINTENANCE_MODEL.md` warns that widening it to fit one backend turns it into a
capability lottery.

---

## D04 — package `__all__` may force optional server names under star import

**OPEN.** `__init__.py:40-56`.

`__all__` includes `_SERVER_EXPORTS` (`CitationOutput`, `SearchDocsOutput`,
`SearchService`, `create_server`), which are served by a PEP 562 module-level
`__getattr__` that imports `_server` — and therefore `pydantic`.

`from scikitplot.mcp import *` resolves **every** name in `__all__`, invoking
`__getattr__` for the lazy ones. Reproduced in isolation (the real package cannot
be imported — see §1a of the checkpoint):

```console
--- plain import ---
   _server loaded? False
--- star import ---
   [!] _server.py EXECUTED -> pydantic would be imported here
   _server loaded? True
```

So the lazy tier is defeated by a star import: on a base install without
`pydantic`, `from scikitplot.mcp import *` raises `ImportError`, while
`import scikitplot.mcp` succeeds.

`test_mcp_import_surface.py` tests `import scikitplot.mcp` but **not** the star
import, so this path is untested.

**Recommendation (M02):** decide deliberately — either drop the server names from
`__all__` (star import then yields the Tier-L surface only), or accept and
**document** that `import *` requires `[mcp]`. Either way, add the star-import
case to `test_mcp_import_surface.py`. Note `__dir__` has the same reach but is
harmless (it does not resolve).

---

## D05 — Corpus adapter import guard may catch overly broad exceptions

**OPEN.** `_corpus_annoy.py:289`.

```python
try:
    from scikitplot.corpus import BuilderConfig, CorpusBuilder, EmbeddingEngine
except Exception as exc:                     # <-- not ImportError
    raise RuntimeError(
        "scikitplot.corpus is required to build the retriever install"
        "the corpus/embedding extras (pip install scikit-plots[corpus])."
    ) from exc
```

Catching `Exception` means a genuine fault *inside* an installed
`scikitplot.corpus` — `TypeError`, `AttributeError`, a misconfigured optional
backend — is reported as *"scikitplot.corpus is required … install the extras"*.
The user is told to install a package that **is** installed but broken:
`BROKEN` misreported as `ABSENT`, the exact distinction
`TRACKER_LOGICAL.md` §3 says matters (*"reinstall versus install"*).

**Inconsistent with its sibling.** `_hybrid.py:350` guards the same class of
operation correctly:

```python
except ImportError as exc:   # pragma: no cover - optional integration path
```

Two adapters, same module family, two different contracts.

Also carries `MCP-M00-08` — the message is malformed by implicit concatenation
and renders as `…build the retriever installthe corpus/embedding extras…`.

**Recommendation:** narrow to `ImportError`; let other exceptions propagate with
their real type. Fix the string. Both are minimal-impact and independent of the
architectural runs.

---

## D06 — packaging regression assertions may be weaker than policy

**OPEN — and worse than "weaker".** See `MCP-M00-10` in the checkpoint for the
full evidence.

Two facts:

1. `tests/test_mcp_import_surface.py::test_mcp_extra_declares_server_dependencies`
   has its policy assertions **commented out**, leaving `sdk_lines` computed and
   unused and `import re as _re` unused. The live assertions check only that
   `pydantic` and `mcp>=2.0.0,<3` appear somewhere in the block.
2. The `[mcp]` extra **violates the policy stated in its own comment**: the
   eleven-line comment says the SDK line must carry no python marker; the line
   carries `; python_version >= "3.10"`.

The disabled assertion is exactly the one that would have caught the live
violation.

**Recommendation (M01):** blocked on `MCP_COMPATIBILITY_POLICY.md`, which is
missing (checkpoint §3). Do not "fix" either side until the policy document is
recovered and states which is authoritative — the comment or the metadata.

---

## D07 — legacy-tier tests may exercise server-tier objects

**OPEN**, with a precise root cause — see `MCP-M00-06`.

`tests/test_mcp_runtime_status.py` declares itself Tier-L in line 1:

> *"SDK-free capability probe + guaranteed SDK-free retrieval (Tier-L, Python 3.8+)."*

and then, on line 9:

```python
from scikitplot.mcp._server import SearchService, server_runtime_status
```

`server_runtime_status` is **defined in `_capabilities.py:129`** — the
pydantic-free module — and only re-exported by `_server.py:35`. The test reaches
a Tier-L function through the Tier-S module gratuitously, contradicting its own
docstring and making itself uncollectable without `pydantic`.

`test_mcp_version_guard.py` (line 11, `from scikitplot.mcp import _server`) and
`test_mcp_hardening.py` (line 11) are also Tier-L in intent but genuinely
exercise `_server`, so they need `importorskip`, not re-pointing.

**Recommendation:** re-point `test_mcp_runtime_status.py` to `_capabilities` —
one line, no behaviour change, removes one of the four collection errors from
`MCP-M00-01`. Guard the other three. Note `test_server.py` also needs `starlette`,
which the `[mcp]` extra does not declare.

---

## D08 — "SDK-free" terminology used where "protocol-neutral" is intended

**OPEN.** Guide §2 draws the distinction and gives the exact test:

> *"If `_core.py` constructs MCP-shaped structures such as `structuredContent` /
> `isError`, it is SDK-independent but still MCP-aware."*

It does, in three places:

```text
_core.py:257,264    "structuredContent": {…}, "isError": False
_core.py:313,321    "structuredContent": {…}, "isError": False
_core.py:351,359    "structuredContent": {…}, "isError": False
```

`README.md:216` nevertheless describes `_core.py` as *"SDK-free contracts"*. By
the guide's own criterion the accurate term is **SDK-independent but MCP-aware**.
"SDK-free" appears 20 times across the module; "protocol-neutral" appears only in
maintenance docs, never in source.

The distinction is load-bearing rather than pedantic: `MAINTENANCE_MODEL.md`
forbids a parallel vocabulary, and `_core.py` emitting wire-shaped dicts is what
makes `MCP-M00-07` (`SearchService` misattributed as protocol-neutral) plausible
enough to have gone unnoticed.

**Recommendation (M05, with `MCP-M00-07`):** fix the terminology once the
ownership question is settled, so prose and structure change together.

---

## D09 — historical maintenance documents may contain superseded facts

**OPEN, extensively.** Full evidence in checkpoint §3 (`MCP-M00-04`) and §5
(`MCP-M00-05`, `MCP-M00-07`).

| Superseded fact | Recorded | Actual |
|---|---:|---:|
| markdown files | 31 | 17 |
| `_maintenance/` markdown | 16 | 8 |
| `_maintenance/history/` markdown | 6 | 0 (contains `.py` only) |
| source files | 16 | 17 |
| source LOC | 3 182 / 3 503 | 3 515 |
| markdown : source ratio | 1.9 | 1.0 |
| `SearchService` module | `_core.py` | `_server.py:110` |
| `SearchService` invariant | "never MCP wire types" | returns pydantic `SearchDocsOutput` |
| boundary invariant 3 | "protected by nothing" | tested in `test_mcp_import_surface.py` |
| `stale_lifecycle.py` path | `_maintenance/` | `_maintenance/history/` |

Plus **18 dangling documentation targets**, three of which (`RULESET.md`,
`MCP_COMPATIBILITY_POLICY.md`, `DESIGN.md`) are marked **KEEP**, are required by
`MAINTAINING.md`'s read order, and are cited from **runtime source**
(`_capabilities.py`, `_core.py`, `_hybrid.py`, `integrations/__init__.py`).

`SUBMODULE_STRUCTURE.md` §3 also instructs *"use `stale_lifecycle.py`"* to manage
archival — but that script has itself been archived into `history/`.

**Recommendation:** D09 is the reason M00 cannot close (checkpoint §7). Resolve
the anchor and the three missing KEEP documents first; then re-derive every count
with `check_trackers.py --update` and extend the gate to compare `markdown_files`
(`MCP-M00-05`).
