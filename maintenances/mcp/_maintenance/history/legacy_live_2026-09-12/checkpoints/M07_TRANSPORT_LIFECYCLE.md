# Run M07 — Transport lifecycle

```text
run_id            M07
date              2026-08-21
source_sha256     ee7593f81b35bd90a1dd2ba03691aadd2fa491599bce5a2b85dd70f88b14bb2c
scope             real-SDK protocol behaviour, error channel, lifecycle, health route
guide             MCP_UNIFIED_DEFERRED_REVIEW_GUIDE.md §12
mode              REVIEW + IMPLEMENTATION INCREMENT
exit gate         PARTIAL — in-process transport fully exercised; stdio/HTTP live -> M12
```

---

## 0. The real SDK is installed

```text
mcp        2.0.0      (satisfies the declared mcp>=2.0.0,<3)
pydantic   2.13.4
starlette  1.6.0
anyio      4.14.2
```

This is the first run able to honour the guide's instruction: *"Do not infer
transport correctness from unit-level direct function calls."* **That instruction
was right, and this run proves it** — see §2.

### Suite, real SDK, no ignores

```text
pristine tree (baseline)   153 passed, 1 skipped, 0 failed
this tree, before M07      166 passed, 1 skipped, 0 failed
this tree, after M07       166 passed, 1 skipped, 0 failed
```

Every collection error and failure recorded in `MCP-M00-01` and `MCP-M00-12` was
an artefact of the missing extra. With `[mcp]` present the suite is green on both
trees, and the M04/M05 increments add 13 tests and break nothing.

Notably `test_search_service_rejects_coerced_max_concurrency` and
`..._acquire_timeout` pass unchanged, which is direct evidence that M05's
delegation of `SearchService` to `SearchCoordinator` preserved behaviour exactly.

`pytest-asyncio` is required to collect `test_protocol_in_memory.py` and *is*
declared in `pyproject.toml:437`; it simply was not installed here.

---

## 1. M06's deferred items — now closed

Both needed a real SDK. Against `mcp 2.0.0`:

```text
arg_model             search_docsArguments
extra policy          forbid
additionalProperties  False
required              ['query']
properties            ['k', 'query']
```

End-to-end over the in-process protocol:

| call | `is_error` | rejected by |
|---|---|---|
| `{"query":"transport","k":2}` | False | — |
| `{"query":"transport","k":2,"evil":"x"}` | **True** | schema (`additionalProperties`) |
| `{"k":2}` | **True** | schema (required) |
| `{"query":"transport","k":999}` | **True** | schema (`le`) |
| `{"query":"transport","k":"two"}` | **True** | schema (StrictInt) |
| `{"query":"x"*2000}` | **True** | schema (`max_length`) |
| `{"query":"   "}` | **True** | `SearchCoordinator` |

Two independent layers fire: the SDK-derived schema, then the Tier-L coordinator
for what a schema cannot express (whitespace-only queries). The
`_forbid_unknown_tool_arguments` seam **works against the real SDK**.

---

## 2. `M07-01` (P2) — M04's status never reached the wire

The first protocol-level measurement contradicted M04's unit-level verification:

```text
strict=False   is_error=False   retrieval_status=None   structured=present
strict=True    is_error=True    retrieval_status=None   structured=NONE
```

M04 had verified `build_search_docs_result` directly and confirmed
`retrieval_status='failed'` with `isError=True`. Both were true at Tier-L and
**neither survived the server tier**:

- `SearchDocsOutput` is a *closed* model (`extra="forbid"`) with a fixed field
  list. `retrieval_status` and `retrieval_errors` were not fields, so pydantic
  **silently dropped them**.
- `SearchService.search` read only `raw["structuredContent"]` and never
  `raw["isError"]`, so the error flag was discarded too.

So the guide's required invariant — *MCP must not say "No matching documentation"
when every retrieval path failed* — held at Tier-L and **failed at the MCP wire**,
which is the only place it is externally observable.

This is a gap in my own M04 verification, and it is exactly the failure mode the
guide's methodology rule names. Recording it as such rather than quietly fixing
it.

**Fixed:** `retrieval_status` and `retrieval_errors` added to `SearchDocsOutput`
as documented optional fields; `SearchService` propagates both.

## 3. `M07-02` (P2) — the success path never reported status

`build_search_docs_result` set `retrieval_status` only in its `if not safe:`
branch. Any result *with* hits carried no status — so **`DEGRADED` was invisible
precisely when it matters most**: a partial result that still returns passages.
A client inspecting only `count` would see a normal answer and never learn part
of the evidence was missing.

**Fixed:** the success path now reports status and surfaces `retrieval_errors`.

## 4. `MCP-D03` / `M06-04` — resolved at the protocol level

After both fixes, measured through a real `Client`:

| retriever | `is_error` | `retrieval_status` | `count` | `retrieval_errors` |
|---|---|---|---|---|
| healthy | False | `success` | 1 | — |
| **fail + ok (fusion)** | False | **`degraded`** | 1 | `['index down']` |
| all-fail, `strict=False` | **True** | — | — | — |
| all-fail, `strict=True` | **True** | — | — | — |

`M06-04` is resolved: both `strict` modes now produce the **same** protocol
outcome, so a construction flag the client cannot observe no longer selects the
error channel. `strict` is back to being a diagnostic control.

The degraded row is the substantive win — the client receives its passages *and*
learns the answer is incomplete, with the reason.

### Accepted trade-off, recorded

A total failure raises, which makes the SDK set `is_error` but discards
`structured_content`, so per-leg detail is lost on that path. The alternative —
returning `is_error=False` with `status="failed"` — would make a total failure
look like a successful call, which is the exact lie M04 exists to remove. The
error text carries the disambiguation (*"this is not a statement that no
documentation matches the query"*), and full detail remains available at Tier-L
for in-process consumers. Emitting `isError` **with** structured content would
require constructing `CallToolResult` directly; deferred to M12 as an SDK-coupling
decision.

---

## 5. Lifecycle verification

**Concurrency semaphore** — `max_concurrency=1`, `acquire_timeout=0.01`, three
concurrent callers against a 0.2 s retriever:

```text
rejected 2 of 3 -> RuntimeError: search service is busy; retry shortly
```

Bounded correctly, fails fast with an actionable message rather than queueing
unboundedly.

**Server close / no orphan tasks** — three full create → connect → call → close
cycles:

```text
orphan tasks: before=1 after=1 leaked=0
```

**Health route validation** — `create_server` rejects malformed paths at build
time:

```text
'/healthz' -> registered      'bad' -> ValueError      '' -> ValueError
None       -> not registered  '//x' -> registered (accepted; minor)
```

### `M03-04` confirmed at runtime

`create_server(r, health_path="/healthz")` registers the route with **no
transport parameter in scope** — it cannot know the transport. The default is
`/healthz`, so the documented default library call registers an HTTP route that
a stdio server cannot serve. The CLI is safe only because `__main__.py` gates it.
Unchanged from M03; now observed rather than read.

---

## 6. `M07-03` (P3) — the SDK-missing guard misreports transitive failures

With `starlette` blocked but `mcp` installed:

```text
create_server(...)  ->  RuntimeError: MCP SDK v2 is required for the server
                        layer; on Python >= 3.10 install it with:
                        pip install "mcp>=2.0.0,<3"
```

The SDK **is** installed. The guard catches an `ImportError` raised from inside
the SDK's own dependency chain and reports it as "SDK not installed", sending the
user to reinstall a package that is already present.

This is the fourth site of one habit, after `M03-03` (`_present()` → `False` for a
*failed* probe), `MCP-D05` (`BROKEN` reported as `ABSENT`) and `M06-01`
(uncoercible score → a legitimate `0.0`). Each converts "something is wrong in a
way I did not check" into a confident, specific, wrong claim.

### `M01-01` severity corrected: P2 → P3

M01 rated the undeclared `starlette` dependency P2 on the grounds that the SDK
might restructure and drop it. This run shows `starlette` is a **hard requirement
of `mcp` itself** — blocking it breaks the SDK import outright, not just the
health route. So the "SDK stops pulling it in" scenario is not a realistic
near-term risk.

The finding **stands** — an undeclared direct import still violates the
dependency policy, and `tests/test_server.py:16` still imports it at module
scope — but the practical risk is lower than I claimed. Downgraded to P3, with
the correction recorded rather than silently amended.

---

## 7. Deferred to M12

The guide's remaining lifecycle items need a live server on a port or a client
that can be severed mid-call:

```text
stdio transport (subprocess)      client disconnect
Streamable HTTP transport          timeout
cancellation                       HTTP status/error shape
```

`tests/integration/test_mcp_http_live.py` exists for exactly this and skips
unless `SCIKITPLOT_MCP_RUN_LIVE=1` against an already-running server. That is the
right shape; M12 should run it rather than reimplement it. **In-process transport
is fully exercised** and is where the invariant defects were found.

---

## 8. Changed files

```text
MOD scikitplot/mcp/_server.py   SearchDocsOutput gains retrieval_status /
                                retrieval_errors; SearchService propagates both;
                                search_docs raises on a FAILED outcome
MOD scikitplot/mcp/_core.py     success path reports status + errors
```

Both edits are confined to the result-shaping path. No change to ranking,
sanitisation, validation bounds, capabilities, packaging or the seam.

---

## 9. Run record

```text
run_id                  M07
environment             FIRST run with the real SDK (mcp 2.0.0, pydantic 2.13.4,
                        starlette 1.6.0, anyio 4.14.2, pytest-asyncio)
suite                   pristine 153 passed / this tree 166 passed / 0 failed both
closed                  M06 deferred items (extra=forbid, unknown arguments) VERIFIED
                        M06-04 (error channel) RESOLVED
                        MCP-D03 now holds AT THE WIRE, not just at Tier-L
new                     M07-01 (P2) status dropped by the closed output model  [FIXED]
                        M07-02 (P2) success path never reported status         [FIXED]
                        M07-03 (P3) SDK guard misreports transitive ImportError
corrected               M01-01 severity P2 -> P3 (starlette is a hard SDK requirement)
confirmed               M03-04 at runtime; semaphore bounds; no orphan tasks
regressions             NONE
deferred to M12         stdio + Streamable HTTP live transports, cancellation,
                        disconnect, timeout, HTTP status shape, isError-with-payload
next exact action       M08 (plugins and integrations). M02 already validated all 17
                        plugin JSON files and found M02-01 (the actionable error is
                        unreachable on the launch path all 8 bundles use) and M02-02
                        (bare `python`). M07 supplies what M02 lacked: with the SDK
                        installed, `python -m scikitplot.mcp` can now be run end to end
                        per bundle, so M08 can verify each declared launch actually
                        starts a server rather than inspecting JSON.
```
