# Runs M08–M12 — Plugins, security, agentic boundary, migration, closure matrix

```text
runs              M08, M09, M10, M11 (COMPLETE) · M12 (PARTIAL)
date              2026-08-21
source_sha256     ee7593f81b35bd90a1dd2ba03691aadd2fa491599bce5a2b85dd70f88b14bb2c
environment       real SDK: mcp 2.0.0, pydantic 2.13.4, starlette 1.6.0, anyio 4.14.2
mode              REVIEW ONLY — no production code changed in these runs
guide             §13–§17, closure definition §18, rejection list §19
```

---

# M08 — Plugins and integrations · COMPLETE

| Guide question | Answer |
|---|---|
| is the command/path real? | **YES — executed, not inspected** (§M08.1) |
| is the install extra real? | **YES** — `[mcp]` resolved and installed `mcp 2.0.0` |
| does metadata drift from runtime? | **NO** — declared schema matches served schema |
| does import stay lazy? | **YES** — verified M02, re-verified after M05 |
| are credentials embedded? | **NO** — scanned all 17 JSON files, zero credential-like keys |
| is transport configuration valid? | **YES** — stdio round-trip green |
| does integration duplicate MCP logic? | **NO — fixed in M05** |

## M08.1 The declared command actually works

All 8 bundles reduce to **one** distinct invocation, `python -m scikitplot.mcp`,
declared by 7 of them (`.claude-plugin/.mcp.json` carries it under `mcpServers`).
Launched as a real MCP client would, over genuine stdio subprocess transport:

```text
STDIO OK tools: ['search_docs']
call            is_error=False  status=success  count=1
call unknown-arg is_error=True
```

This answers M08's central question by execution. It also turns M12's
`stdio round-trip` item green.

*Note:* `list_resources` returns `[]` because `docs://chunk/{doc_id}` is a
resource **template**, surfaced through `list_resource_templates`. Correct MCP
behaviour, not a defect.

## M08.2 Integrations no longer duplicate protocol logic

M05 closed `MCP-D02`: `integrations/agno` now depends on the Tier-L
`SearchCoordinator`, not `_server`. `integrations/openclaw` was always
config-only. Neither reimplements protocol handling — they are packaging/config
artifacts, as the guide requires.

## M08.3 Open, unchanged from M02

- **`M02-01` (P2)** — on a base install `python -m scikitplot.mcp` still emits a
  raw `ModuleNotFoundError: pydantic` traceback instead of the actionable message
  at `_server.py:280`, because `__main__.py:51` imports the `_server` *module*
  first. This is the launch path all 7 bundles declare. Untouched by M04–M07.
- **`M02-02` (P3)** — bare `python` in every bundle.

---

# M09 — Security / trust boundaries · COMPLETE

Verified through the real protocol, not by reading code.

| Item | Result |
|---|---|
| prompt injection | **contained** — payload delivered with `UNTRUSTED` notice |
| URI sanitization | **enforced** — `file:///etc/passwd` → `''` |
| path leakage | **none** — `_DOC_ID_RE` rejects `/`, traversal blocked (M06) |
| oversized content | **capped** — `MAX_CHUNK_CHARS`, `_MAX_RESOURCE_CHARS` 20 000 |
| resource enumeration | **bounded** — single template, id-validated |
| tool metadata trust | **enforced, not hinted** — see below |
| raw exception exposure | **none** — §M09.1 |
| health endpoint leakage | **none** — §M09.2 |
| trace/log secrets | **see M09.3** |
| DoS via k / query length | **bounded** — `k ≤ 20`, query ≤ 1024, rejected at two layers |
| concurrent request pressure | **bounded** — semaphore, 2 of 3 rejected fast (M07) |

## M09.1 Raw exceptions are not exposed

A retriever raising with a live credential in the message:

```text
RuntimeError: psycopg2 connect failed dsn=postgres://u:AKIA_SUPER_SECRET_KEY_12345@db/x

client sees:  Error executing tool search_docs: documentation search is
              temporarily unavailable
secret leaked: False        is_error: True
```

`SearchCoordinator` wraps retriever exceptions in a generic `RuntimeError` and
logs the original server-side. Backend internals do not cross the wire.

## M09.2 Health endpoint

Returns `{status, service, version}` only, with `Cache-Control: no-store` and
`include_in_schema=False`. No environment, filesystem, backend or secret detail.

## M09.3 One caveat worth recording

The original exception — including any credential in it — **is** written to the
server log via `logger.exception`. That is correct for diagnostics and is not a
wire leak, but operators shipping logs to a third-party sink inherit whatever
secrets a backend puts in its exception text. Worth one line in the deployment
docs; not a code defect.

## M09.4 Annotations do not replace enforcement — satisfied

`read_only`/`idempotent`/`open_world:false` are *advertised*, and separately
**enforced**: exactly one tool exists (`search_docs`), it performs no writes, and
the resource path is read-only. The guide's warning is respected.

---

# M10 — Agentic capability boundary · COMPLETE

The preferred deployment shape holds exactly:

```text
Agent/host → MCP client → scikitplot.mcp → read-only Corpus retrieval
```

Scanned for any autonomous or server-initiated surface:

```text
sampling / create_message / elicit / roots / subscribe   -> NONE
write-capable tools                                      -> NONE (search_docs only)
```

The server is a passive read-only capability provider. It does not act as an
agent, and the conditional requirements (capability negotiation, budget,
recursion depth, cancellation, user control) **do not apply** because no
server-to-client sampling exists.

This is a deliberate design property worth preserving: the rejection list forbids
making the MCP server the primary autonomous agent, and nothing in M04–M07 moved
it in that direction.

---

# M11 — Compatibility / migration · COMPLETE

M04, M05 and M07 changed production code, so this run verified they did not break
consumers. Measured against the **pristine tree**:

| Surface | Result |
|---|---|
| public imports (`__all__`) | **removed: NONE** · added: `SearchCoordinator` |
| `SearchService` location | **unchanged** — still `scikitplot.mcp._server` |
| `RetrievedChunk` mapping | **unchanged** |
| tool input schema | **identical**: properties `['k','query']`, required `['query']` |
| tool output required fields | **identical**: `citations, count, passages, query, security` |
| CLI command | **unchanged** — `python -m scikitplot.mcp` |
| plugin configs | **unchanged** |
| README examples | **unchanged** (integration prose corrected in M05 to be *true*) |

**All changes are additive.** `retrieval_status` and `retrieval_errors` are
optional and therefore absent from the schema's `required` list, so an existing
client that ignores them behaves exactly as before. `SearchCoordinator` is a new
name, not a relocation — no migration adapter is needed because nothing moved.

The one behavioural change a client can observe is intended and is the point of
M04/M07: a total retrieval failure now returns `is_error=True` instead of a
successful empty result. A client that treated "empty" as "no matches" was
already being misinformed.

---

# M12 — Final real-SDK closure matrix · PARTIAL

## Green

```text
[x] base / lazy imports              M02, re-verified M07
[x] in-process round-trip            M07
[x] stdio round-trip                 M08.1  <- newly green
[x] tool list / call                 M07, M08
[x] resource list / read             M08 (template semantics confirmed)
[x] strict unknown args              M07 — rejected at the protocol
[x] structured output                M07
[x] shutdown                         M07 — 0 orphan tasks over 3 cycles
[x] failed-vs-empty mapping          M07 — success/degraded/failed all distinct
[x] effective capabilities           M03
```

## Not run, with cause

```text
[ ] build wheel / install wheel[mcp]   needs the meson-python + Cython/C++ toolchain;
                                       no wheel was built, so any metadata claim
                                       would be INFERRED presented as VERIFIED
[ ] Python 3.10 + minimum MCP v2       only 3.12 available here
[ ] Python 3.10 + latest MCP v2        "
[ ] next-Python prerelease lane        3.14 claimed in classifiers, untested (M01-04)
[ ] Streamable HTTP                    needs a bound port; the harness already exists
                                       at tests/integration/test_mcp_http_live.py and
                                       should be RUN, not reimplemented
[ ] timeouts / cancellation            needs a severable live client
```

`M01-05` remains true and now matters: `mcp 2.0.0` is the only stable 2.x
release, so "minimum" and "latest" are the same version today. The matrix is
degenerate until 2.1 ships — which is exactly when it starts mattering, silently.

---

# Closure assessment (guide §18)

```text
[x] deferred queue reverified                        M00, M00-R — all 9 MCP-Dxx
[ ] packaging/install contract green                 MCP-M00-10 open (marker vs comment);
                                                     M01-01 open (starlette undeclared)
[x] optional import boundary deliberate/tested       M02 under an __import__ blocker
[ ] compatible SDK version detected correctly        MCP-D01 / M03-01..03 — no sdk_version
[x] neutral Corpus outcomes mapped correctly         M04, corrected at the wire in M07
[x] backend failure never masquerades as empty       M07 — verified at the protocol
[x] integrations/plugins do not duplicate protocol   M05, M08
[ ] min/latest real SDK matrix green                 degenerate (M01-05); one stable 2.x
[x] stdio green                                      M08.1
[ ] Streamable HTTP green                            deferred
[ ] cancellation/shutdown green                      shutdown green; cancellation deferred
[x] private SDK seam has drift gates                 seam isolated to one function (M06);
                                                     fails closed (M00); verified against
                                                     the real SDK (M07)
[ ] historical maintenance state marked current      3 KEEP docs still absent; anchor stale
```

**8 of 13 met. `scikitplot.mcp` is NOT closed**, and should not be marked so.

## Rejection list (§19) — compliance

Every item **complied with**, and two were actively corrected during the campaign:

```text
swallow retrieval failure into []        <- WAS violated; fixed M04 + M07
trust package presence as compatibility  <- STILL violated: MCP-D01 open
expose raw backend exceptions            <- complied (M09.1)
spread private SDK access                <- complied (one function, M06)
advertise capabilities not configured    <- complied (M03)
make MCP SDK a base dependency           <- complied (M02)
move vector/graph logic into MCP         <- complied
make agent framework mandatory           <- complied (M05, M08)
add legacy JSON-RPC fallback             <- complied
copy another MCP stack                   <- complied
```

`trust package presence as compatibility` is the one live violation of the
project's own rejection list. That makes `MCP-D01` the highest-priority remaining
item, not merely an open finding.

---

# The five blockers to closure

1. **`MCP-D01` / `M03-01..03` — capability truth.** `sdk_version` and
   `sdk_compatible` do not exist, so the declared range is never enforced. On the
   rejection list. Fix as one class with `M03-03`, `MCP-D05`, `M06-01` and
   `M07-03` — five sites of the same habit: reporting a confident wrong answer
   where the honest answer is "I did not check".
2. **`MCP-M00-10` — packaging.** Adjudicated in M01 (remove the marker); needs
   the decision applied and the commented-out assertion re-enabled.
3. **`M01-01` (P3) — undeclared `starlette`.**
4. **M12's untested lanes** — wheel build, Python 3.10, Streamable HTTP,
   cancellation.
5. **Maintenance state** — `RULESET.md`, `MCP_COMPATIBILITY_POLICY.md`,
   `DESIGN.md` still absent (`DESIGN.md` cited from `_hybrid.py:22,346`); anchor
   still stale (`MCP-M00-11`).

---

# Run record

```text
runs                    M08, M09, M10, M11 COMPLETE · M12 PARTIAL
commands                stdio subprocess round-trip via the declared plugin command
                        credential scan across 17 plugin JSON files
                        secret-bearing exception through the real protocol
                        injection payload through the real protocol
                        public-surface and tool-schema diff vs the pristine tree
                        autonomous-surface scan (sampling/elicitation/write tools)
new findings            NONE — M08-M11 confirmed existing behaviour
production code changed NO
closure                 8 of 13 criteria met; scikitplot.mcp NOT closed
```
