# Run M12 (continued) — Live Streamable HTTP closure

```text
run_id            M12-LIVE
date              2026-08-21
source_sha256     9be9218a2465bfe146bc28e60b1b1ff58bad2fe99fb60e87834a8f4307ac76eb
scope             the transport lanes M07/M12 deferred for want of a bound port
mode              REVIEW ONLY — no production code changed
result            Streamable HTTP, cancellation and graceful shutdown all GREEN
```

---

## 1. What changed since M12's first pass

Nothing in the code. M12 deferred four items *"needs a bound port"*, and this
container can bind `127.0.0.1`. The existing harness was **run**, not
reimplemented, exactly as the M08–M12 assessment recommended.

One practical note for anyone reproducing this: a server started with plain
`nohup … &` does **not** survive between tool invocations — the first attempt
died and produced 20 `ConnectionRefusedError` failures that looked like 20
defects. `setsid` fixes it. The 20 failures were harness-environment artefacts,
not findings; recorded here so nobody re-diagnoses them.

## 2. Server startup

```console
$ python -m scikitplot.mcp --transport streamable-http --host 127.0.0.1 --port 8000
INFO __main__: starting MCP Streamable HTTP bind=127.0.0.1:8000 endpoint=/mcp
     health=/healthz stateful=False
INFO mcp.server.streamable_http_manager: StreamableHTTP session manager started
INFO Uvicorn running on http://127.0.0.1:8000

$ curl -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/healthz
200
```

`__main__.py` correctly passes `health_path` only for `streamable-http`, so the
`M03-04` default change (health route now opt-in in `create_server`) leaves the
CLI behaviour intact — confirmed by the health endpoint answering 200 here.

## 3. The live suite

```console
$ SCIKITPLOT_MCP_RUN_LIVE=1 pytest tests/integration/test_mcp_http_live.py -q
27 passed, 1 skipped
```

The skip is `SCIKITPLOT_MCP_CANARY_TOKEN`, which verifies a *real indexed
document* and is deployment-specific — correctly optional.

Covered by those 27: tool listing, tool call contract, citation numbering and
uniqueness, untrusted-passage prefix, 20 000-char passage cap, the full invalid
argument matrix (negative/oversized/boolean/float/string `k`, wrong-typed and
boolean and array `query`), malformed HTTP, health repeatability, and parallel
request pressure.

## 4. Round-trip and strict validation over HTTP

```text
HTTP tools:            ['search_docs']
HTTP call              is_error=False  status=success  count=1
HTTP unknown-arg       is_error=True
```

`extra="forbid"` and the `_forbid_unknown_tool_arguments` seam hold over the
network transport, not merely in-process. M04/M07's `retrieval_status` reaches
the client over HTTP as well.

## 5. Cancellation

```text
cancellation:            client call cancelled cleanly
after cancellation,      health = 200
```

A call aborted mid-flight raises `CancelledError` client-side and leaves the
server serving. No wedged session, no partial-response corruption.

## 6. Concurrency pressure

```text
25 concurrent calls -> exceptions=0  is_error=0  ok=25
after pressure,         health = 200
```

Every request succeeded. Note the contrast with the Tier-L semaphore test in M07
(`max_concurrency=1` → 2 of 3 rejected fast): the default `max_concurrency` is
comfortably above this load, so the bound is doing its job without being the
bottleneck. Both behaviours are correct; they are different configurations.

## 7. Graceful shutdown

```console
$ kill -TERM <pid>
INFO:     Waiting for application shutdown.
INFO mcp.server.streamable_http_manager: StreamableHTTP session manager shutting down
INFO:     Application shutdown complete.
INFO:     Finished server process [513]

process:  gone
port:     connection refused (released)
```

Ordered shutdown, session manager closed, socket released, no orphan process.

---

## 8. Closure status

```text
[x] deferred queue reverified
[x] packaging/install contract green
[x] optional import boundary deliberate/tested
[x] compatible SDK version detected correctly
[x] neutral Corpus outcomes mapped correctly
[x] backend failure never masquerades as empty
[x] integrations/plugins do not duplicate protocol logic
[ ] min/latest real SDK matrix green            M01-05: mcp 2.0.0 is the ONLY
                                                stable 2.x release, so min == latest.
                                                Degenerate until 2.1 ships.
[x] stdio green                                 M08
[x] Streamable HTTP green                       THIS RUN
[x] cancellation/shutdown green                 THIS RUN
[x] private SDK seam has drift gates
[ ] historical maintenance state marked current RULESET.md,
                                                MCP_COMPATIBILITY_POLICY.md and
                                                DESIGN.md have never appeared in
                                                any supplied archive
```

**11 of 13 → 12 of 13**, counting the min/latest matrix as unmet because it is
degenerate rather than green.

## 9. The two remaining items — and why neither is a code defect

**`M01-05` — the min/latest SDK matrix.** `mcp 2.0.0` is the only stable 2.x
release on PyPI (the other 2.x entries are `a1`–`a3`, `b1`, `b2` prereleases).
Under `mcp>=2.0.0,<3`, minimum and latest are the same version, so the matrix
cannot be meaningfully green today. It is **satisfiable the day 2.1 ships**, and
should be pinned as a CI gate now so it does not silently pass forever.

The wheel-build and Python 3.10 lanes belong here too: they need a
meson-python/Cython toolchain and multiple interpreters, neither of which exists
in this container. Asserting wheel metadata without a built wheel would be
`INFERRED` presented as `VERIFIED`, which this campaign has consistently refused
to do.

**Maintenance state.** `RULESET.md`, `MCP_COMPATIBILITY_POLICY.md` and
`DESIGN.md` are marked KEEP, sit in `MAINTAINING.md`'s mandated read order, and
`DESIGN.md` is still cited from runtime source (`_hybrid.py:22,346`). They have
been absent from all four archives supplied during this campaign. Only the
maintainer can resolve this — either supply them, or record them as lost and
regenerate. The anchor should also move to `9be9218a…` (`MCP-M00-11`).

---

## 10. Run record

```text
run_id                  M12-LIVE
commands                live server on 127.0.0.1:8000 (streamable-http)
                        SCIKITPLOT_MCP_RUN_LIVE=1 pytest test_mcp_http_live.py -> 27 passed
                        cancellation probe -> cancelled cleanly, health 200
                        25-way concurrency probe -> 0 exceptions, health 200
                        SIGTERM -> ordered shutdown, port released, no orphan
production code changed NO
new findings            NONE
closure                 12 of 13; remaining two need infrastructure or the
                        maintainer's documents, not code
next action             Pin the min/latest SDK matrix as a CI gate (it becomes
                        meaningful when mcp 2.1 ships), run the wheel/3.10 lanes in
                        CI where a toolchain exists, and resolve the three KEEP
                        documents. Then scikitplot.mcp can be marked CLOSED.
```
