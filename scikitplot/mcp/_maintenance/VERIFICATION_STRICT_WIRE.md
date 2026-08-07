# Strict wire and idempotence verification — 2026-08-06

## User-environment evidence before correction

Default suite:

```text
84 passed, 1 skipped
```

This demonstrated that unit, service, SDK in-memory, health, retrieval, URI,
score, and deterministic-output tests were healthy.

Opt-in live suite reached the Boolean case after eleven successful live tests:

```text
1 failed, 11 passed
boolean k unexpectedly succeeded
arguments = {"query": "transport", "k": true}
```

The response contained one result, proving that the MCP/Pydantic layer converted
`true` to `1` before invoking the already-strict service method.

## Corrective controls

- MCP tool annotations use `StrictStr` and `StrictInt`.
- The service retains independent direct-call validation.
- Real-SDK and SDK-double tests reject Boolean, numeric-string, and float limits.
- Live health checks compare the running version with the local package version.
- Optional SDK tests use `pytest.importorskip("mcp")`.
- Non-empty responses reject non-null status messages.
- Constructor concurrency/timeout values reject Boolean and string coercion.

## Sandbox verification

The sandbox does not provide the optional `mcp>=2,<3` package, so real-SDK tests
are intentionally skipped there. All dependency-independent tests passed:

```text
pytest -q scikitplot/mcp/tests
94 passed, 2 skipped
```

The two skips are:

1. real MCP SDK in-memory protocol tests because the SDK is unavailable;
2. live HTTP acceptance because `SCIKITPLOT_MCP_RUN_LIVE` is not enabled.

Collection with an MCP SDK-compatible stub confirms the expected user-side
matrix after this update:

```text
default suite: 109 test items plus one live-module skip
live suite:    137 test items
```

When the live command does not configure a canary token, the real-index canary
case intentionally skips. The wrapper configures the built-in canary by default.

Additional checks performed during packaging:

```text
compileall: PASS
bash syntax: PASS
normal suite repeated: PASS
reverse-order dependency-independent suite: 94 passed
self-test byte comparison: PASS
archive extraction and test: PASS
```

## Required Docker verification

After applying the update, rebuild and recreate the running process. Restarting
is essential because live tests target the process already bound to port 8000.

```bash
docker compose build mcp
docker compose up -d --force-recreate mcp
```

Confirm the new version:

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/healthz; echo
```

Expected:

```json
{"status":"ok","service":"scikitplot-docs","version":"0.2.2"}
```

Then run:

```bash
SCIKITPLOT_MCP_RUN_LIVE=1 pytest scikitplot/mcp/ -vv
```

or the version-aware wrapper:

```bash
bash scikitplot/mcp/tests/test_mcp_search_docs.sh
```

The previously failing Boolean case must now return an MCP validation/tool error,
not a successful `k=1` search.
