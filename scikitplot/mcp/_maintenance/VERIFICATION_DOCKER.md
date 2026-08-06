# Docker, protocol, and idempotence verification — 2026-08-06

> **0.2.2 update:** see `VERIFICATION_STRICT_WIRE.md` for the later live
> Boolean-to-integer coercion finding, strict MCP input correction, stale-server
> version gate, and current verification commands. The historical counts below
> describe the preceding 0.2.1 stage.

## Scope

Verified the updated `scikitplot.mcp` package after adding and reconciling:

- explicit `--docker` CLI profile and `/healthz`;
- built-in `--probe` and backend-only `--self-test`;
- a stable no-result contract;
- Pydantic output invariants;
- strict input-type handling rather than implicit coercion;
- deterministic retrieval and repeated-call assertions;
- real MCP SDK in-memory protocol tests when `mcp>=2,<3` is installed;
- opt-in live Streamable HTTP acceptance tests;
- bounded, read-only concurrency/load scripts;
- multi-stage wheel-based Docker deployment.

## Defect reproduced and corrected

The live server returned this internally inconsistent result:

```json
{
  "count": 0,
  "passages": ["No matching documentation was found for this query."],
  "citations": []
}
```

The status sentence was human-readable output, not a retrieved passage. It now
lives in `message`, while the invariant below is enforced at runtime and in all
test layers:

```text
count == len(passages) == len(citations)
```

Correct empty result:

```json
{
  "count": 0,
  "passages": [],
  "citations": [],
  "message": "No matching documentation was found for this query."
}
```

## Additional defect found by the fail-closed canary

The demo tokenizer kept trailing separator punctuation, so a document containing
`MCP_CANARY_7F3A91C2.` did not match the query `MCP_CANARY_7F3A91C2`. Boundary
punctuation is now stripped while internal separators remain available for module
names and identifiers. The exact-canary self-test caught this before packaging.

## Commands and results

Self-contained suite, first run:

```text
pytest -q scikitplot/mcp/tests
81 passed, 2 skipped
```

Immediate second run against the same tree:

```text
pytest -q scikitplot/mcp/tests
81 passed, 2 skipped
```

The skips are intentional environmental gates:

1. real SDK in-memory tests skip only when `mcp>=2,<3` is unavailable;
2. external Streamable HTTP tests skip unless `SCIKITPLOT_MCP_RUN_LIVE=1`.

Compilation and shell validation:

```text
python -m compileall -q scikitplot/mcp
PASS

bash -n scikitplot/mcp/tests/test_mcp_search_docs.sh
PASS

bash -n scikitplot/mcp/tests/integration/test_mcp_load.sh
PASS
```

Backend self-test repeated twice:

```text
python -m scikitplot.mcp --self-test --self-test-query transport
PASS
second output byte-identical to first
```

Docker-profile resolution:

```text
python -m scikitplot.mcp --docker --print-effective-config
PASS
transport=streamable-http
host=0.0.0.0
port=8000
path=/mcp
health_path=/healthz
```

## User-environment protocol evidence

The running Docker service already passed a real SDK v2 Streamable HTTP call:

```text
tools/list -> search_docs
tools/call -> structured result with passages, citations, and security marker
```

After installing this patch, rerun the live acceptance wrapper. It specifically
verifies the corrected empty-result contract, idempotent repeated calls, tool
annotations, malformed HTTP containment, repeated sessions, concurrency, and
the real indexed canary.

## Deployment invariant

Do not run the production Docker service from a Meson editable install. Python
can invoke the generated editable loader and Ninja while resolving the parent
`scikitplot` package, before `scikitplot.mcp.__main__` executes. Build a wheel in
a builder stage and install it into a clean runtime stage.
