# Strict MCP wire validation and repeatable live acceptance

## Defect

The service layer already contained:

```python
if isinstance(k, bool) or not isinstance(k, int):
    raise ValueError("k must be an integer")
```

That check was insufficient at the MCP boundary. With a non-strict Pydantic
`int` annotation, JSON `true` was converted to Python integer `1` before the tool
function called `SearchService.search()`. The downstream service therefore saw a
plain `int` and accepted the request.

## Correct boundary

The registered MCP tool now uses `StrictStr` and `StrictInt`:

```python
query: Annotated[StrictStr, Field(...)]
k: Annotated[StrictInt, Field(ge=1, le=MAX_RESULTS)] = 5
```

The advertised JSON Schema remains `type: string` and `type: integer`, while the
runtime validator no longer coerces adjacent JSON/Python types.

## Why both validation layers remain

1. **MCP/Pydantic boundary** rejects malformed wire values before invocation.
2. **SearchService boundary** protects direct Python callers and alternate
   transports that bypass MCP validation.

Removing either layer recreates a bypass path.

## Stale-process prevention

Live tests operate against an already-running process. Source changes do not
change that process. The acceptance wrapper now reads the local `_version.py`
without importing the parent package, compares it with `/healthz`, and stops with
an explicit rebuild/restart message on mismatch.

Override only when deliberately testing a remote release:

```bash
SCIKITPLOT_MCP_EXPECTED_VERSION=0.2.2 \
SCIKITPLOT_MCP_RUN_LIVE=1 \
pytest -vv scikitplot/mcp/tests/integration/test_mcp_http_live.py
```

## Repeatable sequence

```bash
# 1. Source-only tests
pytest -q scikitplot/mcp/tests

# 2. Rebuild/restart the server after any source change
docker compose build mcp
docker compose up -d --force-recreate mcp

# 3. Version-aware live acceptance
bash scikitplot/mcp/tests/test_mcp_search_docs.sh

# 4. Repeat the same acceptance run
bash scikitplot/mcp/tests/test_mcp_search_docs.sh
```

Both live runs should produce the same successful contract results for an
immutable backend generation.
