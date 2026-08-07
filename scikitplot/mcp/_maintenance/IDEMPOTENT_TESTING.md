# Idempotent MCP testing and acceptance

## Definition used by this submodule

For the read-only `search_docs` operation, idempotence means that the same
immutable backend generation, query, result limit, and configuration produce the
same structured response without mutating server or corpus state. It does not
mean that results must remain identical after the corpus or index generation is
changed.

The stable output invariant is:

```text
count == len(passages) == len(citations)
```

Additional invariants:

- citation numbers are contiguous and one-based;
- non-empty citation `doc_id` values are unique;
- a zero-result response has `passages=[]`, `citations=[]`, and a non-empty
  `message`;
- a non-empty response has `message=null`;
- every passage is explicitly labelled as untrusted reference data;
- scores are finite JSON numbers;
- no request writes files, changes the index, or opens a second index generation.
- JSON Boolean, numeric-string, and floating-point values are not accepted for
  the integer `k` field; validation is strict before Python invocation.
- the live server version must match the checked-out package version unless an
  explicit `SCIKITPLOT_MCP_EXPECTED_VERSION` override is supplied.

## Test layers

### Layer 1 — pure unit and contract tests

```bash
pytest -q scikitplot/mcp/tests \
  --ignore=scikitplot/mcp/tests/integration/test_mcp_http_live.py
```

This covers the core formatter, BM25 demo backend, hybrid fusion, adapters, CLI
resolution, service bounds, SDK registration double, and malformed values.

### Layer 2 — real MCP SDK without networking

With `mcp>=2,<3` installed, these tests connect a real client directly to the
server object. They exercise initialization, schema discovery, tool calls, and
resource reads without a port or external process:

```bash
pytest -vv scikitplot/mcp/tests/test_protocol_in_memory.py
```

### Layer 3 — backend self-test

```bash
python -m scikitplot.mcp \
  --docs-jsonl /data/scikitplot-docs.jsonl \
  --self-test \
  --self-test-query MCP_CANARY_7F3A91C2 \
  --self-test-require-match \
  --self-test-expected-doc-id scikitplot-canary-001
```

This fails before serving when the corpus cannot load, search raises, or the
structured contract is inconsistent. Repeat twice and compare output when the
backend is expected to be deterministic.

### Layer 4 — live Streamable HTTP acceptance

Start the Docker service, then run:

```bash
SCIKITPLOT_MCP_RUN_LIVE=1 \
SCIKITPLOT_MCP_CANARY_TOKEN=MCP_CANARY_7F3A91C2 \
SCIKITPLOT_MCP_CANARY_DOC_ID=scikitplot-canary-001 \
pytest -vv scikitplot/mcp/tests/integration/test_mcp_http_live.py
```

Or use the wrapper:

```bash
bash scikitplot/mcp/tests/test_mcp_search_docs.sh
```

The wrapper reads `_version.py` without importing `scikitplot`, checks
`/healthz`, and fails before protocol tests when a stale container or wrong image
is serving the port. This avoids confusing failures caused by testing new source
files against an old long-running process.

Strict wire-type cases covered by both in-memory and live protocol tests:

```text
k=true    -> rejected
k=false   -> rejected
k="2"     -> rejected
k=2.0     -> rejected
query=true -> rejected
```

The live suite is opt-in by design. Ordinary `pytest scikitplot/mcp` must not
depend on a stale process, fixed port, network namespace, or previous test run.

### Layer 5 — bounded load and resource observation

```bash
SCIKITPLOT_MCP_LOAD_REQUESTS=500 \
SCIKITPLOT_MCP_LOAD_CONCURRENCY=25 \
SCIKITPLOT_MCP_LOAD_TIMEOUT=180 \
bash scikitplot/mcp/tests/integration/test_mcp_load.sh
```

In parallel, observe CPU, memory, file descriptors, and restarts:

```bash
docker stats scikitplot-mcp
docker exec scikitplot-mcp sh -c 'ls /proc/1/fd | wc -l'
```

Run the same load profile again. A healthy read-only server should return the
same baseline response and settle near its previous idle resource range.

## Canary generations

A production index should publish an immutable generation identifier and a
canary record. Recommended deployment sequence:

1. build a new index in a temporary generation directory;
2. validate its manifest, schema, embedding model/revision, dimension, metric,
   document count, checksum, and canary;
3. run `--self-test` against that generation;
4. run in-memory and live MCP acceptance tests;
5. atomically switch the active generation pointer;
6. retain the previous generation for rollback;
7. never mutate the active generation in place.

This makes retries safe and prevents partially rebuilt indexes from becoming
visible to readers.

## CI recommendation

Use separate jobs so failures identify their layer:

```text
unit-contract       no MCP dependency, no network
protocol-in-memory  MCP SDK installed, no external process
container-smoke     wheel-built image, health + canary
http-acceptance     real Streamable HTTP client/server
load-bounded        scheduled or release-gated
```

Store test reports and image/index generation identifiers as artifacts. Do not
make external live tests part of a default developer run.

## Reproducible source manifest

After changing package files, atomically refresh the package-relative SHA-256
manifest:

```bash
python scikitplot/mcp/_maintenance/update_artifact_manifest.py --write
```

Verify without changing timestamps or bytes:

```bash
python scikitplot/mcp/_maintenance/update_artifact_manifest.py --check
```

The default unit suite also verifies exact manifest coverage and hashes. A
second `--write` on an unchanged tree is a no-op.
