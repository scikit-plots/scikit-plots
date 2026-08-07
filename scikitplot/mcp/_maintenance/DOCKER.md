# Docker execution and health checks for `scikitplot.mcp`

## Correct command

Prefer the explicit Docker profile:

```bash
python -m scikitplot.mcp --docker
```

It resolves to:

```bash
python -m scikitplot.mcp \
  --transport streamable-http \
  --host 0.0.0.0 \
  --port 8000 \
  --path /mcp \
  --health-path /healthz
```

Do not insert blank lines between shell continuation lines, and do not put any
space after a trailing backslash. The following is invalid because `\ ` does not
escape the newline:

```bash
--host 0.0.0.0 \
```

A single-line command avoids this entire class of shell failure:

```bash
python -m scikitplot.mcp --docker --port 8000 --path /mcp
```

## Endpoints

With the default Docker profile:

- MCP Streamable HTTP: `http://127.0.0.1:8000/mcp` from the Docker host.
- Health: `http://127.0.0.1:8000/healthz` from the Docker host.
- Container-to-container MCP URL in Compose: `http://mcp:8000/mcp`.

`/healthz` intentionally returns only:

```json
{"status":"ok","service":"scikitplot-docs","version":"0.2.2"}
```

It discloses no paths, environment variables, backend details, indexes, or
secrets. The route is public and is only a process/HTTP readiness signal. A full
MCP functional probe should additionally connect with an MCP client and call
`search_docs`.

## Built-in health probe

Inside a non-editable runtime image:

```bash
python -m scikitplot.mcp --docker --probe
```

Expected output:

```text
HEALTHY http://127.0.0.1:8000/healthz
```

The probe:

- converts wildcard bind addresses (`0.0.0.0`, `::`) to loopback addresses;
- disables environment HTTP proxies;
- enforces a timeout;
- requires HTTP 200 and JSON `status == "ok"`;
- exits zero only when healthy.

Configure the probe with environment variables or CLI arguments:

```bash
SCIKITPLOT_MCP_PORT=8123 \
SCIKITPLOT_MCP_HEALTH_PATH=/live \
python -m scikitplot.mcp --docker --probe --probe-timeout 3
```

## Backend self-test before serving

Validate corpus loading, retrieval, and the structured-output invariants without
opening a socket or starting an MCP transport:

```bash
python -m scikitplot.mcp --self-test --self-test-query transport
```

For a real JSONL corpus:

```bash
python -m scikitplot.mcp \
  --docs-jsonl /data/scikitplot-docs.jsonl \
  --self-test \
  --self-test-query MCP_CANARY_7F3A91C2 \
  --self-test-require-match \
  --self-test-expected-doc-id scikitplot-canary-001
```

Run the same command twice and compare the emitted JSON. An immutable index and
deterministic ranker should return byte-identical output for the same inputs.
The self-test is read-only and never starts the server, so it is suitable as a
Docker build validation or an init-container readiness gate.

## Acceptance and bounded load tests

Normal unit tests are self-contained and do not require a running HTTP service:

```bash
pytest -q scikitplot/mcp/tests
```

Live HTTP tests are explicitly opt-in:

```bash
SCIKITPLOT_MCP_RUN_LIVE=1 \
pytest -vv scikitplot/mcp/tests/integration/test_mcp_http_live.py
```

The wrapper prechecks health, requires the running version to match the local
`_version.py`, and verifies the built-in canary. This catches a stale container
before protocol assertions produce misleading failures:

```bash
bash scikitplot/mcp/tests/test_mcp_search_docs.sh
```

Run a bounded, read-only concurrency test:

```bash
SCIKITPLOT_MCP_LOAD_REQUESTS=500 \
SCIKITPLOT_MCP_LOAD_CONCURRENCY=25 \
bash scikitplot/mcp/tests/integration/test_mcp_load.sh
```

See `_maintenance/IDEMPOTENT_TESTING.md` for the complete contract and CI matrix.

## Docker port publication

For local development, bind only to the Docker host loopback interface:

```bash
docker run --rm \
  --name scikitplot-mcp \
  -p 127.0.0.1:8000:8000 \
  scikitplot-mcp:latest
```

Do not publish an unauthenticated MCP server to all host interfaces unless an
authenticated reverse proxy, firewall, or trusted private network protects it.
`--docker` acknowledges the container-network bind; it does not implement
identity, authorization, rate limits, or TLS.

## Why the Ninja traceback occurred

`meson-python` editable installs add a generated finder such as:

```text
_scikit_plots_editable_loader.py
```

Python imports the parent `scikitplot` package before it can execute
`scikitplot.mcp.__main__`. The editable finder can invoke Ninja during that
parent-package resolution. Therefore a failure or hang here occurs before any
MCP CLI code can catch it:

```text
_scikit_plots_editable_loader.py -> self._rebuild() -> subprocess.run(ninja)
```

The warning:

```text
ninja: warning: premature end of file; recovering
```

usually indicates an interrupted or damaged Ninja metadata/log file. The later
`KeyboardInterrupt` means the rebuild was manually stopped. The subsequent
`bash: --transport: command not found` messages came from broken shell line
continuation, not from MCP.

## Development-container recovery

First inspect the generated loader without importing `scikitplot`:

```bash
LOADER=/root/micromamba/envs/py311/lib/python3.11/site-packages/_scikit_plots_editable_loader.py
grep -nE 'build_path|build_cmd|ninja|install\(' "$LOADER"
```

Stop only stale build processes owned by the current container, then inspect the
build directory printed by the loader:

```bash
ps -ef | grep -E '[n]inja|[m]eson'
```

In that build directory, remove only Ninja's regenerable state files and rebuild
verbosely:

```bash
rm -f /path/from/loader/.ninja_log /path/from/loader/.ninja_deps
ninja -C /path/from/loader -v
```

When the generated build tree itself is invalid, recreate the editable install
from the project root:

```bash
python -m pip uninstall -y scikit-plots
python -m pip install --no-build-isolation -e /work
```

Do not run a production service from this editable installation. Every fresh
Python interpreter—including a Docker health check—may trigger the rebuild hook.

## Production rule: install a wheel, not `-e`

Use a multi-stage image. Build scikit-plots and all runtime dependencies into
wheels in the builder stage; install those wheels into a clean runtime stage.
The included `docker/Dockerfile.mcp` is a reference template.

This separation provides:

- deterministic startup without Ninja-on-import;
- a smaller runtime image without compilers/build tools;
- repeatable health checks;
- simpler rollback by image digest;
- reduced writable build state and attack surface.

## Effective configuration

Validate Docker and environment resolution without starting the server:

```bash
python -m scikitplot.mcp --docker --print-effective-config
```

Supported environment variables:

```text
SCIKITPLOT_MCP_DOCKER
SCIKITPLOT_MCP_TRANSPORT
SCIKITPLOT_MCP_HOST
SCIKITPLOT_MCP_PORT
SCIKITPLOT_MCP_PATH
SCIKITPLOT_MCP_HEALTH_PATH
SCIKITPLOT_MCP_MAX_CONCURRENCY
SCIKITPLOT_MCP_MAX_REQUEST_BODY
SCIKITPLOT_MCP_PROBE_TIMEOUT
SCIKITPLOT_MCP_SELF_TEST
SCIKITPLOT_MCP_SELF_TEST_QUERY
SCIKITPLOT_MCP_SELF_TEST_REQUIRE_MATCH
SCIKITPLOT_MCP_SELF_TEST_EXPECTED_DOC_ID
SCIKITPLOT_MCP_ALLOW_UNAUTHENTICATED_REMOTE
SCIKITPLOT_MCP_LOG_LEVEL
```
