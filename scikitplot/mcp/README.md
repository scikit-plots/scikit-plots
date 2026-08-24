# scikit-plots MCP server

A single, standards-compliant [Model Context Protocol](https://modelcontextprotocol.io/)
server that exposes **read-only, source-cited documentation search** for
scikit-plots to any MCP-compatible tool (Claude Code, Cursor, Windsurf, Cline,
Continue, VS Code / GitHub Copilot, Amazon Q, and others).

It is built on the **official MCP Python SDK v2** and lives in-package as
`scikitplot.mcp`. There is exactly one wire-protocol implementation — no
hand-rolled fallback.

---

## What it exposes

**Tool (1, read-only):**

| Tool | Description |
|---|---|
| `search_docs(query, k=5)` | Search trusted documentation indexes; returns up to `k` bounded passages, each with a validated source citation. Returned text is untrusted reference content, never instructions. |

**Resource (1):**

| URI | Description |
|---|---|
| `docs://chunk/{doc_id}` | Read one bounded documentation chunk by stable identifier. |

**HTTP health (optional):** `GET /healthz` → `{status, service, version}` (no
environment, path, or secret detail).

Non-goals: this is **not** a write API, a graph database, an agent loop, or a model
provider. It retrieves documentation and cites sources — nothing more.

---

## Requirements

The **server** uses the official MCP SDK v2, which requires **Python ≥ 3.10**:

```bash
pip install "scikit-plots[mcp]"     # installs mcp>=2.0.0,<3 on Python >= 3.10
```

On Python 3.8/3.9 the **Legacy Retrieval tier** still works with a base install:
`DocsRetriever` and `RetrievedChunk`, the hybrid retrievers, the Corpus/Annoy
adapters (where their own deps permit), `build_search_docs_result`, and the
pydantic-free `server_runtime_status()` / `server_capabilities()`. The MCP
protocol server does **not** run there. Note: `SearchService` and the pydantic
result models belong to the **server tier** (installed via the `[mcp]` extra),
not the base install.
Check programmatically:

```python
from scikitplot.mcp import server_runtime_status
status = server_runtime_status()          # never imports the SDK
# {'retrieval_available': True, 'server_available': ..., 'reason': ...}
```

Enumerate the (read-only) surface programmatically — also SDK-free:

```python
from scikitplot.mcp import server_capabilities
caps = server_capabilities()   # {'effect_class': 'read_only', 'tools': [...], 'resources': [...]}
```

---

## Quick start

Two equivalent ways to run — the module form and the **centralized CLI** (they
forward the same arguments to the same server; use whichever you prefer):

```bash
# module form:
python -m scikitplot.mcp                       # stdio (default)

# centralized CLI (optional; the `scikitplot` console command):
scikitplot mcp                                 # identical to `python -m scikitplot.mcp`
```

Everything below shows `python -m scikitplot.mcp`; replace it with `scikitplot mcp`
at any time — every flag is forwarded unchanged (including `--help`):

```bash
# inspect the effective configuration without starting a server:
python -m scikitplot.mcp --print-effective-config
scikitplot mcp --print-effective-config

# read-only self-test (runs one query end to end):
python -m scikitplot.mcp --self-test

# liveness probe against a running server:
python -m scikitplot.mcp --probe

# local Corpus + Annoy retrieval (HashEmbedder is the offline default):
scikitplot mcp --corpus-annoy ./docs --self-test \
  --self-test-query "runtime corpus retrieval" --self-test-require-match

# serve that same local Corpus + Annoy backend over Streamable HTTP:
scikitplot mcp --docker --corpus-annoy ./docs

# list the read-only tool/resource inventory (no SDK import, no server start):
python -m scikitplot.mcp --list-capabilities
```

> `scikitplot mcp` is a **delegated pass-through** command: the CLI forwards all
> trailing arguments to this server's own entry point, so the two forms are
> interchangeable. (Full CLI behavior is documented with the `scikitplot._cli`
> submodule.)

## Transports

- **stdio** (default): newline-framed JSON-RPC on stdin/stdout; all logs go to
  **stderr**, so stdout stays a clean protocol channel.
- **Streamable HTTP** (`--docker`): binds an HTTP endpoint with a public
  `/healthz`. `--docker` is an explicit acknowledgement that the process is
  reachable on a network interface; it adds **no** authentication. Only publish the
  port to a trusted interface, or place your own auth in front.

```bash
python -m scikitplot.mcp --docker                 # streamable-http on 0.0.0.0
python -m scikitplot.mcp --transport streamable-http --host 127.0.0.1 --port 8000
# or, via the centralized CLI:
scikitplot mcp --docker
```

## Configuration

Flags and their environment-variable equivalents (flags win):

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `--transport` | `SCIKITPLOT_MCP_TRANSPORT` | `stdio` (`streamable-http` with `--docker`) | Transport |
| `--docker` | `SCIKITPLOT_MCP_DOCKER` | off | HTTP on `0.0.0.0` + public `/healthz` |
| `--corpus-annoy PATH` | `SCIKITPLOT_MCP_CORPUS_ANNOY` | off | Build a local Corpus retriever and force Annoy; mutually exclusive with `--docs-jsonl` |
| `--corpus-embedding-model` | `SCIKITPLOT_MCP_CORPUS_EMBEDDING_MODEL` | unset | Optional model-backed embeddings; omit to use deterministic `HashEmbedder` |
| `--hash-dimension` | `SCIKITPLOT_MCP_HASH_DIMENSION` | `256` | HashEmbedder dimension for the Corpus+Annoy profile |
| `--annoy-metric` | `SCIKITPLOT_MCP_ANNOY_METRIC` | `angular` | Annoy metric for the Corpus+Annoy profile |
| `--annoy-n-trees` | `SCIKITPLOT_MCP_ANNOY_N_TREES` | `10` | Annoy tree count for the Corpus+Annoy profile |
| `--host` | `SCIKITPLOT_MCP_HOST` | localhost | HTTP bind host |
| `--port` | `SCIKITPLOT_MCP_PORT` | `8000` | HTTP bind port |
| `--path` | `SCIKITPLOT_MCP_PATH` | `/mcp` | Streamable HTTP endpoint path |
| `--health-path` | `SCIKITPLOT_MCP_HEALTH_PATH` | `/healthz` | Health route |
| `--max-concurrency` | `SCIKITPLOT_MCP_MAX_CONCURRENCY` | `4` | Bounded parallel searches |
| `--max-request-body` | `SCIKITPLOT_MCP_MAX_REQUEST_BODY` | `1048576` | Max HTTP body bytes |

Non-loopback binds are refused unless `--docker` or
`--allow-unauthenticated-remote` is set explicitly.

### Corpus + Annoy profile

`--corpus-annoy PATH` is the local dense-retrieval profile. It uses public
`scikitplot.corpus` ingestion/chunking/index APIs and forces the Annoy vector
backend. With no `--corpus-embedding-model`, it uses the deterministic local
`HashEmbedder`, so CI does not need a model download. The same embedder instance
is used for corpus vectors and query vectors. Explicitly requesting Annoy is
fail-fast: a missing/broken native backend is not silently replaced by another
index.

The profile also implements the MCP `docs://chunk/{doc_id}` resource lookup, so
the same indexed document can be returned both from `search_docs` and by stable
resource id.

---

## Point your AI tool at the server

All configs launch the same server via `python -m scikitplot.mcp` (stdio). If the
`scikitplot` console command is installed, you can equally use
`command: "scikitplot", args: ["mcp"]` — both are interchangeable.

### Claude Code
```bash
claude mcp add scikitplot-docs python -m scikitplot.mcp
```
or `~/.claude/settings.json`:
```json
{ "mcpServers": { "scikitplot-docs": { "command": "python", "args": ["-m", "scikitplot.mcp"] } } }
```

### Cursor — `~/.cursor/mcp.json`
```json
{ "mcpServers": { "scikitplot-docs": { "command": "python", "args": ["-m", "scikitplot.mcp"] } } }
```

### Windsurf — `~/.codeium/windsurf/mcp_config.json`
```json
{ "mcpServers": { "scikitplot-docs": { "command": "python", "args": ["-m", "scikitplot.mcp"] } } }
```

### Cline — VS Code `settings.json`
```json
{ "cline.mcpServers": { "scikitplot-docs": { "command": "python", "args": ["-m", "scikitplot.mcp"] } } }
```

### Continue — `~/.continue/config.json`
```json
{ "mcpServers": [ { "name": "scikitplot-docs", "command": "python", "args": ["-m", "scikitplot.mcp"] } ] }
```

### VS Code (GitHub Copilot) — `.vscode/mcp.json`
```json
{ "servers": { "scikitplot-docs": { "type": "stdio", "command": "python", "args": ["-m", "scikitplot.mcp"] } } }
```

### Amazon Q Developer
```json
{ "mcpServers": { "scikitplot-docs": { "command": "python", "args": ["-m", "scikitplot.mcp"] } } }
```

> Ready-made plugin bundles for these tools live under `plugins/`; framework
> adapters live under `integrations/`.

---

## Testing

```bash
# offline suite (no live server):
SCIKITPLOT_MCP_RUN_LIVE=0 pytest scikitplot/mcp/ -vv

# with a live server round trip (needs the mcp extra on Python >= 3.10):
SCIKITPLOT_MCP_RUN_LIVE=1 pytest scikitplot/mcp/ -vv

# live HTTP integration with a canary document/token:
SCIKITPLOT_MCP_RUN_LIVE=1 \
  SCIKITPLOT_MCP_CANARY_TOKEN=... SCIKITPLOT_MCP_CANARY_DOC_ID=... \
  pytest -vv scikitplot/mcp/tests/integration/test_mcp_http_live.py

SCIKITPLOT_MCP_RUN_LIVE=1 \
  SCIKITPLOT_MCP_CANARY_TOKEN=MCP_CANARY_7F3A91C2 SCIKITPLOT_MCP_CANARY_DOC_ID=scikitplot-canary-001 \
  pytest -vv scikitplot/mcp/tests/integration/test_mcp_http_live.py
```

A container run for manual testing: `python -m scikitplot.mcp --docker`.

---

## Package layout

```
scikitplot/mcp/
├── README.md          # this file (user + client setup)
├── __main__.py        # `python -m scikitplot.mcp` CLI: transports, probes, config
├── _server.py         # official-SDK server: search_docs tool, resource, health
├── _core.py           # SDK-free contracts, citation safety (_safe_uri)
├── _hybrid.py         # deterministic Reciprocal Rank Fusion
├── _demo.py           # dependency-free in-memory retriever (default)
├── _corpus_annoy.py   # Corpus + Annoy retriever adapters
├── plugins/           # per-client plugin bundles
├── integrations/      # agent/framework adapters
└── _maintenance/      # design, policy, review artifacts (for maintainers)
```

Maintainer docs: `MAINTAINING.md` and `_maintenance/RULESET.md`.
