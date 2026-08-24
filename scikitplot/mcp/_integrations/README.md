# scikit-plots MCP integrations

Optional, **read-only** adapters that expose scikit-plots documentation retrieval
to agent frameworks. All adapters:

- expose only `search_docs` (no writes, no graph logic);
- import their framework lazily and fail with an actionable message if it is absent;
- do **not** require the MCP SDK *or* `pydantic` — the in-process toolkit is backed
  by `SearchCoordinator`, the Tier-L orchestration, so it also runs on the Legacy
  Retrieval tier (Python 3.8+).

| Folder | What it provides |
|---|---|
| `agno/` | `ScikitplotDocsToolkit` (framework-neutral, testable) + `build_agno_toolkit()` |
| `openclaw/` | `OpenClawMcpConfig` — emits the gateway MCP-config entry (zero-code wiring) |

Retrieved passages are untrusted reference content; cite `source_uri` values.
