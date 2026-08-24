# scikit-plots MCP plugins

Ready-made bundles that register the **read-only** scikit-plots documentation MCP server (`python -m scikitplot.mcp`, or the equivalent `scikitplot mcp`) with common AI tools. Each folder holds that client's config format:

| Folder | Client | Config file |
|---|---|---|
| `.claude-plugin/` | Claude Code | `.mcp.json` + `plugin.json` + `marketplace.json` |
| `.cursor-plugin/` | Cursor | `mcp.json` |
| `.windsurf-plugin/` | Windsurf | `mcp_config.json` |
| `.continue-plugin/` | Continue | `config.json` |
| `.cline-plugin/` | Cline (VS Code) | `settings.json` |
| `.vscode-plugin/` | VS Code / Copilot | `mcp.json` |
| `.codex-plugin/` | Codex | `mcp.json` |
| `.openclaw-plugin/` | OpenClaw | `mcp.json` |

`agents/`, `hooks/`, `skills/` hold reusable, read-only building blocks that use the `search_docs` tool. All bundles launch the same server; the tool is read-only and every passage carries a validated source citation. See the module `README.md` for per-client setup details.
