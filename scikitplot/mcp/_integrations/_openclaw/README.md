# OpenClaw integration

Wire the **read-only** scikit-plots documentation MCP server into an OpenClaw-style
gateway with no adapter code:

```python
from scikitplot.mcp.integrations.openclaw import OpenClawMcpConfig

print(OpenClawMcpConfig().to_json())
# paste into your gateway's MCP config, then restart the gateway
```

The `search_docs` tool becomes available to the host. It is read-only and every
passage carries a validated source citation. The console form
`scikitplot mcp` is equivalent to `python -m scikitplot.mcp`.
