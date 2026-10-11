.. _mcp-capabilities-and-runtime:

Capabilities and runtime
========================

Use ``server_capabilities`` for the vocabulary the server understands and
``effective_server_capabilities`` for the subset that is actually active under
a particular configuration. ``server_runtime_status`` provides operational
status without requiring a caller to infer readiness from imports.

Capability vocabulary is cross-checked against Corpus where the domains
intersect. Do not invent a second spelling for the same capability in MCP merely
for UI convenience; adapters should translate at their boundary when needed.
