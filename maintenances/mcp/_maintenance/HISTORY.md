# History

## 2026-09-12 — maintenance-plane reset

The inherited MCP maintenance campaign contained useful M00–M14 findings but
its active entry points had drifted away from the repository layout: the checker
resolved `maintenances/mcp` as though it were `scikitplot/mcp`, read-order files
were absent, and docs still instructed maintainers to run a runtime-local
`_maintenance` path that does not exist in this snapshot.

The old live surface was preserved under
`history/legacy_live_2026-09-12/`. The active plane now follows the same
runtime/maintenance/skill separation used by the newer submodule campaigns.
