# Corpus live registry

| ID | Status | Finding / action |
|---|---|---|
| CORPUS-001 | BLOCKING | Supplied `scikitplot/corpus/` directory is empty. Restore/provide the actual runtime before semantic review or release claims. |
| CORPUS-002 | RESOLVED-MAINT | Inherited entry points said review/implementation were COMPLETE despite the absent runtime. Live docs now classify those claims as historical-only. |
| CORPUS-003 | RESOLVED-MAINT | Legacy commands pointed at `scikitplot/corpus/_maintenance/...` although maintenance lives under `maintenances/corpus/`. Canonical commands now use the real maintenance plane. |
| CORPUS-004 | RESOLVED-MAINT | `skills/corpus/` had no maintainer entry point. `skills/corpus/SKILL.md` now defines authority, ownership, fail-closed continuation, and verification. |
| CORPUS-005 | OPEN-REVALIDATE | Historical tracker names the core runtime contracts listed in `MAINTENANCE.json`. Revalidate each path/symbol against the restored runtime; do not rename/fabricate source to satisfy this maintenance slice. |
| CORPUS-006 | OPEN-REVALIDATE | Historical final suite count (`3206/27/4`) is not current evidence. Replace it only with a fresh canonical run on the restored source. |
| CORPUS-007 | OPEN-INTEGRATION | MCP/Annoy/CLI boundary claims require their current consumer code and tests; keep them UNAVAILABLE until verified. |

The next exact action is CORPUS-001. Do not use `--update` while it remains unresolved.
