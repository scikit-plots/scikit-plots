# Maintaining `scikitplot.mcp`

This is the active entry point for the MCP submodule. Historical M00–M14 campaign
material is provenance under `_maintenance/history/`; it is not current authority.

## First commands

```sh
python -B maintenances/mcp/_maintenance/check_trackers.py --json
python -B maintenances/mcp/_maintenance/review_subsystem.py --json
python -B -m unittest discover -s maintenances/mcp/_maintenance/tests -p 'test_*.py' -v
```

The wide repository root is the directory containing `scikitplot/`,
`maintenances/`, and `skills/`. Never infer the runtime root from the location of
the maintenance script.

## Ownership rule

MCP owns the **wire and adapter semantics**. Corpus owns retrieval semantics.
Annoy owns vector/native mechanics. The external MCP SDK owns protocol
implementation. Keep those planes distinct.

Two invariants are especially important:

1. importing `scikitplot.mcp` must remain SDK/pydantic optional;
2. there is one MCP protocol implementation: the supported official SDK, never
   an in-package JSON-RPC fallback branded as MCP.

Current status is recorded in `_maintenance/STATE.json`; current evidence in
`_maintenance/EVIDENCE.json`.
