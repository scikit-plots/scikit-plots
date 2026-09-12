# Fresh-chat handoff — MCP

MCP owns the wire, adapter validation, transport wiring, capability reporting,
and MCP-specific client integration. Corpus owns retrieval semantics; Annoy owns
vector/native mechanics. Do not collapse those owners into MCP.

Start with:

```sh
python -B maintenances/mcp/_maintenance/check_trackers.py --json
python -B maintenances/mcp/_maintenance/review_subsystem.py --json
```

The supplied 2026-09-12 snapshot has a healthy static runtime architecture but
an obsolete inherited maintenance plane. The new gate resolves the wide
repository explicitly instead of treating `maintenances/mcp` as runtime.

Release is separate from structural health. A release claim needs current
official-SDK in-memory, stdio, Streamable HTTP, Corpus+Annoy, and packaging-extra
evidence. Missing optional dependencies or a sliced archive are `UNAVAILABLE`,
not PASS.

Historical M00–M14 checkpoints explain earlier decisions only. Never copy their
status line forward without re-running the present gate/evidence lanes.
