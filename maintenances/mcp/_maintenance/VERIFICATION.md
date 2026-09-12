# Verification

## Structural contract

```sh
python -B maintenances/mcp/_maintenance/check_trackers.py --json
python -B maintenances/mcp/_maintenance/review_subsystem.py --json
python -B -m unittest discover -s maintenances/mcp/_maintenance/tests -p 'test_*.py' -v
```

Structural PASS means the runtime files/symbols, optional-dependency boundary,
single-protocol rule, security markers, plugin/integration surface, maintenance
handoff, and inventory agree with the current snapshot. It does **not** prove a
live MCP transport.

## Runtime lanes

The useful release lanes are deliberately independent:

- **tier_l_tests** — SDK-free retrieval, outcomes, hybrid logic, import surface;
- **server_model_tests** — pydantic wire models/service behavior without
  claiming a real MCP protocol round trip;
- **sdk_in_memory** — official MCP SDK `Client` against the in-process server;
- **stdio_live** — actual stdio protocol process;
- **http_live** — actual Streamable HTTP process and health/tool/resource path;
- **corpus_annoy_integration** — real local Corpus+Annoy profile, no silent
  backend substitution;
- **packaging_extra** — `[mcp]` dependency metadata/resolver behavior from the
  full packaging checkout.

Never infer one lane from another. In particular, static `mcp` imports and
pydantic model tests cannot make `sdk_in_memory` green.

## Updating inventory

`--update` may refresh `TRACKER.json` only when the structural contract is clean.
Evidence must then be regenerated for the new runtime fingerprint. Do not edit a
fingerprint by hand to hide drift.
