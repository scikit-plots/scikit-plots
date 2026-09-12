# Verification

## Maintenance-tooling gate

```sh
python -B maintenances/corpus/_maintenance/tests/test_contract.py
python -B maintenances/corpus/_maintenance/check_trackers.py --json
```

The regression suite proves the checker fails closed on missing runtime contracts, stale inventory, forbidden runtime-plane imports, empty skills, unsafe review metadata, and foreign working directories.

## Runtime gate

A complete runtime must satisfy every `runtime_contract.required_contracts` entry in `MAINTENANCE.json`, contain at least the declared source/test floor, and avoid runtime imports of maintenance/skill planes. These are restoration tripwires, not proof of semantic correctness.

After source restoration, run the canonical Corpus test suite from the repository configuration. Do not reuse the historical `3206 passed / 27 skipped / 4 xfailed` count as current evidence.

## Examples/gallery gate

Re-run Corpus examples/gallery in the supported documentation execution modes, including offline/path handling where applicable. Historical checkpoint prose is not execution evidence.

## Cross-module gate

Verify MCP, Annoy adapter, and CLI consumer boundaries against the complete current tree before marking the cross-module gate GREEN.

`--release` remains blocked until every release gate in `REVIEW.json` is current `GREEN` in `EVIDENCE.json`.
