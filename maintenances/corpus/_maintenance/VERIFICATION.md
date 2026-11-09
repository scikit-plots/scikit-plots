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

## Free-text search lane (CX-01, CX-05)

```sh
python -B maintenances/corpus/_maintenance/evidence/probe_fts5_text.py
python -B maintenances/corpus/_maintenance/evidence/probe_fts5_sqlite.py
python -m pytest scikitplot/corpus/_storage/tests/test__storage.py -k TestFreeTextSearch
```

`probe_fts5_text.py` sends 13 queries (NUL, lone surrogates, other control
characters, a BOM, a noncharacter) through `search_text`,
`StorageQuery.full_text` and the MCP lexical leg in default and strict mode,
each on a fresh store; it exits 0 only when none raises and no leg fails (28
failures on the tree before PR 919). `probe_fts5_sqlite.py` needs only the
standard library, so it also runs where corpus does not import (Python 3.8):
it checks the quoting helpers against the interpreter's own SQLite (60
failures before PR 919). The test class pins the replaced characters to
exactly NUL and U+D800 to U+DFFF and holds every query to the result of the
same query with a space in their place. Run both probes on every supported
interpreter when the quoting or SQLite changes.

`--release` remains blocked until every release gate in `REVIEW.json` is current `GREEN` in `EVIDENCE.json`.
