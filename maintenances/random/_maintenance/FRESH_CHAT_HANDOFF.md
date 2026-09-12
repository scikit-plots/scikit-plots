# Fresh-chat handoff — `scikitplot.random`

## Authority and read order

1. User-supplied current workspace and failure logs.
2. `STATE.json` and current runtime files.
3. `MAINTENANCE.json`, `REVIEW.json`, `TRACKER.json`, and `EVIDENCE.json`.
4. Current tests and build files.
5. Historical notes under `history/`.

Do not inherit old archive hashes, counts, findings or campaign sequencing as current facts. Revalidate them.

## Planes

```text
scikitplot/random/                 runtime source + tests
maintenances/random/                  maintenance state + dev-only tools
skills/random/                  fresh-chat skill entry
```

Runtime code must not import `maintenances` or `skills`. Maintenance code inspects runtime read-only unless the user explicitly authorizes a runtime edit.

## Owner boundary

This subsystem owns kiss seed/bit-generator/generator/state apis, cython wrapper/declarations/helpers, typing, meson extension definition, and random-specific tests. The shared `kissrandom.h` is upstream at `scikitplot/cexternals/_annoy/src/kissrandom.h` and belongs to `scikitplot.cexternals._annoy`. If the defect is in that header, route the edit upstream and verify every declared consumer rather than forking a local copy.

## First commands

```sh
python -B maintenances/random/_maintenance/check_trackers.py --json
python -B maintenances/random/_maintenance/review_subsystem.py --json
python -B maintenances/random/_maintenance/tests/test_contract.py
```

Read `EVIDENCE.json` before saying a build or runtime test passed. `--release` is intentionally blocked while required native/platform gates are `UNAVAILABLE`.
