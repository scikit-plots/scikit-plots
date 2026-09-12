# Verification

Run the maintenance contract first:

```sh
python -B maintenances/annoy/_maintenance/check_trackers.py --json
python -B maintenances/annoy/_maintenance/review_subsystem.py --json
python -B maintenances/annoy/_maintenance/tests/test_contract.py
```

Interpret the three statuses independently:

- `maintenance_status`: whether the maintainer metadata/tools/handoff are sound.
- `runtime_status`: whether current static runtime/build contracts are intact.
- `release_status`: whether current compile/test/platform evidence is complete.

For the supplied snapshot, runtime is expected to fail on the missing Tempita
helper. Once that is repaired, regenerate from both templates in a clean build,
compile `scikitplot.annoy._annoy.annoylib`, build the independent
`scikitplot.cexternals._annoy` extension, and run focused tests for dtype,
persistence/error ownership, RNG/concurrency and the public high-level wrapper.

A static declaration-path check is not ABI proof. A green release requires fresh
compiled evidence.


The active repository build requires **C++17**. Template-local C++14 comments/directives are historical/stale and must not be treated as the current compiler contract.
