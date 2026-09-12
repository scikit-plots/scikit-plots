# Maintaining `scikitplot.random`

This directory owns maintenance state for `scikitplot/random`. The runtime code stays under `scikitplot/`; project-local onboarding stays under `skills/random/`. A fresh session should not need chat history.

**Ownership:** KISS seed/bit-generator/generator/state APIs, Cython wrapper/declarations/helpers, typing, Meson extension definition, and random-specific tests. It consumes `kissrandom.h` from `scikitplot/cexternals/_annoy/src/`; that shared header is owned by the separate `cexternals/_annoy` subsystem. Do not patch or vendor the header here to make this consumer pass.

Read `_maintenance/FRESH_CHAT_HANDOFF.md`, then `_maintenance/STATE.json`, `_maintenance/FAMILY.md`, and `_maintenance/VERIFICATION.md`. Machine-readable contracts are `MAINTENANCE.json` and `REVIEW.json`; the skill entry point is `/skills/random/SKILL.md`.

From the repository root:

```sh
python -B maintenances/random/_maintenance/check_trackers.py
python -B maintenances/random/_maintenance/review_subsystem.py --json
python -B maintenances/random/_maintenance/tests/test_contract.py
```

The maintenance gate is structural evidence only. A runtime or release claim additionally requires a clean native build, the module test suite, and appropriate platform evidence. `UNAVAILABLE` is never promoted to `PASS`. After an intentional runtime structural change, review the diff and use `check_trackers.py --update`; refresh refuses to bless broken dependency/build/plane contracts.
