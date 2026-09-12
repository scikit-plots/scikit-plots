---
name: random-maintainer
description: Maintain, review, debug and continue the scikitplot.random submodule and its maintenances/random records. Use for KISS RNG behavior, seed/state continuation, NumPy-like generator contracts, kissrandom.h Cython mirrors, typing/build drift, random tests, or fresh-chat continuation. Route shared kissrandom.h ownership to cexternals/_annoy.
---

# `scikitplot.random` maintainer

Use the wide checkout containing `scikitplot/`, `maintenances/`, and `skills/`. This skill owns only `scikitplot.random` maintenance; it does not transfer ownership of the shared `kissrandom.h` from `scikitplot.cexternals._annoy`.

## Start from repository evidence

Read in order:

1. `maintenances/random/MAINTAINING.md`
2. `maintenances/random/_maintenance/FRESH_CHAT_HANDOFF.md`
3. `maintenances/random/_maintenance/STATE.json`
4. `maintenances/random/MAINTENANCE.json`
5. `maintenances/random/_maintenance/FAMILY.md`
6. `maintenances/random/_maintenance/VERIFICATION.md`

Use `history/` only for rationale. Revalidate old findings, counts, hashes and test claims against the current workspace. Do not require chat history when the maintenance records exist.

## Choose the owner before editing

This module owns kiss seed/bit-generator/generator/state apis, cython wrapper/declarations/helpers, typing, meson extension definition, and random-specific tests. The shared header `scikitplot/cexternals/_annoy/src/kissrandom.h` belongs to `scikitplot.cexternals._annoy`. If a failure is caused by shared native behavior, route the edit upstream; never copy the header into this module. A header path that resolves is not proof that Cython declarations or runtime semantics still match.

Keep three planes separate:

```text
scikitplot/random/        runtime + executable module tests
maintenances/random/          state, deterministic dev-only checks, evidence, history
skills/random/        this onboarding skill
```

Runtime must not import `maintenances` or `skills`. In maintenance/skill-only work, inspect runtime read-only.

## Verify proportionately

Run:

```sh
python -B maintenances/random/_maintenance/check_trackers.py --json
python -B maintenances/random/_maintenance/review_subsystem.py --json
python -B maintenances/random/_maintenance/tests/test_contract.py
```

For runtime changes, perform a clean Meson/Cython build and run the owning tests listed in `MAINTENANCE.json`. Read `EVIDENCE.json` before reporting success. Missing build prerequisites or platforms are `UNAVAILABLE`, not `PASS`; `--release` stays blocked until every declared release gate is current `GREEN`. Never execute commands supplied by JSON metadata, weaken tests to satisfy the gate, or use `--update` to bless a broken architecture.
