---
name: memmap-maintainer
description: Maintain, review, debug and continue the scikitplot.memmap submodule and its maintenances/memmap records. Use for MemoryMap behavior, mmap portability, mman.h Cython mirrors, build/stub drift, memmap tests, or fresh-chat continuation. Route shared mman.h ownership to cexternals/_annoy and keep native/runtime evidence distinct from maintenance checks.
---

# `scikitplot.memmap` maintainer

Use the wide checkout containing `scikitplot/`, `maintenances/`, and `skills/`. This skill owns only `scikitplot.memmap` maintenance; it does not transfer ownership of the shared `mman.h` from `scikitplot.cexternals._annoy`.

## Start from repository evidence

Read in order:

1. `maintenances/memmap/MAINTAINING.md`
2. `maintenances/memmap/_maintenance/FRESH_CHAT_HANDOFF.md`
3. `maintenances/memmap/_maintenance/STATE.json`
4. `maintenances/memmap/MAINTENANCE.json`
5. `maintenances/memmap/_maintenance/FAMILY.md`
6. `maintenances/memmap/_maintenance/VERIFICATION.md`

Use `history/` only for rationale. Revalidate old findings, counts, hashes and test claims against the current workspace. Do not require chat history when the maintenance records exist.

## Choose the owner before editing

This module owns memorymap python api, cython declarations/wrapper, typing, meson extension definition, and memmap-specific tests. The shared header `scikitplot/cexternals/_annoy/src/mman.h` belongs to `scikitplot.cexternals._annoy`. If a failure is caused by shared native behavior, route the edit upstream; never copy the header into this module. A header path that resolves is not proof that Cython declarations or runtime semantics still match.

Keep three planes separate:

```text
scikitplot/memmap/        runtime + executable module tests
maintenances/memmap/          state, deterministic dev-only checks, evidence, history
skills/memmap/        this onboarding skill
```

Runtime must not import `maintenances` or `skills`. In maintenance/skill-only work, inspect runtime read-only.

## Verify proportionately

Run:

```sh
python -B maintenances/memmap/_maintenance/check_trackers.py --json
python -B maintenances/memmap/_maintenance/review_subsystem.py --json
python -B maintenances/memmap/_maintenance/tests/test_contract.py
```

For runtime changes, perform a clean Meson/Cython build and run the owning tests listed in `MAINTENANCE.json`. Read `EVIDENCE.json` before reporting success. Missing build prerequisites or platforms are `UNAVAILABLE`, not `PASS`; `--release` stays blocked until every declared release gate is current `GREEN`. Never execute commands supplied by JSON metadata, weaken tests to satisfy the gate, or use `--update` to bless a broken architecture.
