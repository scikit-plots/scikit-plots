<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# annoy deep-review campaign — summary

A run-by-run deep semantic review + hardening of the `annoy` subsystem
(C++ `annoylib.h`, Cython `annoylib.pyx.in`/`.pxd.in`, `annoymodule.cc`, memmap,
KISS RNG). Zero-hallucination, minimal root-cause fixes, every closed finding
backed by a permanent regression guard. Full per-finding evidence in
`_maintenance/RUN_*.md`; the closed-findings table is in `MAINTAINING.md`.

## Closed (each with a regression guard)
- **CY-001** duplicate Cython template `annoylib_pyx.in` removed (fix-trap).
- **CY-002** unconsumed + stale `annoylib.pxi` removed.
- **CY-003** installed generated `.pxd` declared a PRIVATE (no-ABI-stability) surface.
- **CY-009** concurrency Policy A: corrected every "thread-safe" overclaim
  (build/query x2/context-manager) + explicit class policy.
- **CY-020** compiler-warning volume: fixed `ftruncate` `warn_unused_result` sites,
  added a strict warning-budget guard for hand-written headers.
- **Native-error ownership sweep**: save/load/on_disk_build/unbuild/serialize/
  deserialize migrated to `ScopedError` RAII.
- **`__contains__`/`__getitem__`** negative-index consistency.
- **Float-type system**: `supported_dtypes()` runtime API, **float80** usable dtype,
  multiprecision 256/512 infra (gated).
- (plus CY-004/005/006/007/008/010/011/012/013/017/018/019 and the §6.x defects
  from prior sessions.)

## Designed, deferred (with rollout plans)
- **CY-015** annoymodule.cc modernization — `_maintenance/CY015_ANNOYMODULE_DESIGN.md`.
- **(A)** multiprecision backend, **(B)** widened I/O bridge —
  `_maintenance/DEFERRED_FUTURE_WORK.md`.

## Gate
Full ring green end-to-end: **256** Python tests (81 session guards + 175 core) +
C++ guards (type-support, warning-budget actionable-clean, float16 fallback),
0 failures.
