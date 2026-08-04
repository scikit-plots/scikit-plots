<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# annoy review campaign — burndown (reconciled)

## Status: CY register substantively COMPLETE

The Cython-side deep-review register (CY-001..020) plus the Part I §6.x defects
that were directly fixable in-sandbox are closed, each with a permanent regression
guard recorded in `../MAINTAINING.md`.

### Closed (with regression guards)
CY-004/005/006/007/008/010/011/012/013/017/018/019 (prior sessions); and this
campaign: **CY-001** (duplicate template removed), **CY-002** (unconsumed/stale
`.pxi` removed), **CY-003** (installed `.pxd` declared PRIVATE), **CY-009**
(concurrency policy A + docstring corrections), **CY-020** (warning-budget guard +
ftruncate `warn_unused_result` fixes). Plus the cross-cutting work: native-error
ownership sweep (save/load/on_disk_build/unbuild/serialize/deserialize →
ScopedError), `__contains__`/`__getitem__` negative-index consistency, and the
float-type system (`supported_dtypes()`, **float80** usable dtype, multiprecision
256/512 infra). CY-014 = no-op. CY-016 addressed via the tier/`supported_dtypes`
work.

### Designed, implementation deferred
- **CY-015** — annoymodule.cc modernization (single-source HTML asset pipeline,
  table-driven dispatch, C++ capability/version API, modularization, UX). Full
  design + rollout order in `CY015_ANNOYMODULE_DESIGN.md`; deferred item (C).
- **(A)** multiprecision backend (boost/MPFR) → float256/512 usable.
- **(B)** widened public I/O bridge (>float64 through add_item/get_item).
  All three in `DEFERRED_FUTURE_WORK.md`.

### Remaining (optional / CI-tier only)
- `backup_template/` — 6 quarantined dead template copies (user-owned; recommend
  removal, not fix-traps).
- Concurrency stress / ThreadSanitizer runs — need CI, not deterministic in the
  sandbox (the CY-009 policy + supported-case guard are in place).
- §6.x design-tier items (e.g. §6.6 format contract, §6.14 KISS modulo bias) —
  maintainer decisions, tracked design-side.

## Gate
Full ring green end-to-end: 256 Python tests (81 session guards + 175 core) + C++
guards (type-support, warning-budget actionable-clean, float16 fallback) = 0
failures.
