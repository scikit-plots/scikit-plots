<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# Run 2 — ANNOY-CONV-001 (guide 6.11) — unsigned count/size Python conversion

**Priority:** P1  **Area:** `cexternals/_annoy/src/annoymodule.cc`  **Gate tier:** ring

## Finding (grounded in guide 6.11)
`annoy_build_summary_dict` converted `n_items64` and `n_trees64` — both declared
`uint64_t` and non-negative by construction (cast from `get_n_items()` /
`get_n_trees()`, which return the centralized `IndexDtype = uint64_t`) — through
the **signed** `PyLong_FromLongLong`. Any value above `LLONG_MAX` is
misrepresented (the sign bit is reinterpreted; `UINT64_MAX` becomes `-1`).

## Design intent (maintainer)
Non-negative counts/sizes keep their full **unsigned** range end-to-end. If a
value can never be negative, it is carried and returned as unsigned so the full
`uint64_t` domain round-trips.

## Root cause
Wrong signedness of the Python constructor for a non-negative wide value — a
signed constructor for an unsigned semantic type. The adjacent byte-size path
already used `PyLong_FromUnsignedLongLong`; the count paths were inconsistent.

## Fix (minimal, uses the centralized convention)
- Added `AnnoyCountToPy` — a checked count/size helper — in
  `src/annoy_pyconv.h`, mirroring the existing `AnnoyIdxToPy` (line 233) and
  tracking the same `IndexDtype` width variant (uint64 → `PyLong_FromUnsignedLongLong`;
  uint32 alternative documented inline). Included from `annoymodule.cc` right
  after the centralized dtype section, so all count/index conversions share one
  source of truth (the guide's "one checked conversion helper per semantic type").
- Repointed the two verified sites: `n_items64` and `n_trees64` now use
  `AnnoyCountToPy`. Diff: `out/annoymodule.cc.run2.diff` (+7/-2 lines).

## Audit (guide's "audit every item/count/size conversion")
- Correct already: item indices → `AnnoyIdxToPy`; `n_neighbors` →
  `PyLong_FromSize_t`; memory bytes / seed → `PyLong_FromUnsignedLongLong`.
- Left signed **by design** (not counts, or sentinel-bearing): `search_k`
  (accepts `-1` = auto), `f`/`n_jobs`/`verbose`/`schema_version`/`metric_id`
  (small bounded config scalars). Noted, not changed — minimal impact.
- The `PyLong_FromLong((long)n_trees)` param sites (build param, may carry a
  pre-build sentinel) are left as-is; only the **built** counts (`get_n_*`) were
  the verified unsigned defect.

## Regression test (permanent)
`tests/test_annoy_pyconv.cpp` — embeds CPython and exercises the **real**
`annoy_pyconv.h`. Seven checks, all pass:
```
[PASS] 0 / 1 / 4096 round-trip
[PASS] LLONG_MAX round-trips
[PASS] LLONG_MAX+1 round-trips (was truncated)
[PASS] UINT64_MAX round-trips (was truncated)
[PASS] old signed PyLong_FromLongLong(UINT64_MAX) yields -1 (the bug)
0 failures
```
Build: `g++ -std=c++17 $(python3-config --includes) -Isrc \
tests/test_annoy_pyconv.cpp $(python3-config --ldflags --embed) -o t && ./t`
(A real index cannot hold 2^63 items, so the boundary is only reachable via this
unit test — the authoritative check for the fidelity contract.)

## Always-green gate (ring)
Rebuilt `cexternals/_annoy/annoylib` (compiles `annoymodule.cc`), restaged, reran:
kiss 105, memmap 49, annoy `types` 8, `euclidean` 12, `_annoy` cython 27 — all
green. Public API smoke: a 10-item / 5-tree index reports `get_n_items()==10`,
`get_n_trees()==5`. No test codified the old behavior, so none needed rewriting.

## Follow-up (noted, not rushed)
Consider routing the build-param `n_trees`/`search_k` display conversions through
typed helpers once their sentinel semantics are pinned (separate from this
verified count defect).
