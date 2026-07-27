<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# Run 8 — ANNOY-STD-001 (guide 6.9) — declared C++ minimum did not match reality

**Priority:** P1  **Area:** `cexternals/_annoy/src/annoylib.h`  **Gate tier:** ring + focused compile check

## Finding (grounded in guide 6.9)
`annoylib.h` declared `#error "Annoy requires at least C++11 or newer"` and kept a
C++11 shared-mutex fallback path, but the code does not compile under strict
C++11. The guide found C++14 as the real floor (multi-statement `constexpr`); it
is in fact **C++17** now — the build sets `cpp_std=c++17` and the code uses
`if constexpr` (including the Run 3 `annoy_int_cmp.h` addition). The declared
minimum, the fallback branch, and several "enforces C++11" comments all
contradicted the snapshot.

## Root cause
The version guard and fallback comments drifted from the actual language features
used and the build's configured standard.

## Fix (set and enforce the true minimum; remove contradictory fallbacks)
- Replaced the C++11 `#error` with an **early** C++17 guard (placed before the
  C++17-only `<shared_mutex>` include), so a lower-standard build fails with an
  accurate message instead of an obscure later error:
  `#if __cplusplus < 201703L → #error "... requires at least C++17 (-std=c++17)"`.
- Removed the dead C++11 shared-mutex `#else` fallback (C++17 always provides
  `<shared_mutex>`); `#include <shared_mutex>` is now unconditional under the
  enforced minimum.
- Updated the stale "Compile-time ABI assertions (C++11)" / "enforces C++11 via
  #error" comments to reflect the C++17 minimum; removed the redundant late
  `#error`. Diff: `out/annoylib.h.run8.diff`.

## Verification
- **Focused compile check** (the guide's method):
  - `-std=c++11` → rejected with "requires at least C++17" ✓
  - `-std=c++14` → rejected with the same accurate message ✓
  - `-std=c++17` → the header syntax-checks standalone ✓
- **Real build (c++17) unchanged:** both `annoylib` extensions rebuilt clean
  (no warnings). The change is c++17-transparent — the removed `#else` branch was
  never compiled under c++17, and the guard passes identically.
- **Ring green:** kiss 111, memmap 49, annoy/_annoy cython 27, portable-blob 5,
  fd-sentinel 1, euclidean 12.

## Note
Scattered `#if __cplusplus >= 201103L` guards elsewhere are now always-true and
harmless (conservative documentation); they carry no contradictory `#else`
fallback, so they were left in place per minimal-impact. Dead `annoylib_v0.h` /
`annoylib_review.h` retain older guards but compile nowhere (CY-001, pruning pass).
