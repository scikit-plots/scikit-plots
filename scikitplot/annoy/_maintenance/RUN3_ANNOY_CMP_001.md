<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# Run 3 — ANNOY-CMP-001 (guide 6.13) — range-safe integer narrowing

**Priority:** P1  **Area:** `cexternals/_annoy/src/annoylib.h`  **Gate tier:** ring

## Finding (grounded in guide 6.13)
`safe_numeric_cast<T,S>` (annoylib.h:1210) bounds-checked by casting the TARGET
bound into the SOURCE type:
```cpp
if (value > static_cast<S>(std::numeric_limits<T>::max())) ...
else if (value < static_cast<S>(std::numeric_limits<T>::lowest())) ...
```
When T's bound is not representable in S (signed↔unsigned, or T wider than S) the
`static_cast<S>(...)` wraps, so the comparison is against a garbage bound. E.g.
`static_cast<int32_t>(INT64_MAX)` → −1, so widening `int32→int64` wrongly clamped
valid positive values; `static_cast<uint32_t>(INT32_MIN)` wraps to a large
positive, corrupting the lower-bound test. Signed/unsigned comparisons were also
implementation-defined (`-Wsign-compare`).

## Root cause
The bounds are transformed into the source domain before comparison. The correct
form compares in a common domain without casting either bound into the other
operand's type.

## Fix (minimal, canonical)
- Added `src/annoy_int_cmp.h`: sign-safe `annoy_cmp_less`/`annoy_cmp_greater`
  (a C++17 backport of C++20 `std::cmp_*`) and `annoy_clamp_cast_int<T,S>`, which
  clamps using those comparisons — **no bound cast into S** — and handles `bool`
  explicitly. Included from `annoylib.h`.
- Rewrote the integer branch of `safe_numeric_cast` to `if constexpr (... is_integer ...)
  return annoy_clamp_cast_int<T,S>(value);` (`if constexpr` so it is not
  instantiated for floating T/S). Float→int and float→float paths unchanged.
- Diff: `out/annoylib.h.run3.diff` (+7/−7). Dead duplicate `annoylib_review.h`
  carries the same bug but is included nowhere (CY-001) — left for the pruning
  pass, noted here so the fix is not "hidden" by the copy.

## Regression test (permanent)
`tests/test_annoy_int_cmp.cpp` — exercises the **real** header across the full
signed/unsigned × width matrix (8×8 integer type pairs) against an `__int128`
oracle that represents every value exactly, plus the specific old-bug cases and
`bool`. All pass:
```
[PASS] full T x S integer clamp matrix matches __int128 oracle
[PASS] widening i32->i64 keeps INT32_MAX (old code clamped)
[PASS] u32->i32 clamps to INT32_MAX
[PASS] i32->u8 negative -> 0
[PASS] cmp_less(-1,0u) / cmp_greater(UINT64_MAX,5) / cmp_less(5u,-1)
[PASS] int->bool / bool->int
0 failures
```
Build: `g++ -std=c++17 -Isrc tests/test_annoy_int_cmp.cpp -o t && ./t`

## Always-green gate (ring)
`annoylib.h` feeds both the C++ module and the Cython `Index`, so both
`annoylib` extensions were rebuilt and restaged. Ring: kiss 105, memmap 49,
annoy/_annoy 333, annoy `types` 8, `euclidean` 12 — all green. No test codified
the old behavior.
