<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# Run 25 — float16 runtime F16C dispatch (fixes the float128/extended SIGILL)

**Priority:** P1 (crash)  **Area:** `cexternals/_annoy/src/annoylib.h` (`float16_t`)  **Gate tier:** ring + C++

## Finding (crash / portability)
`float16_t` chose its implementation at COMPILE time (`#if defined(__F16C__)`).
Combined with `cpu-dispatch=max`, F16C (and AVX512) intrinsics were baked into
dispatch variants and raised **SIGILL (Illegal instruction)** on hosts that report
but don't fully honor those features. Verified in a fresh build of the uploaded
tree: `test_float128_precision_contract.py` and
`test_extended_dtype_index_combinations.py` crashed, and the whole `_annoy` suite
died with "Illegal instruction" (both exercise float16 conversion paths).

## Fix (runtime dispatch; AVX512 dispatch untouched, as requested)
On x86 GCC/Clang, `float16_t` now compiles BOTH converters unconditionally — the
portable scalar path AND the F16C path via `__attribute__((target("f16c")))` (so
it builds with no global `-mf16c`) — and selects per conversion using a cached
`__builtin_cpu_supports("f16c")` probe. The binary now runs on ANY x86 CPU and
only executes F16C when the running CPU actually supports it. ARM stays native
`__fp16`; MSVC/non-x86 stay scalar. New macro `ANNOY_HAS_RUNTIME_DISPATCH_FLOAT16`.
`cpu-dispatch=max`/AVX512 was left unchanged per instruction; removing the F16C
landmine alone resolved the crash.

## Verification
- **SIGILL resolved in the uploaded build tree (vrepo):** after applying the same
  fix, `test_float128_precision_contract` 4 passed, `test_extended_dtype_index_combinations`
  115 passed, and the full `_annoy` suite went from "Illegal instruction" → **434
  passed**.
- **Portability:** the header compiles cleanly with NO `-mf16c`, with `-mf16c`,
  and with `-march=native` (all 0 failures).
- **Correctness:** `tests/test_float16_portable_fallback.cpp` proves the scalar
  path is bit-for-bit equal to F16C on every round-trip, so runtime selection
  changes speed only, never results. On an F16C host the HW path is taken and is
  correct.
- **Capability:** `tests/test_float_capability_report.cpp` now reports
  `float16 tier = runtime-dispatched (scalar + F16C, chosen per-CPU); portable`.
- Ring green in the R23 repo: float128 4, extended-dtype 115, doc-parity 15,
  metric-aliases 18, euclidean 12, kiss 123; float16 dtype builds/queries via the
  high-level Index.

## Note
This is design-fix #1 from FLOAT_TYPE_ANALYSIS_AND_DESIGN.md. The remaining
proposals (conservative cpu-dispatch — explicitly deferred by request; the
`supported_dtypes()` API; multiprecision 96/256/512 tier) stay open.
