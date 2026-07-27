<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# Run 4 — ANNOY-RNG-001 (guide 6.15) — canonical [0, 1) float conversion

**Priority:** P1  **Area:** `random/_kiss/kiss_random.pyx`  **Gate tier:** submodule (+ cross-ring sanity)

## Finding (grounded in guide 6.15)
`random()` is documented as returning floats in `[0, 1)`, but every path divided
a 64-bit draw by `2**64 - 1` (`UINT64_MAX`). `(double)UINT64_MAX` rounds up to
`2**64`, so a maximal draw yields exactly `1.0` — breaking the half-open contract
and risking out-of-range use in `int(r * n)` indexing.

## Root cause
Non-power-of-two divisor plus rounding: the scale `UINT64_MAX`/`2**64` is not
exactly representable, so the top of the range maps to `1.0`.

## Fix (canonical, the guide's prescription)
Use the top mantissa bits and an exact power-of-two scale (numpy's method):
- Added constants `KISS_F64_SHIFT = 11` (64−53) and `KISS_F64_SCALE = 2**-53`.
- `kiss64_next_double` (NumPy Generator C callback): `(val >> 11) * 2**-53`.
- `random()` scalar path: `(raw >> 11) * 2**-53`.
- `random()` array paths: `float64` → `(raw >> 11) * 2**-53`; `float32` →
  `(raw >> 40) * 2**-24` (24 mantissa bits); other dtypes computed in float64
  then cast. Max possible value is `(2**53 - 1) * 2**-53 < 1.0` — the boundary is
  unreachable by construction, not merely improbable.
- Diff: `out/kiss_random.pyx.run4.diff`.

## Reproducibility scope
Only the user-facing random **float** stream changes (sanctioned as a defect fix
with reference vectors). The **integer** stream used for annoy tree splitting is
untouched, so index structure / saved-index reproducibility is unaffected. No
existing test encoded the old float formula (all 105 prior kiss tests still pass).

## Regression test (permanent)
`tests/test_kiss_random_unit_interval.py` (6 tests): scalar `[0,1)` + on the
`2**-53` grid over 200k draws (`KissGenerator` and `KissRandomState`); float64 and
float32 array `[0,1)`; the NumPy `Generator.random()` C-callback path over 500k
draws; and explicit reference vectors incl. maximal draw → `1 - 2**-53`.

## Always-green gate
`kiss_random` rebuilt: full suite **111 passed** (105 + 6 new). Cross-ring
sanity (unaffected): memmap 49, annoy/_annoy cython 27.

## Deferred — §6.14 (KISS modulo bias), with rationale
`index(n)` uses `random_value % n` (biased). Fixing it (rejection sampling)
changes the **integer** stream, which drives tree splitting and shuffles —
altering index structure and breaking saved-index / seed reproducibility. Per the
guide's own review decision this needs a policy call ("is unbiased bounded
sampling a public guarantee?") and a preserved legacy-stream mode. That is a
design decision, not a hardening patch, so it is staged with a documented
invariant (rule L-PERF / defer-risky) rather than rushed here.
