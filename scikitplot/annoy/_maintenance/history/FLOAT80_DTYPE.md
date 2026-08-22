<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# float80 (long double) wired as a usable dtype

Extends the ladder: `float16 → float32 → float64 → float80 → float128`. float80
is x87 80-bit extended precision (`long double`, 64-bit mantissa) — a distinct,
usable tier between float64 (53-bit) and float128 (113-bit) on GCC/Clang.

## Change (data-driven)
The dtype dispatch generates from a `data_types` list, so float80 is added as one
entry (aliases `float80`/`longdouble`/`extended`/`fp80`/…) which auto-wires the
`DataTypeId` enum, `parse_data_dtype`, and the construction dispatch. Both
`annoylib.pxd.in` and `annoylib.pyx.in` `data_types` lists were updated (they must
match). C++ (`annoylib.h`): `typedef long double float80_t`; added `long double` to
`is_valid_data_type` and the HammingWrapper whitelist; `TypeName<float80_t>` guarded
by `ANNOY_HAS_FLOAT128` to avoid the MSVC collision (there `float128_t` IS
`long double`). `supported_dtypes()` reports float80 `usable_as_dtype=true`
(the dispatch always accepts it; float128-distinctness only changes the note).

## Verification (uploaded tree, vrepo2)
Builds clean; float80 works across all four metrics with save/load round-trip;
`supported_dtypes()` usable set = float16/32/64/80/128; ring green (extended-dtype
115, metric-aliases 18, float128 4); `test_supported_dtypes.py` 17 passed;
`test_type_support.cpp` 0 failures.

## Notes
* On MSVC/generic (`float128_t` == `long double`), float80 is redundant with
  float128 but the generated typedef stays valid (same underlying type); it self-
  reports as such in the `note`.
* I/O is still capped at double via the `_w` bridge; float80's benefit is
  higher-precision internal distance arithmetic (64-bit mantissa) than float64.
* float256/512 remain gated-unavailable (no boost/MPFR backend) — honest ceiling.
