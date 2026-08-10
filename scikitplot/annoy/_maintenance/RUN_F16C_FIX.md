<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# float16 F16C: linker-agnostic detection + bit-exact scalar converters

Two independent fixes in `annoylib.h`, both verified here on x86.

## 1. Detection: CPUID instead of __builtin_cpu_supports (macOS-Intel link fix)
`has_f16c()` now reads CPUID leaf 1 / ECX[29] via `__get_cpuid` (added
`#include <cpuid.h>`). `__builtin_cpu_supports` pulls in compiler-rt's
`__cpu_features2`, which becomes an *illegal text relocation* under the classic
macOS linker (ld64) — the macOS-15 Intel link failure. CPUID references no external
symbol, so the float16 path links under ANY linker, on all OSes, and avoids LLVM's
Darwin CPU-feature-init fragility. Still runtime-dispatched (no SIGILL); handles
missing CPUID leaf 1 (→ scalar path); the cached static is thread-safe/idempotent.
Complements the project's `-ld_classic` removal (defense in depth).

Guard: `tests/check_no_cpu_feature_reloc.sh` fails if any compiler-rt CPU-feature
global is reintroduced into the float16 object.

## 2. Converters: bit-exact round-to-nearest-even + subnormals
The scalar converters previously **truncated** (f32->f16) and **flushed subnormals
to zero** (both directions), diverging from F16C hardware by up to 1 ULP — so an
index built on a non-F16C CPU did not match one built on an F16C CPU. Both are now
rewritten to IEEE-754 round-to-nearest-even with full subnormal + overflow + Inf +
NaN-payload handling.

**Verified bit-for-bit against the F16C hardware converters (not "within 1 ULP"):**
  * f32 -> f16: 0 mismatches over ALL 2^32 float inputs.
  * f16 -> f32: 0 mismatches over ALL 2^16 half inputs.
So the scalar and hardware paths now produce identical bits on every CPU — indexes
are reproducible regardless of F16C support. The prior "bit-for-bit equal" comment
is now actually true.

Guard: `tests/test_float16_scalar_matches_hardware.cpp` asserts scalar == hardware
exactly (fast comprehensive sweep by default; `ANNOY_F16_EXHAUSTIVE=1` sweeps all
2^32). On non-F16C hosts it no-ops (the existing fallback test covers those).

## Verification summary (x86, here)
compiles clean; object references no `__cpu_features2`; f32->f16 0/2^32 and
f16->f32 0/2^16 vs hardware; float16 index builds+queries; the functional
scalar-vs-hardware mismatch count went 49762 -> 0. Needs your macOS-Intel CI only
to confirm the full extension links (expected — the symbol is gone).
