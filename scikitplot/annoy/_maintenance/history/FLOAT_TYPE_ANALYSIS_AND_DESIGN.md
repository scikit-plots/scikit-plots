<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# Float-type infrastructure: deep analysis + future-oriented design

Scope: a full audit of how `annoylib.h` (+ `typenames.h`) represents floating
point, the root cause of the `float128`/extended-dtype **SIGILL**, and a design
for robust, portable support across the whole range **bool → 8 → 16 → 32 → 64 →
80/96 → 128 → 256 → 512**. Grounded entirely in the current source.

---

## Part 1 — What exists today (deep search)

### 1.1 The type ladder
| Type | Definition in `annoylib.h` | Width | Tier |
|---|---|---|---|
| `float16_t` | ARM `__fp16`; x86 `#if __F16C__` F16C struct; else scalar-bit struct | 2 B | native / **accelerated** / portable |
| `float32_t` | `typedef float` | 4 B | native IEEE binary32 |
| `float64_t` | `typedef double` | 8 B | native IEEE binary64 |
| `float128_t` | GCC/Clang `__float128`; MSVC/other `long double` | 16 B (or 10/12) | **native quad** / emulated / fallback |

Integers (`int8..int64`, `uint8..uint64`) are documented exhaustively in
`typenames.h`, but `typeName<T>()` there stops at `double` — the float16/128
names live only implicitly in the Cython dtype maps.

### 1.2 The double-precision "widened bridge"
All public I/O (`add_item`/`get_item`/queries) flows through the `_w` methods,
which are `double*`. So regardless of `dtype`, values are narrowed/widened to
`double` at the boundary (CY-012). `float16` loses precision on input;
`float128` gains **no input precision** — its only benefit is higher-precision
*internal* distance arithmetic via `annoy_sqrt<float128_t>` / `annoy_fabs<T>`
(specialised around lines 1300–1332, used in the Distance hot paths).

### 1.3 Capability macros (defined, unused)
`ANNOY_HAS_F16C_FLOAT16`, `ANNOY_HAS_FLOAT128`, `ANNOY_HAS_FLOAT128_EMULATED`,
`ANNOY_HAS_FLOAT128_FALLBACK`, `ANNOY_FLOAT16_DEFINED`, `ANNOY_FLOAT128_DEFINED`
are set as a side effect of type selection but never surfaced. They are exactly
the raw material for a support-tier report (Part 3).

---

## Part 2 — Root cause of the SIGILL (portability)

Two facts combine into an illegal-instruction landmine:

1. **Compile-time ISA guards.** The float16 acceleration is chosen with
   `#if defined(__F16C__)`. Whether F16C intrinsics (`_mm_cvtps_ph` /
   `_mm_cvtph_ps`) are emitted is decided **when the translation unit is
   compiled**, not by the CPU that later runs it.
2. **`cpu-dispatch = max`** (meson.options) with `cpu-baseline = min`. numpy-style
   dispatch compiles many ISA variants up to the highest the *compiler* supports
   (here through `avx512vbmi/ifma`). Inside those high-ISA variants `__F16C__`
   (and AVX512) are defined, so the accelerated code is baked in.

Result: a binary can contain F16C/AVX512 code selected by runtime dispatch, and
on a CPU — or a **virtualized host** — that reports but does not fully honor those
sub-features, executing them raises `SIGILL`. This is why the crash is
build-config/host specific: it reproduces in a fresh `cpu-dispatch=max` build on a
host with imperfect AVX512, and NOT in a build whose dispatch resolved lower.

**The portable scalar float16 fallback already exists and is bit-for-bit correct**
— validated by `tests/test_float16_portable_fallback.cpp` (scalar path == F16C
path on every round-trip). The problem is purely that the safe path is chosen at
compile time only.

---

## Part 3 — Future-oriented design (bool → 512)

### 3.1 Principle: a runtime *support-tier* registry, not compile landmines
Every element type declares a tier that is **queryable at runtime**:

- `native`      — a first-class hardware/IEEE type (float32, float64, ARM fp16).
- `accelerated` — portable semantics with an optional hardware fast path chosen
  **at runtime** after a CPU-feature probe (float16 via F16C *iff* detected).
- `emulated`    — correct but slower/soft (float128 via long double or software
  quad; 256/512 via multiprecision).
- `unavailable` — not built on this platform; constructing it raises a clear
  error rather than mis-dispatching.

`tests/test_float_capability_report.cpp` is the seed: it already reports the tier
+ real mantissa width per build. Promote it to a small `annoy_type_support()`
C++ API and expose it in Cython as `Index.supported_dtypes()` / a capability dict,
so callers (and wheels/CI) can see "float16=accelerated(F16C)" vs
"float16=portable" and "float128=native-quad" vs "float128=long-double".

### 3.2 Kill the compile-time ISA landmine
Two complementary fixes:

1. **Runtime dispatch for float16.** Compile BOTH the scalar and the F16C
   converters unconditionally; pick between them once, guarded by a runtime CPU
   probe (`__builtin_cpu_supports("f16c")` on GCC/Clang, `__cpuidex` on MSVC).
   The scalar path is proven equivalent, so this changes speed only, never
   results — and never faults on a non-F16C host.
2. **Constrain dispatch for this extension.** Until (1) lands, build the annoy
   extension with a conservative `cpu-dispatch` (e.g. drop `avx512vbmi/ifma`, or
   pin a tested set) so no untrusted sub-ISA is emitted. `cpu-baseline=min` should
   stay; the risk is entirely in the high dispatch variants.

### 3.3 The width ladder, made explicit and extensible
| Width | Type | How | Status |
|---|---|---|---|
| 1-bit | `bool` | bit-packed vectors (Hamming already packs bits) | reuse Hamming packing |
| 8-bit | `int8`/`uint8` | quantized embeddings; scale/zero-point metadata | design: add a quantization tier |
| 16-bit | `float16` | scalar (portable) + runtime F16C; ARM `__fp16` | **make runtime-dispatched** |
| 32/64-bit | `float`/`double` | native IEEE | done |
| 80/96-bit | `long double` | x87 extended; explicit `float80` tier | **DONE — usable dtype** |
| 128-bit | `float128` | `__float128`/libquadmath (true quad) | native where present; report emulation |
| 256/512-bit | `float256`/`float512` | **no native C++ type** — provide via `boost::multiprecision` (`cpp_bin_float`) or MPFR behind the `emulated` tier, compiled only when the dep is available | design: optional-dep tier |

Key point for 256/512: there is no hardware or standard-library scalar of that
width, so future support must be an **optional multiprecision backend** selected
through the same tier registry — never a silent `long double` alias (which is
what today's `float128` fallback does, and is a correctness trap).

### 3.4 Honesty rules the ladder must keep (already partly enforced)
- The double `_w` bridge caps I/O precision — any type wider than double gains
  only internal-arithmetic precision (documented, CY-012). A future "wide I/O"
  path would need a widened bridge (ABI/format change; ties to §6.6).
- `float128` that is really `long double` must self-report as `emulated`
  (Part 3.1), never claim 128-bit precision — the capability probe already
  distinguishes them by `sizeof`.
- Every accepted dtype must round-trip through save/load/pickle for its tier, or
  be rejected at construction (parity with the R20 advisory-dtype validation).

---

## Part 4 — What is validated now vs. proposed

Validated in this pass (safe, no header change, standalone C++):
- `test_float16_portable_fallback.cpp` — the portable scalar float16 path is
  bit-for-bit equal to F16C hardware (the SIGILL-free escape hatch is correct).
- `test_float_capability_report.cpp` — per-build support-tier + true-width report
  (the runtime-introspection seed).

Proposed (design decisions, need sign-off before code):
- [DONE — Run 25] Runtime F16C dispatch: float16 now compiles both paths and
  selects via cached __builtin_cpu_supports("f16c"). SIGILL resolved (full _annoy
  suite 434 passed in the previously-crashing build). See RUN25 doc.
- Conservative `cpu-dispatch` for the annoy extension — immediate mitigation.
- [DONE] `annoy_type_support.h` registry + `supported_dtypes()` Cython surface
  (report tier/size/precision; usable set cross-checked against Index acceptance).
- [INFRA DONE] float80 native tier + float256/512 gated behind
  ANNOY_ENABLE_MULTIPRECISION+boost (honest 'unavailable' when absent); wiring them
  as USABLE dtypes (the ~160-specialization change) remains.
- Widened I/O bridge (only if >double public precision is a real requirement;
  large ABI/format change).
