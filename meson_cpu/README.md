# meson_cpu

A standalone, architecture-aware CPU feature detection and runtime-dispatch layer extracted from the NumPy Meson build system.

> **Origin:** adapted from [NumPy's `meson_cpu`](https://github.com/mesonbuild/meson/issues/12572) infrastructure.
> Upstream reference: [https://github.com/numpy/numpy/tree/main/meson_cpu](https://github.com/numpy/numpy/tree/main/meson_cpu)

### highway

- https://github.com/google/highway
- https://google.github.io/highway/en/master/
- https://chromium.googlesource.com/external/github.com/google/highway/+/refs/heads/upstream/master/

See Also:
---------
- https://en.wikipedia.org/wiki/Single_instruction,_multiple_data
- https://en.cppreference.com/cpp/experimental/simd
- [simplifycpp.org](https://simplifycpp.org):
    - [mini_booklet_SIMD_in_Modern_CPP.pdf](https://simplifycpp.org/books/minibooklet/mini_booklet_SIMD_in_Modern_CPP.pdf)
    - https://simplifycpp.org/?id=cpu
- [github.com/ermig1979/Simd](https://github.com/ermig1979/Simd)
    - https://github.com/ermig1979/Simd
    - https://ermig1979.github.io/Simd/
    - https://ermig1979.github.io/Simd/help/index.html
    - https://simd.sourceforge.net/index.html
- https://github.com/jfalcou/eve

---

## Overview

`meson_cpu` is a Meson-based build subsystem that manages **CPU baseline selection**, **feature probing**, and **runtime dispatch configuration** across multiple architectures.

Its purpose is to help performance-sensitive native projects compile optimized code paths while preserving portability.

Instead of hardcoding compiler flags or manually maintaining architecture matrices, `meson_cpu` centralizes feature definitions and exposes a deterministic interface for:

* selecting a **minimum supported CPU baseline**
* enabling **optional runtime-dispatched instruction sets**
* generating **compile-time configuration headers**
* validating compiler support for requested optimizations
* normalizing behavior across architectures and compilers

---

## Why This Exists

Modern CPUs expose increasingly complex SIMD and ISA extensions. Naively enabling all optimizations can break compatibility, produce invalid binaries, or create inconsistent behavior across toolchains.

`meson_cpu` separates:

* **baseline features** → always available in produced binaries
* **dispatch features** → optional accelerated paths selected at runtime

This enables safe portability with maximum performance.

```python
# TODO: Numpy like cpu optimization for annoy vector DB?
# Source distribution (raw source code archive)
# Think of -march as a strict requirement and -mtune as a strong suggestion.
# - march=cpu-type (Machine Architecture): Dictates the minimum hardware requirement.
# It allows the compiler to use special instruction sets (like SSE4, AVX, AVX2) specific to that CPU.
# Code compiled with a specific -march will not run on processors that do not support those instructions.
# - mtune=cpu-type (Machine Tune): Optimizes the ordering and scheduling of instructions to run as fast as possible on the specified CPU,
# but it does not use instructions that would break compatibility.
# The code will still run everywhere, it just might be slightly less efficient on CPUs other than the tuned target.
# For most extensions, you should rely on the default settings of setuptools, scikit-build, or maturin (for Rust).
# They default to safe baselines. If you are passing CFLAGS (and CXXFLAGS for C++) manually, use:
#     CFLAGS="-O3 -march=x86-64 -mtune=generic"
#     CFLAGS="-O3 -march=x86-64-v2 -mtune=generic"  # For safer, broader compatibility (2009+)
#     CFLAGS="-O3 -march=x86-64-v3 -mtune=generic"  # For maximum performance on 95% of modern hardware (2013+)
#     extra_compile_args = -march=x86-64-v4 -mtune=generic
# (Note: If you want to drop support for ancient pre-2009 CPUs, -march=x86-64-v2 is becoming the new modern baseline).
# v1 (x86-64)	 Baseline (SSE2)       	 2003+	Extreme legacy support. Slowest for math. Original 64-bit CPUs (AMD Opteron, Intel Core 2)
# v2	         SSSE3,  SSE4.2, POPCNT	 2009+	Safe Baseline. Supports almost all active PCs/Servers. Intel Nehalem (2008), AMD Jaguar
# v3	         AVX, AVX2, BMI2, FMA	   2013+	High Performance. Required for fast vector math. Intel Haswell (2013), AMD Zen
# v4         	 AVX-512	               2017+  Intel Skylake-X (2017), AMD Zen 4
# -march=native	    0/10 (Crashes others)	    10/10	Local builds / Private servers
# -march=x86-64	   10/10 (Works on everything)	3/10	Basic CLI tools, non-math libs
# -march=x86-64-v3	8/10 (2013+ CPUs)	        9/10	Vector DBs, AI, Data Science
```

---

## What It Provides

* CPU feature modeling across architectures
* baseline / dispatch parsing
* compiler validation using test programs
* configuration header generation
* cross-compiler normalization

---

## Supported Architectures

| Architecture  | Representative Features                  |
| ------------- | ---------------------------------------- |
| x86 / x86_64  | `X86_V2`, `X86_V3`, `X86_V4`, `AVX512_*` |
| ARM / AArch64 | `NEON`, `ASIMD`, `SVE`                   |
| PPC64         | `VSX*`                                   |
| s390x         | `VX`, `VXE*`                             |
| RISC-V 64     | `RVV`                                    |
| LoongArch64   | `LSX`                                    |

---

## Project Structure

```text
meson_cpu/
├── meson.build
├── main_config.h.in
├── x86/
├── arm/
├── ppc64/
├── s390x/
├── riscv64/
└── loongarch64/
```

---

## Example Usage

```meson
subdir('meson_cpu')
message('Baseline: ' + ' '.join(CPU_BASELINE_NAMES))
message('Dispatch: ' + ' '.join(CPU_DISPATCH_NAMES))
```

```bash
meson setup builddir -Dcpu-baseline=min -Dcpu-dispatch=max
```

---

## Design Principles

* deterministic
* portable
* fail-fast
* reusable
* future-proof

---

## References

* NumPy Meson CPU subsystem
* Meson Build System documentation
* compiler ISA flag references

---

## License / Attribution

Derived from concepts and implementation patterns used in NumPy's build infrastructure. Preserve upstream attribution when redistributing.
