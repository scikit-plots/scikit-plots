# meson_cpu

A standalone, architecture-aware CPU feature detection and runtime-dispatch layer extracted from the NumPy Meson build system.

> **Origin:** adapted from NumPy's `meson_cpu` infrastructure.
> Upstream reference: [https://github.com/numpy/numpy/tree/main/meson_cpu](https://github.com/numpy/numpy/tree/main/meson_cpu)

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
