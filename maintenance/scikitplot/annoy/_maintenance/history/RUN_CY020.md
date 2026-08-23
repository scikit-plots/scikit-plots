<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# CY-020 — compile-warning volume: surface + enforce the actionable ones

**Priority:** P2  **Area:** `cexternals/_annoy/src/annoylib.h` + guard  **Gate tier:** C++ strict-compile

## Finding
The generated `annoylib.pyx.cpp` is compiled with ~35 `-Wno-*` suppressions to
silence unavoidable Cython noise. Because the generated code and our hand-written
`annoylib.h` share ONE translation unit, those suppressions also hide *actionable*
warnings in our own C++. A strict compile of the hand-written headers alone
(`-Wall -Wextra -Wconversion -Wshadow`) surfaced 49 warnings, of which the
actionable ones were: **1 `-Wunused-result`** (later 2 sites) ignoring `ftruncate`'s
return, 2 `-Wshadow`, and 10 `-Wconversion`.

## Fix
- **Actionable (fixed):** two `ftruncate` best-effort rollback calls were
  `(void)`-cast, but GCC's `warn_unused_result` ignores the `(void)` cast — so a
  failed rollback truncate was silently dropped. Changed to
  `if (ftruncate(...) != 0) { /* non-fatal */ }`, which consumes the result and
  keeps the best-effort intent. Headers are now `-Wunused-result`-clean.
- **Intentional (kept, documented):** the 2 `-Wshadow` are by design — each build
  thread gets its own `Random _random` shadowing the shared member (thread-local
  RNG seed); renaming would fight the upstream design.
- **Benign (documented budget):** 36 `-Wunused-parameter` (API-symmetry stubs) and
  10 `-Wconversion` (bounded int->float / size->int, value-safe).

## Guard (measurable exit criterion)
`tests/test_warning_budget.cpp` compiles the hand-written headers ALONE and is
intended to be built in CI with `-Werror=unused-result -Werror=return-type
-Werror=uninitialized`. It instantiates the templates across float/double/long
double so the generic bodies are actually diagnosed. This keeps the actionable
categories enforced independent of the generated build's blanket suppressions; the
benign budget is documented in the file.

## Verification
Strict-actionable compile passes (0). cexternals + annoy rebuild clean;
`on_disk_build` (exercises the `ftruncate` resize path) works (50 items);
ring green (save_load 5, serialize 3, memmap 49).
