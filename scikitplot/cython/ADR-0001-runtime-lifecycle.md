<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# ADR 0001 — Native-extension runtime lifecycle contract (CYTHON-ABI-001)

Status: Accepted · Date: 2026-07-24

## Context

`scikitplot.cython` compiles and imports native extensions at runtime. Native
extensions have hard lifecycle constraints that are **not** universally
guaranteed and were previously undocumented and unenforced (finding
CYTHON-ABI-001, an OPEN QUESTION in the review):

- **Unload / replacement.** CPython cannot generally unload a native extension.
  Re-importing a changed artifact under the same module name does not reliably
  replace the already-loaded C code.
- **Free-threaded CPython (`Py_GIL_DISABLED`).** Arbitrary compiled extensions
  are not guaranteed thread-safe without the GIL.
- **Subinterpreters.** Extensions without per-interpreter GIL support
  (`Py_mod_multiple_interpreters`) must not be imported into a non-main
  subinterpreter.
- **Fork after load.** POSIX `fork` after loading an extension is only safe when
  the child does not depend on threads/locks created before the fork.
- **Finalization.** Native finalizers run at interpreter shutdown in an order
  the library does not control.

## Decision

1. **Declare the contract explicitly.** `runtime_capabilities()` returns a
   `RuntimeCapabilities` record: `gil_enabled`, `free_threaded_build`,
   `in_main_interpreter`, `supports_unload` (always `False`),
   `supports_fork_after_load`, and `platform`.

2. **Safe defaults that error, not surprise.** `check_runtime_supported()`
   raises `UnsupportedRuntimeError` by default on a free-threaded interpreter or
   in a non-main subinterpreter. Callers who have verified their extensions may
   opt in with `allow_free_threaded=True` / `allow_subinterpreter=True`.

3. **No false unload promise.** The library does not claim to unload or hot-swap
   native modules. `supports_unload` is `False`; callers needing a new artifact
   should use a fresh module name (the cache key already varies by content and
   toolchain, so distinct artifacts get distinct names).

4. **Cache correctness across ABIs.** The cache fingerprint already includes the
   free-threaded flag (`gil_disabled`) and the resolved compiler/ABI
   (CYTHON-CACHE-003 / CYTHON-PORT-001), so an artifact built under one ABI is
   never reused under an incompatible one.

## Consequences

- Callers get a deterministic, documented failure on unsupported runtimes
  instead of an opaque crash or silent state loss.
- The free-threaded / subinterpreter / fork / finalization behavior is now a
  declared, testable matrix (`tests/test__runtime_capabilities.py`) rather than
  an open question.
- Full end-to-end subinterpreter/free-threaded/fork execution in subprocesses
  remains a CI concern (a special interpreter build is needed to exercise the
  free-threaded and subinterpreter branches for real); the in-process contract
  and its guards are unit-tested here by simulating each configuration.

## Alternatives considered

- **Attempt real unload/replacement.** Rejected: CPython provides no safe
  general mechanism; pretending to would be worse than a clear error.
- **Silently allow free-threaded/subinterpreter.** Rejected: risks data races
  and crashes; the safe default is to error and let verified callers opt in.
