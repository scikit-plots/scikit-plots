<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# Run 1 — ANNOY-MMAN-001 (guide 6.1) — Windows `ftruncate` failure detection

**Priority:** P0  **Area:** `cexternals/_annoy/src/mman.h`  **Gate tier:** ring (A+B)

## Finding (grounded in guide 6.1)
`mman.h:277–303` (Windows branch): the resize path tested
`SetFilePointerEx(...) == ~0`. `SetFilePointerEx` returns a Win32 `BOOL` whose
**failure value is 0**, not `~0`, so a genuine failure never matched and the
error path was skipped — a silent, potentially corrupting failure. It also wrote
to `stderr`, bypassing structured error propagation.

## Root cause
Wrong sentinel for a `BOOL` API (`~0` instead of `0`), compounded by a missing
resolved-handle validation and a direct `fprintf(stderr, …)` side effect.

## Fix (minimal, root-cause; externed signature preserved)
- Extracted the logic into `src/mman_ftruncate_win.h` (`annoy_win_set_file_size`
  + `annoy_win_ftruncate`) so control flow and error mapping are unit-testable
  with mocked Win32 on any host (the guide's "one platform-neutral truncate
  adapter" decision).
- Correct `BOOL` failure checks (`== 0`), validate `_get_osfhandle` against
  `INVALID_HANDLE_VALUE` (→ `EBADF`), deterministic `GetLastError` → errno
  mapping, and **no** `stderr` write.
- `mman.h` keeps the exact externed shim `inline int ftruncate(int fd, int64_t
  size)` (used by `mem_map.pyx:204`), now delegating to the adapter.
- Diff: `out/mman.h.run1.diff` (net −18 lines in mman.h + new 62-line header).

## Regression test (permanent)
`tests/test_mman_ftruncate.cpp` — mocks the minimal Win32 surface with fault
injection and compiles the **real** adapter header. Nine checks, all pass:
```
[PASS] success resize -> 0
[PASS] zero size -> 0
[PASS] large size -> 0
[PASS] first seek fail -> -1/EIO
[PASS] second seek fail -> -1/EIO
[PASS] SetEndOfFile fail -> -1/EIO
[PASS] fd<0 -> -1/EBADF
[PASS] INVALID_HANDLE -> -1/EBADF
[PASS] ERROR_INVALID_HANDLE -> EBADF
0 failures
```
Build: `g++ -std=c++17 -Wall -Iscikitplot/cexternals/_annoy/src \
scikitplot/cexternals/_annoy/tests/test_mman_ftruncate.cpp -o t && ./t`

## Always-green gate (ring)
Rebuilt the `mman.h`-including targets (`mem_map`, cexternals `annoylib`) and reran:
- `memmap/_memmap/tests`: **49 passed**
- `random/_kiss/tests`: **105 passed** (unaffected sanity)
The edit is inside the `#ifdef _WIN32` branch, so the Linux `.so` is unchanged;
the rebuild confirms no accidental breakage. No test codified the old behavior,
so none needed rewriting.

## Deferred (honesty)
Real Windows execution of this path → Windows CI (the host test proves the
control flow and mapping; it cannot exercise the actual Win32 kernel calls).
