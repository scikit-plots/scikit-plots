<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# Run 6 — ANNOY-FD-001 (guide 6.8) — file descriptor 0 as "not open" sentinel

**Priority:** P1  **Area:** `cexternals/_annoy/src/annoylib.h`  **Gate tier:** ring

## Finding (grounded in guide 6.8)
The on-disk build descriptor `_fd` used `0` as the "not open" sentinel
(`int _fd; // 0 = not open`), initialized `_fd(0)`, reset `_fd = 0` on close, and
tested open-ness by truthiness: `if (_fd)`, `if (_on_disk && _fd)`,
`if (_fd || _nodes)`. File descriptor `0` is **valid** (stdin closed/redirected),
so an index whose on-disk file landed on fd 0 was treated as "not open" —
`unload()` skipped the close and munmap, leaking the descriptor and the mapping.

## Root cause
`0` is a legal descriptor; using it as an invalid sentinel conflates a valid
resource with "absent".

## Fix (minimal, canonical sentinel)
Per the review decision, use `-1`:
- `int _fd;` comment → `(-1 = not open; fd 0 is valid)`; member init `_fd(-1)`.
- All 16 close/reset sites `_fd = 0;` → `_fd = -1;`.
- All three open-ness checks now compare against the sentinel:
  `if (_fd != -1)`, `if (_on_disk && _fd != -1)`, `if (_fd != -1 || _nodes)`.
- `open()`/`_open()` already return `-1` on failure and the existing
  `if (_fd == -1)` failure checks now align with the sentinel. Diff:
  `out/annoylib.h.run6.diff`.
- Dead duplicates `annoylib_v0.h` / `annoylib_review.h` carry the same pattern
  but are compiled nowhere (CY-001) — left for the pruning pass, noted so the fix
  is not shadowed.

## Verification
- **Discriminating regression** `tests/test_fd_sentinel.py`: in a subprocess it
  closes fd 0, runs `on_disk_build` (which then opens onto fd 0), and asserts
  `unload()` **closes** fd 0 and query results are unchanged. Measured:
  `used_fd0=True, still_open_after_unload=False, nns_match=True`. With the old
  truthiness, `still_open` would be `True` (the leak). Runs in a subprocess so
  closing stdin cannot disrupt the pytest runner; skips if fd 0 is not obtained.
- **Ring green** (both `annoylib` extensions rebuilt): kiss 111, memmap 49,
  annoy/_annoy cython 27, portable-blob 5, euclidean 12. Small `on_disk_build`
  round-trip verified equivalent.

## Note
Full on-disk fault-injection (ENOSPC / crash mid-build) remains a CI concern
(sandbox OOMs the large on-disk suites); this run fixes and verifies the sentinel
contract itself.
