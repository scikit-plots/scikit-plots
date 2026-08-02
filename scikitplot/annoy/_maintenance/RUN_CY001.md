<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# CY-001 — duplicate Cython template removed (single source of truth)

**Priority:** P1  **Area:** `annoy/_annoy/annoylib_pyx.in`  **Gate tier:** ring

## Finding
Two near-identical templates existed: `annoylib.pyx.in` (built; meson selects it)
and `annoylib_pyx.in` (NOT built). Duplication lets a fix land in the dead copy and
appear complete during review, yet never ship (the guide's named risk).

## Evidence the alternate was a stale fix-trap
Diff confirmed the canonical is strictly ahead: the alternate still had the OLD raw
`char* error`+`free(error)` pattern the canonical had migrated to `ScopedError`
(7 sites), and had **0** of `supported_dtypes`, `Float80`, `ScopedError`, or the
metric-alias work — all present in the canonical. Its "unique" lines were
superseded code, not fixes to salvage.

## Fix
Removed `annoylib_pyx.in`. The build is unaffected (canonical regenerates cleanly),
proving the alternate controlled no packaged code. Guard
`tests/test_single_canonical_template.py` (3) asserts `_annoy/` holds exactly the
canonical `annoylib.pyx.in` + `annoylib.pxd.in` and that `annoylib_pyx.in` does not
return.

## Note
`_annoy/backup_template/{annoylib.pyx.in, annoylib2.pyx.in}` are additional dead
copies, but quarantined by folder name (not built, not sibling to the canonical) so
they are not fix-traps. Recommend removing them too for cleanliness — left in place
as they are an explicit user backup folder; the guard ignores subdirectories.

## Verification
Guard 3 passed; canonical `.pyx` regenerates (exit 0); ring unaffected.
