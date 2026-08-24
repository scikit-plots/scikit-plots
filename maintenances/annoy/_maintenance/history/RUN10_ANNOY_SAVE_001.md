<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# Run 10 — ANNOY-SAVE-001 (guide 6.5) — transactional `save()`

**Priority:** P1  **Area:** `cexternals/_annoy/src/annoylib.h`  **Gate tier:** ring

## Finding (grounded in guide 6.5)
`save()` was not failure-atomic. It (1) `unlink`ed the existing target, (2) wrote
directly to the final filename, (3) closed it, (4) `unload()`ed the in-memory
index, (5) `load()`ed the just-written file. A partial write therefore lost the
previous file, and a failed reload lost the in-memory usable state.

## Root cause
Destructive in-place write with no staging: the target and the in-memory backing
were mutated before the new data was known-good and durable.

## Fix (transactional save)
Per the review decision — stage, flush, atomically replace, switch backing only
after commit, preserve the original until then:
- Write to a same-directory temporary file `"<filename>.tmp-<pid>"` (never
  `unlink` the target).
- `fflush` + `fsync` (POSIX) / `_commit` (Windows) for durability before commit.
- Atomically replace: `rename()` (POSIX) / `MoveFileEx(..., MOVEFILE_REPLACE_EXISTING)`
  (Windows). On any failure the temp is removed and the function returns without
  touching the target.
- Only after a successful commit does it `unload()` + `load()`. A write failure
  now leaves BOTH the previous file and the in-memory index intact.
- Diff: `out/annoylib.h.run10.diff`. Externed `save()` signature unchanged.

## Verification
`tests/test_save_atomicity.py` (3 tests):
- **Complete file, no litter:** after save the file is loadable and matches, the
  object is still usable, and no `*.tmp-*` remains.
- **Atomic replace:** saving over an existing target succeeds and leaves no temp.
- **Clean failure preserves state:** a save to an invalid path fails, the
  in-memory index stays usable, and no partial/temp files are created.
- Ring green: kiss 123, memmap 49, annoy/_annoy cython 27, portable-blob 5,
  fd-sentinel 1, serialize_test 3, euclidean 12.

## Deferred to CI (honesty)
Crash / `ENOSPC` mid-write fault injection (proving the previous file survives a
truncated write) needs an injecting filesystem; the atomicity is guaranteed by
construction (the target is never modified until the rename commits) and verified
here for the happy path, atomic replace, and clean-failure cases. The Windows
`MoveFileEx` path is compile-guarded and runs on Windows CI.

## Related, not in this run
- §6.6 (on-disk vs normal-save format contract) is a format-matrix **decision**
  (legacy/native vs headered vs converted-on-finalize) — staged for a design call.
- §6.7 (failed final truncate corrupts the on-disk build) applies to the
  `on_disk_build` finalization path (distinct from `save()`); same transactional
  principle, separate run.
