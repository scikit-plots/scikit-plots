<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# Run 11 — ANNOY-SAVE-002 (guide 6.7) — corrupt on-disk file on failed finalize

**Priority:** P1  **Area:** `cexternals/_annoy/src/annoylib.h`  **Gate tier:** ring

## Finding (grounded in guide 6.7)
In `build()`'s on-disk finalization, if `remap_memory_and_truncate` (the final
`ftruncate` to the exact node size) failed, the code returned an error but left
the partially finalized, header-less file at the user's path. The source even
carried the acknowledgement: `// TODO: this probably creates an index in a
corrupt state... not sure what to do`. A later `load()` could accept that file as
a valid (corrupt) index.

## Root cause
The failure path abandoned a half-written on-disk file without removing it, so it
remained exposed as loadable.

## Fix (never expose the corrupt file)
Per the review decision ("never exposed as valid"):
- Added `std::string _on_disk_path`, set in `on_disk_build()` once the fd is open
  and cleared in `reinitialize()`.
- On finalize-truncate failure, `unlink`/`_unlink` the file so it can never be
  loaded as a corrupt index, then return the error. Unlinking an open file is
  POSIX-safe — the fd/mmap stay valid and are torn down normally by
  `unload()`/destructor — so no lifecycle change is needed here (the fragile
  mmap teardown, guide 6.2/6.3, is deliberately untouched).
- Removed the `TODO ... corrupt state` comment. Diff: `out/annoylib.h.run11.diff`.

## Verification
`tests/test_on_disk_finalize.py` (2 tests): a **successful** on-disk build leaves
a complete, loadable file whose queries match the in-memory index, and an
on-disk build/reload round-trip preserves ids + distances (the happy path — the
file must NOT be removed on success). Ring green: kiss 123, memmap 49,
annoy/_annoy cython 27, save-atomicity 3, on-disk-finalize 2, fd-sentinel 1,
euclidean 12.

## Deferred to CI (honesty)
Triggering the finalize-`ftruncate` failure deterministically needs filesystem
fault injection (ENOSPC / a bad fd), so the removal-on-failure branch is verified
by construction + inspection here and exercised under a CI fault matrix (same
posture as ANNOY-SAVE-001's ENOSPC case). Crash *during* the incremental on-disk
build (before finalization) is a broader guarantee that would need a temp-path +
atomic-rename or a header "incomplete" flag — tied to §6.6 (on-disk format) and
the mmap-lifecycle work; noted as a follow-up, not rushed here.
