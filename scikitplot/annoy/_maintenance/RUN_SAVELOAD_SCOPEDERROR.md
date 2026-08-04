<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# save/load native-error ownership → ScopedError (R14/R19 follow-up)

**Priority:** P2  **Area:** `annoy/_annoy/annoylib.pyx.in` (`save`/`load`)  **Gate tier:** ring

## Finding
`save()` and `load()` used the raw `char* error = NULL` + manual `free(error)`
pattern. It freed correctly today, but was fragile: any future early-return or
exception inserted between the native call and `free()` would leak the strdup'd
error string, and a success-with-non-NULL-error edge would leak. CY-005 and the
load-helper already use `ScopedError` (RAII); save/load were inconsistent.

## Fix
Migrated both to own the native error via `ScopedError` (`&err.err`), which frees
the string in `__dealloc__` on every control-flow path. No behaviour change: the
same `IOError("save failed: …")` / `IOError("load failed: …")` messages are
raised, now leak-safe against future edits. No manual `free()` remains in the
save/load region.

## Verification
`tests/test_save_load_scopederror.py` (4): round-trip; load-missing → IOError;
save-bad-path → IOError; and 200× repeated failures followed by a working
save/load (a leak/double-free would surface under repetition). Ring green:
serialize 3, supported_dtypes 17, no-orphan-pxi 2.
