---
title: "_decorates: singleton load unpickles a file with no trust guard"
status: open
kind: "security"
area: "scikitplot/_decorates"
discovered_during: "Package sweep, tools/maint_tools/sweep_static.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot._decorates"
towncrier_type: "fix"
towncrier_fragment: ""
---

# _decorates: singleton load unpickles a file with no trust guard

## Summary

`_decorates/_decorate.py` loads a singleton instance with `joblib.load(filename, ...)`, which unpickles the file: loading runs any code the file contains. Nothing asks whether the file is trusted.

## Why it matters

`scikitplot.corpus.load_documents` refuses pickle and joblib files unless `trusted=True` and checks an optional SHA-256 first (CORPUS-SEC-001); the same risk elsewhere in the package has no such guard, so the protection depends on which submodule a user calls.

## Current evidence

- `python -B tools/maint_tools/sweep_static.py --list joblib-load` lists `scikitplot/_decorates/_decorate.py:1233: cls._instance[cls] = joblib.load(filename, *args, **kwargs)`.
- `scikitplot/corpus/_export/_export.py` calls `_pickle_safety_guard(format.value, trusted)` and `_verify_artifact_integrity(path, expected_sha256)` before `pickle.load`.

## Root cause / current understanding

Two submodules implemented deserialisation independently; the guard was never shared.

## Expected behavior

One shared guard: pickle-based loading requires an explicit `trusted=True` (or a matching SHA-256), and the error message says why.

## Affected paths and ownership

- `scikitplot/_decorates/_decorate.py`
- `scikitplot/corpus/_export/_export.py` (the guard to share)

## Constraints and non-goals

No change for callers that load their own trusted files once they pass `trusted=True`.

## Edge cases to cover

- A file that is not a pickle
- A tampered file with the right extension
- A path given as `str` and as `Path`

## Proposed direction

Move the corpus guard to a shared private helper and call it from every pickle/joblib load in first-party code.

## Verification / acceptance criteria

- Loading without `trusted=True` raises before any byte is unpickled (test with a file whose unpickling would write a marker)

## Documentation impact

none

## Release-note promotion

Required: `scikitplot._decorates` / `fix`, one bullet, when the change lands.
