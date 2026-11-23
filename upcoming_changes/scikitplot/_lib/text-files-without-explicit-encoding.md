---
title: "_lib: text files read or written without an explicit encoding (2 places)"
status: open
kind: "compatibility"
area: "scikitplot/_lib"
discovered_during: "Package sweep, tools/maint_tools/sweep_static.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot._lib"
towncrier_type: "fix"
towncrier_fragment: ""
---

# _lib: text files read or written without an explicit encoding (2 places)

## Summary

In shared helpers, 2 calls open, read or write a text file without `encoding=`. Python then uses the locale's encoding: UTF-8 on most Linux and macOS systems, cp1252 or another code page on Windows and in some containers.

## Why it matters

The same file reads differently on two machines: non-ASCII text (names, accents, symbols) becomes mojibake or raises `UnicodeDecodeError` on Windows, and a file written there cannot be read back on Linux. Python 3.15 makes UTF-8 mode the default, which changes these calls' behaviour once more.

## Current evidence

- `python -B tools/maint_tools/sweep_static.py --list open-without-encoding` and `--list path-text-without-encoding` (main cba9383):
- `scikitplot/_lib/_testutils.py:282`: `with open("/proc/meminfo") as f:`
- `scikitplot/_lib/_testutils.py:320`: `with open(native_file, "w") as f:`

## Root cause / current understanding

Each call relies on `locale.getpreferredencoding()`; nothing states which encoding the file is in.

## Expected behavior

Every text-mode open, `read_text` and `write_text` names its encoding (`utf-8`, or `utf-8-sig` when a byte-order mark may be present), and a decode error says which file and encoding failed.

## Affected paths and ownership

- `scikitplot/_lib` (files above)
- `tools/maint_tools/sweep_static.py` (the check)

## Constraints and non-goals

Files the package writes are UTF-8 from now on; files a user hands in keep their own encoding rules (readers that detect encodings are not changed here).

## Edge cases to cover

- A path or content with non-ASCII characters on Windows (cp1252) and on a C/POSIX locale
- A file with a UTF-8 byte-order mark
- A file written by an older release in the locale encoding

## Proposed direction

Add `encoding="utf-8"` to each call; enable ruff's `PLW1514` (unspecified-encoding) for the area so new calls are caught; then `sweep_static.py --fail-on open-without-encoding` stays at zero for it.

## Verification / acceptance criteria

- `sweep_static.py --list open-without-encoding` and `--list path-text-without-encoding` print nothing for the area
- The area's tests pass under `PYTHONUTF8=0 LC_ALL=C` and on Windows CI

## Documentation impact

none

## Release-note promotion

Required: `scikitplot._lib` / `fix`, one bullet, when the change lands.
