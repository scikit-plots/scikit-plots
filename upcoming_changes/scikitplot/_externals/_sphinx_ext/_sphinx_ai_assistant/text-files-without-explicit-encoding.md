---
title: "_sphinx_ext/_sphinx_ai_assistant: text files read or written without an explicit encoding (21 places)"
status: open
kind: "compatibility"
area: "scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant"
discovered_during: "Package sweep, tools/maint_tools/sweep_static.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "unknown"
towncrier_type: "fix"
towncrier_fragment: ""
---

# _sphinx_ext/_sphinx_ai_assistant: text files read or written without an explicit encoding (21 places)

## Summary

In the AI assistant's documentation proxy, 21 calls open, read or write a text file without `encoding=`. Python then uses the locale's encoding: UTF-8 on most Linux and macOS systems, cp1252 or another code page on Windows and in some containers.

## Why it matters

The same file reads differently on two machines: non-ASCII text (names, accents, symbols) becomes mojibake or raises `UnicodeDecodeError` on Windows, and a file written there cannot be read back on Linux. Python 3.15 makes UTF-8 mode the default, which changes these calls' behaviour once more.

## Current evidence

- `python -B tools/maint_tools/sweep_static.py --list open-without-encoding` and `--list path-text-without-encoding` (main cba9383):
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/ci/redis_chaos_attestation.py:125`: `with path.open("rb") as handle:`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/archive_native_status_evidence.py:108`: `with path.open("rb") as handle:`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/audit_archive_retention.py:113`: `with path.open("rb") as f:`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/continue_archive_merkle_authority.py:122`: `with path.open("rb") as f:`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/continue_recovered_trust.py:97`: `with path.open("rb") as handle:`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/finalize_publication.py:56`: `with path.open("rb") as handle:`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/govern_archive_merkle_log_authority.py:86`: `with path.open("rb") as f:`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/govern_release_history.py:78`: `with path.open("rb") as handle:`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/preserve_archive_merkle_rebridge.py:118`: `with Path(path).open("rb") as f:`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/preserve_release_history.py:55`: `with path.open("rb") as handle:`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/promote_release.py:61`: `with path.open("rb") as handle:`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/promote_release.py:382`: `with path.open("rb") as src, zf.open(info, "w", force_zip64=False) as dst:`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/publish_release.py:51`: `with path.open("rb") as handle:`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/rebridge_archive_merkle_authority.py:106`: `with Path(path).open("rb") as f:`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/seal_release_governance.py:80`: `with path.open("rb") as handle:`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/verify_archive_merkle_transparency.py:110`: `with path.open("rb") as f:`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/verify_attestation_lifecycle.py:97`: `with path.open("rb") as handle:`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/verify_native_status_provenance.py:86`: `with path.open("rb") as handle:`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/verify_release_evidence.py:123`: `with path.open("rb") as handle:`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/witness_archive_health.py:114`: `with path.open("rb") as f:`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/witness_publication.py:53`: `with path.open("rb") as handle:`

## Root cause / current understanding

Each call relies on `locale.getpreferredencoding()`; nothing states which encoding the file is in.

## Expected behavior

Every text-mode open, `read_text` and `write_text` names its encoding (`utf-8`, or `utf-8-sig` when a byte-order mark may be present), and a decode error says which file and encoding failed.

## Affected paths and ownership

- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant` (files above)
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

Required: `unknown` / `fix`, one bullet, when the change lands.
