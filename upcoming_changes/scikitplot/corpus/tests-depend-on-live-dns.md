---
title: "corpus: 22 downloader and web-reader tests need live DNS"
status: open
kind: "reliability"
area: "scikitplot/corpus"
discovered_during: "Real-life probe on main cba9383 (2026-11-24)"
release_note: "not-required"
towncrier_section: "unknown"
towncrier_type: "unknown"
towncrier_fragment: ""
---

# corpus: 22 downloader and web-reader tests need live DNS

## Summary

With name resolution unavailable (an offline CI runner, a sandbox, some proxies), 22 tests fail: 11 in `test__web.py`, 8 in `_downloader/tests/test__downloader.py`, 3 in `test__url_and_archive_handler.py`. They mock HTTP but not DNS, and the SSRF check resolves the host first.

## Why it matters

Unit tests should not depend on the network; these fail with an error that looks like a product bug (`'example.com' does not appear to be an IPv4 or IPv6 address` in the context).

## Current evidence

- Corpus suite on main and on #957's branch (same environment, DNS unavailable): the same 22 failures.
- `_url_handler._resolve_and_validate` raises `ValueError: cannot resolve host ... (SSRF fail-closed)` as designed; the tests never stub `socket.getaddrinfo`.

## Root cause / current understanding

The fail-closed resolution (CORPUS-NET-002) was added after the tests, which still assume resolution succeeds.

## Expected behavior

Tests stub the resolver (a fixture returning a public address) and one test checks the fail-closed path explicitly.

## Affected paths and ownership

- `scikitplot/corpus/_readers/tests/test__web.py`
- `scikitplot/corpus/_downloader/tests/test__downloader.py`
- `scikitplot/corpus/tests/test__url_and_archive_handler.py`

## Constraints and non-goals

Production code unchanged; the fail-closed behaviour stays.

## Edge cases to cover

- IPv6-only resolution
- A host resolving to a private address (must still be refused)

## Proposed direction

Add a shared `fake_dns` fixture in the corpus `conftest.py`.

## Verification / acceptance criteria

- The suite passes with networking disabled (`unshare -n` or a blocking socket fixture)

## Documentation impact

none

## Release-note promotion

Not required (no shipped behaviour changes).
