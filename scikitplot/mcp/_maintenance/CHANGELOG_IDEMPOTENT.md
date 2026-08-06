# Idempotent test and protocol hardening

## 0.2.2 — strict MCP wire validation

- Fixed a protocol-boundary coercion bug where JSON Boolean `true` was accepted
  as `k=1` before `SearchService.search()` could reject it. Python's `bool` is
  an `int` subclass, and non-strict Pydantic integer validation converted the
  value before the service boundary.
- Declared MCP tool inputs with `StrictStr` and `StrictInt`, preserving the same
  JSON Schema while rejecting Booleans, numeric strings, and integral floats.
- Added SDK-double and real-SDK regression tests for `true`, `false`, `"2"`,
  `2.0`, and non-string queries.
- Made constructor configuration strict for `max_concurrency` and
  `acquire_timeout_seconds`; accidental Boolean/string coercion is now rejected.
- Enforced the documented inverse output invariant: non-empty results must have
  `message=null`.
- Restored optional MCP test dependency gates so source-only environments skip
  protocol tests cleanly instead of failing collection.
- Added a live-server version gate. Acceptance tests now fail early with a clear
  stale-image message when the running server does not match the checked-out
  submodule version.
- Advanced the service version to `0.2.2`.

## Correctness fixes

- Corrected zero-result structured output: status text moved from `passages` to
  `message`, preserving `count == len(passages) == len(citations)`.
- Added a Pydantic model validator for result cardinality, citation numbering,
  unique chunk identifiers, and mandatory empty-result messages.
- Rejected implicit input coercion for `query` and `k`, including Boolean `k`.
- Normalized token-boundary punctuation so exact canary identifiers in prose are
  retrievable without including sentence punctuation.
- Centralized the service version in `_version.py` and advanced it to `0.2.1`.

## Idempotence and deployment

- Added deterministic `--self-test` with optional fail-closed match and exact
  `doc_id` requirements.
- Restored stable JSON output for effective configuration and self-tests.
- Added a Docker image-build self-test executed twice and compared byte-for-byte.
- Kept live HTTP acceptance tests opt-in so ordinary test runs do not depend on
  an external process, stale port, or previous run.
- Added real MCP SDK in-memory protocol tests when the optional SDK is installed.
- Added bounded live concurrency/load tests that perform no writes.
- Documented immutable index generations, canary validation, atomic activation,
  and rollback.

## Verification

```text
self-contained run 1: 81 passed, 2 skipped
self-contained run 2: 81 passed, 2 skipped
reverse collected order: 81 passed
compileall: PASS
shell syntax: PASS
exact-canary self-test repeated byte-identically: PASS
```

## 0.2.3 — closed tool schemas and stable source manifests

- Reject unknown `search_docs` arguments instead of silently dropping them.
- Publish `additionalProperties: false` in the MCP input schema.
- Reject undeclared fields in structured output models.
- Centralize manifest inclusion policy in the atomic manifest writer.
- Ignore root-level release artifacts and build/cache state without hiding
  intentional source fixtures.
- Add live and in-memory regressions for extra arguments and manifest drift.
