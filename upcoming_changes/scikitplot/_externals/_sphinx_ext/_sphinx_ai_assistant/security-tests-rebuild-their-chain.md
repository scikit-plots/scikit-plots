---
title: "The release-security tests rebuild the whole Run 160-167 chain per test or per module, and load their modules many times"
status: open
kind: "reliability"
area: "scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/tests/_hf_spaces_proxy/security"
discovered_during: "TP-1, recording and profiling the security suite with the test-performance recorder, 2026-10-23"
release_note: "not-required"
towncrier_section: "unknown"
towncrier_type: "other"
towncrier_fragment: ""
---

# The release-security tests rebuild the whole Run 160-167 chain per test or per module, and load their modules many times

## Summary

Each Run N test module needs the signed, verified state of Runs 160 to N-1.
It builds that state itself: Runs 161-164 per test (a helper such as
`_fixture(tmp_path)` or `_witnessed(tmp_path)` called in every test), Runs
165-168 per module, and Run 168's module fixture rebuilds Run 167's module
fixture by calling `t167.base.__wrapped__`. Every epoch advance also verifies
the whole earlier history in production code. To reach their predecessors the
modules put `tests/` and `_hf_spaces_proxy/security/` on `sys.path` and import
test and source modules as top-level names, and `_load(name, path)` helpers
execute the same files under further names.

## Why it matters

- Time: the 24 files of `security/` took 321 s here (527 tests, CPython 3.11,
  2 CPUs; 1 661 s of test time for all of `_sphinx_ai_assistant` in the
  uploaded `_externals` profile). Most of a chain test is the rebuild, not
  the check: `test_run164_wrong_root_pin_rejected` takes 1.93 s, of which
  1.93 s is `_fixture` (the Run 160-163 chain) and 0.00 s is the
  verification it tests; `test_run162_witness_signature_mutation_fails` spends
  its 1.49 s in `_witnessed`.
- Growth: every new Run N adds tests that rebuild a longer chain, so the
  suite grows faster than the number of tests.
- Evidence quality: one file runs up to 29 times in one process
  (`verify_attestation_lifecycle.py`, re-executed by the imports of 7 test
  files; 34 files run more than once). cProfile then reports one copy's
  cumulative time and loses calls into the others: in the full-suite profile
  37.1 s of Run 168's `base` fixture has no callee (Rule 131).
- Isolation: `sys.path.insert` at import time is never undone (Rule 69), so
  every later test in the process resolves those names to these copies.

## Current evidence

Recorded with `SKPLT_TEST_PERF_DIR` (`scikitplot/_testing/_perf_telemetry.py`)
and analysed with `.github/scripts/ci_test_perf.py report`:

| test file | tests | total s | setup s |
|---|---|---|---|
| `test_preserve_archive_merkle_rebridge_history.py` (Run 168) | 23 | 75.4 | 25.6 |
| `test_rebridge_archive_merkle_authority.py` (Run 167) | 20 | 54.0 | 7.6 |
| `test_anchor_archive_health.py` (Run 163) | 22 | 28.4 | 3.7 |
| `test_witness_archive_health.py` (Run 162) | 32 | 27.8 | 0.1 |
| `test_audit_archive_retention.py` (Run 161) | 40 | 27.5 | 0.1 |
| `test_verify_archive_merkle_transparency.py` (Run 164) | 21 | 27.3 | 0.0 |
| `test_continue_archive_merkle_authority.py` (Run 166) | 16 | 20.5 | 3.7 |

Run 168's module fixture `base` profiled alone (`SKPLT_TEST_PERF_PROFILE`, its
file only, so every module ran once), edge-exact:

```text
base (run168)                       25.73 s
  t167._advance             x3      11.78 s   (advance_archive_merkle_rebridge 11.75 s)
  t167.base.__wrapped__     x1       7.95 s   (the same build as Run 167's own base: 7.58 s)
      t166._advance_run163  x2       4.23 s
      t164._fixture         x1       1.96 s
      continue_merkle_authority      1.08 s
  t166._advance_run163      x2       6.00 s
_advance_run163 (all 4 calls)       10.23 s   anchor_archive_health 6.76 s, witness 1.83 s, audit 1.63 s
```

## Root cause / current understanding

Verified: each module computes its predecessors' state from scratch because
the state is produced by helper functions of other test modules, not by a
shared fixture that pytest could cache, and Runs 161-164 call those helpers
inside each test. The repeated module executions follow from the `sys.path`
imports and the `_load(name, path)` helpers (the session's
`modules_reexecuted` lists every file and count). Not verified: whether any
test depends on getting a freshly generated key set rather than the same
chain as its neighbours.

## Expected behavior

Each stage of the chain (Runs 160 to 167) is built once per session; a test
that only reads a stage uses it as it is, and a test that changes files works
on its own copy. Each source and test module is executed once per process.

## Affected paths and ownership

`scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/tests/_hf_spaces_proxy/security/`
(owner `maintenances/_externals/_sphinx_ext/_sphinx_ai_assistant/`).

## Constraints and non-goals

- Never weaken what a test proves: every negative test still feeds its
  hostile input to the real verifier, and every positive test still verifies
  the full history.
- Production verification keeps verifying the whole history; making it
  incremental is a separate security design question, not a test speed-up.

## Edge cases to cover

- A test that mutates a stage's files (tamper, drift, symlink tests) gets a
  private copy; a shared stage is made read-only so that an accidental write
  fails at once instead of leaking into the next test.
- Tests that compare two independently generated chains still get two.
- Running one file alone, the whole directory, and `-n auto`.

## Proposed direction

1. A `security/conftest.py` with session-scoped fixtures, one per stage, each
   built from the previous stage's fixture, returning its paths and keys; and
   a helper that copies a stage into `tmp_path` for tests that write.
2. Import the stage builders as one package module each (no `sys.path` edits,
   no `_load(name, path)` copies of an already imported file).
3. Keep each test's assertion code as it is; only where its inputs come from
   changes.

## Verification / acceptance criteria

- Same tests, same outcomes; `module-re-executed` no longer fires for files
  under `security/` (recorder session line `modules_reexecuted`).
- `ci_test_perf.py compare --base <before> --head <after>` on the same
  machine shows the setup and call time of Runs 161-168 and the total of the
  directory; record the numbers in the area's `todo/todo.md`.

## Documentation impact

None outside the area's maintenance notes.

## Release-note promotion

Not required: tests only.
