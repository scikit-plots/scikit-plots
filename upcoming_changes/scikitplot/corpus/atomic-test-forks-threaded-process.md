---
title: "On CPython 3.15 the atomic-publication test forks a threaded process and the fork warning fails it"
status: open
kind: "compatibility"
area: "scikitplot/corpus"
discovered_during: "PR 919: Python 3.8 to 3.15 matrix of corpus and mcp (finding CX-10)"
release_note: "not-required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "unknown"
towncrier_fragment: ""
---

# On CPython 3.15 the atomic-publication test forks a threaded process and the fork warning fails it

## Summary

`scikitplot/corpus/tests/test__atomic.py::TestConcurrentPublication::
test_contended_target_stays_consistent` asks for the `fork` start method
and starts a pool of 8 processes. On CPython 3.15.0b4, when the test process
already runs other threads (as under `pytest -n 4` beside other tests),
`os.fork()` emits `DeprecationWarning: This process (pid=...) is
multi-threaded, use of fork() may lead to deadlocks in the child.`, and the
repository's `filterwarnings = error` turns it into a failure.

## Why it matters

The corpus suite is red on the next CPython for a reason that is not a
defect of the code under test, and the forked children of a threaded parent
can deadlock: the warning names a real hazard of the test itself.

## Current evidence

`pytest scikitplot/corpus/tests/test__atomic.py scikitplot/corpus/_storage
-n 4`, three runs per interpreter, `main` at `6734def` and PR 919 alike:

- CPython 3.15.0b4: 1 failed in each of 3 runs (the warning above, raised
  from `os.fork()` in `multiprocessing.popen_fork`);
- CPython 3.12.13, 3.13.18, 3.13.14t, 3.14.6, 3.14.6t: 0 failures;
- the test alone on 3.15.0b4: passed 5 of 5.

## Root cause / current understanding

The test forces `mp.get_context("fork")`. Whether CPython warns depends on
the threads alive at the fork, which other tests in the same worker create;
3.15.0b4 reports it where 3.12 to 3.14 did not under the same load. Why the
detection differs in 3.15 is not established here.

## Expected behavior

The test passes on every supported interpreter, alone and under `-n`,
with warnings as errors.

## Affected paths and ownership

- `scikitplot/corpus/tests/test__atomic.py`
- Owner: `maintenances/corpus/`

## Constraints and non-goals

Tests only. Keep the property the test proves: concurrent publishers leave
exactly one complete payload and no temporary files. Do not silence the
warning.

## Edge cases to cover

`-n 0` and `-n 4`; CPython 3.12 to 3.15; free-threaded builds; platforms
without `fork`.

## Proposed direction

Use a start method that never forks a threaded parent (`forkserver` or
`spawn`; the worker `_publish_worker` is module level, so it pickles).

## Verification / acceptance criteria

The command above passes three times on each of CPython 3.12 to 3.15 and
the free-threaded builds; the test still fails if publication is made
non-atomic.

## Documentation impact

None.

## Release-note promotion

Not required (tests only).
