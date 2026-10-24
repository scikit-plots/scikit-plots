---
title: "run_with_timeout rejects fractional seconds, and three of its tests expect a timeout that cannot happen"
status: promoted
kind: "reliability"
area: "scikitplot/_utils (timeout.py)"
discovered_during: "TP-1, the garbage-collection measurements over scikitplot/_utils, 2026-10-23"
release_note: "unknown"
towncrier_section: "scikitplot._utils"
towncrier_type: "fix"
towncrier_fragment: "docs/source/whats_new/upcoming_changes/scikitplot._utils/867.fix.rst"
---

# run_with_timeout rejects fractional seconds, and three of its tests expect a timeout that cannot happen

## Summary

`scikitplot/_utils/timeout.py` passes `seconds` to `signal.alarm`, which takes
whole seconds only; `run_with_timeout(0.1)` raises `TypeError` before the
block runs. Separately, `test_timeout_raises_scikitplot_timeout_error`,
`test_timeout_error_message_contains_seconds` and
`test_handler_raises_scikitplot_timeout_error` sleep 0.01 s inside a
1-second timeout and expect the timeout to fire.

## Current evidence

Every run of `scikitplot/_utils` here (CPython 3.11, Linux, six runs under
three GC policies) fails the same four tests of
`scikitplot/_utils/tests/test_timeout.py`:

```text
scikitplot/_utils/timeout.py:38: in run_with_timeout
    _signal.alarm(seconds)
E   TypeError: 'float' object cannot be interpreted as an integer
        seconds    = 0.1
```

## Root cause / current understanding

Verified: `signal.alarm` takes an `int`. The three timeout tests sleep 100
times shorter than the timeout (`time.sleep(0.01)  # far exceeds 1 second`).
Not verified: whether fractional seconds are meant to be supported
(`signal.setitimer(signal.ITIMER_REAL, seconds)` accepts them).

## Expected behavior

Either fractional seconds work (`setitimer`) or a non-integer is refused with
a clear `ValueError`; the tests sleep longer than the timeout they test.

## Affected paths and ownership

`scikitplot/_utils/timeout.py`, `scikitplot/_utils/tests/test_timeout.py`.

## Verification / acceptance criteria

`python -m pytest scikitplot/_utils/tests/test_timeout.py` passes on Linux
and macOS; on Windows the tests are skipped as the function refuses.

## Release-note promotion

A `fix` fragment if fractional seconds become supported.

## Resolution (2026-10-24, TF-1 step 2)

`signal.setitimer(ITIMER_REAL)` replaces `signal.alarm`, so fractions work; a
limit that is not a finite number above 0 is a `TypeError`/`ValueError`; an
already armed timer is refused (`RuntimeError`) instead of being cancelled;
the previous `SIGALRM` handler is put back. The tests block 30 s against a
0.05 s limit (Rule 156) and patch `is_windows` on the module object (Rule
144): 26 passed under pytest 9.0.3 and 9.1.1.
