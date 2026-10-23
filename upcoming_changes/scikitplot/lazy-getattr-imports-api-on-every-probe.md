---
title: "Any unknown attribute of scikitplot imports scikitplot.api, and a failed import runs again on every probe"
status: open
kind: "reliability"
area: "scikitplot/__init__.py"
discovered_during: "TP-1, the test-performance recorder's module-execution count, 2026-10-23"
release_note: "unknown"
towncrier_section: "unknown"
towncrier_type: "efficiency"
towncrier_fragment: ""
---

# Any unknown attribute of scikitplot imports scikitplot.api, and a failed import runs again on every probe

## Summary

`scikitplot.__getattr__` calls `_api_names()`, which imports `scikitplot.api`
to learn the names it re-exports, for any attribute the module does not have.
Tools probe modules for optional attributes all the time; pytest does it for
every package it collects. So any test session under `scikitplot/` imports the
whole plotting API (scikit-learn, `scipy.stats`, the vendored seaborn), and when
that import fails, Python removes the half-imported modules and the next probe
executes them all again.

## Why it matters

- Cost: the import of `scikitplot.api` was 17.2 s of a profiled
  `scikitplot/_externals/` session (`__init__.py:317(__getattr__)`, 6 calls;
  `_api_names`, 5 calls) whose tests never use the API.
- Correctness: with `scikitplot.api` broken (here, seaborn missing), five
  probes execute `scikitplot/_compat/__init__.py`, `scikitplot/api/__init__.py`
  and seven more module bodies five times each in one process; module state
  is rebuilt each time, and any error inside them surfaces as an unrelated
  attribute lookup.

## Current evidence

Recorder output (`SKPLT_TEST_PERF_DIR`, session line `modules_reexecuted`) of
one test of `scikitplot/_testing/tests/`, with seaborn absent:

```text
['scikitplot/_compat/__init__.py', 5], ['scikitplot/api/__init__.py', 5],
['scikitplot/api/estimators/__init__.py', 5], ... (9 files, 5 executions each)
```

The stack at each execution (an `exec` audit hook):

```text
_pytest/python.py:678 setup -> importtestmodule -> config.consider_module
    -> scikitplot/__init__.py:394 __getattr__("pytest_plugins") -> _api_names
_pytest/python.py:682 setup -> _get_first_non_fixture_func("setUpModule", "setup_module", ...)
    -> scikitplot/__init__.py:394 __getattr__ -> _api_names
```

`import scikitplot.api` alone raises `ModuleNotFoundError: No module named 'seaborn'`
in that environment.

## Root cause / current understanding

Verified: `__getattr__` treats every unknown name as a possible re-export and
resolves it by importing the API; nothing remembers that the import failed or
that the name is not an API name. pytest's `Package.setup` asks for
`pytest_plugins`, `setUpModule`, `setup_module`, `tearDownModule` and
`teardown_module` (pytest 9.1.1).

## Expected behavior

Looking up a name that is not part of the public API raises `AttributeError`
without importing `scikitplot.api`; a failed import of `scikitplot.api` is not
retried by every later lookup (the error is kept and raised again, or the
names are known without the import).

## Affected paths and ownership

`scikitplot/__init__.py` (`__getattr__`, `_api_names`, `__dir__`); no
`maintenances/` area owns the package root.

## Constraints and non-goals

`dir(scikitplot)` and `scikitplot.<api name>` keep working; a partial
distribution without `scikitplot.api` keeps returning no re-exports.

## Edge cases to cover

- dunder and underscore names, and pytest/unittest hook names;
- API present and importable; present but broken; absent (partial distribution);
- concurrent first access from two threads.

## Proposed direction

Either a static list of the API's exported names (generated and checked, like
the other generated configurations, Rule 70) consulted before any import, or a
negative cache of the import error; plus an early `AttributeError` for names
starting with `_` and for the hook names above.

## Verification / acceptance criteria

With the recorder on, a session under `scikitplot/_externals/` executes
`scikitplot/api/__init__.py` zero times (`module_exec` and the session's
`modules_reexecuted`), and with seaborn removed, no module is executed twice.

## Documentation impact

None.

## Release-note promotion

Probably an `efficiency` fragment for `scikitplot`, decided when implemented.
