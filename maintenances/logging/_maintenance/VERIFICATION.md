# Verification

Use these lanes independently:

1. `python -B maintenances/logging/_maintenance/check_trackers.py --json`
2. `python -B -m pytest maintenances/logging/_maintenance/tests -q -p no:cacheprovider`
3. Native focused suite: `python -B -m pytest scikitplot/logging/tests/test__logging.py -q`
4. Diagnostic harness only: change the test import to `_logging` in a temporary copy; current result is 171 passed.
5. Public package probes for `__all__`, `getLogger`/`getLevelName`, `dir()`, and import safety.
6. Direct probes for ERR/ENV/HDL/CLI/CALL/ATTR/FMT findings.
7. Complete package aliases, processes, notebooks, redirected streams, and platform matrix.

A harness-only PASS cannot replace lane 3. A test suite that asserts defective behavior cannot close its corresponding finding.
