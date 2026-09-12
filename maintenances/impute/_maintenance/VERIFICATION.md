# Verification policy

Use multiple lanes and keep their meaning separate.

## Lane A — static maintenance contract

Run `check_trackers.py --json` and the maintenance regression tests. This lane
checks ownership boundaries, public/experimental markers, index-policy markers,
plane separation, evidence integrity, and known structural findings.

## Lane B — source tests that require no scikitplot package shell

`test__base.py` + `test_common.py` can execute directly. They currently pass,
but note that they exercise sklearn implementations rather than local `_base.py`.

## Lane C — impute logic harness

A throwaway harness may provide only missing surrounding package pieces and an
Annoy-compatible test double. Use this to distinguish impute algorithm failures
from snapshot/build gaps. Never report this as native Annoy integration.

## Lane D — real integration

Requires a complete `scikitplot` package root, real `_path`/`_time`, packaging
shim, experimental enable hook, and built `scikitplot.annoy._annoy`. Run the full
suite against the real compiled backend. Optional Voyager coverage requires the
real external package.

## Release

Release stays blocked until the structural findings are resolved or explicitly
accepted, a complete-package full suite passes, real Annoy integration passes,
and supported-platform/package evidence is recorded.
