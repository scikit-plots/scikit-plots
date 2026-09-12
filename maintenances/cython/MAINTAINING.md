# Maintaining `scikitplot.cython`

`scikitplot.cython` is a runtime compilation service: it validates and compiles
**caller-supplied programs**, publishes/imports native artifacts, and maintains
cache/lock/pin state.  That trust boundary is different from project build-time
Cython modules such as `scikitplot.annoy`.

Start with `_maintenance/FRESH_CHAT_HANDOFF.md`, then run:

```sh
python -B maintenances/cython/_maintenance/check_trackers.py --json
python -B maintenances/cython/_maintenance/review_subsystem.py --json
python -B -m pytest maintenances/cython/_maintenance/tests -q -p no:cacheprovider
```

A green maintenance suite does not imply release readiness.  Read
`_maintenance/EVIDENCE.json` and the skip reasons from the runtime suite.

The supplied snapshot has two active findings: a checked-in `__pycache__` and
order-dependent setuptools/distutils test pollution that causes three real-build
tests to self-skip in a full run even though they pass individually.  Do not
convert either into a green tracker entry without changing the underlying tree.

Historical pre-contract maintenance material is preserved under
`_maintenance/history/legacy_live_2026-09-12/` and is provenance only.
