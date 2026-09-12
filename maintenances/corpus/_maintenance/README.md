# Corpus maintenance plane

This directory contains deterministic developer-only state and checks for `scikitplot.corpus`. It is not importable runtime functionality.

The current archive deliberately demonstrates a fail-closed continuation case: maintenance records exist, but the runtime source is absent. The tooling must preserve that distinction instead of converting historical review completion into current runtime evidence.

Run `check_trackers.py`, `review_subsystem.py`, and `tests/test_contract.py`. Read `EVIDENCE.json` before making build/test/release claims.
