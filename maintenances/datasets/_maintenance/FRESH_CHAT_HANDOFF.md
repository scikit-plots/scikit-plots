# Fresh-chat handoff: datasets

Current campaign status: **maintenance PASS / runtime FAIL / integration UNAVAILABLE / release BLOCKED**.

Read `REVIEW.json` for the owned findings. The strongest behavioral baseline is the package-root harness: **209 passed, 38 skipped, 19 subtests**. All 38 skips are Parquet lanes because no `pyarrow`/`fastparquet` engine is installed. The general `_data_loader.py` surface has no focused test module and independent probes reproduce ZIP, upload and database-default failures.

Run:

```sh
python -B maintenances/datasets/_maintenance/check_trackers.py --json
python -B maintenances/datasets/_maintenance/review_subsystem.py --json
python -B -m pytest maintenances/datasets/_maintenance/tests -q -p no:cacheprovider
```

Do not turn harness stubs for the missing root package/appdirs/logger/safe_import into release evidence. They only isolate datasets-owned behavior.
