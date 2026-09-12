# Fresh-chat handoff — `scikitplot.impute`

Read `MAINTAINING.md`, `STATE.json`, `FAMILY.md`, and `VERIFICATION.md` first.
Then run:

```sh
python -B maintenances/impute/_maintenance/check_trackers.py --json
python -B maintenances/impute/_maintenance/review_subsystem.py --json
python -B -m pytest maintenances/impute/_maintenance/tests -q -p no:cacheprovider
```

Current intended state is **maintenance PASS / runtime structural FAIL / release
BLOCKED**. Do not change the status merely because the partial archive cannot
collect the whole runtime suite.

Open local findings:

1. `_ann.py` catches `Exception` around the in-tree Annoy import and falls back
   to external `annoy`; narrow this boundary before calling the backend contract
   fail-closed.
2. `_ann.py` imports pandas at module scope although `pd` is unused and the
   tests treat pandas as optional.
3. `_base.py` is an sklearn-derived local mirror that is not used by the public
   re-exports or ANNImputer, while its nominal tests import sklearn's classes.
   Decide whether it is a maintained local implementation, a compatibility
   reference, or removable history; then align tests/API accordingly.

Evidence from a throwaway harness shows the impute-owned algorithmic tests are
healthy: 255 passed plus 32 unittest subtests when the missing package-shell
utilities and an Annoy-compatible test double are supplied. That harness is
**not** evidence for real Annoy integration.
