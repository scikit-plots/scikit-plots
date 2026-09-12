---
name: impute-maintainer
description: Maintain scikitplot.impute, especially ANNImputer, its experimental public gate, approximate-neighbor backend dispatch, index access/persistence policy, sklearn compatibility surface, tests, and release evidence. Keep Annoy/Voyager implementation ownership outside this skill.
---

# `scikitplot.impute` maintainer

Work from the wide checkout containing `scikitplot/`, `maintenances/`, and
`skills/`. This skill owns **imputation semantics and composition**, not Annoy,
Voyager, or generic package utilities.

## Read first

1. `maintenances/impute/MAINTAINING.md`
2. `maintenances/impute/_maintenance/FRESH_CHAT_HANDOFF.md`
3. `maintenances/impute/_maintenance/STATE.json`
4. `maintenances/impute/_maintenance/FAMILY.md`
5. `maintenances/impute/_maintenance/VERIFICATION.md`

Then run:

```sh
python -B maintenances/impute/_maintenance/check_trackers.py --json
python -B maintenances/impute/_maintenance/review_subsystem.py --json
python -B -m pytest maintenances/impute/_maintenance/tests -q -p no:cacheprovider
```

## Keep backend ownership explicit

`ANNImputer` may use `scikitplot.annoy._annoy.Index`; that Cython/native backend
belongs to the Annoy maintainers. Voyager belongs to its external package.
Impute owns selection, parameter mapping, neighbor-to-imputation semantics, and
what happens when a requested backend is unavailable.

Do not silently switch implementations after an arbitrary backend exception.
Import fallback must distinguish **backend unavailable** from **backend broken**.
The current `except Exception` around the in-tree Annoy import is therefore an
open structural finding.

## Preserve estimator and persistence semantics

Keep sklearn estimator conventions (`fit` returns self, fitted attributes,
feature names, indicators, clone/get_params behavior) and make missing-value
behavior deterministic. Preserve the three index-access modes:

- `public`: in-memory index may be exposed through `train_index_`;
- `private`: in-memory index is usable internally but public access is denied;
- `external`: only path/metadata remain on the estimator and a loader recreates
  the runtime index.

`PrivateIndexMixin` is an accidental-access policy, **not a security boundary**.
Do not strengthen documentation claims beyond what same-process Python can
actually enforce. External file deletion/error behavior must fail visibly.

## Treat optionality as a contract

Do not add module-scope imports for optional data/backend packages unless they
are genuinely required to import the feature. The current `_ann.py` imports
pandas even though it has no runtime `pd` use and tests treat pandas as
optional; keep this finding visible until resolved.

Voyager absence must remain representable without preventing the Annoy backend
from importing. Real Voyager evidence is a separate lane.

## Decide the `_base.py` role explicitly

`_base.py` is copied/adapted from sklearn, but the public module currently
re-exports sklearn's `SimpleImputer`/`MissingIndicator`, `ANNImputer` inherits
sklearn's `_BaseImputer`, and the nominal base/common tests also import sklearn.
That means local `_base.py` can drift without the current suite noticing.

Before changing it, decide one of three models: maintained local implementation,
compatibility reference, or historical/removable code. Align imports, tests, and
public API with that decision; do not merely update copied code by habit.

## Experimental public gate

`ANNImputer` is intentionally unavailable from `scikitplot.impute` until the
experimental enable hook runs. Preserve the informative ImportError and ensure
the activation module is tested in a complete checkout. Do not interpret a
missing experimental package in a partial review archive as an impute code bug.

## Evidence rules

Separate:

1. static contract health;
2. direct sklearn-derived/common tests;
3. throwaway harness tests that isolate impute logic;
4. real complete-package + compiled Annoy integration;
5. optional Voyager integration;
6. release/platform evidence.

A test double can validate impute algorithms but cannot prove native Annoy ABI,
serialization, memory mapping, or compiled-backend behavior. Never collapse
those evidence levels into one green status.

Current partial-snapshot harness evidence is 255 passed plus 32 subtests, but
release remains blocked until real integration is available and the structural
findings are resolved or formally accepted.
