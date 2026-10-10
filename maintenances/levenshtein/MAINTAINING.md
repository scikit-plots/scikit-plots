# Maintaining `scikitplot.levenshtein`

This is the maintenance entry point for the public Levenshtein facade.

The subsystem is deliberately small:

```text
scikitplot/levenshtein/
├── __init__.py
├── _core.py
├── meson.build
└── tests/test_levenshtein.py
```

Its implementation dependencies are intentionally asymmetric:

```text
                       auto selection
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
scikitplot.cexternals   RapidFuzz (MIT)   pure Python
   _editdistance            optional       always present
       bundled
              \
               \ explicit only
                v
          Levenshtein
          GPL-2.0-or-later
```

`auto` must never select the GPL backend.

## Read first

1. `_maintenance/RESUME.md`
2. `_maintenance/DESIGN.md`
3. `_maintenance/VERIFICATION.md`
4. `REVIEW.json`
5. `_maintenance/STATE.json`
6. `skills/levenshtein/SKILL.md`

Then run:

```sh
python -B maintenances/levenshtein/_maintenance/tools/check_contract.py --json
python -B maintenances/levenshtein/_maintenance/tools/review_subsystem.py --json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -c /dev/null \
  scikitplot/levenshtein/tests/test_levenshtein.py \
  maintenances/levenshtein/_maintenance/tests \
  docs/source/user_guide/levenshtein \
  -q -p no:cacheprovider \
  --confcutdir=maintenances/levenshtein
```

The repository-wide pytest configuration can require Sphinx plugins that are
not needed for this subsystem. `-c /dev/null` is an accepted focused-lane
escape hatch; it is not a substitute for the release-wide test lane.

## The invariants

### L1 — the facade is always importable

`import scikitplot.levenshtein` must not import RapidFuzz or `Levenshtein`.
Optional accelerators are resolved on first use.

### L2 — `auto` is license-safe

Automatic selection is:

```text
bundled internal -> RapidFuzz -> pure Python
```

The external GPL `Levenshtein` package is supported only when explicitly
requested.

### L3 — pure Python is the availability floor

The subsystem must still compute a correct distance when every accelerator is
missing or broken.

### L4 — metric semantics are backend-neutral

For unit-cost edit distance:

- `distance` is the minimum number of insertions, deletions and substitutions;
- `similarity = max(len(a), len(b)) - distance`;
- `normalized_distance = distance / max(len(a), len(b))`;
- `normalized_similarity = 1 - normalized_distance`.

The empty/empty normalized similarity is `1.0`.

### L5 — explicit backend requests are observable

An unavailable explicit backend:

- warns and falls back when `strict=False`;
- raises `ImportError` when `strict=True`.

`KeyboardInterrupt`, `SystemExit`, and `MemoryError` are never swallowed.

### L6 — ranking is deterministic

Ties are ordered by input order after similarity and distance. `limit=0`
must not consume the choices iterable.

### L7 — provenance describes the implementation that actually ran

`Match.backend` and Corpus retrieval metadata must report the actual backend,
including runtime fallback to Python.

### L8 — Corpus is an optional bridge, not an import dependency

`make_corpus_scorer()` imports Corpus only inside the returned scorer.

### L9 — top-level discovery stays lazy

`scikitplot.levenshtein` must be reachable from the top-level lazy submodule
surface without forcing heavy API discovery or optional accelerators.

### L10 — examples and user guide are executable contracts

Gallery examples use deterministic local data and no network access. The
user-guide synchronization tests derive public names and known backend names
from runtime code.

## Known open work

`REVIEW.json` is authoritative.

Two current items are intentionally left open rather than hidden by the docs:

- `LV-001`: a runtime failure of the chosen safe accelerator falls directly to
  pure Python instead of trying the next remaining safe accelerator.
- `LV-002`: `rank()` resolves the selected backend for each choice; this can
  repeat capability probes and warnings.

Neither changes result correctness today. Both should be solved with one
per-call execution plan rather than ad-hoc exception nesting.

## When adding a backend

Do not add it directly to the `auto` chain.

First define:

1. license and redistribution posture;
2. import name and distribution name;
3. supported input domain;
4. distance semantics;
5. whether it is safe for automatic selection;
6. runtime-failure policy;
7. deterministic cross-backend conformance tests.

A backend may be supported explicitly without being eligible for `auto`.

## When changing ranking

Preserve:

```text
sort key = (-normalized_similarity, distance, original_index)
```

A `score_cutoff` is currently a backend-neutral **post-computation semantic
filter**, not a promise of backend-native pruning. If optimized cutoff support
is added, the returned match set must remain identical.

## When changing the Corpus bridge

The bridge is intentionally one-way:

```text
scikitplot.levenshtein  --lazy adapter-->  scikitplot.corpus
```

Do not import Corpus at facade import time.

## Evidence discipline

`STATE.json` can say runtime/maintenance PASS while release remains UNVERIFIED.
A lane that did not run is never upgraded to PASS.

Update `EVIDENCE.json` only from commands that actually ran.
