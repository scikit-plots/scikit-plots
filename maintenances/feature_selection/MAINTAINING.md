# Maintaining `scikitplot.feature_selection`

`scikitplot.feature_selection` profiles wide pandas tables into a fitted
`FeatureSchema`, traces fitted scikit-learn pipelines into a `FeatureLineage`,
and freezes drift references. Runtime: `scikitplot/feature_selection/`
(pure Python on pandas, NumPy, SciPy and scikit-learn; PyYAML and `tomli-w`
optional). Skill: [`plugins/scikitplot-dev/skills/scikitplot-feature-selection-dev/SKILL.md`](../../plugins/scikitplot-dev/skills/scikitplot-feature-selection-dev/SKILL.md).

Start a session with [`maintenances/README.md`](../README.md), then the active
[`todo/todo.md`](todo/todo.md) and this area's [`todo/lessons.md`](todo/lessons.md).

## Module map

| Module | Owns |
|---|---|
| `_types.py` | the closed vocabularies (state, family, kind, role, relation, quality) |
| `_errors.py` | error types; every one also derives from the built-in a caller expects |
| `_serialize.py` | json / jsonl / yaml / toml, each write verified by reading it back |
| `_policy.py` | `FeaturePolicy` and its sections; strict `from_dict` |
| `_stats.py` | `FeatureStats`, dtype family, one pass per column |
| `_assessments.py` | providers, the shared registry, built-in questions |
| `_binning.py` | bin recommendations (Knuth exhaustive search, NumPy rules) |
| `_schema.py` | `FeatureSpec`, `FeatureSchema`, `profile_features` |
| `_rename.py` | `RenamePlan` |
| `_estimators.py` | `FeatureProfiler`, `FeatureFilter`, `FeatureColumnSelector`, `FeatureRenamer` |
| `_lineage.py` | `trace_pipeline`, `FeatureLineage`, `FeatureMapping`, `Attribution` |
| `_drift.py` | `ReferenceProfile` |
| `_analysis.py` | `analysis_plan`, `recommend_treatments` (advisory tables) |

## Invariants (each has tests in `scikitplot/feature_selection/tests/`)

1. Data-derived state is learned in `fit` / `profile_features` only; `transform`,
   `audit` and `drift` never refit.
2. `unknown` and `not_applicable` are never `false`; selections count only `true`.
3. Default provider parameters are defined once, on the providers; the policy's
   built-in specs carry none. Kind inference reads the `binary` and `discrete`
   assessments (after overrides).
4. A manual override keeps `inferred_state`.
5. Lineage edges come from an estimator's `get_feature_lineage`, public fitted
   attributes or column-wise class contracts; names are never parsed. A step
   whose values combine columns (row-wise `OneToOneFeatureMixin` classes,
   KNN/iterative imputers, PCA-like) is `mixed`; a step without a contract is
   `unknown`. Neither is attributed. Ground truth for an adapter: perturb one
   raw column and see which outputs change (`_changed_outputs` in
   `test_fs_review_findings.py`).
6. Every traced stage's count equals `len(get_feature_names_out())`, and the
   observed width when a sample is given; otherwise `LineageError`.
7. Every persisted artefact has `schema_version` and `artifact`; another
   generation raises `IncompatibleSchemaError`; every field is type-checked.
8. Values that a format cannot hold are refused at write time (Rule 107).

## Verification

```sh
python -B -m pytest scikitplot/feature_selection -q --doctest-modules -p no:cacheprovider
python -I scikitplot/_build_utils/generate_apis_reference/__init__.py --module scikitplot.feature_selection check --strict-coverage
```

Run once with the optional libraries present (`pyyaml`, `tomli-w`, `pyarrow`,
`pydantic`) and once at the dependency floor (see the results in `todo/todo.md`).
