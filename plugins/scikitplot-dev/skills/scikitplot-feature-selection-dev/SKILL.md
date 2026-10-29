---
name: scikitplot-feature-selection-dev
description: Maintain scikitplot.feature_selection; protect train-only fitting, four-valued assessments, single-source thresholds, exact-or-declared pipeline lineage with width reconciliation, strict persisted artefacts, and sklearn clone/CV behaviour.
metadata:
  scikitplot-mirror: "feature_selection"
---

# `scikitplot.feature_selection` maintainer

## Read first

1. `maintenances/feature_selection/MAINTAINING.md` (module map, invariants)
2. `maintenances/feature_selection/todo/todo.md` (active work, last results)
3. `maintenances/feature_selection/todo/lessons.md`
4. `plugins/_shared/skills/scikitplot-engineering-rules/SKILL.md` and the rules it names (contracts, testing, packaging)

## When changing assessments

- Add or change a built-in in `_assessments.py` only; `DEFAULT_ASSESSMENTS` in
  `_policy.py` lists names, never parameters.
- Bump the provider `version` when the meaning of an answer changes.
- Return `not_applicable` when the question has no meaning and `unknown` when
  evidence is short; add a row for each to `test_builtin_states`.
- Validate parameter ranges in the provider's `validate` so a bad policy
  fails before any column is profiled.

## When adding a lineage adapter

- Use public fitted attributes only, and let the tracer compare your count
  with `get_feature_names_out`; never infer parents from output names.
- Add the transformer to `test_known_adapters_are_exact_and_reconcile` with a
  sample `X`, and check its oldest supported scikit-learn behaviour
  (`remainder` was a string before 1.7).
- An adapter that cannot be exact must return `None` so the step is `unknown`.

## When changing a persisted artefact

- Bump its version constant, keep the old reader refusing with
  `IncompatibleSchemaError`, and extend the strict `from_dict`.
- Run the json, yaml and toml round-trip tests; anything one format cannot hold
  must fail at write time.

## Estimators

Constructor arguments are stored verbatim (`clone`). Nothing that cannot be
deep-copied (the registry, locks) is a parameter. `transform` never refits.
Test every change under `cross_val_score`, `GridSearchCV`, `pickle` and
`set_output(transform="pandas")`.
