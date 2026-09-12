---
name: preprocessing-maintainer
description: Maintain scikitplot.preprocessing GetDummies and DummyCodeEncoder; protect feature-local category identity, token expansion, unknown/drop/infrequent semantics, inverse transforms, sklearn estimator/set_output integration, sparse/dense outputs, tests, and release evidence.
---

# `scikitplot.preprocessing` maintainer

Work from the wide checkout containing `scikitplot/`, `maintenances/`, and `skills/`. This skill owns the two encoders shipped by `scikitplot.preprocessing`, not pandas, sklearn internals, SciPy sparse matrices, or root-package version infrastructure.

## Read first

1. `maintenances/preprocessing/MAINTAINING.md`
2. `maintenances/preprocessing/_maintenance/FRESH_CHAT_HANDOFF.md`
3. `maintenances/preprocessing/_maintenance/STATE.json`
4. `maintenances/preprocessing/_maintenance/FAMILY.md`
5. `maintenances/preprocessing/_maintenance/VERIFICATION.md`

Then run:

```sh
python -B maintenances/preprocessing/_maintenance/check_trackers.py --json
python -B maintenances/preprocessing/_maintenance/review_subsystem.py --json
python -B -m pytest maintenances/preprocessing/_maintenance/tests -q -p no:cacheprovider
```

## Treat `GetDummies` and `DummyCodeEncoder` separately

They intentionally differ. `GetDummies` is pandas-backed, can preserve non-encoded columns in dense DataFrame output, and returns only the numeric dummy block in sparse mode. `DummyCodeEncoder` is a sklearn-style categorical encoder with all input features participating in its encoded matrix. Do not force superficial parity between them.

## Feature identity is `(feature, category)`

This is the load-bearing `DummyCodeEncoder` invariant. Never key a global output mapping by raw category value alone. `"x"` in input column A and `"x"` in input column B are different encoded dimensions. For every fit/transform configuration, assert that transformed width, `get_feature_names_out()`, `_n_features_outs`, drop offsets, and inverse decoding agree. Current finding **PRE-DCE-001** is a critical violation of this rule.

## Infrequent categories must affect the actual matrix

`min_frequency` and `max_categories` are not metadata-only features. Once fitted, the same grouping must be reflected in the returned matrix, feature names, drop bookkeeping, `infrequent_categories_`, inverse transform, and unknown-category behavior. Current finding **PRE-DCE-002** tracks a runtime where bookkeeping says categories are grouped but transform still returns separate raw-category columns.

Add tests that assert values as well as shapes. In particular, two infrequent raw labels must activate the same grouped output dimension, not merely reduce `_n_features_outs` on paper.

## Validate documented parameter domains

Enum-like public parameters must reject unsupported values predictably. For `GetDummies`, `handle_unknown` currently documents only `"error"` and `"ignore"`; arbitrary strings/objects must not silently behave as ignore. Prefer sklearn-compatible parameter validation or an explicit fit-time check. Keep clone/get_params semantics intact by storing constructor arguments verbatim.

## Preserve unknown/drop/inverse consistency

For each `handle_unknown` mode, test known + unknown tokens across multiple features. For every `drop` mode, test dense and CSR outputs, feature names, and inverse transform. Unknowns encoded as all-zero feature groups must decode according to the documented contract. A test that only checks “no exception” is not sufficient.

## Separator and token semantics are public behavior

Literal separators, regex separators, and callables must remain deterministic. Preserve treatment of whitespace, duplicate tokens within a cell, `None`/NaN/infinite values, numeric categories, manual categories, and mixed dtypes. A callable separator is user code: propagate its exceptions rather than disguising them as category errors.

## Sklearn integration is part of the API

Maintain `BaseEstimator` constructor discipline, `clone`, `get_params`, `feature_names_in_`, `n_features_in_`, `get_feature_names_out`, `Pipeline`, and `set_output`. Dense pandas/polars output adapters and sparse restrictions must follow the sklearn version actually supported by the project. Avoid copying private sklearn behavior without tests because private utility APIs can move between releases.

## Keep optional/container dependencies honest

`pandas` is operationally required for `GetDummies`, but it should not become an accidental import-time requirement for unrelated code paths beyond the submodule's documented dependency policy. Test supported pandas/sklearn/scipy version ranges in complete-package CI. Missing optional container libraries such as polars/pyarrow are `UNAVAILABLE` evidence unless the project declares them required.

## Test isolation claims must be executable

The focused tests contain `_helpers.install_scikitplot_stub`, but current bootstrap calls and path setup are commented while docstrings claim direct standalone execution. Either maintain a real standalone mode or document only package/module pytest execution. Never rely on commented bootstrap code as evidence. This is tracked as **PRE-TEST-001**.

## Evidence ladder

Keep these distinct:

1. static contract + mutation tests;
2. GetDummies focused tests;
3. DummyCodeEncoder focused tests;
4. advanced/cross-encoder tests;
5. negative semantic probes for duplicate labels, infrequent grouping, invalid parameters, width/name/inverse consistency;
6. complete-package installed import and supported dependency-version tests;
7. docs/example and supported-platform evidence.

Release remains blocked if the feature-identity or infrequent-matrix contracts are red, documented parameter modes are silently ignored, or level 6 is unavailable.

## Do not bless contradictory metadata

A suite can pass while fitted metadata and returned matrices disagree. Before closing any encoder change, compare actual matrix width and values against feature-name metadata and inverse-transform expectations on adversarial multi-feature inputs. If those disagree, runtime remains red regardless of nominal unit-test counts.

## Runtime edits require focused regressions

When repairing an owned finding, add the smallest regression that would have failed before the change, then rerun all three focused test modules and the maintenance mutation suite. Record dependency versions with the evidence so private sklearn/pandas behavior is not mistaken for a timeless contract.
