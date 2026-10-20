# `scikitplot.feature_selection` maintenance: active tasks

Rules that apply to every increment: [`skills/_engineering/`](../../../skills/_engineering/SKILL.md).
Release-note PR for this round: **866** (given by the maintainer, Rule 100).

## Task: FS-1, the first `scikitplot.feature_selection` (2026-10-19)

### Context

- **Goal.** A pandas-native subsystem that answers, for a wide table (100+
  columns), three questions people currently answer by hand:
  1. *What is each feature?* Role, physical family, semantic kind, and
     policy-backed traits ("am I skewed / discrete / constant / binary /
     identifier-like / high-cardinality ...?"), each with evidence and an
     explicit UNKNOWN / NOT_APPLICABLE state, fitted on training data only.
  2. *What happened to each feature in a fitted pipeline?* Which steps
     touched it, which outputs it became (one-hot children, polynomial
     terms, PCA components), where it was dropped, and how a transformed
     feature importance maps back to the raw feature.
  3. *Has it drifted?* A frozen training reference compared with later data,
     kind-aware (numeric bins, category frequencies, missingness).
- **Rationale.** The prototype `classify_variable_type` (supplied
  `simple_pure_pandas_based.txt`) answers question 1 with one mutable dict.
  Its defects, verified by reading it: ID-like columns are labelled
  constant (`unique_ratio >= uniq_thr` in the constant branch); `bin_enc_thr`
  compares a ratio with a count (25); docstring and signature defaults
  differ; `target_enc_cols` is unused; the column name `tenure` is hard
  coded; `feature` is reset to every column at the end; the categories are
  mutually exclusive although "skewed" and "continuous" are independent; a
  wall-clock timestamp sits inside the result; and nothing is fitted, so
  val/test re-profiling recategorises and leaks.
- **Reference, not ground.** `scikitplot_feature_selection_architecture_review_v3`
  (supplied) is read as a design checklist only. Its 15-subpackage layout
  is not adopted: Simplicity First keeps one flat package of small modules.

### Big picture

```text
FeaturePolicy (TOML/YAML/JSON, frozen dataclass, digest)
   │
   ▼
profile_features(X_train) / FeatureProfiler.fit(X_train)
   │   FeatureStats per column  ──►  assessment providers (registry)
   │                                 binning recommendation
   ▼
FeatureSchema (fitted, immutable, serialisable)
   ├── schema["age"].assess("skewed")  -> Assessment(state, evidence, reasons)
   ├── schema.select(kind=..., is_=..., tag=...) / groups() / summary()
   ├── schema.audit(X_val)             -> structural mismatches (no refit)
   ├── analysis_plan / recommend_treatments (advisory rule tables)
   └── ReferenceProfile.fit(X_train) -> drift(X_new)
            │
sklearn Pipeline / ColumnTransformer (FeatureColumnSelector, FeatureFilter,
FeatureRenamer, FeatureProfiler as steps; any sklearn transformer)
            │
trace_pipeline(fitted) -> FeatureLineage (stage DAG)
   ├── origins / descendants / history / dropped / stage_summary
   └── attribute(importances) -> raw-feature importances (exact only)
```

### Invariants

1. **Fit on train, observe later.** Every data-derived decision (kind,
   assessment, bin edges, selection mask, rename map, drift reference) is
   learned in `fit`; `transform` and `audit` never refit.
2. **UNKNOWN is not FALSE.** Too little evidence is `unknown`; a question
   that does not apply is `not_applicable`. Selections treat only `true` as
   true.
3. **Multi-axis.** Role, family, kind, assessments and user tags are
   independent fields of one `FeatureSpec`.
4. **One threshold, one place.** Kind inference reads the `binary` and
   `discrete` assessments; there is no second threshold table.
5. **Manual overrides keep the evidence.** An override records
   `origin="manual"` and the inferred state it replaced.
6. **Lineage is never fabricated.** Exact edges come from fitted
   attributes or class contracts; PCA-like steps are `mixed`; anything else
   is `unknown`. Attribution sums only single-origin, non-unknown paths and
   returns the rest as unattributed.
7. **Width reconciliation.** A traced stage's output count equals
   `len(get_feature_names_out())` and, when a sample is given, the actual
   transformed width; a disagreement raises `LineageError`.
8. **Config is policy; fitted state is separate.** Callables are never
   serialised; every write is verified by reading it back (Rule 107).

### Failure modes handled explicitly

Empty frame; all-null column; non-string or duplicate column labels;
unhashable cells (lists/dicts); infinities; nullable and pyarrow-backed
dtypes; ordered categoricals; booleans; tiny `n` (skewness needs a floor);
missing / extra / reordered / retyped columns at transform; unknown
transformers in a pipeline; `remainder` as a string (sklearn < 1.7) or a
`FunctionTransformer` (newer); duplicate output names; YAML aliases and
dates; TOML's lack of null.

### Module map (`scikitplot/feature_selection/`)

| Module | Responsibility |
|---|---|
| `_types.py` | enums: state, family, kind, role, relation, lineage quality |
| `_errors.py` | error types (policy, schema mismatch, incompatible schema, lineage) |
| `_serialize.py` | json / jsonl / yaml / toml text I/O, strict and verified |
| `_policy.py` | `ColumnRule`, `AssessmentSpec`, `BinningPolicy`, `DriftPolicy`, `FeaturePolicy` |
| `_stats.py` | `FeatureStats`, dtype family, one pass per column |
| `_assessments.py` | `Assessment`, `AssessmentContext`, provider registry, built-ins |
| `_binning.py` | `BinningRecommendation`, Knuth and NumPy bin rules |
| `_schema.py` | `FeatureSpec`, `FeatureSchema`, `profile_features` |
| `_rename.py` | `RenamePlan` (dict / list / callable / template) |
| `_estimators.py` | `FeatureProfiler`, `FeatureFilter`, `FeatureColumnSelector`, `FeatureRenamer` |
| `_lineage.py` | `trace_pipeline`, `FeatureLineage`, `Attribution` |
| `_drift.py` | `ReferenceProfile`, `drift` report |
| `_analysis.py` | `analysis_plan`, `recommend_treatments` |

### Implementation steps

- [x] 1. Plan (this section).
- [x] 2. Core contracts: types, errors, serialisation, policy.
- [x] 3. Profiling: stats, assessments, binning, schema.
- [x] 4. Estimators and rename plan.
- [x] 5. Lineage tracer and attribution.
- [x] 6. Drift and analysis routing.
- [x] 7. Tests.
- [x] 8. Repository integration: root `_submodules`, stub, Towncrier
      section and fragment 866, labeler, maintenance plane, skill.
- [x] 9. Verification: newest and floor dependency sets, doctests,
      formatter, repository checkers, independent review (Rule 92).

### Acceptance criteria

- [x] Every prototype defect above has a regression test that the new API passes.
- [x] A fitted `FeatureProfiler` in a `Pipeline` survives `clone`, `pickle`,
      `GridSearchCV` and `set_output(transform="pandas")`; validation data
      never changes the schema.
- [x] `trace_pipeline` reconciles widths on a pipeline with
      `ColumnTransformer` (imputer with indicator, scaler, one-hot with drop
      and infrequent), `PolynomialFeatures`, `SelectKBest`, `PCA` in a
      `FeatureUnion`, and remainder passthrough; attribution of a one-hot
      importance sums to the raw column exactly.
- [x] A policy round-trips through JSON, YAML and TOML equal to itself;
      off-contract records (unknown key, bool for int, YAML date, alias) are
      refused.
- [x] Suite passes on the newest dependencies and on Python 3.11 with
      NumPy 2.0 / pandas 2.2.2 / scikit-learn 1.4.2.

### Risks and mitigation

- sklearn private attributes move between releases → use public fitted
  attributes only (`categories_`, `drop_idx_`, `infrequent_categories_`,
  `n_bins_`, `powers_`, `statistics_`, `indicator_.features_`,
  `output_indices_`, `get_support`), and check every adapter's count
  against `get_feature_names_out`.
- Profiling cost on wide tables → one pass per column, no pairwise work by
  default.
- Scope creep → plotting and statistical tests are *routed* (named), not
  executed; execution stays with `scikitplot.stats`, plotting modules and
  sklearn.

## Results review - FS-1 (2026-10-19)

### Implementation summary

- New package `scikitplot/feature_selection/` (13 modules, pure Python) and
  its suite `tests/` (11 files).
- Repository planes: root `_submodules` entry; Towncrier section
  (`generate_towncrier_sections.py sync --apply`) and fragment
  `scikitplot.feature_selection/866.feature.rst` (one bullet); labeler
  (`generate_pr_labeler.py sync --apply`); API reference blueprint entry and
  regenerated `docs/source/apis_reference.py` (`rebuild --apply`); user guide
  `docs/source/user_guide/feature_selection/` (5 pages and an executing sync
  test) linked from the user-guide index; gallery
  `galleries/examples/feature_selection/`; this maintenance plane and
  `skills/feature_selection/SKILL.md`. The root `__init__.pyi` lists no
  subpackages, so it needed no change (plan step 8 named it; no deviation in
  behaviour).

### Verification evidence

| Lane | Environment | Result |
|---|---|---|
| Source tree, newest | Python 3.13, NumPy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, SciPy 1.18.1, pyarrow, pydantic, PyYAML, tomli-w | 408 passed (suite, doctests, guide), 0 skipped |
| Floor | Python 3.11, NumPy 2.0.0, pandas 2.2.2, scikit-learn 1.4.2, SciPy 1.13.1 (Rule 11 pair) | 406 passed, 2 skipped (pydantic, pyarrow not installed there) |
| Oldest Python | Python 3.9.25, same floor pins, tomli, pydantic | 407 passed, 1 skipped (pyarrow) — needed `Optional[int]` for pydantic on 3.9 and a FeatureUnion test guarded for scikit-learn < 1.5 |
| Installed full build | meson build of a copy (placeholder `CITATION.bib`/`README.md`, absent from the upload), final pure-Python package copied over the installed one, repo `pytest.ini` + `conftest.py`, warnings as errors | 405 passed + guide 3 passed; earlier `-n 4` ×3 green |
| User guide | every `python` block executed in order; TOML example loaded | 3 passed |
| API reference | `rebuild` reproducible; runtime `check --strict-coverage --module scikitplot.feature_selection` | stale=0 missing=0 |
| Repository checks | `generate_towncrier_sections.py check`, `generate_pr_labeler.py check`, `check_path_lengths.py check` | pass; `check_records.py` output identical before and after (its router/rules/archives failures exist on the uploaded tree) |
| Formatting | black 26.5.1 (repo config), codespell (repo ignore file absent from upload, default dictionary) | clean |
| Performance | 500 columns x 10,000 rows | 1.2 s default, 0.9 s without binning, 6.2 s with `rule="knuth"` |

Pre-existing, not touched: `test_api_reference_sync.py[scikitplot.cexternals]`
fails on the uploaded tree too (stale `_f2py`, `_numcpp`).

### Independent review (Rule 92)

A reviewer who had not seen the work probed a snapshot read-only, with
ground truth from perturbing one raw column and watching which outputs move.
Confirmed correct: SplineTransformer order, PolynomialFeatures, KBins one-hot,
OneHotEncoder with every `drop` / infrequent / `max_categories` combination,
TargetEncoder, selectors, imputer indicators on all-NaN columns, every
ColumnTransformer column-spec and remainder form, sparse output, `set_output`,
fold-local selection under `GridSearchCV`, PSI and JS against SciPy.
Ten findings, each fixed with a regression in `tests/test_fs_review_findings.py`
(31 of its 32 tests fail on the reviewed snapshot):

| # | Severity | Finding | Fix |
|---|---|---|---|
| 1 | high | categorical `value_counts()` counted unused categories (constant/binary wrong) | count positive entries only |
| 2 | high | container without names invented duplicate names; attribution by name merged them | refuse duplicate fallback names (`LineageError`); refuse name-aligned values when names repeat |
| 3 | high | `Normalizer`, `TfidfTransformer`, KNN/iterative imputers marked exact one-to-one | `_COLUMNWISE` allowlist; other one-to-one classes and model-based imputers are `mixed`; per-output qualities keep indicators exact |
| 4 | medium | `FeatureLineage.dumps("toml")` wrote what the loader refused | omitted `None` keys accepted; every dump verified by reading back |
| 5 | medium | lineage loader accepted bad ids and types | per-field record contract (`_record`, `_ids`), order and stage checks |
| 6 | medium | other loaders: wrong-length proportions, reversed cuts, nested params, string reasons, inconsistent names/positions | explicit checks in each `from_dict` |
| 7 | medium | drift parsed numeric strings as numbers | reference records the training dtype family; another family is `type_changed` |
| 8 | medium | a final model with `transform` (LDA, KMeans) was traced as a transformer | last top-level step with `predict` is the model; `trace_final=True` opts in |
| 9 | low | a manual `binary=false` did not change a boolean's kind | respected (`categorical_nominal`) |
| 10 | low | NumPy integers in a policy broke `digest` | sections and params store built-in `int`/`float` |

Re-review of the fixes: all ten reproductions now give the expected result;
the `_COLUMNWISE` allowlist matches every `OneToOneFeatureMixin` class of
scikit-learn 1.9 (nine column-wise, `Normalizer` and `TfidfTransformer`
mixed). Three new findings, fixed with regressions that fail on the
re-reviewed snapshot (4 of 4):

| # | Severity | Finding | Fix |
|---|---|---|---|
| R1 | medium | a final step that is itself a `Pipeline` was taken as the model | recurse; its own last `predict` step is the model |
| R2 | low | equal-width edges narrower than float spacing repeated, so `dumps` refused a fitted schema | unique edges, reason `tied_edges_merged` |
| R3 | medium | references without `family` refused; all-missing or object-dtype numbers reported `type_changed` | missing `family` loads as unknown; no family check without values; object columns of numbers are `integer`/`float` |

Not changed: `check_estimator(FeatureRenamer())` disagrees with scikit-learn's
generic `n_features_in_` message (it raises `SchemaMismatchError`, a
`ValueError`); a convention, recorded here for a later round.

### Deviations from plan

- No `__init__.pyi` change (the root stub lists no subpackages).
- Added during the round: `AssessmentRegistry.copy()`, `trace_final=`,
  `FeatureMapping.output_qualities`, identifier-like features skipped by
  drift, integer identifiers opt-in (`include_integers`).

### Technical debt

- Profiling is single-threaded; 500 x 10,000 takes about 1 s (6 s with Knuth).
- Datetime and text features are not compared by `ReferenceProfile`.
- Lineage of third-party transformers without `get_feature_lineage` is
  `unknown` by design; adapters for category_encoders / feature-engine are
  future work.
