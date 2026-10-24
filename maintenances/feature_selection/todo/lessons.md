# `scikitplot.feature_selection`: lessons that only this area needs

- **TOML has no null, so `None` means "default" everywhere a value can be
  persisted.** Policy parameters, provider parameters recorded in an
  `Assessment`, and evidence values that are `None` are omitted, never stored.
  Found when the first TOML round trip of a schema failed on
  `max_unique_ratio = None` (FS-1). Check: `test_schema_round_trips[toml]`.
- **A lone surrogate is valid in a Python string and invalid in a UTF-8 file.**
  `tomli-w` writes it raw; `_serialize.dumps` therefore encodes the text
  strictly before returning. Check: `test_toml_refuses_a_lone_surrogate_at_write_time`.
- **`SimpleImputer(add_indicator=True)` keeps an all-missing column alive as its
  indicator.** The values are dropped, the indicator survives, so the raw
  column is not "dropped" in lineage. Check:
  `test_simple_imputer_dropping_an_empty_column`.
- **Unique integers are not identifiers by default.** Integer measurements
  (amounts, counts) are often all distinct in small samples;
  `identifier_like` checks integers only with `include_integers=True`, and
  integer keys should be declared with `roles.identifier`.
- **`OneToOneFeatureMixin` is a naming contract, not a dependency contract.**
  `Normalizer` and `TfidfTransformer` keep one output per input name but scale
  each row by a norm over all columns; KNN and iterative imputers fill a
  column from the others. Only the column-wise classes in `_COLUMNWISE` are
  exact. Found by the FS-1 review (finding 3) by perturbing one raw column.
  Check: `test_steps_that_combine_columns_are_not_exact`.
- **A categorical's `value_counts()` lists unused categories with count 0.**
  Count only positive entries. Found by the FS-1 review (finding 1).
- **`to_numpy(dtype="float64")` parses numeric strings.** A drift reference
  records the training dtype family and reports another family as
  `type_changed` instead of comparing parsed values (finding 7).
