.. _preprocessing-known-limits-and-troubleshooting:

Known limits and troubleshooting
================================

Train/test columns differ
   Fit once on training data and transform every later split with the same
   fitted encoder.

Unexpected all-zero category
   Check unknown-category policy and whether the value was present at fit time.

Duplicate/collapsed categories
   The current maintenance snapshot records an open cross-feature category
   collapse defect in ``DummyCodeEncoder``. Prefer focused regression coverage
   for multi-column/category-overlap cases.

Infrequent grouping differs from expectation
   An open finding records that the configured grouping path is not fully
   applied. Treat it as experimental behavior until repaired.
