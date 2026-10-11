.. _preprocessing-sklearn-integration:

Scikit-learn integration
========================

Both encoders follow estimator conventions closely enough to participate in
pipeline-style workflows. Keep the encoder inside the fitted preprocessing
pipeline so category state is learned only from training data.

Use ``get_feature_names_out`` when downstream diagnostics or model explainers
need a stable mapping from encoded columns back to source features. Sparse
output can reduce memory use for high-cardinality categories, but downstream
estimators must accept sparse input.

Do not call ``fit_transform`` independently on train and test sets.
