.. _impute-ann-imputer:

ANNImputer
==========

:class:`ANNImputer` builds an approximate vector index and retrieves candidate
neighbors to fill missing features. The implementation supports Annoy and
Voyager-oriented paths and exposes controls such as ``n_neighbors``, ``metric``,
``n_trees``, ``search_k``, weighting, initial strategy, and on-disk index build.

Approximation and preprocessing
-------------------------------

Rows with missing values still need a query representation suitable for the
selected vector backend. Initial imputation/normalization choices therefore
influence which neighbors are retrieved. Treat those choices as part of the
model pipeline and evaluate them using held-out data.

Do not compare ANN and exact KNN results by expecting every imputed value to be
identical. Compare downstream quality, latency, memory, and reproducibility under
a fixed configuration.
