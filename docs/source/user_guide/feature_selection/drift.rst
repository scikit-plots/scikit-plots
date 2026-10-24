.. currentmodule:: scikitplot.feature_selection

.. _feature_selection-drift:

Drift
=====

:class:`ReferenceProfile` freezes, per feature, the buckets of the training
data: quantile bins for continuous numbers, the most frequent values for
binary, discrete and categorical features, and one bucket for missing values.
Later data is put into the *same* buckets.

.. code-block:: python

   from scikitplot.feature_selection import ReferenceProfile

   reference = ReferenceProfile.fit(X_train, schema)
   reference.save("reference.json")

   later = X_train.assign(income=X_train["income"] * 2)
   report = reference.drift(later)
   report[["feature", "psi", "js_distance", "missing_delta", "unseen_ratio", "status"]]

Statuses are ``stable``, ``warn``, ``drift`` (population stability index at
``drift.psi_warn`` / ``drift.psi_alert`` of the policy), ``insufficient``
(fewer than ``drift.min_count`` values), ``missing_column``, ``type_changed`` and
``not_applicable`` (datetime, text, other kinds).

For transformed features, fit a reference on ``pipeline[:-1].transform(X_train)``
and map findings back to raw columns with :meth:`FeatureLineage.origins`.
