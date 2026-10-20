.. currentmodule:: scikitplot.feature_selection

.. _feature_selection-lineage:

Pipelines and lineage
=====================

Route columns by what they are
------------------------------

:class:`FeatureColumnSelector` is a ``ColumnTransformer`` column selector. It
profiles the frame it is given at fit time, which inside cross-validation is
the training fold, and ``ColumnTransformer`` keeps the selection for later
data:

.. code-block:: python

   from sklearn.compose import ColumnTransformer
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
   from sklearn.linear_model import LogisticRegression
   from scikitplot.feature_selection import FeatureColumnSelector, FeatureFilter

   pre = ColumnTransformer(
       [
           ("skewed", PowerTransformer(),
            FeatureColumnSelector(kind="numeric_continuous", is_=("skewed",))),
           ("numeric", StandardScaler(),
            FeatureColumnSelector(kind=("numeric_continuous", "numeric_discrete"),
                                  is_not=("skewed",))),
           ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            FeatureColumnSelector(kind="categorical_nominal")),
       ]
   )
   model = Pipeline([("filter", FeatureFilter()), ("pre", pre),
                     ("lr", LogisticRegression(max_iter=1000))])

:class:`FeatureFilter` removes constant and identifier-like columns and every
column whose role is not ``feature`` (so a target column cannot leak), with
``get_support`` and a ``report_`` of reasons. :class:`FeatureProfiler` keeps the
training schema inside the pipeline, and :class:`FeatureRenamer` applies a
rename plan (mapping, list, callable or ``"col-{index}"`` template) frozen at
fit time.

Trace what happened
-------------------

.. code-block:: python

   from scikitplot.feature_selection import trace_pipeline

   y_train = X_train.pop("churn")
   model.fit(X_train, y_train)
   lineage = trace_pipeline(model, X=X_train.head(50))

   lineage.stage_summary()          # one row per step: relation, n_in -> n_out
   lineage.treatments()             # raw column -> chain of transformers
   lineage.history("income")        # every step that touched it
   lineage.origins("categorical__city_izmir")   # ('city',)
   lineage.descendants("city")      # the model inputs derived from it
   lineage.dropped()                # {'customer_id': 'filter'}

Passing ``X=`` runs every stage on the sample and checks its real output width
against the traced count; a disagreement raises :class:`LineageError`.

Lineage quality is explicit. Edges taken from fitted attributes or
column-wise class contracts are ``exact``. Steps whose outputs combine
several inputs are ``mixed`` even when the names stay one-to-one: PCA,
``Normalizer`` and ``TfidfTransformer`` (each row is scaled by a norm over all
columns), and KNN or iterative imputers (filled values come from other
columns; their missing indicators stay exact). Steps without any contract are
``unknown``. Neither is attributed to a single raw column. In a top-level
pipeline the last step with ``predict`` is the model and is not traced, even
if it also has ``transform``; pass ``trace_final=True`` to trace it. Give your own transformer
a ``get_feature_lineage(input_features)`` method returning a
:class:`FeatureMapping` to make it exact.

Importances per raw column
--------------------------

.. code-block:: python

   import numpy as np

   result = lineage.attribute(np.abs(model[-1].coef_[0]))
   result.by_input        # one value per raw column, one-hot columns summed
   result.unattributed    # interactions, PCA components, unknown steps

It works with any library's importances or SHAP values (a 2-D array is summed
per sample) as long as they are aligned with ``lineage.output_names``.
