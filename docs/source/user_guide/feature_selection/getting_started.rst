.. currentmodule:: scikitplot.feature_selection

.. _feature_selection-getting-started:

Getting started
===============

Profile the training split:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from scikitplot.feature_selection import profile_features

   rng = np.random.default_rng(0)
   X_train = pd.DataFrame(
       {
           "customer_id": [f"c{i}" for i in range(500)],
           "income": rng.lognormal(10, 1, 500),
           "kids": rng.integers(0, 4, 500),
           "city": rng.choice(["ankara", "izmir", "bursa"], 500),
           "churn": rng.integers(0, 2, 500),
       }
   )
   schema = profile_features(X_train, {"roles": {"target": ["churn"]}})

Every column is now a :class:`FeatureSpec` that can be asked about itself:

.. code-block:: python

   spec = schema["income"]
   spec.kind                  # 'numeric_continuous'
   spec.is_("skewed")         # True only when the state is "true"
   spec.state("skewed")       # 'true' | 'false' | 'unknown' | 'not_applicable'
   spec.assess("skewed").evidence_dict   # {'direction': 'right', 'skewness': ..., ...}
   print(spec.explain())

``unknown`` (too little evidence) and ``not_applicable`` (the question has no
meaning, such as the skewness of a string) are never treated as ``false``.

Group and select:

.. code-block:: python

   schema.groups(by="kind")         # {'categorical_nominal': [...], ...}
   schema.groups(by="assessment")   # {'skewed': ['income'], ...}
   schema.select(kind="numeric_continuous", is_not=("skewed",))
   schema.summary()                 # one row per feature
   schema.feature_names             # role == "feature" only; the target is excluded

Plan the analysis and the preprocessing; neither runs anything:

.. code-block:: python

   from scikitplot.feature_selection import analysis_plan, recommend_treatments

   analysis_plan(schema, target="churn")   # plots, summaries, scipy.stats tests
   recommend_treatments(schema)            # ordered sklearn steps per feature

Keep the schema and check later data against it:

.. code-block:: python

   schema.save("schema.json")
   X_test = X_train.sample(100, random_state=1)
   schema.audit(X_test)     # missing / extra / reordered / retyped columns
