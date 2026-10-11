.. currentmodule:: scikitplot.preprocessing

.. _preprocessing-index:

Categorical preprocessing
=========================

``scikitplot.preprocessing`` currently provides two categorical encoders: ``GetDummies`` for pandas-style dummy expansion and ``DummyCodeEncoder`` for a fitted scikit-learn-style transform/inverse-transform workflow.

.. important::

   Fit categorical state on training data and reuse it for validation/test data. Creating dummies independently for each split can produce incompatible columns and is a common source of silent model errors.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   get_dummies
   dummy_code_encoder
   sklearn_integration
   python_api
   known_limits_and_troubleshooting

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Getting started
      :link: preprocessing-getting-started
      :link-type: ref

      Choose pandas-style GetDummies or sklearn-style DummyCodeEncoder.

   .. grid-item-card:: GetDummies
      :link: preprocessing-get-dummies
      :link-type: ref

      Column selection, separators, drop behavior, sparse output, dtype, and unknown categories.

   .. grid-item-card:: DummyCodeEncoder
      :link: preprocessing-dummy-code-encoder
      :link-type: ref

      Fit/transform/inverse-transform categorical encoding with estimator semantics.

   .. grid-item-card:: Scikit-learn integration
      :link: preprocessing-sklearn-integration
      :link-type: ref

      Pipelines, feature names, training/test separation, and sparse output.

   .. grid-item-card:: Python API
      :link: preprocessing-python-api
      :link-type: ref

      The two public encoder classes and their main contracts.

   .. grid-item-card:: Known limits and troubleshooting
      :link: preprocessing-known-limits-and-troubleshooting
      :link-type: ref

      Current correctness findings and validation mismatches.

Which page should I use?
------------------------

Start with **Getting started**. Then choose the page that owns the decision
you are making: architecture and lifecycle, configuration, persistence,
performance, public API, or limits. The guide intentionally separates those
concerns so examples do not accidentally become policy.

Examples
--------

See :ref:`preprocessing_examples` for executable examples that accompany this guide.
