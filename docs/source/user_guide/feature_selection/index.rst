.. currentmodule:: scikitplot.feature_selection

.. _feature_selection-index:

Feature Selection
=================

Feature profiling, lineage and drift
----------------------------------------------------------------------

``scikitplot.feature_selection`` helps with tables that have too many columns
to reason about one by one. It answers three questions that are usually
answered by hand and forgotten:

* **What is each feature?** Role, dtype family, semantic kind, user tags, a bin
  recommendation, and policy-backed assessments such as *skewed*, *discrete*,
  *constant*, *binary* or *identifier-like*, each with evidence and an explicit
  ``unknown`` / ``not_applicable`` state.
* **What happened to each feature in a fitted pipeline?** Which steps touched
  it, which columns it became, where it was dropped, and how a model's
  importances map back to raw columns.
* **Has it drifted?** A frozen training reference compared with later data.

.. important::

   Profile the **training** split only. Validation, test and production data
   are audited against the fitted schema and the frozen reference; they are
   never profiled into them.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   configuration
   pipelines_and_lineage
   drift

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Getting started
      :link: feature_selection-getting-started
      :link-type: ref

      Profile a table, ask features about themselves, group and select them.

   .. grid-item-card:: Configuration
      :link: feature_selection-configuration
      :link-type: ref

      Policies in TOML, YAML or JSON; thresholds; overrides; custom assessments.

   .. grid-item-card:: Pipelines and lineage
      :link: feature_selection-lineage
      :link-type: ref

      Route columns in a ColumnTransformer, trace a fitted pipeline, attribute importances.

   .. grid-item-card:: Drift
      :link: feature_selection-drift
      :link-type: ref

      Freeze a training reference and judge later data feature by feature.

Examples
--------

See :ref:`feature_selection_examples` for executable examples that accompany this guide.
