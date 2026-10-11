.. currentmodule:: scikitplot.decile

.. _decile-index:

Decile analysis
===============

``scikitplot.decile`` groups model scores into ordered quantiles/ntiles so you can explain response concentration, lift, cumulative gain, KS separation and business value to technical and non-technical stakeholders.

.. important::

   A decile plot only makes sense when the score column and positive-class semantics are correct. Verify ``pos_label``/``class_index`` and score ordering before interpreting lift, gain, or financial curves.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   kds
   modelplotpy
   business_plots_and_financial
   compatibility_and_limits

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Getting started
      :link: decile-getting-started
      :link-type: ref

      Create a decile table or model plot from labels and predicted scores.

   .. grid-item-card:: KDS functions
      :link: decile-kds
      :link-type: ref

      Decile tables, lift, cumulative gain, KS statistic, and reports.

   .. grid-item-card:: ModelPlotPy
      :link: decile-modelplotpy
      :link-type: ref

      Current top-level ModelPlotPy surface and legacy compatibility namespace.

   .. grid-item-card:: Business and financial plots
      :link: decile-business-plots-and-financial
      :link-type: ref

      Response, cumulative lift/gain, costs, profit, ROI, and selection summaries.

   .. grid-item-card:: Compatibility and limits
      :link: decile-compatibility-and-limits
      :link-type: ref

      Positive-label semantics, legacy behavior, RNG side effects, and current snapshot limitations.

Which page should I use?
------------------------

Start with **Getting started**. Then choose the page that owns the decision
you are making: architecture and lifecycle, configuration, persistence,
performance, public API, or limits. The guide intentionally separates those
concerns so examples do not accidentally become policy.

Examples
--------

See :ref:`decile_examples` for executable examples that accompany this guide.
