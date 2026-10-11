.. currentmodule:: scikitplot.seaborn

.. _seaborn-index:

Seaborn-style model plots
=========================

``scikitplot.seaborn`` provides a small seaborn-like high-level plotting API for model evaluation and decile analysis. It is a Scikit-Plots wrapper layer, not a replacement for upstream seaborn.

.. important::

   The repository maintenance snapshot currently marks this submodule as not release-ready. Treat these plotting wrappers as experimental and pin/test the exact dependency versions used by your application.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   auc_and_evaluation
   decile_and_model_plots
   data_semantics_and_styling
   python_api
   known_limits_and_troubleshooting

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Getting started
      :link: seaborn-getting-started
      :link-type: ref

      Choose the high-level plotting wrapper that matches the evaluation question.

   .. grid-item-card:: AUC and evaluation
      :link: seaborn-auc-and-evaluation
      :link-type: ref

      Use aucplot/evalplot with explicit data, x/y/hue, and classification semantics.

   .. grid-item-card:: Decile and model plots
      :link: seaborn-decile-and-model-plots
      :link-type: ref

      Use decileplot/modelplot and interpret lift/gain-style outputs.

   .. grid-item-card:: Data semantics and styling
      :link: seaborn-data-semantics-and-styling
      :link-type: ref

      Long-form data, hue/grouping, kind selection, Axes return values, and plotting side effects.

   .. grid-item-card:: Python API
      :link: seaborn-python-api
      :link-type: ref

      aucplot, evalplot, decileplot, modelplot, and print_labels.

   .. grid-item-card:: Known limits and troubleshooting
      :link: seaborn-known-limits-and-troubleshooting
      :link-type: ref

      Experimental status, upstream-private compatibility, and current open maintenance findings.

Which page should I use?
------------------------

Start with **Getting started**. Then choose the page that owns the decision
you are making: architecture and lifecycle, configuration, persistence,
performance, public API, or limits. The guide intentionally separates those
concerns so examples do not accidentally become policy.

Examples
--------

See :ref:`seaborn_examples` for executable examples that accompany this guide.

See also
--------

* :ref:`decile-index`

Further reading
---------------
