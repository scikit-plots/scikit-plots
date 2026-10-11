.. _decile-modelplotpy:

ModelPlotPy
===========

The top-level :class:`scikitplot.decile.ModelPlotPy` is the current
Scikit-Plots implementation. ``scikitplot.decile.modelplotpy`` remains a legacy
compatibility namespace and should be treated separately when validating
behavior.

The current top-level plotting functions include:

* ``plot_response`` and ``plot_cumresponse``;
* ``plot_cumlift`` and ``plot_cumgains``;
* ``plot_all``;
* ``plot_costsrevs``, ``plot_profit`` and ``plot_roi``; and
* ``summarize_selection``.

Use the current top-level surface for new code unless you specifically need to
maintain compatibility with the legacy namespace.

The maintenance review also records that the legacy implementation mutates
NumPy's global RNG state. Avoid depending on global NumPy RNG state around
legacy plotting code until that behavior is resolved.
