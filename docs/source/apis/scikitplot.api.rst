:html_theme.sidebar_secondary.remove:

.. _api_ref:

scikitplot.api
==============

.. automodule:: scikitplot.api

**User guide.** See the :ref:`api-index` section for further details.

.. _api_ref-plot-a-pca-representation:

Plot a PCA representation
-------------------------

.. automodule:: scikitplot.api.decomposition

.. currentmodule:: scikitplot.api

**User guide.** See the :ref:`decomposition-index` section for further details.

.. autosummary::
  :nosignatures:
  :toctree: ../modules/generated/
  :template: base.rst

  decomposition.plot_pca_2d_projection
  decomposition.plot_pca_component_variance

.. _api_ref-plot-estimators--model--object-instances:

Plot Estimators (model) object instances
----------------------------------------

.. automodule:: scikitplot.api.estimators

.. currentmodule:: scikitplot.api

**User guide.** See the :ref:`estimators-index` section for further details.

.. autosummary::
  :nosignatures:
  :toctree: ../modules/generated/
  :template: base.rst

  estimators.plot_feature_importances
  estimators.plot_learning_curve
  estimators.plot_elbow

.. _api_ref-plot-model-evaluation-metrics:

Plot model evaluation metrics
-----------------------------

.. automodule:: scikitplot.api.metrics

.. currentmodule:: scikitplot.api

**User guide.** See the :ref:`metrics-index` section for further details.

.. autosummary::
  :nosignatures:
  :toctree: ../modules/generated/
  :template: base.rst

  metrics.plot_residuals_distribution
  metrics.plot_classifier_eval
  metrics.plot_confusion_matrix
  metrics.plot_precision_recall
  metrics.plot_roc
  metrics.plot_calibration
  metrics.plot_silhouette

.. _api_ref-api-development-utilities:

API Development Utilities
-------------------------

**Developer guide.** See the :ref:`developers-guide-index` section for further details.

.. autosummary::
  :nosignatures:
  :toctree: ../modules/generated/
  :template: base.rst

  _utils.validate_labels
  _utils.cumulative_gain_curve
  _utils.binary_ks_curve
  _utils.validate_plotting_kwargs
  _utils.save_figure
  _utils.save_plot
