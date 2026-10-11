.. Do not use here directive ``autosummary``

.. _impute-python-api:

Python API
==========

Experimental import
-------------------

``ANNImputer`` is added only after:

.. code-block:: python

   from scikitplot.experimental import enable_ann_imputer  # noqa: F401

The estimator follows fit/transform conventions and exposes
``get_feature_names_out``. ``VoyagerKNNImputer`` is an implementation-level
specialization; prefer the unified ``ANNImputer(backend=...)`` surface for new
application code unless a specific compatibility requirement says otherwise.
