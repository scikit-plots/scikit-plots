.. currentmodule:: scikitplot.impute

.. _impute-index:

Missing-value imputation
========================

``scikitplot.impute`` combines standard scikit-learn-style imputation surfaces with an experimental approximate-nearest-neighbor imputer for larger or index-oriented workflows.

.. important::

   ``ANNImputer`` is experimental and is not imported into the public namespace until you explicitly enable it. Approximate-neighbor imputation changes the computational trade-off; it does not guarantee the same neighbors or values as exact KNN imputation.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   standard_imputers
   ann_imputer
   index_access_and_privacy
   python_api
   limits_and_troubleshooting

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Getting started
      :link: impute-getting-started
      :link-type: ref

      Use stable sklearn-compatible imputers or explicitly enable ANNImputer.

   .. grid-item-card:: Standard imputers
      :link: impute-standard-imputers
      :link-type: ref

      Understand the re-exported MissingIndicator and SimpleImputer surface.

   .. grid-item-card:: ANNImputer
      :link: impute-ann-imputer
      :link-type: ref

      Approximate-neighbor imputation with Annoy or Voyager backends.

   .. grid-item-card:: Index access and privacy
      :link: impute-index-access-and-privacy
      :link-type: ref

      Choose private/public/external index ownership and understand data exposure.

   .. grid-item-card:: Python API
      :link: impute-python-api
      :link-type: ref

      Stable imports, experimental activation, fit/transform behavior, and feature names.

   .. grid-item-card:: Limits and troubleshooting
      :link: impute-limits-and-troubleshooting
      :link-type: ref

      Approximation, fallback behavior, optional backends, and current review findings.

Which page should I use?
------------------------

Start with **Getting started**. Then choose the page that owns the decision
you are making: architecture and lifecycle, configuration, persistence,
performance, public API, or limits. The guide intentionally separates those
concerns so examples do not accidentally become policy.

Examples
--------

See :ref:`impute_examples` for executable examples that accompany this guide.

See also
--------

* :ref:`annoy-index`

Further reading
---------------

`Voyager` is optimized for modern nearest-neighbor search.

`HNSW (Hierarchical Navigable Small World)` provides better accuracy and speed,
outperforming Annoy in most use cases, especially when precision is important.

.. seealso::

   * `github: ANNoy based on random projection (hyperplane) trees <https://github.com/spotify/annoy>`__
   * `pypi: ANNoy based on random projection (hyperplane) method <https://pypi.org/project/annoy>`__
   * `github: Voyager based on HNSW algorithm (hnswlib) <https://github.com/spotify/voyager>`__
   * `pypi: Voyager based on HNSW algorithm (hnswlib) <https://pypi.org/project/voyager>`__
   * `github: HNSW implementation Header-only C++/python <https://github.com/nmslib/hnswlib>`__
   * `pypi: HNSW implementation Header-only C++/python <https://pypi.org/project/hnswlib>`__
