..
  https://devguide.python.org/documentation/markup/#sections
  https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#sections
  # with overline, for parts    : ######################################################################
  * with overline, for chapters : **********************************************************************
  = for sections                : ======================================================================
  - for subsections             : ----------------------------------------------------------------------
  ^ for subsubsections          : ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  " for paragraphs              : """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. # https://rsted.info.ucl.ac.be/
.. # https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#paragraph-level-markup
.. # https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#footnotes
.. # https://documatt.com/restructuredtext-reference/element/admonition.html
.. # attention, caution, danger, error, hint, important, note, tip, warning, admonition, seealso
.. # versionadded, versionchanged, deprecated, versionremoved, rubric, centered, hlist

.. currentmodule:: scikitplot.impute

.. _impute-index:

======================================================================
Impute
======================================================================

This module contains some functions related to :py:mod:`~.impute`.

`Annoy (Approximate Nearest Neighbors Oh Yeah)` is a C++ library with Python bindings
to search for points in space that are close to a given query point.
It also creates large read-only file-based data structures
that are mmapped into memory so that many processes may share the same data.

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

.. _ann_imputer-index:

ANNImputer
**********************************************************************

This module contains some functions related to :py:class:`~.ANNImputer`.

TL;DR
------------
- Purpose: Approximate k-nearest-neighbors (KNN) imputation
- Import path: `from scikitplot.impute import ANNImputer`
- Functionality: Replaces missing values using neighbors retrieved via Annoy
- Parameters: n_neighbors, n_trees, metric, optional search_k, etc.

Overview
--------
:py:class:`~.ANNImputer` (from :py:mod:`~scikitplot.impute`) is an approximate nearest-neighbors
imputer that uses Annoy to fill missing values in datasets. It replaces missing
entries by querying the nearest neighbors of samples with missing values and
computing imputations from those neighbors.

Motivation
----------
Unlike exact KNN imputation :py:class:`~sklearn.impute.KNNImputer`, using Annoy allows:
- Faster neighbor retrieval in high-dimensional data
- Memory-efficient indexing of large datasets
- Sharing of prebuilt indexes across processes

Example: Your exact NumPy array example:

.. .. jupyter-execute
.. .. code-block:: python
.. prompt:: python >>>

    import numpy as np
    from scikitplot.experimental import enable_ann_imputer
    from scikitplot.impute import ANNImputer

    X = np.array([[1, 2, np.nan],
                  [3, 4, 3],
                  [np.nan, 6, 5],
                  [8, 8, 7]])

    imputer = ANNImputer(n_trees=5, n_neighbors=5)
    X_imputed = imputer.fit_transform(X)

    print(X_imputed)
    # Output:
    # array([[1., 2., 5.],
    #        [3., 4., 3.],
    #        [4., 6., 5.],
    #        [8., 8., 7.]])

Mechanism
---------
- Builds an Annoy index from complete samples
- Queries nearest neighbors for incomplete samples
- Imputes missing values based on neighbor vectors
- Integer identifiers are used internally; memory allocated to max(id)+1

Notes
-----
- Memory-efficient and fast for large datasets
- Approximate neighbors; exact KNN may differ slightly
- Shares indexes across processes using mmap
- Behavior depends on `n_trees` and `search_k` parameters

Comparison
----------
- Similar in usage to :py:class:`~sklearn.impute.KNNImputer`, but faster on large, high-dimensional datasets
- Provides a trade-off between accuracy and speed via Annoy parameters
