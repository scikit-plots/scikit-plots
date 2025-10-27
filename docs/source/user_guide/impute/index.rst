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

.. _impute-index:
======================================================================
Impute (experimental)
======================================================================

This module contains some functions related to :py:mod:`~.impute`.

.. seealso::

   * https://pypi.org/project/annoy


Annoy (Approximate Nearest Neighbors Oh Yeah) is a C++ library with Python bindings
to search for points in space that are close to a given query point.
It also creates large read-only file-based data structures
that are mmapped into memory so that many processes may share the same data.

.. _annoy_knn_imputer-index:

AnnoyKNNImputer
**********************************************************************

This module contains some functions related to :py:class:`~.AnnoyKNNImputer`.

TL;DR
------------
- Purpose: Approximate nearest-neighbors-based imputation
- Import path: `from scikitplot.impute import AnnoyKNNImputer``
- Functionality: Replaces missing values using neighbors retrieved via Annoy
- Parameters: n_neighbors, n_trees, metric, optional search_k, etc.

Overview
--------
`AnnoyKNNImputer` (from :py:mod:`~scikitplot.impute`) is an approximate nearest-neighbors
imputer that uses Annoy to fill missing values in datasets. It replaces missing
entries by querying the nearest neighbors of samples with missing values and
computing imputations from those neighbors.

Motivation
----------
Unlike exact KNN imputation, using Annoy allows:
- Faster neighbor retrieval in high-dimensional data
- Memory-efficient indexing of large datasets
- Sharing of prebuilt indexes across processes

Example: Your exact NumPy array example:

.. prompt:: python >>>

    import numpy as np
    from scikitplot.experimental import enable_annoyknn_imputer
    from scikitplot.impute import AnnoyKNNImputer

    X = np.array([[1, 2, np.nan],
                  [3, 4, 3],
                  [np.nan, 6, 5],
                  [8, 8, 7]])

    imputer = AnnoyKNNImputer(n_trees=5, n_neighbors=5)
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
- Similar in usage to `sklearn.impute.KNNImputer`, but faster on large, high-dimensional datasets
- Provides a trade-off between accuracy and speed via Annoy parameters
