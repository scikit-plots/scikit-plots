.. _impute-index:

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
.. # attention, caution, danger, error, hint, important, note, tip, warning, admonition, seealso
.. # versionadded, versionchanged, deprecated, versionremoved, rubric, centered, hlist

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
