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

.. currentmodule:: scikitplot.api

.. _api-index:
======================================================================
Metric Performance
======================================================================

This module contains functions related to :py:mod:`~.api`.
For model evaluation metric performance.

.. grid:: 1 1 2 2

    .. grid-item-card::
        :padding: 2

        **decomposition**
        ^^^
        .. toctree::
            :maxdepth: 2

            decomposition.rst

    .. grid-item-card::
        :padding: 2

        **estimators**
        ^^^
        .. toctree::
            :maxdepth: 2

            estimators.rst

    .. grid-item-card::
        :padding: 2

        **metrics**
        ^^^
        .. toctree::
            :maxdepth: 2

            metrics.rst
