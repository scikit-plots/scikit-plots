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

.. currentmodule:: scikitplot.experimental

.. _experimental-index:

======================================================================
Experimental (experimental)
======================================================================

.. grid:: 1 1 1 1

    .. grid-item-card::
        :padding: 3

        **Pipeline**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./pipeline/index.rst

    .. grid-item-card::
        :padding: 3

        **Do-Re-Mi**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./_doremi/index.rst

    .. grid-item-card::
        :padding: 3

        **Large Language Models**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./_llm_provider/index.rst
