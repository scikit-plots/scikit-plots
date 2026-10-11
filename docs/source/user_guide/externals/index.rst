..
   "navigation_depth": 3,  # Allow up to four levels in the left sidebar
   "show_nav_level": 1,    # as the initially shown levels left-hand, in-page TOC, to set how many levels are expanded initially.
   "show_toc_level": 1,    # as the initially shown levels right-hand, in-page TOC, to set how many levels are expanded initially.
   For the left .bd-sidebar-primary, set PyData’s navigation_depth to the deepest level you want anywhere—4 in your case.
   That option is site-wide. Then use :maxdepth: on each section’s toctree to limit particular branches to 3 or 4 levels.
   Sphinx uses :maxdepth: to limit the document tree; :tocdepth: is for the page’s right-hand, within-page TOC.
   https://www.sphinx-doc.org/en/master/usage/restructuredtext/field-lists.html#special-metadata-fields

.. Navigation tree:
   Put :maxdepth: inside a .. toctree:: directive.
   It controls how many levels of linked pages appear in that navigation tree.
   For example, :maxdepth: 4 shows up to four levels.
   It only changes what is displayed; it does not remove or disable deeper pages.
   :tocdepth: 2 → document links:  X → Y → Z → T

.. This page's table of contents:
   Put :tocdepth: near the top of this .rst file.
   It controls how many levels of this page's headings appear in its table of contents.
   It does not change the navigation tree or limit child pages.
   :tocdepth: 2 → headings inside a page:  up to 2 heading levels

..
  https://devguide.python.org/documentation/markup/#sections
  https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#sections
  # with overline, for parts    : ######################################################################
  * with overline, for chapters : **********************************************************************
  = for sections                : ======================================================================
  - for subsections             : ----------------------------------------------------------------------
  ^ for subsubsections          : ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  " for paragraphs              : """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

..
  # https://rsted.info.ucl.ac.be/
  # https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#paragraph-level-markup
  # https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#footnotes
  # https://documatt.com/restructuredtext-reference/element/admonition.html
  # attention, caution, danger, error, hint, important, note, tip, warning, admonition, seealso
  # versionadded, versionchanged, deprecated, versionremoved, rubric, centered, hlist

.. currentmodule:: scikitplot.externals

.. _externals-index:

======================================================================
Externals (experimental)
======================================================================

.. grid:: 1 1 1 1

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **Data API Standards**
        ^^^
        .. toctree::
            :maxdepth: 2

            array_api_compat <./array_api_compat/index.rst>

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **Data API Standards**
        ^^^
        .. toctree::
            :maxdepth: 2

            array_api_extra <./array_api_extra/index.rst>

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **probscale**
        ^^^
        .. toctree::
            :maxdepth: 2

            ./_probscale/index.rst

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **seaborn**
        ^^^
        .. toctree::
            :maxdepth: 2

            ./_seaborn/index.rst

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **sphinxext**
        ^^^
        .. toctree::
            :maxdepth: 2

            Matplotlib Sphinx Ext <./_sphinxext/index.rst>

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **tweedie**
        ^^^
        .. toctree::
            :maxdepth: 2

            ./_tweedie/index.rst
