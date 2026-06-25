:html_theme.sidebar_secondary.remove:

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
  # attention, caution, danger, error, hint, important, note, tip, warning, admonition, seealso
  # versionadded, versionchanged, deprecated, versionremoved, rubric, centered, hlist

.. _brand-index:

================
Brand
================

..
  toctree::
    :maxdepth: 2

    CLI Banner <_banner.rst>
    Logo <_logo.rst>

.. grid:: 1 1 2 2

    .. grid-item-card::
        :padding: 2
        :columns: 12 12 6 6

        **cli-banner-mark**
        ^^^
        .. toctree::
            :maxdepth: 2

            CLI Banner <_banner.rst>

    .. grid-item-card::
        :padding: 2
        :columns: 12 12 6 6

        **logo**
        ^^^
        .. toctree::
            :maxdepth: 2

            Logo <_logo.rst>
