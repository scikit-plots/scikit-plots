.. _scikit-plots-documentation:

.. https://devguide.python.org/documentation/markup/#substitutions
.. Welcome to Scikit-plots 101 |br| |release| - |today|
..
    substitutions don’t work in .. raw:: html
    .. raw:: html

    <div style="text-align: center"><strong>
    Welcome to Scikit-plots 101<br>|full_version| - |today|
    </strong></div>
..
    https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-centered
    .. centered:: Welcome to Scikit-plots 101 :raw-html:`<br />` |full_version| - |today|
    .. centered::
        **Scikit-plots Documentation** :raw-html:`<br />` |full_version| - |today|

.. https://docutils.sourceforge.io/docs/ref/rst/directives.html#custom-interpreted-text-roles
.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

:raw-html:`<div style="text-align: center"><strong>` Scikit-plots Documentation
|br| |full_version| - |today|
:raw-html:`</strong></div>`

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

.. _user-guide-index:

======================================================================
User Guide
======================================================================

.. grid:: 1 1 1 1

    .. grid-item-card::
        :padding: 3

        **Decile-Wise Perf**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./decile_wise_perf/index.rst

    .. grid-item-card::
        :padding: 3

        **Metric Perf**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./api/index.rst

    .. grid-item-card::
        :padding: 3

        **Preprocessing**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./preprocessing/index.rst

    .. grid-item-card::
        :padding: 3

        **Snsx**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./snsx/index.rst

    .. grid-item-card::
        :padding: 3

        **Stats**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./stats/index.rst

    .. grid-item-card::
        :padding: 3

        **Visualkeras**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./visualkeras/index.rst

    .. grid-item-card::
        :padding: 3

        **Logging System**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./sp_logging/index.rst

    .. grid-item-card::
        :padding: 3

        **Array API Dispatching**
        ^^^
        .. toctree::
            :maxdepth: 2

            ./_lib/index.rst


.. _under-development:

Under Development
----------------------------------------------------------------------

.. toctree::
   :caption: Development
   :maxdepth: 1
   :titlesonly:

   ./cexperimental/index.rst
   ./cexternals/index.rst
   ./experimental/index.rst
   ./externals/index.rst
