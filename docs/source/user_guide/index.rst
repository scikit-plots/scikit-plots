:html_theme.sidebar_secondary.remove:

..
  # https://devguide.python.org/documentation/markup/#substitutions

.. Welcome to Scikit-plots 101 |br| |release| - |today|

..
    substitutions don’t work in .. raw:: html
    .. raw:: html

    <div style="text-align: center"><strong>
    Welcome to Scikit-plots 101<br>|full_version| - |today|
    </strong></div>
..
    # https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-centered
    .. centered:: Welcome to Scikit-plots 101 :raw-html:`<br />` |full_version| - |today|
    .. centered::
        **Scikit-plots Documentation** :raw-html:`<br />` |full_version| - |today|

..
  # https://docutils.sourceforge.io/docs/ref/rst/directives.html#custom-interpreted-text-roles

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _scikit-plots-documentation:

:raw-html:`<div style="text-align: center"><strong>` 📚 Scikit-plots Documentation
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

..
  # https://rsted.info.ucl.ac.be/
  # https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#paragraph-level-markup
  # https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#footnotes
  # https://documatt.com/restructuredtext-reference/element/admonition.html
  # attention, caution, danger, error, hint, important, note, tip, warning, admonition, seealso
  # versionadded, versionchanged, deprecated, versionremoved, rubric, centered, hlist

.. _user-guide-index:

======================================================================
User Guide
======================================================================

.. grid:: 1 1 1 1

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **nearest neighbor**
        ^^^
        .. toctree::
            :maxdepth: 2

            ANNoy <./annoy/index.rst>

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **metric analysis**
        ^^^
        .. toctree::
            :maxdepth: 2

            ./api/index.rst

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **pseudonymization engine for LLM**
        ^^^
        .. toctree::
            :maxdepth: 2

            CleanPrompt <./cleanprompt/index.rst>

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **remarks citation generation**
        ^^^
        .. toctree::
            :maxdepth: 2

            Corpus <./corpus/index.rst>

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **live, on demand generation**
        ^^^
        .. toctree::
            :maxdepth: 2

            Cython <./cython/index.rst>

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **decile-wise analysis**
        ^^^
        .. toctree::
            :maxdepth: 2

            ./decile/index.rst

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **data imputation**
        ^^^
        .. toctree::
            :maxdepth: 2

            ./impute/index.rst

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **edit distance / fuzzy matching**
        ^^^
        .. toctree::
            :maxdepth: 3

            Levenshtein <./levenshtein/index.rst>

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **logging system**
        ^^^
        .. toctree::
            :maxdepth: 2

            Logging <./logging/index.rst>

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **memory mapping**
        ^^^
        .. toctree::
            :maxdepth: 2

            MemMap <./memmap/index.rst>

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **model context protocol**
        ^^^
        .. toctree::
            :maxdepth: 2

            Mcp <./mcp/index.rst>

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **workflow automation**
        ^^^
        .. toctree::
            :maxdepth: 2

            MLflow <./mlflow/index.rst>

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **lightweight high-performance**
        ^^^
        .. toctree::
            :maxdepth: 2

            Nc <./nc/index.rst>

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **data preprocessing**
        ^^^
        .. toctree::
            :maxdepth: 2

            ./preprocessing/index.rst

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **random generator**
        ^^^
        .. toctree::
            :maxdepth: 2

            ./random/index.rst

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **lexical document ranking**
        ^^^
        .. toctree::
            :maxdepth: 2

            Rank-BM25 <./rank_bm25/index.rst>

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **seaborn based**
        ^^^
        .. toctree::
            :maxdepth: 2

            Seaborn <./seaborn/index.rst>

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **extended by astropy**
        ^^^
        .. toctree::
            :maxdepth: 2

            ./stats/index.rst

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **tensorflow keras**
        ^^^
        .. toctree::
            :maxdepth: 2

            ./visualkeras/index.rst

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **array api dispatching**
        ^^^
        .. toctree::
            :maxdepth: 2

            ./_lib/index.rst

    .. grid-item-card::
        :columns: 12 12 6 6
        :padding: 2

        **branding**
        ^^^
        .. toctree::
            :maxdepth: 2

            ./_brand/index.rst


.. _under-development:

Under Development
----------------------------------------------------------------------

.. toctree::
   :caption: development
   :maxdepth: 1
   :titlesonly:

   ./cexperimental/index.rst
   ./cexternals/index.rst
   ./experimental/index.rst
   ./externals/index.rst
   ./_externals/index.rst
