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

.. _scikit-plots-documentation:

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
.. # https://documatt.com/restructuredtext-reference/element/admonition.html
.. # attention, caution, danger, error, hint, important, note, tip, warning, admonition, seealso
.. # versionadded, versionchanged, deprecated, versionremoved, rubric, centered, hlist

.. _user-guide-index:

======================================================================
User Guide
======================================================================

.. grid:: 1 1 1 1

    .. grid-item-card::
        :padding: 3

        **nearest neighbor**
        ^^^
        .. toctree::
            :maxdepth: 3

            ANNoy <./annoy/index.rst>

    .. grid-item-card::
        :padding: 3

        **metric analysis**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./api/index.rst

    .. grid-item-card::
        :padding: 3

        **remarks generation**
        ^^^
        .. toctree::
            :maxdepth: 3

            Corpus <./corpus/index.rst>

    .. grid-item-card::
        :padding: 3

        **live, on demand generation**
        ^^^
        .. toctree::
            :maxdepth: 3

            Cython <./cython/index.rst>

    .. grid-item-card::
        :padding: 3

        **decile-wise analysis**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./decile/index.rst

    .. grid-item-card::
        :padding: 3

        **data imputation**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./impute/index.rst

    .. grid-item-card::
        :padding: 3

        **logging system**
        ^^^
        .. toctree::
            :maxdepth: 3

            Logging <./logging/index.rst>

    .. grid-item-card::
        :padding: 3

        **memory mapping**
        ^^^
        .. toctree::
            :maxdepth: 3

            MemMap <./memmap/index.rst>

    .. grid-item-card::
        :padding: 3

        **workflow automation**
        ^^^
        .. toctree::
            :maxdepth: 3

            MLflow <./mlflow/index.rst>

    .. grid-item-card::
        :padding: 3

        **lightweight high-performance**
        ^^^
        .. toctree::
            :maxdepth: 3

            Nc <./nc/index.rst>

    .. grid-item-card::
        :padding: 3

        **data preprocessing**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./preprocessing/index.rst

    .. grid-item-card::
        :padding: 3

        **random generator**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./random/index.rst

    .. grid-item-card::
        :padding: 3

        **seaborn based**
        ^^^
        .. toctree::
            :maxdepth: 3

            Seaborn <./seaborn/index.rst>

    .. grid-item-card::
        :padding: 3

        **extended by astropy**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./stats/index.rst

    .. grid-item-card::
        :padding: 3

        **tensorflow keras**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./visualkeras/index.rst

    .. grid-item-card::
        :padding: 3

        **array api dispatching**
        ^^^
        .. toctree::
            :maxdepth: 2

            ./_lib/index.rst

    .. grid-item-card::
        :padding: 3

        **brand logo**
        ^^^
        .. toctree::
            :maxdepth: 2

            ./_logo.rst


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
