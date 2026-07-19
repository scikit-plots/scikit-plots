:html_theme.sidebar_secondary.remove:

..
  # https://devguide.python.org/documentation/markup/#substitutions

.. Welcome to Scikit-plots 101 |br| |release| - |today|

..
    ✨ substitutions don’t work in .. raw:: html
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

.. _scikit-plots-learn:

:raw-html:`<div style="text-align: center"><strong>` 🤗 Scikit-plots Learn
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

.. _learn-index:

======================================================================
Learn
======================================================================

.. grid:: 1 1 1 1

    .. grid-item-card::
        :padding: 2
        :columns: 12 12 6 6

        **terminologies**
        ^^^
        .. toctree::
            :maxdepth: 2

            Terminology <./terminology/index.rst>

    .. grid-item-card::
        :padding: 2
        :columns: 12 12 6 6

        **external resources**
        ^^^
        .. toctree::
            :maxdepth: 2

            External Resources <./resources/index.rst>

    .. grid-item-card::
        :padding: 2
        :columns: 12 12 6 6

        **data analytics**
        ^^^
        .. toctree::
            :maxdepth: 2

            Data Analytics <./data_analytics/index.rst>

    .. grid-item-card::
        :padding: 2
        :columns: 12 12 6 6

        **data analysis**
        ^^^
        .. toctree::
            :maxdepth: 2

            Data Preparation & Analysis <./data_preparation_and_analysis/index.rst>

    .. grid-item-card::
        :padding: 2
        :columns: 12 12 6 6

        **bayesian data analysis**
        ^^^
        .. toctree::
            :maxdepth: 2

            Bayesian Data Analysis <./bayesian_data_analysis/index.rst>

    .. grid-item-card::
        :padding: 2
        :columns: 12 12 6 6

        **time series**
        ^^^
        .. toctree::
            :maxdepth: 2

            Time Series <./time_series/index.rst>

    .. grid-item-card::
        :padding: 2
        :columns: 12 12 6 6

        **deep learning**
        ^^^
        .. toctree::
            :maxdepth: 2

            Deep Learning <./deep_learning/index.rst>

    .. grid-item-card::
        :padding: 2
        :columns: 12 12 6 6

        **hands-on materials**
        ^^^
        .. toctree::
            :maxdepth: 2

            Hands-On Materials <./hands-on/index.rst>

    .. grid-item-card::
        :padding: 2
        :columns: 12 12 6 6

        **cheatsheets**
        ^^^
        .. toctree::
            :maxdepth: 2

            Cheatsheet <./cheatsheet/index.rst>

    .. grid-item-card::
        :padding: 2

        **glossaries**
        ^^^
        .. toctree::
            :maxdepth: 2

            Glossary <glossary/index.rst>
