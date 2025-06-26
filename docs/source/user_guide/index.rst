.. _scikit-plots-documentation:

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

.. _user-guide-index:

======================================================================
User Guide
======================================================================

.. grid:: 1 1 1 1

    .. grid-item-card::
        :padding: 3

        **Metric Perf**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./api/index.rst

    .. grid-item-card::
        :padding: 3

        **Decile-Wise Perf**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./decile_wise_perf/index.rst


    .. grid-item-card::
        :padding: 3

        **stats**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./stats/index.rst

    .. grid-item-card::
        :padding: 3

        **visualkeras**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./visualkeras/index.rst

    .. grid-item-card::
        :padding: 3

        **DoReMi**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./doremi/index.rst

    .. grid-item-card::
        :padding: 3

        **Logging System**
        ^^^
        .. toctree::
            :maxdepth: 3

            ./sp_logging/index.rst

    .. grid-item-card::
        :padding: 3

        **Dispatching**
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

   ./_astropy/index.rst
   ./experimental/index.rst
   ./_seaborn/index.rst
   ./probscale/index.rst
   ./_tweedie/index.rst
   ./__lightnumpy/index.rst
