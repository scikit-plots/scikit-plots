.. _scikit-plots-documentation:

.. https://docutils.sourceforge.io/docs/ref/rst/directives.html#custom-interpreted-text-roles
.. role:: raw-html(raw)
   :format: html

.. role:: raw-latex(raw)
   :format: latex

.. |br| raw:: html

   <br/>

.. https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-centered
   centered:: Scikit-plots Documentation :raw-html:`<br />` |release|

:raw-html:`<div style="text-align: center"><strong>`
Scikit-plots Documentation |br| |release| - |today|
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

            ./_xp_core_lib/index.rst


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
