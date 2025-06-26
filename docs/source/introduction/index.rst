.. _welcome:

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

:raw-html:`<div style="text-align: center"><strong>` Welcome to Scikit-plots 101
|br| |full_version| - |today|
:raw-html:`</strong></div>`

.. _introduction-index:

======================================================================
Introduction
======================================================================

.. grid:: 1 1 2 2

    .. grid-item-card::
        :padding: 2
        :columns: 12 12 6 6

        **Installation**
        ^^^
        .. toctree::
            :maxdepth: 2

            Installation <../install/installation.rst>

    .. grid-item-card::
        :padding: 2
        :columns: 12 12 6 6

        **Quickstart Tutorial**
        ^^^
        .. toctree::
            :maxdepth: 2

            Quick Start <quick_start.rst>

    .. grid-item-card::
        :padding: 2
        :columns: 12 12 12 12

        **Getting Started Guide**
        ^^^
        .. grid:: 1 1 2 2
            :class-row: sd-align-minor-center

            .. grid-item::

                .. toctree::
                    :maxdepth: 2

                    Getting Started <getting_started.rst>

            .. grid-item::

                .. External links.
                .. _special-names: https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#special-names

                *Indices and Tables*:

                - :ref:`genindex`
                - :ref:`modindex`
                - :ref:`search`
