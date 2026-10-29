:html_theme.sidebar_secondary.remove:

.. _ipymagic-index:

======================================================================
Notebook magics
======================================================================

The ``scikitplot`` command line and a profiler, in any IPython notebook or
console (Jupyter, JupyterLab, VS Code, Colab, terminal IPython):

.. code-block:: ipython

   %load_ext scikitplot

======================================  ==========================================
Magic                                   Does
======================================  ==========================================
``%scikitplot ARGS``                    runs ``scikitplot ARGS``, returns a result
``%%scikitplot ARGS`` + cell            the same, the cell on standard input
``%scikitplot_profile STATEMENT``       profiles a statement, shows a table
``%%scikitplot_profile`` + cell         profiles a cell
``%scikitplot_config key=value``        settings of the session
======================================  ==========================================

``pip install scikit-plots[ipymagic]`` installs IPython and snakeviz (the
interactive profile viewer); the rest needs nothing.

Runnable examples, each step from the command line and from Python:
:ref:`ipymagic__examples`.

.. toctree::
   :maxdepth: 2

   Commands in a notebook <commands.rst>
   Profiling <profiling.rst>
   CI, scripts and agents <automation.rst>
