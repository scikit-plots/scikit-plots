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

.. currentmodule:: scikitplot.mlflow

.. _mlflow-index:

MLflow Workflow Automation
======================================================================

Examples relevant to the :py:mod:`~.mlflow` module.

.. .. jupyter-execute
.. .. code-block:: python
.. prompt:: python >>>

   import scikitplot as sp

   sp.mlflow.workflow(
       profile="local",
       open_ui_seconds=30,
       experiment_name="my-first-project",
       fmt="toml",
       overwrite=True,  # Config already exists: ./configs/mlflow.toml (use overwrite=True).
   )
