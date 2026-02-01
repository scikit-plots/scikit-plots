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

.. rubric:: Examples

* :ref:`sphx_glr_auto_examples_mlflow_plot_mlflow.py`: Example usage of
  :func:`~.workflow` using template.


.. rubric:: Quiskstart Template: Beginner workflow demo

.. .. jupyter-execute
.. .. code-block:: python
.. prompt:: python >>>

  import os
  import scikitplot as sp

  # print(sp.mlflow.DEFAULT_PROJECT_MARKERS)
  # Walk upward from `start` until a directory containing any marker is found.
  # export SCIKITPLOT_PROJECT_MARKERS='[".git","pyproject.toml","README.txt","configs/mlflow.toml"]'
  os.environ["SCIKITPLOT_PROJECT_MARKERS"]='[".git","pyproject.toml","README.txt","configs/mlflow.toml"]'

  sp.mlflow.workflow(
      profile="local",
      open_ui_seconds=30,
      experiment_name="my-first-project",
      fmt="toml",
      overwrite=True,  # If config already exists: ./configs/mlflow.toml (use overwrite=True).
  )
