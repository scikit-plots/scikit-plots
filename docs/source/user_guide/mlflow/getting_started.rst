.. _mlflow-getting-started:

Getting started
===============

For an opinionated local project setup, :func:`workflow` is the shortest path:

.. code-block:: python

   import scikitplot.mlflow as sml

   paths = sml.workflow(
       profile="local",
       experiment_name="demo",
       fmt="toml",
       overwrite=True,
       open_ui_seconds=0,
   )

For application code that already has configuration, use :func:`session` or
``session_from_file`` directly rather than invoking the whole workflow helper.

MLflow itself remains optional. Import helpers distinguish “Scikit-Plots
integration is importable” from “the external MLflow package is installed and
usable”.
