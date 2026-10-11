.. currentmodule:: scikitplot.mlflow

.. _mlflow-index:

MLflow project integration
==========================

``scikitplot.mlflow`` is an orchestration layer around the external MLflow package. It centralizes project discovery, shared configuration, environment application, local store layout, server lifecycle, sessions, provider seams and workflow helpers.

.. important::

   Scikit-Plots does not own MLflow internals. A green unit test of command construction is not proof that every installed MLflow version supports the same CLI/server behavior. Check readiness and CLI capabilities in the deployment environment.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   project_configuration
   sessions_and_facades
   server_and_cli
   providers_and_workflows
   security_and_readiness
   python_api
   troubleshooting

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Getting started
      :link: mlflow-getting-started
      :link-type: ref

      Create or load project configuration and open a managed session.

   .. grid-item-card:: Project configuration
      :link: mlflow-project-configuration
      :link-type: ref

      Discover project roots, markers, TOML/YAML config, local stores, and environment overrides.

   .. grid-item-card:: Sessions and facades
      :link: mlflow-sessions-and-facades
      :link-type: ref

      Use MlflowHandle, artifact/model facades, and controlled session lifetime.

   .. grid-item-card:: Server and CLI
      :link: mlflow-server-and-cli
      :link-type: ref

      Build supported MLflow server commands, spawn a process, and check readiness.

   .. grid-item-card:: Providers and workflows
      :link: mlflow-providers-and-workflows
      :link-type: ref

      Customize the integration provider and use reproducible convenience workflows.

   .. grid-item-card:: Security and readiness
      :link: mlflow-security-and-readiness
      :link-type: ref

      Separate policy intent, CLI compatibility, installed-package readiness, and actual server readiness.

   .. grid-item-card:: Python API
      :link: mlflow-python-api
      :link-type: ref

      Configuration, project, session, provider, workflow, facade, and error surfaces.

   .. grid-item-card:: Troubleshooting
      :link: mlflow-troubleshooting
      :link-type: ref

      Missing MLflow, incompatible CLI flags, store paths, server startup, and current security-policy caveat.

Which page should I use?
------------------------

Start with **Getting started**. Then choose the page that owns the decision
you are making: architecture and lifecycle, configuration, persistence,
performance, public API, or limits. The guide intentionally separates those
concerns so examples do not accidentally become policy.

Examples
--------

See :ref:`mlflow_examples` for executable examples that accompany this guide.

Further reading
---------------
