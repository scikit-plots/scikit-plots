.. Do not use here directive ``autosummary``

.. _mlflow-python-api:

Python API
==========

Configuration and project
   ``SessionConfig``, ``ServerConfig``, ``ProjectConfig``, project-marker and
   config loader/dumper helpers.

Lifecycle
   ``MlflowHandle``, ``session``, ``session_from_file``, ``session_from_toml``.

Server/readiness
   ``build_server_args``, ``build_server_command``, ``SpawnedServer``,
   ``spawn_server``, ``wait_tracking_ready``, CLI capability helpers.

Customization and convenience
   ``MlflowProvider``, provider context helpers, artifact/model facades,
   ``workflow`` and built-in configuration helpers.

Errors
   Integration-specific exception types distinguish missing MLflow, incompatible
   CLI, server startup, and policy violations.
