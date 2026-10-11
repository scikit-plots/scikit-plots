.. _mlflow-server-and-cli:

Server and CLI
==============

Use ``get_mlflow_server_cli_caps`` and ``ensure_flags_supported`` before
constructing assumptions about an installed MLflow CLI. ``build_server_args``
and ``build_server_command`` create a command from validated configuration;
``spawn_server`` owns the subprocess wrapper.

``wait_tracking_ready`` checks service readiness after process start. A process
existing is not the same as the tracking HTTP endpoint being ready.

Container detection is available through ``running_in_docker`` because bind and
host defaults can differ between a developer shell and a container runtime.
