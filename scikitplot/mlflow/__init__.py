# scikitplot/mlflow/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
``MLflow UI | SERVER`` share the exact same settings.

Adds a **project-level configuration** mechanism so multiple scripts
(e.g., `train.py`, `hpo.py`, `predict.py`) share the exact same MLflow settings,
regardless of current working directory.

Examples
--------
Quiskstart Template: Beginner workflow demo

>>> import os
>>> import scikitplot as sp
>>> # print(sp.mlflow.DEFAULT_PROJECT_MARKERS)
>>> # Walk upward from `start` until a directory containing any marker is found.
>>> # export SCIKITPLOT_PROJECT_MARKERS='[".git","pyproject.toml","README.txt","configs/mlflow.toml"]'
>>> os.environ["SCIKITPLOT_PROJECT_MARKERS"] = (
...     '[".git","pyproject.toml","README.txt","configs/mlflow.toml"]'
... )
>>> sp.mlflow.workflow(
...     profile="local",
...     open_ui_seconds=30,
...     experiment_name="my-first-project",
...     fmt="toml",
...     overwrite=True,  # If config already exists: ./configs/mlflow.toml (use overwrite=True).
... )

CLI

>>> # Walk upward from `start` until a directory containing any marker is found.
>>> # export SCIKITPLOT_PROJECT_MARKERS='[".git","pyproject.toml","README.txt","configs/mlflow.toml"]'
>>> python -m scikitplot.mlflow --profile local --open-ui-seconds 5
"""

from __future__ import annotations

# Public surface: keep stable and small.
# Implementation modules are private (prefixed with `_`).
from ._config import ServerConfig, SessionConfig
from ._errors import (
    MlflowCliIncompatibleError,
    MlflowIntegrationError,
    MlflowNotInstalledError,
    MlflowServerStartError,
)
from ._project import (
    DEFAULT_PROJECT_MARKERS,
    ProjectConfig,
    dump_project_config_yaml,
    find_project_root,
    load_project_config,
    load_project_config_toml,
)
from ._project_session import session_from_file, session_from_toml
from ._session import MlflowHandle, session
from ._workflow import (
    builtin_config_path,
    default_project_paths,
    export_builtin_config,
    workflow,  # alias for run_demo
)

__all__ = [  # noqa: RUF022
    # config
    "ServerConfig",
    "SessionConfig",
    "ProjectConfig",
    # errors
    "MlflowIntegrationError",
    "MlflowNotInstalledError",
    "MlflowCliIncompatibleError",
    "MlflowServerStartError",
    # session
    "session",
    "MlflowHandle",
    "session_from_file",
    "session_from_toml",
    # project config helpers
    "DEFAULT_PROJECT_MARKERS",
    "find_project_root",
    "load_project_config",
    "load_project_config_toml",
    "dump_project_config_yaml",
    # workflow helpers
    "workflow",
    "builtin_config_path",
    "export_builtin_config",
    "default_project_paths",
]
