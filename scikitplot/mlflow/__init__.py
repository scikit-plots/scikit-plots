# scikitplot/mlflow/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.mlflow
==================
``MLflow UI | SERVER`` share the exact same settings.

Adds a **project-level configuration** mechanism so multiple scripts
(e.g., `train.py`, `hpo.py`, `predict.py`) share the exact same MLflow settings,
regardless of current working directory.

Examples
--------
Quiskstart Template: Beginner workflow demo

>>> import os
>>> import scikitplot as sp
>>>
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
"""  # noqa: D205, D400

from __future__ import annotations

# Public surface: keep stable and small.
# Implementation modules are private (prefixed with `_`).
from . import (
    _cli_caps,
    _compat,
    _config,
    _container,
    _custom,
    _env,
    _errors,
    _facade,
    _project,
    _readiness,
    _security,
    _server,
    _session,
    _utils,
    _workflow,
)
from ._cli_caps import *  # noqa: F403
from ._compat import *  # noqa: F403
from ._config import *  # noqa: F403
from ._container import *  # noqa: F403
from ._custom import *  # noqa: F403
from ._env import *  # noqa: F403
from ._errors import *  # noqa: F403
from ._facade import *  # noqa: F403
from ._project import *  # noqa: F403
from ._readiness import *  # noqa: F403
from ._security import *  # noqa: F403
from ._server import *  # noqa: F403
from ._session import *  # noqa: F403
from ._utils import *  # noqa: F403
from ._workflow import *  # noqa: F403

__all__ = []
__all__ += _cli_caps.__all__
__all__ += _compat.__all__
__all__ += _config.__all__
__all__ += _container.__all__
__all__ += _custom.__all__
__all__ += _env.__all__
__all__ += _errors.__all__
__all__ += _facade.__all__
__all__ += _project.__all__
__all__ += _readiness.__all__
__all__ += _security.__all__
__all__ += _server.__all__
__all__ += _session.__all__
__all__ += _utils.__all__
__all__ += _workflow.__all__
