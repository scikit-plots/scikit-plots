# scikitplot/mlflow/_project_session.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
_project_session.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ._project import ProjectConfig, load_project_config, load_project_config_toml
from ._session import MlflowHandle, session


@contextmanager
def session_from_toml(
    toml_path: str | Path,
    *,
    profile: str = "local",
) -> Iterator[MlflowHandle]:
    """
    Create an MLflow session using a shared project TOML config.

    Notes
    -----
    TOML reading is supported via stdlib `tomllib` (Python 3.11+).
    TOML writing is not supported in stdlib; use YAML if you need read/write.
    """
    cfg: ProjectConfig = load_project_config_toml(Path(toml_path), profile=profile)
    with session(
        config=cfg.session, server=cfg.server, start_server=cfg.start_server
    ) as h:
        yield h


@contextmanager
def session_from_file(
    config_path: str | Path,
    *,
    profile: str = "local",
) -> Iterator[MlflowHandle]:
    """
    Create an MLflow session using a shared project config file (TOML or YAML).

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to a project config file. Supported extensions: .toml, .yaml, .yml
    profile : str, default="local"
        Profile to load.

    Returns
    -------
    Iterator[MlflowHandle]
        Session handle proxying the upstream `mlflow` module.
    """
    cfg: ProjectConfig = load_project_config(Path(config_path), profile=profile)
    with session(
        config=cfg.session, server=cfg.server, start_server=cfg.start_server
    ) as h:
        yield h
