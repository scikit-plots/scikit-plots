# scikitplot/mlflow/_compat.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
_compat.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

from ._errors import MlflowNotInstalledError


def import_mlflow() -> Any:
    """
    Import MLflow lazily with a user-friendly error.

    Returns
    -------
    module
        Imported `mlflow` module.

    Raises
    ------
    MlflowNotInstalledError
        If MLflow is not installed.
    """
    spec = importlib.util.find_spec("mlflow")
    if spec is None:
        raise MlflowNotInstalledError(
            "MLflow is not installed. Install it via one of:\n"
            "  - pip install mlflow\n"
            "  - pip install scikitplot[mlflow]  (if you define an extra)\n"
        )
    return importlib.import_module("mlflow")


def resolve_download_artifacts(
    mlflow_module: Any, *, client: Any | None = None
) -> Callable[..., str]:
    """
    Resolve a canonical artifact download function across MLflow versions.

    Parameters
    ----------
    mlflow_module : module
        Imported `mlflow` module.
    client : Any or None, default=None
        Optional MLflow client bound to the desired tracking URI. If provided, it will be
        used for fallback APIs to avoid accidentally downloading from a different server.

    Returns
    -------
    callable
        A callable compatible with `mlflow.artifacts.download_artifacts(...)`.

    Raises
    ------
    AttributeError
        If no supported artifact download API is found.

    Notes
    -----
    Preference order is deterministic:
    1) `mlflow.artifacts.download_artifacts` (public modern API)
    2) `client.download_artifacts` (session-bound)
    3) `MlflowClient.download_artifacts` (legacy, constructed without explicit URI)
    """
    # Preferred, documented modern API.
    if hasattr(mlflow_module, "artifacts") and hasattr(
        mlflow_module.artifacts, "download_artifacts"
    ):
        return mlflow_module.artifacts.download_artifacts  # type: ignore[return-value]

    # Fallback: use the provided client first.
    if client is not None and hasattr(client, "download_artifacts"):

        def _dl(
            *, run_id: str, artifact_path: str, dst_path: str | None = None, **_: Any
        ) -> str:
            return client.download_artifacts(run_id, artifact_path, dst_path)  # type: ignore[misc]

        return _dl

    # Last resort: older API existed on MlflowClient.
    if hasattr(mlflow_module, "tracking") and hasattr(
        mlflow_module.tracking, "MlflowClient"
    ):
        legacy = mlflow_module.tracking.MlflowClient()
        if hasattr(legacy, "download_artifacts"):

            def _dl2(
                *,
                run_id: str,
                artifact_path: str,
                dst_path: str | None = None,
                **_: Any,
            ) -> str:
                return legacy.download_artifacts(run_id, artifact_path, dst_path)  # type: ignore[misc]

            return _dl2

    raise AttributeError(
        "No supported artifact download API found in this MLflow installation. "
        "Expected mlflow.artifacts.download_artifacts or MlflowClient.download_artifacts."
    )
