# scikitplot/mlflow/_facade.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
_facade.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from mlflow.tracking import MlflowClient  # noqa: F401

_DEFAULT_MLFLOW_MODULE: Any = None
_DEFAULT_MLFLOW_CLIENT: Any = None


@dataclass(frozen=True)
class ArtifactsFacade:
    """
    Artifact helper facade bound to a specific MLflow client/URI.

    Parameters
    ----------
    mlflow_module : module
        Imported `mlflow` module.
    client : MlflowClient
        MLflow client bound to the session tracking URI.

    Notes
    -----
    The implementation is version-robust and deterministic:

    - Prefer the public modern API: `mlflow.artifacts.download_artifacts`.
    - Otherwise fallback to the session-bound client's `download_artifacts`.

    This avoids accidental use of a *different* tracking URI (e.g., when a new client is
    constructed without explicit configuration).
    """

    mlflow_module: Any = _DEFAULT_MLFLOW_MODULE
    client: Any = _DEFAULT_MLFLOW_CLIENT

    def list(self, run_id: str, artifact_path: str | None = None) -> list[Any]:
        """
        List artifacts for a run.

        Parameters
        ----------
        run_id : str
            MLflow run ID.
        artifact_path : str or None, default=None
            Optional artifact subdirectory.

        Returns
        -------
        list
            List of artifact infos (type depends on MLflow version).
        """
        if artifact_path is None:
            return list(self.client.list_artifacts(run_id))  # type: ignore[misc]
        return list(self.client.list_artifacts(run_id, path=artifact_path))  # type: ignore[misc]

    def download(
        self, run_id: str, artifact_path: str, dst_path: str | None = None
    ) -> Path:
        """
        Download an artifact from a run.

        Parameters
        ----------
        run_id : str
            MLflow run ID.
        artifact_path : str
            Path relative to the run artifact root (e.g., "model/MLmodel").
        dst_path : str or None, default=None
            Optional destination directory.

        Returns
        -------
        pathlib.Path
            Local path to the downloaded file or directory.

        Raises
        ------
        AttributeError
            If no compatible artifact download API is available.
        """
        # Preferred, documented modern API.
        m = self.mlflow_module
        if hasattr(m, "artifacts") and hasattr(m.artifacts, "download_artifacts"):
            p = m.artifacts.download_artifacts(
                run_id=run_id, artifact_path=artifact_path, dst_path=dst_path
            )
            return Path(p)

        # Fallback: older client API, bound to our session.
        if hasattr(self.client, "download_artifacts"):
            p = self.client.download_artifacts(run_id, artifact_path, dst_path)  # type: ignore[misc]
            return Path(p)

        raise AttributeError(
            "No supported artifact download API found. Expected mlflow.artifacts.download_artifacts "
            "or MlflowClient.download_artifacts."
        )

    def log_file(
        self, local_path: str | Path, artifact_path: str | None = None
    ) -> None:
        """
        Log a local file as an artifact.

        Parameters
        ----------
        local_path : str or pathlib.Path
            Path to a local file.
        artifact_path : str or None, default=None
            Optional destination path within the run artifact root.

        Returns
        -------
        None
        """
        p = str(local_path)
        if artifact_path is None:
            self.mlflow_module.log_artifact(p)
        else:
            self.mlflow_module.log_artifact(p, artifact_path=artifact_path)

    def log_files(
        self, local_paths: Sequence[str | Path], artifact_path: str | None = None
    ) -> None:
        """
        Log multiple local files as artifacts.

        Parameters
        ----------
        local_paths : Sequence[str or pathlib.Path]
            Paths to local files.
        artifact_path : str or None, default=None
            Optional destination path within the run artifact root.

        Returns
        -------
        None
        """
        for p in local_paths:
            self.log_file(p, artifact_path=artifact_path)


@dataclass(frozen=True)
class ModelsFacade:
    """
    Model helper facade bound to a session-bound MLflow client.

    Parameters
    ----------
    mlflow_module : module
        Imported `mlflow` module.
    client : MlflowClient
        Client bound to the session tracking/registry URIs.

    Notes
    -----
    This facade intentionally stays thin and uses MLflow public APIs.
    """

    mlflow_module: Any = _DEFAULT_MLFLOW_MODULE
    client: Any = _DEFAULT_MLFLOW_CLIENT

    def load_model(self, model_uri: str, *, flavor: str | None = None) -> Any:
        """
        Load a model by URI.

        Parameters
        ----------
        model_uri : str
            Model URI (e.g., "runs:/<run_id>/model" or "models:/Name/Stage").
        flavor : str or None, default=None
            Optional flavor to load. If provided, attempts to load via `mlflow.<flavor>.load_model`.
            Otherwise, uses `mlflow.pyfunc.load_model`.

        Returns
        -------
        Any
            Loaded model object.

        Raises
        ------
        AttributeError
            If requested flavor is not available.
        """
        if flavor:
            mod = getattr(self.mlflow_module, flavor, None)
            if mod is None or not hasattr(mod, "load_model"):
                raise AttributeError(f"MLflow flavor {flavor!r} is not available.")
            return mod.load_model(model_uri)  # type: ignore[misc]
        return self.mlflow_module.pyfunc.load_model(model_uri)  # type: ignore[misc]

    def register_model(self, model_uri: str, name: str) -> Any:
        """
        Register a model version.

        Parameters
        ----------
        model_uri : str
            Model source URI.
        name : str
            Registered model name.

        Returns
        -------
        Any
            Model version object (type depends on MLflow version).
        """
        return self.mlflow_module.register_model(model_uri, name)  # type: ignore[misc]
