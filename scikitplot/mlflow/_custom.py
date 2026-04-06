# scikitplot/mlflow/_custom.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
_custom.

Fully customizable mlflow like library or inconsistent versions.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Callable

__all__ = [
    "MlflowProvider",
    "get_provider",
    "set_provider",
    "use_provider",
]

_GLOBAL_PROVIDER: MlflowProvider | None = None


@dataclass
class MlflowProvider:
    """
    A customizable provider for MLflow-like libraries.

    This class acts as an abstraction layer to support inconsistent MLflow
    versions, internal wrappers, or alternative tracking libraries that
    mimic the MLflow API. By defining a custom provider, you can override
    how modules are imported, clients are instantiated, and artifacts are downloaded.

    Parameters
    ----------
    module : Any
        The custom MLflow-like module object (can be a mock or wrapper).
    version : str or None, default=None
        A static version string to bypass dynamic package resolution.
    client_factory : Callable[[str, str | None], Any] or None, default=None
        A callable taking (tracking_uri, registry_uri) and returning an MLflow client.
    artifact_downloader : Callable[..., str] or None, default=None
        A callable mimicking `mlflow.artifacts.download_artifacts`.
    """

    module: Any
    version: str | None = None
    client_factory: Callable[[str, str | None], Any] | None = None
    artifact_downloader: Callable[..., str] | None = None

    def get_client(self, tracking_uri: str, registry_uri: str | None = None) -> Any:
        """Instantiate the MLflow-like client."""
        if self.client_factory is not None:
            return self.client_factory(tracking_uri, registry_uri)

        # Standard MLflow fallback
        try:
            return self.module.tracking.MlflowClient(
                tracking_uri=tracking_uri, registry_uri=registry_uri
            )
        except TypeError:
            return self.module.tracking.MlflowClient(tracking_uri=tracking_uri)

    def get_artifact_downloader(self, client: Any) -> Callable[..., str]:
        """Resolve the artifact download callable."""
        if self.artifact_downloader is not None:
            return self.artifact_downloader

        from ._compat import resolve_download_artifacts  # noqa: PLC0415

        return resolve_download_artifacts(self.module, client=client)


def get_provider() -> MlflowProvider | None:
    """Retrieve the currently active MLflow provider."""
    return _GLOBAL_PROVIDER


def set_provider(provider: MlflowProvider | None) -> None:
    """Set the active MLflow provider globally."""
    global _GLOBAL_PROVIDER  # noqa: PLW0603
    _GLOBAL_PROVIDER = provider


@contextlib.contextmanager
def use_provider(provider: MlflowProvider | None) -> Iterator[None]:
    """Temporarily set the MLflow provider for a context block."""
    global _GLOBAL_PROVIDER  # noqa: PLW0603
    old = _GLOBAL_PROVIDER
    try:
        set_provider(provider)
        yield
    finally:
        _GLOBAL_PROVIDER = old
