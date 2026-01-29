# from __future__ import annotations

from typing import Any, Iterator, Mapping

from _facade import ArtifactsFacade, ModelsFacade

from ._config import ServerConfig, SessionConfig

class MlflowHandle:
    _mlflow_module: Any
    _tracking_uri: str
    _registry_uri: str | None
    _ui_url: str
    _client: Any
    artifacts: Any
    models: Any
    server: Any
    version: Any
    experiment_name: str | None
    default_run_name: str | None
    default_run_tags: Mapping[str, str] | None

    @property
    def mlflow_module(self) -> Any: ...
    @property
    def tracking_uri(self) -> str: ...
    @property
    def registry_uri(self) -> str | None: ...
    @property
    def ui_url(self) -> str: ...
    @property
    def client(self) -> Any: ...
    @property
    def artifacts(self) -> ArtifactsFacade: ...
    @property
    def models(self) -> ModelsFacade: ...
    def __getattr__(self, name: str) -> Any: ...
    def start_run(self, *args: Any, **kwargs: Any) -> Iterator[Any]: ...

def session(
    *,
    config: SessionConfig | None = ...,
    server: ServerConfig | None = ...,
    start_server: bool = ...,
) -> Iterator[MlflowHandle]: ...
