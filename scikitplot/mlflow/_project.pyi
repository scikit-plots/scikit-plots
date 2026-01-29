# from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence  # noqa: F401
from dataclasses import dataclass
from pathlib import Path
from typing import overload

from ._config import ServerConfig, SessionConfig

DEFAULT_PROJECT_MARKERS: tuple[str, ...]

def get_project_markers(*, config_path: Path | None = ...) -> tuple[str, ...]: ...
def set_project_markers(markers: Sequence[str] | None) -> None: ...
def project_markers(markers: Sequence[str] | None) -> Iterator[None]: ...
def find_project_root(
    start: Path | None = ...,
    *,
    markers: Sequence[str] | None = ...,
    config_path: Path | None = ...,
) -> Path: ...
def normalize_mlflow_store_values(
    *,
    backend_store_uri: str | None,
    default_artifact_root: str | None,
    base_dir: Path,
) -> tuple[str | None, str | None]: ...
def ensure_local_store_layout(
    *,
    backend_store_uri: str | None,
    default_artifact_root: str | None,
) -> None: ...
@dataclass(frozen=True)
class ProjectConfig:
    profile: str = ...
    session: SessionConfig = ...
    server: ServerConfig | None = ...
    start_server: bool = ...

def load_project_config_toml(
    path: Path,
    *,
    profile: str = ...,
    project_root: Path | None = ...,
) -> ProjectConfig: ...
def load_project_config_yaml(
    path: Path,
    *,
    profile: str = ...,
    project_root: Path | None = ...,
) -> ProjectConfig: ...
def load_project_config(
    path: Path,
    *,
    profile: str = ...,
    project_root: Path | None = ...,
) -> ProjectConfig: ...
@overload
def dump_project_config_yaml(
    cfg: ProjectConfig,
    path: Path,
    *,
    profile: str = ...,
    source_config_path: Path | None = ...,
) -> Path: ...
@overload
def dump_project_config_yaml(
    cfg: None = ...,
    path: None = ...,
    *,
    profile: str = ...,
    source_config_path: Path | None = ...,
) -> Path: ...
