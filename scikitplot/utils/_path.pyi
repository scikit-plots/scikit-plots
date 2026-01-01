# from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

def normalize_directory_path(
    path: str | Path,
    *,
    expand_user: bool = ...,
    expand_vars: bool = ...,
    resolve: bool = ...,
) -> Path: ...
def sanitize_path_component(
    name: str | None,
    *,
    default: str = ...,
    max_len: int = ...,
) -> str: ...
def normalize_extension(ext: str | None) -> str: ...
@dataclass(frozen=True)
class PathNamer:
    root: Path
    default_prefix: str
    default_ext: str
    by_day: bool
    add_secret: bool
    private: bool
    mkdir: bool

    def __init__(
        self,
        root: Path = ...,
        default_prefix: str = ...,
        default_ext: str = ...,
        by_day: bool = ...,
        add_secret: bool = ...,
        private: bool = ...,
        mkdir: bool = ...,
    ) -> None: ...
    def make_filename(
        self,
        prefix: str | None = ...,
        ext: str | None = ...,
        *,
        now: datetime | None = ...,
    ) -> str: ...
    def make_path(
        self,
        prefix: str | None = ...,
        ext: str | None = ...,
        *,
        subdir: str | None = ...,
        now: datetime | None = ...,
    ) -> Path: ...

def make_path(
    root: str | Path = ...,
    prefix: str = ...,
    ext: str = ...,
    *,
    by_day: bool = ...,
    add_secret: bool = ...,
    private: bool = ...,
    mkdir: bool = ...,
    subdir: str | None = ...,
    now: datetime | None = ...,
) -> Path: ...
