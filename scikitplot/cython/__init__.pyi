# scikitplot/cython/__init__.pyi
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Mapping, Sequence, TypeAlias

PathLikeAny: TypeAlias = str | bytes | Path | os.PathLike[str] | os.PathLike[bytes]
ProfileName: TypeAlias = Literal["fast-debug", "release", "annotate"]

class TemplateInfo:
    template_id: str
    path: Path
    meta_path: Path | None
    category: str
    language: str
    level: str
    summary: str
    description: str
    requires_numpy: bool
    requires_cpp: bool
    demo_calls: tuple[dict[str, Any], ...]
    support_paths: tuple[str, ...]
    extra_sources: tuple[str, ...]
    tags: tuple[str, ...]
    schema_version: int
    meta: Mapping[str, Any]

def read_template_info(
    template_id: str, *, encoding: str = "utf-8"
) -> TemplateInfo: ...

class CacheEntry:
    key: str = ""
    build_dir: Path = ...
    module_name: str = ""
    artifact_path: Path = ...
    created_utc: str | None = None
    fingerprint: Mapping[str, Any] | None = None

class PackageCacheEntry:
    key: str = ""
    build_dir: Path = ...
    package_name: str = ""
    modules: tuple[str, ...] = ...
    artifacts: tuple[Path, ...] = ...
    created_utc: str | None = None
    fingerprint: Mapping[str, Any] | None = None

class BuildResult:
    module: ModuleType = ...
    key: str = ""
    module_name: str = ""
    build_dir: Path = ...
    artifact_path: Path = ...
    used_cache: bool = False
    created_utc: str | None = None
    fingerprint: Mapping[str, Any] | None = None
    source_sha256: str | None = None
    meta: Mapping[str, Any] = ...

class PackageBuildResult:
    package_name: str = ""
    key: str = ""
    build_dir: Path = ...
    results: tuple[BuildResult, ...] = ...
    used_cache: bool = False
    created_utc: str | None = None
    fingerprint: Mapping[str, Any] | None = None
    meta: Mapping[str, Any] = ...

    @property
    def modules(self) -> Sequence[ModuleType]: ...

class CacheStats:
    cache_root: Path = ...
    n_modules: int = 0
    n_packages: int = 0
    total_bytes: int = 0
    pinned_aliases: int = 0
    pinned_keys: int = 0
    newest_mtime_utc: str | None = None
    oldest_mtime_utc: str | None = None

class CacheGCResult:
    cache_root: Path = ...
    deleted_keys: tuple[str, ...] = ...
    skipped_pinned_keys: tuple[str, ...] = ...
    skipped_missing_keys: tuple[str, ...] = ...
    freed_bytes: int = 0

def get_cache_dir(cache_dir: str | Path | None = ...) -> Path: ...
def purge_cache(cache_dir: str | Path | None = ...) -> None: ...
def check_build_prereqs(*, numpy: bool = ...) -> dict[str, Any]: ...
def compile_and_load_result(
    source: str,
    *,
    module_name: str | None = ...,
    cache_dir: str | Path | None = ...,
    use_cache: bool = ...,
    force_rebuild: bool = ...,
    verbose: int = ...,
    profile: ProfileName | None = ...,
    annotate: bool = ...,
    view_annotate: bool = ...,
    numpy_support: bool = ...,
    numpy_required: bool = ...,
    include_dirs: Sequence[PathLikeAny] | None = ...,
    library_dirs: Sequence[PathLikeAny] | None = ...,
    libraries: Sequence[str] | None = ...,
    define_macros: Sequence[tuple[str, str | None]] | None = ...,
    extra_compile_args: Sequence[str] | None = ...,
    extra_link_args: Sequence[str] | None = ...,
    compiler_directives: Mapping[str, Any] | None = ...,
    extra_sources: Sequence[PathLikeAny] | None = ...,
    support_files: Mapping[str, str | bytes] | None = ...,
    support_paths: Sequence[PathLikeAny] | None = ...,
    include_cwd: bool = ...,
    lock_timeout_s: float = ...,
    language: str | None = ...,
) -> BuildResult: ...
def compile_and_load(
    source: str, *, module_name: str | None = ..., **kwargs: Any
) -> ModuleType: ...
def cython_import_result(
    pyx_path: str | bytes | Path,
    *,
    module_name: str | None = ...,
    **kwargs: Any,
) -> BuildResult: ...
def cython_import(
    pyx_path: str | bytes | Path, *, module_name: str | None = ..., **kwargs: Any
) -> ModuleType: ...
def cython_import_all(
    directory: str | Path,
    *,
    pattern: str = ...,
    recursive: bool = ...,
    **kwargs: Any,
) -> dict[str, BuildResult]: ...
def import_cached_result(
    key: str, *, cache_dir: str | Path | None = ...
) -> BuildResult: ...
def import_cached(key: str, *, cache_dir: str | Path | None = ...) -> ModuleType: ...
def import_cached_by_name(
    module_name: str, *, cache_dir: str | Path | None = ...
) -> ModuleType: ...
def import_cached_package_result(
    key: str, *, cache_dir: str | Path | None = ...
) -> PackageBuildResult: ...
def import_cached_package(
    key: str, *, cache_dir: str | Path | None = ...
) -> Sequence[ModuleType]: ...
def list_cached(cache_dir: str | Path | None = ...) -> list[CacheEntry]: ...
def list_cached_packages(
    cache_dir: str | Path | None = ...,
) -> list[PackageCacheEntry]: ...
def cache_stats(cache_dir: str | Path | None = ...) -> CacheStats: ...
def gc_cache(
    *,
    cache_dir: str | Path | None = ...,
    keep_n_newest: int | None = ...,
    max_age_days: int | None = ...,
    max_bytes: int | None = ...,
    dry_run: bool = ...,
    lock_timeout_s: float = ...,
) -> CacheGCResult: ...
def pin(
    key: str,
    *,
    alias: str,
    cache_dir: str | Path | None = ...,
    overwrite: bool = ...,
    lock_timeout_s: float = ...,
) -> str: ...
def unpin(
    alias: str, *, cache_dir: str | Path | None = ..., lock_timeout_s: float = ...
) -> bool: ...
def list_pins(cache_dir: str | Path | None = ...) -> dict[str, str]: ...
def import_pinned_result(
    alias: str, *, cache_dir: str | Path | None = ...
) -> BuildResult | PackageBuildResult: ...
def import_pinned(
    alias: str, *, cache_dir: str | Path | None = ...
) -> ModuleType | Sequence[ModuleType]: ...
def register_cached_artifact_path(
    artifact_path: str | bytes | Path,
    *,
    module_name: str,
    cache_dir: str | Path | None = ...,
    copy: bool = ...,
) -> BuildResult: ...
def register_cached_artifact_bytes(
    data: bytes,
    *,
    module_name: str,
    artifact_filename: str,
    cache_dir: str | Path | None = ...,
    temp_dir: str | Path | None = ...,
) -> BuildResult: ...
def import_artifact_path(
    artifact_path: str | bytes | Path, *, module_name: str | None = ...
) -> ModuleType: ...
def import_artifact_bytes(
    data: bytes,
    *,
    module_name: str,
    artifact_filename: str,
    temp_dir: str | Path | None = ...,
    key: str | None = ...,
) -> ModuleType: ...
def build_package_from_code_result(
    modules: Mapping[str, str],
    *,
    package_name: str,
    profile: ProfileName | None = ...,
    **kwargs: Any,
) -> PackageBuildResult: ...
def build_package_from_code(
    modules: Mapping[str, str], *, package_name: str, **kwargs: Any
) -> Sequence[ModuleType]: ...
def build_package_from_paths_result(
    modules: Mapping[str, str | Path],
    *,
    package_name: str,
    profile: ProfileName | None = ...,
    **kwargs: Any,
) -> PackageBuildResult: ...
def build_package_from_paths(
    modules: Mapping[str, str | Path], *, package_name: str, **kwargs: Any
) -> Sequence[ModuleType]: ...
def export_cached(key: str, *, dest_dir: str | Path) -> Path: ...

# Templates / workflows

def template_root() -> Path: ...
def list_templates(*, kind: Literal["cython", "python"] = ...) -> list[str]: ...
def get_template_path(
    template_id: str, *, kind: Literal["cython", "python"] | None = ...
) -> Path: ...
def read_template(
    template_id: str,
    *,
    kind: Literal["cython", "python"] | None = ...,
    encoding: str = ...,
) -> str: ...
def load_template_metadata(
    template_id: str, *, kind: Literal["cython", "python"] | None = ...
) -> dict[str, Any]: ...
def compile_template_result(
    template_id: str,
    *,
    module_name: str | None = ...,
    cache_dir: str | Path | None = ...,
    use_cache: bool = ...,
    force_rebuild: bool = ...,
    verbose: int = ...,
    profile: ProfileName | None = ...,
    numpy_support: bool = ...,
    numpy_required: bool | None = ...,
    annotate: bool = ...,
    view_annotate: bool = ...,
    compiler_directives: Mapping[str, Any] | None = ...,
    include_dirs: list[str | Path] | None = ...,
    extra_compile_args: list[str] | None = ...,
    extra_link_args: list[str] | None = ...,
    extra_sources: list[str | Path] | None = ...,
    support_files: Mapping[str, str | bytes] | None = ...,
    support_paths: list[str | Path] | None = ...,
    include_cwd: bool = ...,
    lock_timeout_s: float = ...,
    language: str | None = ...,
) -> BuildResult: ...
def compile_template(
    template_id: str, *, module_name: str | None = ..., **kwargs: Any
) -> ModuleType: ...

# Package examples (multi-module builds)

def list_package_examples() -> list[str]: ...
def get_package_example_path(name: str) -> Path: ...
def load_package_example_metadata(name: str) -> dict[str, Any]: ...
def build_package_example_result(
    name: str,
    *,
    cache_dir: str | Path | None = ...,
    use_cache: bool = ...,
    force_rebuild: bool = ...,
    verbose: int = ...,
    profile: ProfileName | None = ...,
    numpy_support: bool = ...,
    numpy_required: bool = ...,
    include_dirs: list[str | Path] | None = ...,
    extra_compile_args: list[str] | None = ...,
    extra_link_args: list[str] | None = ...,
    compiler_directives: Mapping[str, Any] | None = ...,
    include_cwd: bool = ...,
    lock_timeout_s: float = ...,
    language: str | None = ...,
) -> PackageBuildResult: ...
def build_package_example(name: str, **kwargs: Any) -> Sequence[ModuleType]: ...
def list_workflows() -> list[str]: ...
def get_workflow_path(name: str) -> Path: ...
def workflow_cli_template_path() -> Path: ...
def copy_workflow(
    name: str, *, dest_dir: str | Path, overwrite: bool = ...
) -> Path: ...
def generate_sphinx_template_docs(
    output_dir: str | Path,
    *,
    title: str = ...,
    include_python: bool = ...,
    include_cython: bool = ...,
) -> list[Path]: ...
