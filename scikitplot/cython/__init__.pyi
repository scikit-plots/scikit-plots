# scikitplot/cython/__init__.pyi
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# from __future__ import annotations

import os
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Mapping, NamedTuple, Self, Sequence, TypeAlias

PathLikeAny: TypeAlias = str | bytes | Path | os.PathLike[str] | os.PathLike[bytes]
ProfileName: TypeAlias = Literal["fast-debug", "release", "annotate"]
PathLikeSeq: TypeAlias = (
    Sequence[str | bytes | os.PathLike[str] | os.PathLike[bytes]] | None
)

# ---------------------------------------------------------------------------
# _security
# ---------------------------------------------------------------------------

class SecurityError(ValueError):
    field: str | None
    def __init__(self, message: str, *, field: str | None = ...) -> None: ...

class SecurityPolicy:
    strict: bool
    allow_absolute_include_dirs: bool
    allow_shell_metacharacters: bool
    allow_reserved_macros: bool
    allow_dangerous_compiler_args: bool
    max_source_bytes: int | None
    max_extra_compile_args: int
    max_extra_link_args: int
    max_include_dirs: int
    max_libraries: int

    def __init__(
        self,
        *,
        strict: bool = ...,
        allow_absolute_include_dirs: bool | None = ...,
        allow_shell_metacharacters: bool | None = ...,
        allow_reserved_macros: bool | None = ...,
        allow_dangerous_compiler_args: bool | None = ...,
        max_source_bytes: int | None = ...,
        max_extra_compile_args: int = ...,
        max_extra_link_args: int = ...,
        max_include_dirs: int = ...,
        max_libraries: int = ...,
    ) -> None: ...
    @classmethod
    def relaxed(cls) -> Self: ...

DEFAULT_SECURITY_POLICY: SecurityPolicy
RELAXED_SECURITY_POLICY: SecurityPolicy

def is_safe_path(
    path: str | os.PathLike[str], *, allow_absolute: bool = ...
) -> bool: ...
def is_safe_macro_name(name: str, *, allow_reserved: bool = ...) -> bool: ...
def is_safe_compiler_arg(
    arg: str, *, allow_shell_meta: bool = ..., allow_dangerous: bool = ...
) -> bool: ...
def validate_build_inputs(
    *,
    policy: SecurityPolicy | None = ...,
    source: str | None = ...,
    define_macros: Sequence[tuple[str, str | None]] | None = ...,
    extra_compile_args: Sequence[str] | None = ...,
    extra_link_args: Sequence[str] | None = ...,
    include_dirs: Sequence[str | os.PathLike[str]] | None = ...,
    libraries: Sequence[str] | None = ...,
) -> None: ...

# ---------------------------------------------------------------------------
# _custom_compiler
# ---------------------------------------------------------------------------

class CustomCompilerProtocol:
    name: str

    def __call__(
        self,
        source: str,
        *,
        build_dir: Path,
        module_name: str,
        **kwargs: Any,
    ) -> Path: ...

class CompilerRegistry:
    def register(
        self, compiler: CustomCompilerProtocol, *, overwrite: bool = ...
    ) -> None: ...
    def get(self, name: str) -> CustomCompilerProtocol: ...
    def list(self) -> list[str]: ...
    def unregister(self, name: str) -> bool: ...

def register_compiler(
    compiler: CustomCompilerProtocol, *, overwrite: bool = ...
) -> None: ...
def get_compiler(name: str) -> CustomCompilerProtocol: ...
def list_compilers() -> list[str]: ...

COMPILER_SPEC_VERSION: int

class CompilerCapabilities:
    name: str
    spec_version: int
    version: str | None
    supported_languages: frozenset[str]
    platforms: frozenset[str] | None
    requires_cpp: bool
    deterministic: bool
    features: Mapping[str, Any]
    def supports_platform(self, platform: str | None = ...) -> bool: ...
    def supports_language(self, language: str) -> bool: ...

def compiler_capabilities(compiler: Any) -> CompilerCapabilities: ...
def pure_python_prereqs() -> dict[str, Any]: ...
def cython_cpp_prereqs() -> dict[str, Any]: ...
def full_stack_prereqs() -> dict[str, Any]: ...
def pybind11_only_prereqs() -> dict[str, Any]: ...
def c_api_prereqs() -> dict[str, Any]: ...
def pybind11_include() -> Path | None: ...
def numpy_include() -> Path | None: ...
def collect_c_api_sources(
    *paths: str | os.PathLike[str],
    recursive: bool = ...,
    suffixes: frozenset[str] | None = ...,
    exclude_patterns: Sequence[str] | None = ...,
) -> list[Path]: ...
def collect_header_dirs(
    *paths: str | os.PathLike[str],
    recursive: bool = ...,
    suffixes: frozenset[str] | None = ...,
) -> list[Path]: ...

class PybindCompiler:
    name: str

    def __call__(
        self,
        source: str,
        *,
        build_dir: Path,
        module_name: str,
        include_dirs: Sequence[str | Path] | None = ...,
        extra_compile_args: Sequence[str] | None = ...,
        extra_link_args: Sequence[str] | None = ...,
        **kwargs: Any,
    ) -> Path: ...

class CApiCompiler:
    name: str

    def __call__(
        self,
        source: str,
        *,
        build_dir: Path,
        module_name: str,
        extra_sources: Sequence[str | Path] | None = ...,
        include_dirs: Sequence[str | Path] | None = ...,
        extra_compile_args: Sequence[str] | None = ...,
        extra_link_args: Sequence[str] | None = ...,
        **kwargs: Any,
    ) -> Path: ...

# ---------------------------------------------------------------------------
# _result
# ---------------------------------------------------------------------------

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
    template_id: str, *, encoding: str = "utf-8", strict: bool = ...
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

# ---------------------------------------------------------------------------
# _public
# ---------------------------------------------------------------------------

def get_cache_dir(cache_dir: str | Path | None = ...) -> Path: ...
def purge_cache(cache_dir: str | Path | None = ...) -> None: ...
def check_build_prereqs(
    *, numpy: bool = ..., pybind11: bool = ...
) -> dict[str, Any]: ...
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
    include_dirs: Sequence[PathLikeAny] | PathLikeAny | None = ...,
    library_dirs: Sequence[PathLikeAny] | PathLikeAny | None = ...,
    libraries: Sequence[str] | None = ...,
    define_macros: Sequence[tuple[str, str | None]] | None = ...,
    extra_compile_args: Sequence[str] | None = ...,
    extra_link_args: Sequence[str] | None = ...,
    compiler_directives: Mapping[str, Any] | None = ...,
    extra_sources: Sequence[PathLikeAny] | PathLikeAny | None = ...,
    support_files: Mapping[str, str | bytes] | None = ...,
    support_paths: Sequence[PathLikeAny] | PathLikeAny | None = ...,
    include_cwd: bool = ...,
    lock_timeout_s: float = ...,
    language: str | None = ...,
    security_policy: SecurityPolicy | None = ...,
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

class PinRegistryError(ValueError): ...

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
def export_cached(
    key: str, *, dest_dir: str | Path, cache_dir: str | Path | None = ...
) -> Path: ...

# ---------------------------------------------------------------------------
# Templates / workflows
# ---------------------------------------------------------------------------

def template_root() -> Path: ...
def list_templates(*, kind: Literal["cython", "python"] = ...) -> list[str]: ...

class PlatformCapabilities(NamedTuple):
    platform: str
    is_browser_wasm: bool
    can_compile_at_runtime: bool
    prebuilt_only: bool
    wasm_package_suffix: str | None

def platform_capabilities() -> PlatformCapabilities: ...
def verify_template_assets() -> list[str]: ...
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

# ---------------------------------------------------------------------------
# Package examples (multi-module builds)
# ---------------------------------------------------------------------------

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

class Stability(str, Enum):
    STABLE: str
    ADVANCED: str
    EXPERIMENTAL: str

API_STABILITY: dict[str, Stability]

def api_stability(name: str) -> Stability: ...
def list_api(tier: Stability | str) -> list[str]: ...

# --- R18 (CYTHON-TYP-001): runtime/stub parity — introspected declarations ---

DEFAULT_COMPILER_DIRECTIVES: dict[str, Any]

class ProfileDefaults:
    annotate: bool
    compiler_directives: Mapping[str, Any]
    extra_compile_args: tuple[str, ...]
    extra_link_args: tuple[str, ...]
    language: str | None

class AppliedProfile:
    annotate: bool
    compiler_directives: dict[str, Any] | None
    extra_compile_args: tuple[str, ...]
    extra_link_args: tuple[str, ...]
    language: str | None

def apply_profile(
    *,
    profile: str | None,
    annotate: bool | None = ...,
    compiler_directives: Mapping[str, Any] | None,
    extra_compile_args: Sequence[str] | None,
    extra_link_args: Sequence[str] | None,
    language: str | None,
) -> AppliedProfile: ...
def build_extension_module(
    *,
    code: str | None,
    source_path: Path | None,
    module_name: str | None,
    cache_dir: str | Path | None,
    use_cache: bool,
    force_rebuild: bool,
    verbose: int,
    profile: str | None = ...,
    annotate: bool,
    view_annotate: bool,
    numpy_support: bool,
    numpy_required: bool,
    include_dirs: PathLikeSeq | None,
    library_dirs: Sequence[str | Path] | None,
    libraries: Sequence[str] | None,
    define_macros: Sequence[tuple[str, str | None]] | None,
    extra_compile_args: Sequence[str] | None,
    extra_link_args: Sequence[str] | None,
    compiler_directives: Mapping[str, Any] | None,
    extra_sources: PathLikeSeq | None = ...,
    support_files: Mapping[str, str | bytes] | None = ...,
    support_paths: PathLikeSeq | None = ...,
    include_cwd: bool = ...,
    lock_timeout_s: float = ...,
    language: str | None = ...,
) -> ModuleType: ...
def build_extension_module_result(
    *,
    code: str | None,
    source_path: Path | None,
    module_name: str | None,
    cache_dir: str | Path | None,
    use_cache: bool,
    force_rebuild: bool,
    verbose: int,
    profile: str | None = ...,
    annotate: bool,
    view_annotate: bool,
    numpy_support: bool,
    numpy_required: bool,
    include_dirs: PathLikeSeq | None,
    library_dirs: PathLikeSeq | None,
    libraries: Sequence[str] | None,
    define_macros: Sequence[tuple[str, str | None]] | None,
    extra_compile_args: Sequence[str] | None,
    extra_link_args: Sequence[str] | None,
    compiler_directives: Mapping[str, Any] | None,
    extra_sources: PathLikeSeq | None = ...,
    support_files: Mapping[str, str | bytes] | None = ...,
    support_paths: PathLikeSeq | None = ...,
    include_cwd: bool = ...,
    lock_timeout_s: float = ...,
    language: str | None = ...,
    build_timeout_s: float | None = ...,
) -> BuildResult: ...
def build_extension_package_from_code_result(
    modules: Mapping[str, str],
    *,
    package_name: str,
    cache_dir: str | Path | None = ...,
    use_cache: bool = ...,
    force_rebuild: bool = ...,
    verbose: int = ...,
    profile: str | None = ...,
    annotate: bool = ...,
    view_annotate: bool = ...,
    numpy_support: bool = ...,
    numpy_required: bool = ...,
    include_dirs: PathLikeSeq | None = ...,
    library_dirs: PathLikeSeq | None = ...,
    libraries: Sequence[str] | None = ...,
    define_macros: Sequence[tuple[str, str | None]] | None = ...,
    extra_compile_args: Sequence[str] | None = ...,
    extra_link_args: Sequence[str] | None = ...,
    compiler_directives: Mapping[str, Any] | None = ...,
    extra_sources: PathLikeSeq | None = ...,
    support_files: Mapping[str, str | bytes] | None = ...,
    support_paths: PathLikeSeq | None = ...,
    include_cwd: bool = ...,
    lock_timeout_s: float = ...,
    language: str | None = ...,
) -> PackageBuildResult: ...
def build_extension_package_from_paths_result(
    modules: Mapping[str, str | Path],
    *,
    package_name: str,
    cache_dir: str | Path | None = ...,
    use_cache: bool = ...,
    force_rebuild: bool = ...,
    verbose: int = ...,
    profile: str | None = ...,
    annotate: bool = ...,
    view_annotate: bool = ...,
    numpy_support: bool = ...,
    numpy_required: bool = ...,
    include_dirs: Sequence[str | Path] | None = ...,
    library_dirs: Sequence[str | Path] | None = ...,
    libraries: Sequence[str] | None = ...,
    define_macros: Sequence[tuple[str, str | None]] | None = ...,
    extra_compile_args: Sequence[str] | None = ...,
    extra_link_args: Sequence[str] | None = ...,
    compiler_directives: Mapping[str, Any] | None = ...,
    extra_sources: Sequence[str | Path] | None = ...,
    support_files: Mapping[str, str | bytes] | None = ...,
    support_paths: Sequence[str | Path] | None = ...,
    include_cwd: bool = ...,
    lock_timeout_s: float = ...,
    language: str | None = ...,
) -> PackageBuildResult: ...
def build_lock(
    lock_dir: Path,
    *,
    timeout_s: float = ...,
    poll_s: float = ...,
    stale_after_s: float | None = ...,
) -> Iterator[None]: ...
def find_entries_by_name(
    cache_dir: str | Path | None, module_name: str
) -> list[CacheEntry]: ...
def find_entry_by_key(cache_dir: str | Path | None, key: str) -> CacheEntry: ...
def find_package_entry_by_key(
    cache_dir: str | Path | None, key: str
) -> PackageCacheEntry: ...
def import_extension(
    *, name: str, path: Path, key: str | None = ..., build_dir: Path | None = ...
) -> ModuleType: ...
def import_extension_from_bytes(
    data: bytes,
    *,
    module_name: str,
    artifact_filename: str,
    temp_dir: str | os.PathLike[str] | None = ...,
    key: str | None = ...,
) -> ModuleType: ...
def import_extension_from_path(
    artifact_path: str | os.PathLike[str] | os.PathLike[bytes] | bytes,
    *,
    module_name: str | None = ...,
    key: str | None = ...,
    build_dir: Path | None = ...,
) -> ModuleType: ...
def is_valid_key(key: str) -> bool: ...
def is_windows() -> bool: ...
def iter_all_entry_dirs(cache_root: str | Path | None) -> list[Path]: ...
def iter_cache_entries(cache_dir: str | Path | None) -> list[CacheEntry]: ...
def iter_package_entries(cache_dir: str | Path | None) -> list[PackageCacheEntry]: ...
def make_cache_key(payload: Mapping[str, Any]) -> str: ...
def peek_cache_dir(cache_dir: str | Path | None) -> Path: ...
def read_meta(build_dir: Path) -> Mapping[str, Any] | None: ...
def register_artifact_path(
    cache_dir: str | Path | None,
    artifact_path: str | os.PathLike[str] | os.PathLike[bytes] | bytes,
    *,
    module_name: str,
    copy: bool = ...,
) -> CacheEntry: ...
def resolve_cache_dir(cache_dir: str | Path | None) -> Path: ...
def resolve_pinned_key(alias: str, *, cache_dir: str | Path | None = ...) -> str: ...
def resolve_profile(profile: str | None) -> ProfileDefaults: ...
def runtime_fingerprint(
    *, cython_version: str, numpy_version: str | None
) -> Mapping[str, Any]: ...
def sanitize(name: str) -> str: ...
def source_digest(data: bytes) -> str: ...
def write_meta(build_dir: Path, meta: Mapping[str, Any]) -> None: ...

# --- R20 (CYTHON-SCH-001): cache metadata schema versioning ---
CACHE_SCHEMA_VERSION: int

def meta_schema_version(meta: Mapping[str, Any] | None) -> int: ...
def is_meta_schema_compatible(meta: Mapping[str, Any] | None) -> bool: ...

# --- R23 (CYTHON-BATCH-001): batch build API ---
class BatchFailure:
    name: str
    source: Path
    error_type: str
    error: str

class BatchBuildResult:
    successes: Mapping[str, BuildResult]
    failures: Sequence[BatchFailure]
    committed: Sequence[str]
    policy: str
    @property
    def ok(self) -> bool: ...
    @property
    def n_succeeded(self) -> int: ...
    @property
    def n_failed(self) -> int: ...

class BatchBuildError(RuntimeError):
    result: BatchBuildResult
    def __init__(self, result: BatchBuildResult) -> None: ...

def cython_import_all_result(
    directory: str | Path,
    *,
    pattern: str = ...,
    recursive: bool = ...,
    collect: bool = ...,
    only: Sequence[str] | None = ...,
    **kwargs: Any,
) -> BatchBuildResult: ...

# --- R25 (CYTHON-PORT-001): resolved-toolchain contract ---
class ResolvedToolchain(NamedTuple):
    compiler_type: str
    cc: str
    cxx: str
    linker: str

def resolved_toolchain() -> ResolvedToolchain: ...

# --- R26 (CYTHON-TPL-002): strict template validation ---
TEMPLATE_SCHEMA_VERSION: int

class TemplateValidationError(ValueError): ...

def validate_template_info(
    info: TemplateInfo, *, raw: Mapping[str, Any] | None = ...
) -> None: ...

# --- R28 (CYTHON-ABI-001): runtime lifecycle contract ---
class UnsupportedRuntimeError(RuntimeError): ...

class RuntimeCapabilities(NamedTuple):
    gil_enabled: bool
    free_threaded_build: bool
    in_main_interpreter: bool
    supports_unload: bool
    supports_fork_after_load: bool
    platform: str

def runtime_capabilities() -> RuntimeCapabilities: ...
def check_runtime_supported(
    *, allow_free_threaded: bool = ..., allow_subinterpreter: bool = ...
) -> None: ...
