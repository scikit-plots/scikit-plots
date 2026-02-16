# scikitplot/cython/_public.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Public API for :mod:`scikitplot.cython`.

This subpackage provides a small, batteries-included Cython "devkit" that lets
users compile and import Cython extension modules at runtime with caching.

Key capabilities:

- Compile and import a single Cython module from a string or ``.pyx`` file.
- Cache compiled artifacts on disk and re-import them after restarts.
- Pin cache keys under human-friendly aliases.
- Deterministic cache garbage collection.
- Build *packages* containing multiple extension modules in one build directory.
- Browse/compile templates shipped as package data.

Security:

Compiling native code executes a compiler toolchain and imports native code into
the current Python process. Do not compile or import untrusted sources.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

from ._builder import (
    build_extension_module_result,
    build_extension_package_from_code_result,
    build_extension_package_from_paths_result,
)
from ._cache import (
    CacheEntry,
    PackageCacheEntry,
    find_entry_by_key,
    find_package_entry_by_key,
    iter_cache_entries,
    iter_package_entries,
    peek_cache_dir,
    register_artifact_path,
    resolve_cache_dir,
)
from ._gc import cache_stats as _cache_stats
from ._gc import gc_cache as _gc_cache
from ._loader import import_extension_from_bytes, import_extension_from_path
from ._pins import list_pins as _list_pins
from ._pins import pin as _pin
from ._pins import resolve_pinned_key as _resolve_pinned_key
from ._pins import unpin as _unpin
from ._profiles import apply_profile
from ._result import BuildResult, CacheGCResult, CacheStats, PackageBuildResult
from ._util import sanitize  # noqa: F401

PathLikeAny = str | bytes | Path | os.PathLike[str] | os.PathLike[bytes]


def get_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """
    Resolve (and create) the cache root directory.

    Parameters
    ----------
    cache_dir : str or pathlib.Path or None, default=None
        Cache root directory override. If None, uses environment override or a
        default cache location.

    Returns
    -------
    pathlib.Path
        Cache root directory.
    """
    return resolve_cache_dir(cache_dir)


def purge_cache(cache_dir: str | Path | None = None) -> None:
    """
    Delete the entire cache directory.

    Parameters
    ----------
    cache_dir : str or pathlib.Path or None, default=None
        Cache root. If None, uses default.

    Raises
    ------
    FileNotFoundError
        If the cache directory does not exist.
    """
    root = peek_cache_dir(cache_dir)
    if not root.exists():
        raise FileNotFoundError(str(root))
    shutil.rmtree(root)


def check_build_prereqs(*, numpy: bool = False) -> dict[str, Any]:
    """
    Check whether build prerequisites are importable.

    Parameters
    ----------
    numpy : bool, default=False
        If True, also check NumPy availability.

    Returns
    -------
    dict[str, Any]
        A dict with keys: ``cython``, ``setuptools``, ``numpy`` (optional).
    """
    out: dict[str, Any] = {}
    try:
        import Cython  # noqa: PLC0415

        out["cython"] = {"ok": True, "version": getattr(Cython, "__version__", None)}
    except Exception as e:
        out["cython"] = {"ok": False, "error": str(e)}

    try:
        import setuptools  # noqa: PLC0415

        out["setuptools"] = {
            "ok": True,
            "version": getattr(setuptools, "__version__", None),
        }
    except Exception as e:
        out["setuptools"] = {"ok": False, "error": str(e)}

    if numpy:
        try:
            import numpy  # noqa: ICN001, PLC0415

            out["numpy"] = {"ok": True, "version": getattr(numpy, "__version__", None)}
        except Exception as e:
            out["numpy"] = {"ok": False, "error": str(e)}
    return out


def compile_and_load_result(
    source: str,
    *,
    module_name: str | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    force_rebuild: bool = False,
    verbose: int = 0,
    profile: str | None = None,
    annotate: bool = False,
    view_annotate: bool = False,
    numpy_support: bool = True,
    numpy_required: bool = False,
    include_dirs: PathLikeAny | None = None,
    library_dirs: PathLikeAny | None = None,
    libraries: Sequence[str] | None = None,
    define_macros: Sequence[tuple[str, str | None]] | None = None,
    extra_compile_args: Sequence[str] | None = None,
    extra_link_args: Sequence[str] | None = None,
    compiler_directives: Mapping[str, Any] | None = None,
    extra_sources: PathLikeAny | None = None,
    support_files: Mapping[str, str | bytes] | None = None,
    support_paths: PathLikeAny | None = None,
    include_cwd: bool = True,
    lock_timeout_s: float = 60.0,
    language: str | None = None,
) -> BuildResult:
    """
    Compile and import a Cython extension module from source text.

    Parameters
    ----------
    source : str
        Cython source text (``.pyx``-like).
    module_name : str or None, default=None
        Module name to compile/import. If None, a unique deterministic name is
        derived from the *full cache key* (which includes compiler options).
        This avoids module-name collisions when building the same source under
        different flags in the same Python session.
    cache_dir, use_cache, force_rebuild, verbose :
        Cache and logging controls.
    profile : {'fast-debug', 'release', 'annotate'} or None, default=None
        Build profile preset. Explicit arguments always override profile defaults.
    annotate, view_annotate :
        Cython annotation controls.
    numpy_support : bool, default=True
        If True, try to include NumPy headers if NumPy is available.
    numpy_required : bool, default=False
        If True, raise if NumPy is not available.
    include_dirs, library_dirs, libraries, define_macros, extra_compile_args, extra_link_args :
        Compilation parameters passed to setuptools/compilers.
    compiler_directives : Mapping[str, Any] or None, default=None
        Cython compiler directives.
    extra_sources : sequence of path-like, optional
        Extra C/C++ source files to compile and link.
    support_files : Mapping[str, str|bytes] or None, default=None
        Extra support files written into the build directory.
    support_paths : sequence of path-like, optional
        Extra support files copied into the build directory.
    include_cwd : bool, default=True
        Include current working directory in include paths.
    lock_timeout_s : float, default=60.0
        Max seconds to wait for the per-key build lock.
    language : {'c', 'c++'} or None, default=None
        Optional language override.

    Returns
    -------
    scikitplot.cython.BuildResult
        Structured build/import result.
    """
    # Best practice: when no explicit module name is provided, let the builder
    # derive a unique, deterministic name from the full cache key (which includes
    # compiler options). This prevents collisions when the same source is built
    # under different flags in the same Python session.
    mod_name = module_name
    annotate2, directives2, cargs2, largs2, lang2 = apply_profile(
        profile=profile,
        annotate=annotate,
        compiler_directives=compiler_directives,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language=language,
    )

    # NOTE:
    # - Builder API is keyword-only.
    # - Builder parameter name is `module_name` (not `name`).
    # Keep this call explicit to prevent signature drift and to remain
    # Sphinx/autodoc friendly.
    return build_extension_module_result(
        code=source,
        source_path=None,
        module_name=mod_name,
        cache_dir=cache_dir,
        use_cache=use_cache,
        force_rebuild=force_rebuild,
        verbose=verbose,
        profile=profile,
        annotate=annotate2,
        view_annotate=view_annotate,
        numpy_support=numpy_support,
        numpy_required=numpy_required,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        define_macros=define_macros,
        extra_compile_args=cargs2,
        extra_link_args=largs2,
        compiler_directives=directives2,
        extra_sources=extra_sources,
        support_files=support_files,
        support_paths=support_paths,
        include_cwd=include_cwd,
        lock_timeout_s=lock_timeout_s,
        language=lang2,
    )


def compile_and_load(
    source: str, *, module_name: str | None = None, **kwargs: Any
) -> ModuleType:
    """
    Compile and import a Cython extension module and return the loaded module.

    Parameters
    ----------
    source : str
        Cython source text.
    module_name : str or None, default=None
        Module name override.
    **kwargs
        Passed to :func:`compile_and_load_result`.

    Returns
    -------
    types.ModuleType
        Loaded module.
    """
    return compile_and_load_result(source, module_name=module_name, **kwargs).module


def cython_import_result(
    pyx_path: str | os.PathLike[str] | os.PathLike[bytes] | bytes,
    *,
    module_name: str | None = None,
    **kwargs: Any,
) -> BuildResult:
    """
    Compile/import a Cython module from a ``.pyx`` file.

    Parameters
    ----------
    pyx_path : path-like
        Path to a ``.pyx`` file.
    module_name : str or None, default=None
        Module name override. If None, derived deterministically from file content.
    **kwargs
        Passed to :func:`compile_and_load_result`. The file's parent directory is
        automatically included in include paths.

    Returns
    -------
    scikitplot.cython.BuildResult
        Build result.
    """
    p = Path(os.fsdecode(os.fspath(pyx_path))).expanduser().resolve()
    source = p.read_text(encoding="utf-8")
    inc = list(kwargs.pop("include_dirs", []) or [])
    inc.append(p.parent)
    return compile_and_load_result(
        source, module_name=module_name, include_dirs=inc, **kwargs
    )


def cython_import(
    pyx_path: str | os.PathLike[str] | os.PathLike[bytes] | bytes,
    *,
    module_name: str | None = None,
    **kwargs: Any,
) -> ModuleType:
    """
    Compile/import a Cython module from a ``.pyx`` file and return the loaded module.
    """
    return cython_import_result(pyx_path, module_name=module_name, **kwargs).module


def import_cached_result(
    key: str,
    *,
    cache_dir: str | Path | None = None,
) -> BuildResult:
    """
    Import a cached *module* entry by cache key.

    Parameters
    ----------
    key : str
        Cache key.
    cache_dir : str or pathlib.Path or None, default=None
        Cache root override.

    Returns
    -------
    scikitplot.cython.BuildResult
        Import result (``used_cache=True``).

    Raises
    ------
    ValueError
        If key refers to a package entry.
    """
    entry = find_entry_by_key(cache_dir, key)
    mod = import_extension_from_path(
        entry.artifact_path,
        module_name=entry.module_name,
        key=entry.key,
        build_dir=entry.build_dir,
    )
    meta = (
        (entry.build_dir / "meta.json").read_text(encoding="utf-8")
        if (entry.build_dir / "meta.json").exists()
        else "{}"
    )
    try:
        import json  # noqa: PLC0415

        meta_dict = json.loads(meta)
        if not isinstance(meta_dict, dict):
            meta_dict = {}
    except Exception:
        meta_dict = {}
    return BuildResult(
        module=mod,
        key=entry.key,
        module_name=entry.module_name,
        build_dir=entry.build_dir,
        artifact_path=entry.artifact_path,
        used_cache=True,
        created_utc=entry.created_utc,
        fingerprint=entry.fingerprint,
        source_sha256=(
            meta_dict.get("source_sha256")
            if isinstance(meta_dict.get("source_sha256"), str)
            else None
        ),
        meta=meta_dict,
    )


def import_cached(key: str, *, cache_dir: str | Path | None = None) -> ModuleType:
    """
    Import a cached *module* entry and return the loaded module.
    """
    return import_cached_result(key, cache_dir=cache_dir).module


def import_cached_package_result(
    key: str, *, cache_dir: str | Path | None = None
) -> PackageBuildResult:
    """
    Import a cached *package* entry by cache key.

    Parameters
    ----------
    key : str
        Cache key.
    cache_dir : str or pathlib.Path or None, default=None
        Cache root override.

    Returns
    -------
    scikitplot.cython.PackageBuildResult
        Package import result.

    Raises
    ------
    ValueError
        If key does not refer to a package entry.
    """
    entry = find_package_entry_by_key(cache_dir, key)
    # Read meta.json for full fidelity
    meta_path = entry.build_dir / "meta.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            import json  # noqa: PLC0415

            meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta_obj, dict):
                meta = meta_obj
        except Exception:
            meta = {}

    # Import each module in deterministic order
    from ._builder import _ensure_package  # noqa: PLC0415

    pkg_fs_dir = entry.build_dir / entry.package_name.replace(".", os.sep)
    _ensure_package(entry.package_name, pkg_fs_dir)

    results: list[BuildResult] = []
    mods = meta.get("modules")
    if not isinstance(mods, list):
        raise RuntimeError("Invalid package metadata: missing modules list")
    used_cache = True
    for m in sorted(mods, key=lambda d: str(d.get("module_name", ""))):
        if not isinstance(m, dict):
            continue
        mn = m.get("module_name")
        ap = m.get("artifact")
        if not isinstance(mn, str) or not isinstance(ap, str):
            continue
        path = (entry.build_dir / ap).resolve()
        mod = import_extension_from_path(
            path, module_name=mn, key=entry.key, build_dir=entry.build_dir
        )
        results.append(
            BuildResult(
                module=mod,
                key=entry.key,
                module_name=mn,
                build_dir=entry.build_dir,
                artifact_path=path,
                used_cache=used_cache,
                created_utc=entry.created_utc,
                fingerprint=entry.fingerprint,
                source_sha256=(
                    m.get("source_sha256")
                    if isinstance(m.get("source_sha256"), str)
                    else None
                ),
                meta=meta,
            )
        )

    return PackageBuildResult(
        package_name=entry.package_name,
        key=entry.key,
        build_dir=entry.build_dir,
        results=tuple(results),
        used_cache=True,
        created_utc=entry.created_utc,
        fingerprint=entry.fingerprint,
        meta=meta,
    )


def import_cached_package(
    key: str, *, cache_dir: str | Path | None = None
) -> Sequence[ModuleType]:
    """
    Import a cached package and return the loaded modules.
    """
    return import_cached_package_result(key, cache_dir=cache_dir).modules


def list_cached(cache_dir: str | Path | None = None) -> list[CacheEntry]:
    """
    List cached *module* entries.
    """
    return iter_cache_entries(cache_dir)


def list_cached_packages(
    cache_dir: str | Path | None = None,
) -> list[PackageCacheEntry]:
    """
    List cached *package* entries.
    """
    return iter_package_entries(cache_dir)


def cache_stats(cache_dir: str | Path | None = None) -> CacheStats:
    """
    Return cache statistics for the given cache root.
    """
    return _cache_stats(cache_dir)


def gc_cache(
    *,
    cache_dir: str | Path | None = None,
    keep_n_newest: int | None = None,
    max_age_days: int | None = None,
    max_bytes: int | None = None,
    dry_run: bool = False,
    lock_timeout_s: float = 60.0,
) -> CacheGCResult:
    """
    Garbage-collect cached builds deterministically.
    """
    return _gc_cache(
        cache_dir=cache_dir,
        keep_n_newest=keep_n_newest,
        max_age_days=max_age_days,
        max_bytes=max_bytes,
        dry_run=dry_run,
        lock_timeout_s=lock_timeout_s,
    )


def pin(
    key: str,
    *,
    alias: str,
    cache_dir: str | Path | None = None,
    overwrite: bool = False,
    lock_timeout_s: float = 60.0,
) -> str:
    """
    Pin a cache key under a human-friendly alias.

    See Also
    --------
    import_pinned
    list_pins
    unpin
    """
    return _pin(
        key,
        alias=alias,
        cache_dir=cache_dir,
        overwrite=overwrite,
        lock_timeout_s=lock_timeout_s,
    )


def unpin(
    alias: str, *, cache_dir: str | Path | None = None, lock_timeout_s: float = 60.0
) -> bool:
    """
    Remove a pinned alias.
    """
    return _unpin(alias, cache_dir=cache_dir, lock_timeout_s=lock_timeout_s)


def list_pins(cache_dir: str | Path | None = None) -> dict[str, str]:
    """
    List alias→key mappings in the pin registry.
    """
    return _list_pins(cache_dir)


def import_pinned_result(
    alias: str, *, cache_dir: str | Path | None = None
) -> BuildResult | PackageBuildResult:
    """
    Import a pinned alias.

    Parameters
    ----------
    alias : str
        Pinned alias.
    cache_dir : str or pathlib.Path or None, default=None
        Cache root override.

    Returns
    -------
    BuildResult or PackageBuildResult
        If the alias points to a module build, returns BuildResult.
        If the alias points to a package build, returns PackageBuildResult.
    """
    key = _resolve_pinned_key(alias, cache_dir=cache_dir)
    # Decide kind by reading meta.json (strict)
    root = peek_cache_dir(cache_dir)
    meta_path = root / key / "meta.json"
    if meta_path.exists():
        try:
            import json  # noqa: PLC0415

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict) and meta.get("kind") == "package":
                return import_cached_package_result(key, cache_dir=cache_dir)
        except Exception:
            pass
    return import_cached_result(key, cache_dir=cache_dir)


def import_pinned(
    alias: str, *, cache_dir: str | Path | None = None
) -> ModuleType | Sequence[ModuleType]:
    """
    Import a pinned alias and return the loaded module(s).

    Returns
    -------
    types.ModuleType or Sequence[types.ModuleType]
        If alias points to a module build, returns a module.
        If alias points to a package build, returns a list of modules.
    """
    res = import_pinned_result(alias, cache_dir=cache_dir)
    if isinstance(res, BuildResult):
        return res.module
    return res.modules


def register_cached_artifact_path(
    artifact_path: str | os.PathLike[str] | os.PathLike[bytes] | bytes,
    *,
    module_name: str,
    cache_dir: str | Path | None = None,
    copy: bool = True,
) -> BuildResult:
    """
    Register an existing compiled extension artifact on disk, then import it.

    Parameters
    ----------
    artifact_path : path-like
        Path to the compiled artifact.
    module_name : str
        Module name used at compilation time.
    cache_dir : str or pathlib.Path or None, default=None
        Cache root override.
    copy : bool, default=True
        If True, copy artifact into the cache.

    Returns
    -------
    scikitplot.cython.BuildResult
        Imported result.
    """
    entry = register_artifact_path(
        cache_dir, artifact_path, module_name=module_name, copy=copy
    )
    mod = import_extension_from_path(
        entry.artifact_path,
        module_name=entry.module_name,
        key=entry.key,
        build_dir=entry.build_dir,
    )
    return BuildResult(
        module=mod,
        key=entry.key,
        module_name=entry.module_name,
        build_dir=entry.build_dir,
        artifact_path=entry.artifact_path,
        used_cache=True,
        created_utc=entry.created_utc,
        fingerprint=entry.fingerprint,
        source_sha256=None,
        meta={
            "kind": "external",
            "module_name": entry.module_name,
            "artifact": str(entry.artifact_path),
        },
    )


def import_artifact_path(
    artifact_path: str | os.PathLike[str] | os.PathLike[bytes] | bytes,
    *,
    module_name: str | None = None,
) -> ModuleType:
    """
    Import a compiled extension artifact from a path.

    Parameters
    ----------
    artifact_path : path-like
        Artifact path.
    module_name : str or None, default=None
        Module name used at compilation time. If None, attempts to read meta.json
        near the artifact.

    Returns
    -------
    types.ModuleType
        Imported module.
    """
    return import_extension_from_path(artifact_path, module_name=module_name)


def import_artifact_bytes(
    data: bytes,
    *,
    module_name: str,
    artifact_filename: str,
    temp_dir: str | os.PathLike[str] | None = None,
    key: str | None = None,
) -> ModuleType:
    """
    Import a compiled extension artifact from raw bytes.
    """
    return import_extension_from_bytes(
        data,
        module_name=module_name,
        artifact_filename=artifact_filename,
        temp_dir=temp_dir,
        key=key,
    )


def build_package_from_code_result(
    modules: Mapping[str, str],
    *,
    package_name: str,
    profile: str | None = None,
    **kwargs: Any,
) -> PackageBuildResult:
    """
    Build and import a multi-module extension package from code strings.

    Parameters
    ----------
    modules : Mapping[str, str]
        Mapping of module short name to Cython code.
    package_name : str
        Package name.
    profile : {'fast-debug', 'release', 'annotate'} or None, default=None
        Optional build profile preset.
    **kwargs
        Passed to the underlying builder.

    Returns
    -------
    scikitplot.cython.PackageBuildResult
        Package build result.
    """
    annotate = bool(kwargs.pop("annotate", False))
    compiler_directives = kwargs.pop("compiler_directives", None)
    extra_compile_args = kwargs.pop("extra_compile_args", None)
    extra_link_args = kwargs.pop("extra_link_args", None)
    language = kwargs.pop("language", None)

    annotate2, directives2, cargs2, largs2, lang2 = apply_profile(
        profile=profile,
        annotate=annotate,
        compiler_directives=compiler_directives,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language=language,
    )
    return build_extension_package_from_code_result(
        modules,
        package_name=package_name,
        profile=profile,
        annotate=annotate2,
        compiler_directives=directives2,
        extra_compile_args=cargs2,
        extra_link_args=largs2,
        language=lang2,
        **kwargs,
    )


def build_package_from_code(
    modules: Mapping[str, str],
    *,
    package_name: str,
    **kwargs: Any,
) -> Sequence[ModuleType]:
    """
    Build and import a multi-module extension package and return loaded modules.
    """
    return build_package_from_code_result(
        modules, package_name=package_name, **kwargs
    ).modules


def build_package_from_paths_result(
    modules: Mapping[str, str | Path],
    *,
    package_name: str,
    profile: str | None = None,
    **kwargs: Any,
) -> PackageBuildResult:
    """
    Build and import a multi-module extension package from ``.pyx`` file paths.
    """
    annotate = bool(kwargs.pop("annotate", False))
    compiler_directives = kwargs.pop("compiler_directives", None)
    extra_compile_args = kwargs.pop("extra_compile_args", None)
    extra_link_args = kwargs.pop("extra_link_args", None)
    language = kwargs.pop("language", None)

    annotate2, directives2, cargs2, largs2, lang2 = apply_profile(
        profile=profile,
        annotate=annotate,
        compiler_directives=compiler_directives,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language=language,
    )
    return build_extension_package_from_paths_result(
        modules,
        package_name=package_name,
        profile=profile,
        annotate=annotate2,
        compiler_directives=directives2,
        extra_compile_args=cargs2,
        extra_link_args=largs2,
        language=lang2,
        **kwargs,
    )


def build_package_from_paths(
    modules: Mapping[str, str | Path],
    *,
    package_name: str,
    **kwargs: Any,
) -> Sequence[ModuleType]:
    """
    Build and import a multi-module extension package and return loaded modules.
    """
    return build_package_from_paths_result(
        modules, package_name=package_name, **kwargs
    ).modules


def export_cached(key: str, *, dest_dir: str | Path) -> Path:
    """
    Export a cache entry directory to a destination folder.

    Parameters
    ----------
    key : str
        Cache key to export.
    dest_dir : str or pathlib.Path
        Destination directory.

    Returns
    -------
    pathlib.Path
        Exported directory path.
    """
    root = peek_cache_dir(None)
    src = root / key
    if not src.exists():
        raise FileNotFoundError(str(src))
    dest_root = Path(dest_dir).expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    dst = dest_root / key
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def import_cached_by_name(
    module_name: str,
    *,
    cache_dir: str | Path | None = None,
) -> ModuleType:
    """
    Import the newest cached module entry matching ``module_name``.

    Parameters
    ----------
    module_name : str
        Exact module name.
    cache_dir : str or pathlib.Path or None, default=None
        Cache root override.

    Returns
    -------
    types.ModuleType
        Loaded module.

    Raises
    ------
    FileNotFoundError
        If no cached entry matches.
    """
    entries = [e for e in iter_cache_entries(cache_dir) if e.module_name == module_name]
    if not entries:
        raise FileNotFoundError(f"No cached entry for module_name={module_name!r}")
    # Newest build_dir mtime wins
    entries.sort(
        key=lambda e: (
            e.build_dir.stat().st_mtime if e.build_dir.exists() else 0.0,
            e.key,
        ),
        reverse=True,
    )
    return import_cached(entries[0].key, cache_dir=cache_dir)


def cython_import_all(
    directory: str | Path,
    *,
    pattern: str = r"*.pyx",
    recursive: bool = False,
    **kwargs: Any,
) -> dict[str, BuildResult]:
    r"""
    Compile and import all ``.pyx`` files in a directory.

    Parameters
    ----------
    directory : str or pathlib.Path
        Directory containing ``.pyx`` files.
    pattern : str, default='*.pyx'
        Glob pattern to match files.
    recursive : bool, default=False
        If True, search recursively.
    **kwargs
        Passed to :func:`cython_import_result`.

    Returns
    -------
    dict[str, BuildResult]
        Mapping of file stem to build result.
    """
    root = Path(directory).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(str(root))
    files = root.rglob(pattern) if recursive else root.glob(pattern)
    out: dict[str, BuildResult] = {}
    for f in sorted(files):
        if f.is_file():
            res = cython_import_result(f, **kwargs)
            out[f.stem] = res
    return out


def register_cached_artifact_bytes(
    data: bytes,
    *,
    module_name: str,
    artifact_filename: str,
    cache_dir: str | Path | None = None,
    temp_dir: str | os.PathLike[str] | None = None,
) -> BuildResult:
    """
    Register a compiled extension artifact from bytes and import it.

    Parameters
    ----------
    data : bytes
        Artifact bytes.
    module_name : str
        Module name used at compilation time.
    artifact_filename : str
        Artifact filename ending with a valid extension suffix (e.g., ``.so`` / ``.pyd``).
    cache_dir : str or pathlib.Path or None, default=None
        Cache root override.
    temp_dir : str or os.PathLike or None, default=None
        Temporary directory used to stage the artifact before registering.

    Returns
    -------
    scikitplot.cython.BuildResult
        Imported result.
    """
    # Import from bytes into deterministic temp path, then register by path (copy).
    mod = import_extension_from_bytes(
        data,
        module_name=module_name,
        artifact_filename=artifact_filename,
        temp_dir=temp_dir,
        key=None,
    )
    # Module has __scikitplot_cython_artifact__ path
    ap = getattr(mod, "__scikitplot_cython_artifact__", None)
    if not isinstance(ap, str):
        # Fallback: write to a temp file
        import tempfile  # noqa: PLC0415

        td = Path(temp_dir) if temp_dir is not None else Path(tempfile.gettempdir())
        td.mkdir(parents=True, exist_ok=True)
        staged = td / artifact_filename
        staged.write_bytes(data)
        ap = str(staged)

    return register_cached_artifact_path(
        ap, module_name=module_name, cache_dir=cache_dir, copy=True
    )
