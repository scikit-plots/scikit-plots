# scikitplot/cython/_public.py
#
# Flake8: noqa: D213
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
from ._loader import import_extension_from_bytes, import_extension_from_path
from ._pins import resolve_pinned_key as _resolve_pinned_key
from ._profiles import apply_profile
from ._result import (  # noqa: F401
    BatchBuildError,
    BatchBuildResult,
    BatchFailure,
    BuildResult,
    CacheGCResult,
    CacheStats,
    PackageBuildResult,
)
from ._utils import sanitize  # noqa: F401

# A single path-like value (str, bytes, or os.PathLike).
PathLikeAny = str | bytes | Path | os.PathLike[str] | os.PathLike[bytes]

# A sequence of path-like values — the correct type for include_dirs etc.
# NOTE: PathLikeAny was incorrectly used as the type for sequence parameters
# (include_dirs, library_dirs, extra_sources, support_paths) in earlier
# versions.  These parameters must be sequences; a bare path-like is coerced
# to a one-element list at the call site for backward compatibility.
PathLikeSeq = Sequence[PathLikeAny] | PathLikeAny | None


def _dedup_paths(paths: list[Any], *, drop: set[Path] | None = None) -> list[Any]:
    """Deduplicate path-likes by normalized identity, preserving order.

    Two entries that resolve to the same absolute path are collapsed to the
    first occurrence; any entry whose normalized form is in ``drop`` is removed
    (CYTHON-PERF-001).  Non-resolvable entries are kept as-is (deduped by their
    string form) so validation semantics are unchanged — this only removes
    redundant duplicates, never a distinct path.
    """
    drop_norm = {str(d) for d in drop or set()}
    seen: set[str] = set()
    out: list[Any] = []
    for item in paths:
        try:
            norm = str(Path(os.fsdecode(os.fspath(item))).expanduser().resolve())
        except (TypeError, ValueError, OSError):
            norm = repr(item)
        if norm in drop_norm or norm in seen:
            continue
        seen.add(norm)
        out.append(item)
    return out


def _coerce_path_seq(
    val: PathLikeSeq,
    param: str,
) -> list[PathLikeAny] | None:
    """
    Coerce a single path-like or a sequence into a list, or return None.

    Parameters
    ----------
    val : sequence of path-like, or path-like, or None
        Value to coerce.
    param : str
        Parameter name for error messages.

    Returns
    -------
    list or None
        Normalized list, or None when *val* is None.

    Notes
    -----
    This function exists to fix a long-standing API inconsistency where
    ``include_dirs``, ``library_dirs``, ``extra_sources``, and
    ``support_paths`` were typed as ``PathLikeAny | None`` instead of
    ``Sequence[PathLikeAny] | None``.  Passing a single path string now
    works correctly instead of being treated as a character-iteration
    source.
    """
    if val is None:
        return None
    # A str/bytes is itself iterable but should be treated as a single path.
    if isinstance(val, (str, bytes, os.PathLike)):
        return [val]
    try:
        return list(val)
    except TypeError:
        raise TypeError(
            f"'{param}' must be a path-like or a sequence of path-like values, "
            f"got {type(val).__name__!r}"
        ) from None


def _validate_build_security(
    *,
    security_policy: Any | None,
    sources: Sequence[str | None] = (),
    define_macros: Any | None = None,
    extra_compile_args: Any | None = None,
    extra_link_args: Any | None = None,
    include_dirs: Any | None = None,
    libraries: Any | None = None,
    trusted_include_dirs: Any | None = None,
) -> None:
    """Apply the security policy to build inputs for **every** build entrypoint.

    This is the single choke point that guarantees single-module and package
    builds are validated identically (CYTHON-SEC-001).  A ``None`` policy means
    the strict :data:`DEFAULT_SECURITY_POLICY`.  Each source string is checked
    (so per-module package sources are all covered), along with the shared
    macro / compile-arg / link-arg / include-dir / library inputs.

    Parameters
    ----------
    security_policy : SecurityPolicy or None
        Policy to enforce; ``None`` selects the strict default.
    sources : sequence of (str or None), default=()
        One or more Cython source strings to validate (e.g. every module of a
        package).  ``None`` entries are skipped.
    define_macros, extra_compile_args, extra_link_args, include_dirs, libraries
        Shared build inputs forwarded to :func:`validate_build_inputs`.
    trusted_include_dirs : sequence of path-like or None, default=None
        Intrinsic include directories derived from the caller's explicit
        arguments (e.g. the directory of a ``.pyx`` file passed to
        :func:`cython_import_result`).  These are checked for path-traversal
        safety but are exempt from the absolute-path restriction, since an
        absolute source location is normal and is not attacker-supplied.  This
        fixes CYTHON-API-001 without weakening the guard on user-supplied
        ``include_dirs``.

    Raises
    ------
    TypeError
        If ``security_policy`` is not a ``SecurityPolicy`` instance.
    SecurityError
        On the first detected violation.
    """
    from ._security import (  # noqa: PLC0415
        DEFAULT_SECURITY_POLICY,
        SecurityError,
        SecurityPolicy,
        is_safe_path,
        validate_build_inputs,
    )

    policy = security_policy if security_policy is not None else DEFAULT_SECURITY_POLICY
    if not isinstance(policy, SecurityPolicy):
        raise TypeError(
            f"security_policy must be a SecurityPolicy instance, "
            f"got {type(policy).__name__!r}"
        )

    inc_list = _coerce_path_seq(include_dirs, "include_dirs")
    # Validate the shared inputs once, plus each source string.  Passing at
    # least one (possibly None) source keeps a single call in the common case.
    srcs = tuple(sources) or (None,)
    for src in srcs:
        validate_build_inputs(
            policy=policy,
            source=src,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=inc_list,
            libraries=libraries,
        )

    # Intrinsic include dirs: still reject traversal, but permit absolute paths.
    trusted = _coerce_path_seq(trusted_include_dirs, "trusted_include_dirs")
    if trusted:
        for p in trusted:
            ps = os.fsdecode(os.fspath(p)) if not isinstance(p, str) else p
            if not is_safe_path(ps, allow_absolute=True):
                raise SecurityError(
                    f"unsafe source include dir (path traversal): {ps!r}",
                    field="include_dirs",
                )


__all__ = [
    "build_package_from_code",
    "build_package_from_code_result",
    "build_package_from_paths",
    "build_package_from_paths_result",
    "check_build_prereqs",
    "compile_and_load",
    "compile_and_load_result",
    "cython_import",
    "cython_import_all",
    "cython_import_all_result",
    "cython_import_result",
    "export_cached",
    "get_cache_dir",
    "import_artifact_bytes",
    "import_artifact_path",
    "import_cached",
    "import_cached_by_name",
    "import_cached_package",
    "import_cached_package_result",
    "import_cached_result",
    "import_pinned",
    "import_pinned_result",
    "list_cached",
    "list_cached_packages",
    "purge_cache",
    "register_cached_artifact_bytes",
    "register_cached_artifact_path",
]


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

    The purge runs under the cache-root GC lock so it cannot race a concurrent
    garbage collection, and it refuses to run while any per-key build lock is
    held so an active build/publish is never destroyed mid-flight
    (CYTHON-GC-001).  Deeper transactional purge (dry-run manifest and recovery
    journal) is tracked separately.

    Parameters
    ----------
    cache_dir : str or pathlib.Path or None, default=None
        Cache root. If None, uses default.

    Raises
    ------
    FileNotFoundError
        If the cache directory does not exist.
    RuntimeError
        If one or more builds are active (a per-key build lock is held).
    """
    from ._lock import build_lock  # noqa: PLC0415

    root = peek_cache_dir(cache_dir)
    if not root.exists():
        raise FileNotFoundError(str(root))

    gc_lock = root / ".gc.lock"
    with build_lock(gc_lock, timeout_s=60.0):
        # Refuse to purge while any build is active; deleting the root would
        # corrupt an in-flight staging/publish transaction.
        active = sorted(
            p.name[: -len(".lock")] for p in root.glob("*.lock") if p.name != ".gc.lock"
        )
        if active:
            raise RuntimeError(
                "cannot purge cache while builds are active "
                f"({len(active)} lock(s) held); retry when idle"
            )
        # Remove all entry/pin contents while holding the GC lock, preserving
        # only the lock directory itself (released below).
        for child in root.iterdir():
            if child == gc_lock:
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:  # ruff:ignore[suppressible-exception]
                    child.unlink()
                except FileNotFoundError:
                    pass

    # The GC lock has been released (its directory removed).  Remove the now
    # empty cache root to preserve the historical "root deleted" contract.
    try:  # ruff:ignore[suppressible-exception]
        root.rmdir()
    except OSError:
        # A concurrent process re-created content or the root; leave it intact.
        pass


def check_build_prereqs(
    *, numpy: bool = False, pybind11: bool = False
) -> dict[str, Any]:
    """
    Check whether build prerequisites are importable.

    Parameters
    ----------
    numpy : bool, default=False
        If True, also check NumPy availability.
    pybind11 : bool, default=False
        If True, also check pybind11 availability (Scenario 3 & 4).

    Returns
    -------
    dict[str, Any]
        Keys: ``cython``, ``setuptools``, optionally ``numpy``,
        ``pybind11``.  Each value: ``{"ok": bool, "version": str}``.

    Notes
    -----
    **Newbie** (Scenarios 1 & 2): run this to understand your environment.
    **Pro/master** (Scenarios 3-5): use the scenario-specific helpers in
    :mod:`scikitplot.cython._custom_compiler` for targeted checks.

    Examples
    --------
    >>> result = check_build_prereqs()
    >>> "cython" in result and "setuptools" in result
    True
    >>> result = check_build_prereqs(numpy=True, pybind11=True)
    >>> all(k in result for k in ("cython", "setuptools", "numpy", "pybind11"))
    True
    """
    out: dict[str, Any] = {}
    try:
        import Cython  # noqa: PLC0415

        out["cython"] = {"ok": True, "version": getattr(Cython, "__version__", None)}
    except Exception as e:  # noqa: BLE001
        out["cython"] = {"ok": False, "error": str(e)}

    try:
        import setuptools  # noqa: PLC0415

        out["setuptools"] = {
            "ok": True,
            "version": getattr(setuptools, "__version__", None),
        }
    except Exception as e:  # noqa: BLE001
        out["setuptools"] = {"ok": False, "error": str(e)}

    if numpy:
        try:
            import numpy as _numpy  # noqa: ICN001, PLC0415

            out["numpy"] = {
                "ok": True,
                "version": getattr(_numpy, "__version__", None),
            }
        except Exception as e:  # noqa: BLE001
            out["numpy"] = {"ok": False, "error": str(e)}

    if pybind11:
        try:
            import pybind11 as _pybind11  # noqa: PLC0415

            out["pybind11"] = {
                "ok": True,
                "version": getattr(_pybind11, "__version__", None),
                "include": str(_pybind11.get_include()),
            }
        except Exception as e:  # noqa: BLE001
            out["pybind11"] = {"ok": False, "error": str(e)}

    return out


def compile_and_load_result(  # noqa: D417
    source: str,
    *,
    module_name: str | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    force_rebuild: bool = False,
    verbose: int = 0,
    profile: str | None = None,
    annotate: bool | None = None,
    view_annotate: bool = False,
    numpy_support: bool = True,
    numpy_required: bool = False,
    include_dirs: PathLikeSeq = None,
    library_dirs: PathLikeSeq = None,
    libraries: Sequence[str] | None = None,
    define_macros: Sequence[tuple[str, str | None]] | None = None,
    extra_compile_args: Sequence[str] | None = None,
    extra_link_args: Sequence[str] | None = None,
    compiler_directives: Mapping[str, Any] | None = None,
    extra_sources: PathLikeSeq = None,
    support_files: Mapping[str, str | bytes] | None = None,
    support_paths: PathLikeSeq = None,
    include_cwd: bool = True,
    lock_timeout_s: float = 60.0,
    language: str | None = None,
    security_policy: Any | None = None,
    _trusted_include_dirs: PathLikeSeq = None,
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

    # --- Coerce single path-like values to sequences (backward compat fix) ---
    inc_dirs_list = _coerce_path_seq(include_dirs, "include_dirs")
    lib_dirs_list = _coerce_path_seq(library_dirs, "library_dirs")
    extra_sources_list = _coerce_path_seq(extra_sources, "extra_sources")
    support_paths_list = _coerce_path_seq(support_paths, "support_paths")
    trusted_inc_list = _coerce_path_seq(_trusted_include_dirs, "_trusted_include_dirs")

    # --- Security validation (applied before any filesystem or compiler ops) ---
    # Single choke point shared with the package build paths (CYTHON-SEC-001).
    # ``_trusted_include_dirs`` (e.g. a .pyx file's own directory) is validated
    # for traversal but exempt from the absolute-path rule (CYTHON-API-001).
    _validate_build_security(
        security_policy=security_policy,
        sources=(source,),
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=inc_dirs_list,
        libraries=libraries,
        trusted_include_dirs=trusted_inc_list,
    )

    # Merge trusted include dirs into the list handed to the builder.
    if trusted_inc_list:
        inc_dirs_list = [*(inc_dirs_list or []), *trusted_inc_list]

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
        include_dirs=inc_dirs_list,
        library_dirs=lib_dirs_list,
        libraries=libraries,
        define_macros=define_macros,
        extra_compile_args=cargs2,
        extra_link_args=largs2,
        compiler_directives=directives2,
        extra_sources=extra_sources_list,
        support_files=support_files,
        support_paths=support_paths_list,
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
    **kwargs : dict
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
    **kwargs : dict
        Passed to :func:`compile_and_load_result`. The file's parent directory is
        automatically included in include paths.

    Returns
    -------
    scikitplot.cython.BuildResult
        Build result.
    """
    p = Path(os.fsdecode(os.fspath(pyx_path))).expanduser().resolve()
    source = p.read_text(encoding="utf-8")
    # Use _coerce_path_seq so that a bare string like include_dirs="include/"
    # is treated as a single path, not iterated character by character.
    raw_inc = kwargs.pop("include_dirs", None)
    inc: list = list(_coerce_path_seq(raw_inc, "include_dirs") or [])
    # The .pyx file's own directory is an intrinsic, trusted include (needed for
    # sibling .pxd/.pxi).  It is passed via ``_trusted_include_dirs`` so it is
    # exempt from the absolute-path guard that (correctly) applies to
    # user-supplied ``include_dirs`` — otherwise importing any .pyx by its
    # normal absolute path fails under the default strict policy
    # (CYTHON-API-001).
    #
    # Deduplicate the user include dirs by NORMALIZED path, and drop any that
    # coincide with the intrinsic parent, so the parent is not compiled in twice
    # (CYTHON-PERF-001).  Dedup runs AFTER validation, so it cannot weaken any
    # security check; it only removes redundant work.
    parent = p.parent
    inc = _dedup_paths(inc, drop={parent})
    return compile_and_load_result(
        source,
        module_name=module_name,
        include_dirs=inc,
        _trusted_include_dirs=[parent],
        **kwargs,
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
    except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
            meta = {}

    # Import each module in deterministic order
    from ._builder import _ensure_package  # noqa: PLC0415

    pkg_fs_dir = entry.build_dir / entry.package_name.replace(".", os.sep)
    _ensure_package(entry.package_name, pkg_fs_dir)

    results: list[BuildResult] = []
    mods = meta.get("modules")
    if not isinstance(mods, list):
        raise RuntimeError(  # noqa: TRY004
            "Invalid package metadata: missing modules list"
        )
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
        except Exception:  # noqa: BLE001
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
    **kwargs : dict
        Passed to the underlying builder.

    Returns
    -------
    scikitplot.cython.PackageBuildResult
        Package build result.
    """
    annotate = kwargs.pop("annotate", None)
    compiler_directives = kwargs.pop("compiler_directives", None)
    extra_compile_args = kwargs.pop("extra_compile_args", None)
    extra_link_args = kwargs.pop("extra_link_args", None)
    language = kwargs.pop("language", None)

    # Route this stable build path through the SAME policy gate as
    # compile_and_load_result (CYTHON-SEC-001).  Validate every module source
    # plus the shared build inputs before any compiler/filesystem work.
    _validate_build_security(
        security_policy=kwargs.pop("security_policy", None),
        sources=tuple(modules.values()),
        define_macros=kwargs.get("define_macros"),
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=kwargs.get("include_dirs"),
        libraries=kwargs.get("libraries"),
    )

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
    annotate = kwargs.pop("annotate", None)
    compiler_directives = kwargs.pop("compiler_directives", None)
    extra_compile_args = kwargs.pop("extra_compile_args", None)
    extra_link_args = kwargs.pop("extra_link_args", None)
    language = kwargs.pop("language", None)

    # Same policy gate as the code-string and single-module paths
    # (CYTHON-SEC-001).  Read each module's source so source-size limits apply
    # to path-based builds too; validate shared build inputs once.
    _pkg_sources: list[str | None] = []
    for _m in modules.values():
        try:
            _pkg_sources.append(
                Path(os.fsdecode(os.fspath(_m)))
                .expanduser()
                .read_text(encoding="utf-8")
            )
        except OSError:
            # Unreadable path: skip source-size check but still validate the
            # shared build inputs below via the None entry.
            _pkg_sources.append(None)
    _validate_build_security(
        security_policy=kwargs.pop("security_policy", None),
        sources=tuple(_pkg_sources),
        define_macros=kwargs.get("define_macros"),
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=kwargs.get("include_dirs"),
        libraries=kwargs.get("libraries"),
    )

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


def export_cached(
    key: str,
    *,
    dest_dir: str | Path,
    cache_dir: str | Path | None = None,
) -> Path:
    """
    Export a cache entry directory to a destination folder.

    Parameters
    ----------
    key : str
        Cache key to export.
    dest_dir : str or pathlib.Path
        Destination directory. Created (including parents) if absent.
    cache_dir : str or pathlib.Path or None, default=None
        Cache root override. If ``None``, uses the environment override or the
        default cache location. Consistent with all other public functions that
        accept ``cache_dir``.

    Returns
    -------
    pathlib.Path
        Path to the exported entry directory inside ``dest_dir``.

    Raises
    ------
    FileNotFoundError
        If ``key`` does not exist in the cache.
    """
    root = peek_cache_dir(cache_dir)
    src = root / key
    if not src.exists():
        raise FileNotFoundError(str(src))
    dest_root = Path(dest_dir).expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    dst = dest_root / key

    # Transactional export (CYTHON-CACHE-004): copy into a staging sibling, then
    # atomically swap it into place.  A failure mid-copy leaves any prior export
    # untouched (restored), never a half-written destination.
    staging = dest_root / f".staging-{key}"
    backup = dest_root / f".backup-{key}"
    if staging.exists():
        shutil.rmtree(staging)
    if backup.exists():
        shutil.rmtree(backup)
    try:
        shutil.copytree(src, staging)
        # Move any existing export aside, then swap staging in.
        if dst.exists():
            os.replace(dst, backup)
        os.replace(staging, dst)
    except BaseException:
        # Roll back: remove partial staging and restore the prior export.
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if not dst.exists() and backup.exists():
            os.replace(backup, dst)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
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


def cython_import_all_result(
    directory: str | Path,
    *,
    pattern: str = r"*.pyx",
    recursive: bool = False,
    collect: bool = False,
    only: Sequence[str] | None = None,
    **kwargs: Any,
) -> BatchBuildResult:
    r"""
    Compile and import all ``.pyx`` files in a directory, with a partial report.

    Unlike :func:`cython_import_all`, this always returns a structured
    :class:`BatchBuildResult` describing ordered successes, ordered failures,
    and the committed native side effects — so a mid-batch failure no longer
    leaves the caller without a report (CYTHON-BATCH-001).

    Parameters
    ----------
    directory : str or pathlib.Path
        Directory containing ``.pyx`` files.
    pattern : str, default='*.pyx'
        Glob pattern to match files.
    recursive : bool, default=False
        If True, search recursively.
    collect : bool, default=False
        Batch policy.  ``False`` is *fail-fast*: stop at the first failure and
        raise :class:`BatchBuildError` (whose ``result`` carries the partial
        outcome, including a resume token).  ``True`` is *collect*: attempt every
        item and return a :class:`BatchBuildResult` with all failures recorded.
    only : Sequence[str] or None, default=None
        If given, restrict the batch to these stems (e.g. a resume token from a
        prior :class:`BatchBuildError`).
    **kwargs : dict
        Passed to :func:`cython_import_result`.

    Returns
    -------
    BatchBuildResult
        Structured batch result (always, under the ``collect`` policy).

    Raises
    ------
    FileNotFoundError
        If ``directory`` does not exist.
    BatchBuildError
        Under the fail-fast policy (``collect=False``) when an item fails; the
        exception's ``result`` attribute holds the partial :class:`BatchBuildResult`.
    """
    root = Path(directory).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(str(root))
    files = root.rglob(pattern) if recursive else root.glob(pattern)
    candidates = [f for f in sorted(files) if f.is_file()]
    if only is not None:
        wanted = set(only)
        candidates = [f for f in candidates if f.stem in wanted]

    successes: dict[str, BuildResult] = {}
    failures: list[BatchFailure] = []
    policy = "collect" if collect else "fail_fast"

    for index, f in enumerate(candidates):
        try:
            successes[f.stem] = cython_import_result(f, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - reported structurally
            failures.append(
                BatchFailure(
                    name=f.stem,
                    source=f,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            )
            if not collect:
                # Fail-fast: attach the partial result (with a resume token of
                # the not-yet-attempted stems) and raise.
                remaining = [c.stem for c in candidates[index + 1 :]]
                result = BatchBuildResult(
                    successes=dict(successes),
                    failures=list(failures),
                    committed=list(successes),
                    policy=policy,
                )
                # Stash the resume token on the exception via the result's
                # failures/committed; expose it as an attribute for convenience.
                err = BatchBuildError(result)
                err.resume_token = tuple(remaining)  # type: ignore[attr-defined]
                raise err from exc

    return BatchBuildResult(
        successes=dict(successes),
        failures=list(failures),
        committed=list(successes),
        policy=policy,
    )


def cython_import_all(
    directory: str | Path,
    *,
    pattern: str = r"*.pyx",
    recursive: bool = False,
    **kwargs: Any,
) -> dict[str, BuildResult]:
    r"""
    Compile and import all ``.pyx`` files in a directory.

    This is the backward-compatible convenience wrapper: it returns a plain
    ``{stem: BuildResult}`` mapping and is fail-fast.  For a structured partial
    report (ordered successes/failures, committed side effects, resume token) or
    a collect-all policy, use :func:`cython_import_all_result`.

    Parameters
    ----------
    directory : str or pathlib.Path
        Directory containing ``.pyx`` files.
    pattern : str, default='*.pyx'
        Glob pattern to match files.
    recursive : bool, default=False
        If True, search recursively.
    **kwargs : dict
        Passed to :func:`cython_import_result`.

    Returns
    -------
    dict[str, BuildResult]
        Mapping of file stem to build result.

    Raises
    ------
    FileNotFoundError
        If ``directory`` does not exist.
    BatchBuildError
        If any file fails; the exception's ``result`` attribute holds the
        partial :class:`BatchBuildResult` describing what was already committed.
    """
    result = cython_import_all_result(
        directory,
        pattern=pattern,
        recursive=recursive,
        collect=False,
        **kwargs,
    )
    return dict(result.successes)


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
