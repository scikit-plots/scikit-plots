# scikitplot/cython/_builder.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Build an extension module from Cython source.

This implementation is intentionally self-contained (no Sage dependency) and is
licensed under the scikitplot project license. It is *inspired by* the Sage user
experience but does not copy or port Sage source code.

Notes
-----
This module performs native compilation. It must never run implicitly on import.
All compilation is opt-in via the public API.

.. rubric:: Security:

* Compiling and importing native code is inherently unsafe with untrusted inputs.
  Treat all inputs as fully trusted.
"""

from __future__ import annotations

import os  # noqa: F401
import shutil
import sys
import webbrowser
from dataclasses import asdict  # noqa: F401
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence, TypeAlias

# Canonical path-like type accepted by public/internal APIs.
PathLike: TypeAlias = os.PathLike[str] | os.PathLike[bytes] | str | bytes
PathLikeSeq: TypeAlias = Sequence[PathLike]

from ._cache import (
    make_cache_key,
    read_meta,
    resolve_cache_dir,
    runtime_fingerprint,
    source_digest,
    write_meta,
)
from ._loader import import_extension
from ._lock import build_lock
from ._result import BuildResult, PackageBuildResult
from ._util import sanitize

_ALLOWED_SUPPORT_NAME = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)

_ALLOWED_EXTRA_SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
}


def _to_path(p: PathLike) -> Path:
    """
    Convert a path-like object to an absolute :class:`~pathlib.Path`.

    Parameters
    ----------
    p : path-like
        A filesystem path as ``str``/``bytes`` or ``os.PathLike``.

    Returns
    -------
    pathlib.Path
        Absolute, resolved path.
    """

    fs = os.fspath(p)
    if isinstance(fs, (bytes, bytearray)):
        fs = os.fsdecode(fs)
    return Path(str(fs)).expanduser().resolve()


def _normalize_extra_sources(
    extra_sources: PathLikeSeq | None,
) -> list[Path]:
    """
    Normalize and validate ``extra_sources``.

    Parameters
    ----------
    extra_sources : sequence of path-like, or None
        Optional sequence of additional C/C++ source files to compile and link into
        the extension module.

        Accepted path types include:

        - ``str`` / ``bytes`` filesystem paths
        - ``os.PathLike[str]`` / ``os.PathLike[bytes]`` (e.g., :class:`pathlib.Path`)

    Returns
    -------
    list[pathlib.Path]
        Absolute, resolved file paths.

    Raises
    ------
    FileNotFoundError
        If any provided source file does not exist.
    ValueError
        If any provided source file has an unsupported suffix or duplicates are found.

    Notes
    -----
    This function performs *path-level* validation only. Build-directory level
    constraints (such as duplicate basenames after copying) are enforced by
    :func:`_copy_extra_sources`.
    """

    if not extra_sources:
        return []

    out: list[Path] = []
    seen: set[str] = set()

    for p in extra_sources:
        fs = os.fspath(p)
        if isinstance(fs, (bytes, bytearray)):
            fs = os.fsdecode(fs)
        src = Path(str(fs)).expanduser().resolve()

        if not src.exists() or not src.is_file():
            raise FileNotFoundError(str(src))
        if src.suffix.lower() not in _ALLOWED_EXTRA_SOURCE_SUFFIXES:
            raise ValueError(
                f"extra source must be one of {sorted(_ALLOWED_EXTRA_SOURCE_SUFFIXES)}, got: {src.name}"
            )

        key = str(src)
        if key in seen:
            raise ValueError(f"duplicate extra source path: {src}")
        seen.add(key)
        out.append(src)

    return out


# Default compiler directives favor correctness and helpful introspection over micro-optimizations.
# Users can override any directive via the public API `compiler_directives=` or via build profiles.
DEFAULT_COMPILER_DIRECTIVES: Mapping[str, Any] = {
    "language_level": 3,
    "embedsignature": True,
}


def build_extension_module(  # noqa: D417
    *,
    code: str | None,
    source_path: Path | None,
    module_name: str | None,
    cache_dir: str | Path | None,
    use_cache: bool,
    force_rebuild: bool,
    verbose: int,
    profile: str | None = None,
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
    # --- new in v6 ---
    extra_sources: PathLikeSeq | None = None,
    support_files: Mapping[str, str | bytes] | None = None,
    support_paths: PathLikeSeq | None = None,
    include_cwd: bool = True,
    lock_timeout_s: float = 60.0,
    language: str | None = None,
) -> ModuleType:
    """
    Compile and import an extension module (module-only convenience wrapper).

    This is the internal implementation used by the public API. It returns only
    the imported module. For structured metadata, use
    :func:`build_extension_module_result`.

    Parameters
    ----------
    All parameters
        See :func:`build_extension_module_result`.

    Returns
    -------
    types.ModuleType
        Imported extension module.
    """
    return build_extension_module_result(
        code=code,
        source_path=source_path,
        module_name=module_name,
        cache_dir=cache_dir,
        use_cache=use_cache,
        force_rebuild=force_rebuild,
        verbose=verbose,
        profile=profile,
        annotate=annotate,
        view_annotate=view_annotate,
        numpy_support=numpy_support,
        numpy_required=numpy_required,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        compiler_directives=compiler_directives,
        extra_sources=extra_sources,
        support_files=support_files,
        support_paths=support_paths,
        include_cwd=include_cwd,
        lock_timeout_s=lock_timeout_s,
        language=language,
    ).module


def build_extension_module_result(  # noqa: PLR0912
    *,
    code: str | None,
    source_path: Path | None,
    module_name: str | None,
    cache_dir: str | Path | None,
    use_cache: bool,
    force_rebuild: bool,
    verbose: int,
    profile: str | None = None,
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
    # --- v6: multi-file/module customization ---
    extra_sources: PathLikeSeq | None = None,
    support_files: Mapping[str, str | bytes] | None = None,
    support_paths: PathLikeSeq | None = None,
    include_cwd: bool = True,
    lock_timeout_s: float = 60.0,
    language: str | None = None,
) -> BuildResult:
    """
    Compile and import an extension module, with deterministic caching.

    Parameters
    ----------
    code : str or None
        Cython source string that is valid for a ``.pyx`` file. Exactly one of
        ``code`` or ``source_path`` must be provided.
    source_path : pathlib.Path or None
        Path to a ``.pyx`` file. Exactly one of ``code`` or ``source_path`` must
        be provided.
    module_name : str or None
        Optional explicit module name. If None, a deterministic name is derived
        from the cache key.
    cache_dir : str or pathlib.Path or None
        Root cache directory for build artifacts.
    use_cache : bool
        If True, reuse a previously compiled artifact when the cache key matches.
    force_rebuild : bool
        If True, rebuild even if a cached artifact exists.
    verbose : int
        Build verbosity. Negative values suppress most build output.
    profile : str or None
        Optional profile label (e.g., "fast-debug", "release", "annotate").
        This value is persisted in metadata for introspection but does not affect
        the cache key by itself (the actual build options do).
    annotate : bool
        If True, generate a Cython annotation HTML file.
    view_annotate : bool
        If True, open the annotation HTML file in a browser (requires annotate=True).
    numpy_support : bool
        If True, attempt to add NumPy include directories when NumPy is installed.
    numpy_required : bool
        If True and numpy_support=True, raise ImportError when NumPy is not installed.
    include_dirs, library_dirs, libraries, define_macros, extra_compile_args, extra_link_args :
        Standard extension build options passed to the C/C++ compiler and linker.
    compiler_directives : Mapping[str, Any] or None
        Cython compiler directives (default includes ``language_level=3``).
    extra_sources : sequence of path-like, optional
        Additional C/C++ sources to compile and link into the extension.
        Only files ending in {allowed} are accepted.
    support_files : Mapping[str, str | bytes] or None, optional
        Extra *support* files written into the build directory when compiling from
        a source string (e.g., ``.pxi`` / ``.pxd`` / headers). Keys must be simple
        filenames (no directories). Values are written as UTF-8 text for ``str``
        and as raw bytes for ``bytes``.
    support_paths : sequence of path-like, optional
        Extra support files copied into the build directory when compiling from a
        source file. Each file is copied by basename into the build directory.
    include_cwd : bool, default=True
        If True, include the current working directory in include paths.
    lock_timeout_s : float, default=60.0
        Maximum seconds to wait for the per-key build lock.
    language : {{'c', 'c++'}} or None, default=None
        Optional explicit language for the extension. If None, the build uses the
        default compiler behavior.

    Returns
    -------
    scikitplot.cython.BuildResult
        Structured build result including the imported module and metadata.

    Raises
    ------
    ValueError
        If inputs are invalid or unsupported.
    ImportError
        If required build dependencies are missing (Cython, setuptools, and optionally NumPy).
    RuntimeError
        If compilation fails.

    Notes
    -----
    The returned module is annotated with:

    - ``__scikitplot_cython_key__``: cache key
    - ``__scikitplot_cython_build_dir__``: build directory
    - ``__scikitplot_cython_artifact__``: compiled artifact path
    """

    if (code is None) == (source_path is None):
        raise ValueError("Provide exactly one of: code or source_path")

    if view_annotate and not annotate:
        raise ValueError("view_annotate=True requires annotate=True")

    if language is not None and language not in {"c", "c++"}:
        raise ValueError("language must be one of: None, 'c', 'c++'")

    # --- imports here to keep scikitplot import lightweight ---
    try:
        import Cython  # noqa: PLC0415
        from Cython.Build import cythonize  # noqa: PLC0415
    except Exception as e:
        raise ImportError(
            "Cython is required for scikitplot.cython. Install with: pip install Cython"
        ) from e

    numpy_include: Path | None = None
    numpy_version: str | None = None
    if numpy_support:
        try:
            import numpy as np  # noqa: F401, PLC0415

            numpy_version = np.__version__
            numpy_include = Path(np.get_include())
        except Exception as e:
            if numpy_required:
                raise ImportError(
                    "NumPy is required when numpy_required=True. "
                    "Install with: pip install numpy  (or pass numpy_required=False)"
                ) from e
            numpy_version = None
            numpy_include = None

    cache_root = resolve_cache_dir(cache_dir)

    # Normalize include dirs (strict: absolute, resolved paths)
    inc_dirs: list[Path] = []
    if include_dirs is not None:
        inc_dirs.extend(_to_path(p) for p in include_dirs)
    if source_path is not None:
        inc_dirs.append(Path(source_path).expanduser().resolve().parent)
    if include_cwd:
        inc_dirs.append(Path.cwd().resolve())
    if numpy_include is not None:
        inc_dirs.append(numpy_include.expanduser().resolve())

    inc_dirs_norm = [p.resolve() for p in inc_dirs]
    lib_dirs_norm = [_to_path(p) for p in (library_dirs or [])]

    directives: dict[str, Any] = dict(DEFAULT_COMPILER_DIRECTIVES)
    if compiler_directives is not None:
        directives.update(dict(compiler_directives))

    cython_version = getattr(Cython, "__version__", "unknown")
    fp = runtime_fingerprint(cython_version=cython_version, numpy_version=numpy_version)

    # Main source hash
    src_bytes = (
        code.encode("utf-8") if code is not None else Path(source_path).read_bytes()
    )
    src_hash = source_digest(src_bytes)

    explicit_name = sanitize(module_name) if module_name is not None else None

    # Support/support_path digests (for caching correctness)
    support_file_digests = _support_files_digest(support_files)
    support_path_digests = _support_paths_digest(support_paths)
    extra_sources_norm = _normalize_extra_sources(extra_sources)
    extra_source_digests = _support_paths_digest(extra_sources_norm)

    # Cache key (strict + deterministic)
    key = make_cache_key(
        {
            "source_sha256": src_hash,
            "source_path": (
                str(Path(source_path).expanduser().resolve())
                if source_path is not None
                else None
            ),
            "explicit_module_name": explicit_name,
            "directives": directives,
            "include_dirs": [p.as_posix() for p in inc_dirs_norm],
            "include_cwd": include_cwd,
            "library_dirs": [p.as_posix() for p in lib_dirs_norm],
            "libraries": list(libraries or []),
            "define_macros": list(define_macros or []),
            "extra_compile_args": list(extra_compile_args or []),
            "extra_link_args": list(extra_link_args or []),
            "fingerprint": dict(fp),
            "annotate": annotate,
            "language": language,
            "support_files": support_file_digests,
            "support_paths": support_path_digests,
            "extra_sources": extra_source_digests,
        }
    )

    default_modname = f"scikitplot_cython_{key[:16]}"
    name = explicit_name or sanitize(default_modname)

    build_dir = cache_root / key
    lock_dir = build_dir.with_suffix(".lock")
    build_dir.mkdir(parents=True, exist_ok=True)

    # Ensure support files exist in build dir *before* compilation
    pyx_path = build_dir / f"{name}.pyx"
    if code is not None:
        pyx_path.write_text(code, encoding="utf-8")
        _write_support_files(build_dir, support_files, reserved={pyx_path.name})
    else:
        srcp = Path(source_path).expanduser().resolve()
        pyx_path.write_bytes(srcp.read_bytes())
        _copy_support_paths(build_dir, support_paths, reserved={pyx_path.name})

    # Copy extra sources into build dir and compile from that location
    extra_source_paths = _copy_extra_sources(
        build_dir, extra_sources_norm, reserved={pyx_path.name}
    )

    # Add build_dir to include dirs to support local includes/includes of copied files
    if build_dir not in inc_dirs_norm:
        inc_dirs_norm = [build_dir, *inc_dirs_norm]

    used_cache = False

    with build_lock(lock_dir, timeout_s=float(lock_timeout_s)):
        ext_path = _find_built_extension(build_dir, name)

        if use_cache and (not force_rebuild) and (ext_path is not None):
            used_cache = True
            _ensure_meta(
                build_dir=build_dir,
                key=key,
                module_name=name,
                artifact_path=ext_path,
                fingerprint=dict(fp),
                source_sha256=src_hash,
                directives=directives,
                include_dirs=[p.as_posix() for p in inc_dirs_norm],
                support_files=support_file_digests,
                support_paths=support_path_digests,
                extra_sources=extra_source_digests,
                language=language,
                profile=profile,
                annotate=annotate,
                view_annotate=view_annotate,
                extra_compile_args=list(extra_compile_args or []),
                extra_link_args=list(extra_link_args or []),
            )
            module = import_extension(
                name=name, path=ext_path, key=key, build_dir=build_dir
            )
            meta = read_meta(build_dir) or {}
            return BuildResult(
                module=module,
                key=key,
                module_name=name,
                build_dir=build_dir,
                artifact_path=ext_path,
                used_cache=True,
                created_utc=meta.get("created_utc"),
                fingerprint=meta.get("fingerprint"),
                source_sha256=meta.get("source_sha256"),
                meta=meta,
            )

        # Clean old *build artifacts* for this key and module name.
        # Never delete the authoritative source file ``{name}.pyx``.
        _clean_build_artifacts(build_dir=build_dir, name=name, keep={pyx_path.name})

        ext_path = _compile(
            name=name,
            pyx_path=pyx_path,
            build_dir=build_dir,
            include_dirs=inc_dirs_norm,
            library_dirs=lib_dirs_norm,
            libraries=list(libraries or []),
            define_macros=list(define_macros or []),
            extra_compile_args=list(extra_compile_args or []),
            extra_link_args=list(extra_link_args or []),
            directives=directives,
            cythonize=cythonize,
            annotate=annotate,
            verbose=verbose,
            extra_sources=extra_source_paths,
            language=language,
        )

        _ensure_meta(
            build_dir=build_dir,
            key=key,
            module_name=name,
            artifact_path=ext_path,
            fingerprint=dict(fp),
            source_sha256=src_hash,
            directives=directives,
            include_dirs=[p.as_posix() for p in inc_dirs_norm],
            support_files=support_file_digests,
            support_paths=support_path_digests,
            extra_sources=extra_source_digests,
            language=language,
            profile=profile,
            annotate=annotate,
            view_annotate=view_annotate,
            extra_compile_args=list(extra_compile_args or []),
            extra_link_args=list(extra_link_args or []),
        )

        if view_annotate:
            html = _find_annotation(build_dir, name)
            if html is not None:
                webbrowser.open(html.as_uri())

        module = import_extension(
            name=name, path=ext_path, key=key, build_dir=build_dir
        )
        meta = read_meta(build_dir) or {}
        return BuildResult(
            module=module,
            key=key,
            module_name=name,
            build_dir=build_dir,
            artifact_path=ext_path,
            used_cache=used_cache,
            created_utc=meta.get("created_utc"),
            fingerprint=meta.get("fingerprint"),
            source_sha256=meta.get("source_sha256"),
            meta=meta,
        )


build_extension_module_result.__doc__ = build_extension_module_result.__doc__.format(
    allowed=sorted(_ALLOWED_EXTRA_SOURCE_SUFFIXES)
)


def _clean_build_artifacts(*, build_dir: Path, name: str, keep: set[str]) -> None:
    artifact_suffixes = {".c", ".cpp", ".html", ".o", ".obj"}
    for p in build_dir.iterdir():
        if not p.is_file():
            continue
        if p.name in keep:
            continue
        is_ext = any(p.name.endswith(suf) for suf in EXTENSION_SUFFIXES)
        is_artifact = p.suffix in artifact_suffixes
        # Strict: only remove files clearly part of this module's output.
        if is_ext or (is_artifact and p.stem.startswith(name)):
            try:  # noqa: SIM105
                p.unlink()
            except FileNotFoundError:
                pass


def _ensure_meta(
    *,
    build_dir: Path,
    key: str,
    module_name: str,
    artifact_path: Path,
    fingerprint: Mapping[str, Any],
    source_sha256: str,
    directives: Mapping[str, Any],
    include_dirs: list[str],
    support_files: list[tuple[str, str]],
    support_paths: list[tuple[str, str]],
    extra_sources: list[tuple[str, str]],
    language: str | None,
    # --- v11.1: persist user-facing build options ---
    profile: str | None = None,
    annotate: bool = False,
    view_annotate: bool = False,
    extra_compile_args: list[str] | None = None,
    extra_link_args: list[str] | None = None,
) -> None:
    meta = read_meta(build_dir) or {}
    meta_out = dict(meta)

    # Annotation HTML is produced by Cython when `annotate=True`.
    # For deterministic reuse, record the path if it exists.
    html_path = build_dir / f"{module_name}.html"
    annotate_html = html_path.as_posix() if html_path.exists() else None

    meta_out.update(
        {
            "key": key,
            "kind": "module",
            "module_name": module_name,
            "artifact": artifact_path.name,
            "artifact_filename": artifact_path.name,  # backward compatibility
            "created_utc": meta.get("created_utc") or _utc_now_iso(),
            "fingerprint": dict(fingerprint),
            "source_sha256": source_sha256,
            "directives": dict(directives),
            "compiler_directives": dict(directives),
            "profile": profile,
            "annotate": bool(annotate),
            "view_annotate": bool(view_annotate),
            "extra_compile_args": list(extra_compile_args or []),
            "extra_link_args": list(extra_link_args or []),
            "annotate_html": annotate_html,
            "annotation_html": annotate_html,
            "include_dirs": list(include_dirs),
            "support_files": list(support_files),
            "support_paths": list(support_paths),
            "extra_sources": list(extra_sources),
            "language": language,
        }
    )
    write_meta(build_dir, meta_out)


def _utc_now_iso() -> str:
    from datetime import datetime, timezone  # noqa: PLC0415

    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _support_files_digest(
    support_files: Mapping[str, str | bytes] | None,
) -> list[tuple[str, str]]:
    if not support_files:
        return []
    out: list[tuple[str, str]] = []
    for name, content in sorted(support_files.items(), key=lambda kv: kv[0]):
        _validate_support_filename(name)
        data = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        out.append((name, source_digest(data)))
    return out


def _support_paths_digest(
    paths: PathLikeSeq | None,
) -> list[tuple[str, str]]:
    if not paths:
        return []
    out: list[tuple[str, str]] = []
    for p in [_to_path(x) for x in paths]:
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(str(p))
        out.append((p.name, source_digest(p.read_bytes())))
    # deterministic order
    out.sort(key=lambda t: t[0])
    return out


def _validate_support_filename(name: str) -> None:
    if not name:
        raise ValueError("support file name must be non-empty")
    if "/" in name or "\\" in name:
        raise ValueError("support file name must be a simple filename (no directories)")
    if any(ch not in _ALLOWED_SUPPORT_NAME for ch in name):
        raise ValueError(f"unsupported character in support filename: {name!r}")


def _write_support_files(
    build_dir: Path, support_files: Mapping[str, str | bytes] | None, reserved: set[str]
) -> None:
    if not support_files:
        return
    for fn, content in support_files.items():
        _validate_support_filename(fn)
        if fn in reserved:
            raise ValueError(f"support file name collides with reserved file: {fn}")
        out = build_dir / fn
        if isinstance(content, str):
            out.write_text(content, encoding="utf-8")
        else:
            out.write_bytes(bytes(content))


def _copy_support_paths(
    build_dir: Path,
    paths: PathLikeSeq | None,
    reserved: set[str],
) -> None:
    if not paths:
        return
    seen: set[str] = set()
    for src in [_to_path(p) for p in paths]:
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(str(src))
        name = src.name
        _validate_support_filename(name)
        if name in reserved:
            raise ValueError(f"support path name collides with reserved file: {name}")
        if name in seen:
            raise ValueError(f"duplicate support path basename: {name}")
        seen.add(name)
        (build_dir / name).write_bytes(src.read_bytes())


def _copy_extra_sources(
    build_dir: Path,
    sources: PathLikeSeq | None,
    reserved: set[str],
) -> list[Path]:
    if not sources:
        return []
    out: list[Path] = []
    seen: set[str] = set()
    for src in [_to_path(p) for p in sources]:
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(str(src))
        if src.suffix.lower() not in _ALLOWED_EXTRA_SOURCE_SUFFIXES:
            raise ValueError(
                f"extra source must be one of {sorted(_ALLOWED_EXTRA_SOURCE_SUFFIXES)}, got: {src.name}"
            )
        name = src.name
        _validate_support_filename(name)
        if name in reserved:
            raise ValueError(f"extra source name collides with reserved file: {name}")
        if name in seen:
            raise ValueError(f"duplicate extra source basename: {name}")
        seen.add(name)
        dst = build_dir / name
        dst.write_bytes(src.read_bytes())
        out.append(dst)
    return out


def _compile(
    *,
    name: str,
    pyx_path: Path,
    build_dir: Path,
    include_dirs: list[Path],
    library_dirs: list[Path],
    libraries: list[str],
    define_macros: list[tuple[str, str | None]],
    extra_compile_args: list[str],
    extra_link_args: list[str],
    directives: Mapping[str, Any],
    cythonize: Any,
    annotate: bool,
    verbose: int,
    extra_sources: list[Path],
    language: str | None,
) -> Path:
    """Compile the extension module into build_dir."""
    # NOTE:
    # Cython compilation failures are frequently reported to stdout/stderr
    # (and may not be included in the raised exception message). To provide
    # user/dev-friendly diagnostics (especially when verbose=0 in docs builds),
    # we capture output during cythonize and include a trimmed tail on failure.
    try:
        from setuptools import Extension  # noqa: PLC0415
        from setuptools.dist import Distribution  # noqa: PLC0415
    except Exception as e:
        raise ImportError("setuptools is required to compile Cython extensions.") from e

    _set_verbosity(verbose)

    sources = [str(pyx_path)] + [str(p) for p in extra_sources]

    ext = Extension(
        name=name,
        sources=sources,
        include_dirs=[str(p) for p in include_dirs],
        library_dirs=[str(p) for p in library_dirs],
        libraries=libraries,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language=language,
    )

    try:
        import contextlib  # noqa: PLC0415
        import io  # noqa: PLC0415
        import sys  # noqa: PLC0415

        class _Tee:
            """Write-through stream for capturing while still displaying output."""

            def __init__(self, primary: Any, secondary: io.StringIO) -> None:
                self._primary = primary
                self._secondary = secondary

            def write(self, s: str) -> int:  # pragma: no cover
                # Best-effort: write to both streams.
                try:  # noqa: SIM105
                    self._secondary.write(s)
                except Exception:
                    pass
                return self._primary.write(s)

            def flush(self) -> None:  # pragma: no cover
                try:  # noqa: SIM105
                    self._secondary.flush()
                except Exception:
                    pass
                try:  # noqa: SIM105
                    self._primary.flush()
                except Exception:
                    pass

        buf = io.StringIO()
        out_stream: Any
        err_stream: Any
        if verbose > 0:
            out_stream = _Tee(sys.stdout, buf)
            err_stream = _Tee(sys.stderr, buf)
        else:
            out_stream = buf
            err_stream = buf

        with contextlib.redirect_stdout(out_stream), contextlib.redirect_stderr(
            err_stream
        ):
            try:
                ext_modules = cythonize(
                    [ext],
                    compiler_directives=dict(directives),
                    annotate=annotate,
                    quiet=(verbose < 0),
                )
            except TypeError:
                # Cython API drift: retry without the optional `quiet` kwarg.
                ext_modules = cythonize(
                    [ext],
                    compiler_directives=dict(directives),
                    annotate=annotate,
                )
    except Exception as e:
        tail = ""
        try:
            txt = buf.getvalue()
            # Trim to a reasonable tail to avoid huge exception messages.
            lines = txt.splitlines()
            tail_lines = lines[-60:] if len(lines) > 60 else lines  # noqa: PLR2004
            if tail_lines:
                tail = "\n" + "\n".join(tail_lines)
        except Exception:
            tail = ""

        raise RuntimeError(
            f"Cythonize failed for module '{name}'.\n"
            f"Source: {pyx_path}\n"
            "Hint: rerun with verbose=1 to see full compiler/Cython output."
            f"{tail}"
        ) from e

    dist = Distribution()
    dist.ext_modules = ext_modules

    try:
        cmd = dist.get_command_obj("build_ext")
    except Exception as e:
        raise ImportError(
            "setuptools build_ext command is unavailable; cannot compile extensions. "
            "Ensure setuptools is installed and not patched by a broken build environment."
        ) from e
    cmd.build_lib = str(build_dir)
    cmd.build_temp = str(build_dir / "build")
    cmd.inplace = False
    cmd.force = True

    try:
        # Ensure options are finalized before invoking the build.
        try:  # noqa: SIM105
            cmd.ensure_finalized()
        except Exception:
            # Best-effort: some setuptools variants finalize during run_command.
            pass
        dist.run_command("build_ext")
    except SystemExit as e:
        raise RuntimeError(
            f"Compilation failed for module '{name}': build_ext exited with code {e.code!r}.\n"
            "Build prerequisites: a working C/C++ compiler toolchain and Python "
            "development headers (e.g., python3-dev on Debian/Ubuntu)."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Compilation failed for module '{name}': {e}\n"
            "Build prerequisites: a working C/C++ compiler toolchain and Python "
            "development headers (e.g., python3-dev on Debian/Ubuntu)."
        ) from e

    ext_path = _find_built_extension(build_dir, name)
    if ext_path is None:
        raise RuntimeError(
            f"Build completed but extension file not found for '{name}' in {build_dir}"
        )
    return ext_path


def _set_verbosity(verbose: int) -> None:
    """Set build verbosity (best-effort)."""
    try:
        from setuptools._distutils.log import set_verbosity  # type: ignore[]  # noqa: I001, PLC0415

        set_verbosity(verbose)
    except Exception:
        return


def _find_built_extension(build_dir: Path, name: str) -> Path | None:
    for suffix in EXTENSION_SUFFIXES:
        for p in build_dir.glob(f"{name}*{suffix}"):
            if p.is_file():
                return p
    return None


def _find_annotation(build_dir: Path, name: str) -> Path | None:
    html = build_dir / f"{name}.html"
    return html if html.exists() else None


# ---------------------------------------------------------------------------
# Package (multi-module) builds
# ---------------------------------------------------------------------------


def build_extension_package_from_code_result(  # noqa: D417, PLR0912
    modules: Mapping[str, str],
    *,
    package_name: str,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    force_rebuild: bool = False,
    verbose: int = 0,
    profile: str | None = None,
    annotate: bool = False,
    view_annotate: bool = False,
    numpy_support: bool = True,
    numpy_required: bool = False,
    include_dirs: PathLikeSeq | None = None,
    library_dirs: PathLikeSeq | None = None,
    libraries: Sequence[str] | None = None,
    define_macros: Sequence[tuple[str, str | None]] | None = None,
    extra_compile_args: Sequence[str] | None = None,
    extra_link_args: Sequence[str] | None = None,
    compiler_directives: Mapping[str, Any] | None = None,
    extra_sources: PathLikeSeq | None = None,
    support_files: Mapping[str, str | bytes] | None = None,
    support_paths: PathLikeSeq | None = None,
    include_cwd: bool = True,
    lock_timeout_s: float = 60.0,
    language: str | None = None,
) -> PackageBuildResult:
    """
    Compile a *package* of multiple extension modules from in-memory code strings.

    Parameters
    ----------
    modules : Mapping[str, str]
        Mapping of ``{module_short_name: pyx_code}``. Module names must be simple
        identifiers (no dots). The compiled modules will be importable as
        ``{package_name}.{module_short_name}``.
    package_name : str
        Python package name (identifier-like; may include dots for a nested package).
    cache_dir, use_cache, force_rebuild, verbose, annotate, view_annotate, numpy_support, numpy_required :
        See :func:`scikitplot.cython.compile_and_load`.
    include_dirs, library_dirs, libraries, define_macros, extra_compile_args, extra_link_args, compiler_directives :
        See :func:`scikitplot.cython.compile_and_load`.
    extra_sources : sequence of path-like, optional
        Additional C/C++ sources compiled and linked into *each* extension module.
        Allowed suffixes: .c, .cc, .cpp, .cxx, .C
    support_files : Mapping[str, str | bytes] or None, default=None
        Support files written into the package directory (e.g., ``.pxi``, ``.pxd``).
        Filenames must be simple (no directories). Collisions are strict.
    support_paths : sequence of path-like, optional
        Support files copied into the package directory. Basenames must be unique.
    include_cwd : bool, default=True
        If True, include the current working directory in include paths.
    lock_timeout_s : float, default=60.0
        Maximum seconds to wait for the per-key build lock.
    language : {'c', 'c++'} or None, default=None
        Optional explicit language for the extensions.

    Returns
    -------
    scikitplot.cython.PackageBuildResult
        Result object containing all loaded modules.
    """
    from ._result import PackageBuildResult  # noqa: PLC0415

    if not modules:
        raise ValueError("modules must be a non-empty mapping")

    # Strict validate module names from dict
    for mn in modules:
        if not isinstance(mn, str) or (not mn) or ("." in mn):
            raise ValueError(f"Invalid module short name: {mn!r}")

    cache_root = resolve_cache_dir(cache_dir)

    try:
        import Cython  # noqa: PLC0415
    except Exception as e:
        raise ImportError("Cython is required to compile extension modules.") from e

    np_version: str | None = None
    np_include: str | None = None
    if numpy_support:
        try:
            import numpy as np  # noqa: PLC0415

            np_version = np.__version__
            np_include = np.get_include()
        except Exception as e:
            if numpy_required:
                raise ImportError(
                    "NumPy is required (numpy_required=True) but is not available."
                ) from e
            np_version = None
            np_include = None

    directives = dict(DEFAULT_COMPILER_DIRECTIVES)
    if compiler_directives is not None:
        directives.update(dict(compiler_directives))

    include_dir_list = [str(_to_path(p)) for p in (include_dirs or [])]
    if include_cwd:
        include_dir_list.append(str(Path.cwd().resolve()))
    if np_include is not None:
        include_dir_list.append(str(Path(np_include).resolve()))

    extra_sources = _normalize_extra_sources(extra_sources)

    # Payload for cache key includes per-module source digests.
    module_digests = [
        (name, source_digest(code.encode("utf-8")))
        for name, code in sorted(modules.items())
    ]

    key_payload = {
        "kind": "package",
        "package_name": package_name,
        "modules": module_digests,
        "directives": dict(directives),
        "include_dirs": list(include_dir_list),
        "library_dirs": [str(_to_path(p)) for p in (library_dirs or [])],
        "libraries": list(libraries or []),
        "define_macros": list(define_macros or []),
        "extra_compile_args": list(extra_compile_args or []),
        "extra_link_args": list(extra_link_args or []),
        "extra_sources": _support_paths_digest(extra_sources),
        "support_files": _support_files_digest(support_files),
        "support_paths": _support_paths_digest(support_paths),
        "language": language,
        "fingerprint": dict(
            runtime_fingerprint(
                cython_version=Cython.__version__, numpy_version=np_version
            )
        ),
        "annotate": bool(annotate),
    }
    key = make_cache_key(key_payload)

    build_dir = cache_root / key
    lock_dir = build_dir.with_suffix(".lock")
    build_dir.parent.mkdir(parents=True, exist_ok=True)

    used_cache = False
    modules_out: list[BuildResult] = []
    meta_out: Mapping[str, Any] = {}

    with build_lock(lock_dir, timeout_s=lock_timeout_s):
        if build_dir.exists() and force_rebuild:
            shutil.rmtree(build_dir)

        build_dir.mkdir(parents=True, exist_ok=True)
        meta_existing = read_meta(build_dir)

        if (
            use_cache
            and (not force_rebuild)
            and meta_existing
            and meta_existing.get("kind") == "package"
        ):
            # Verify artifacts exist
            ok = True
            mods = meta_existing.get("modules")
            if not isinstance(mods, list) or not mods:
                ok = False
            else:
                for m in mods:
                    if not isinstance(m, dict):
                        ok = False
                        break
                    ap = m.get("artifact")
                    if not isinstance(ap, str) or not ap:
                        ok = False
                        break
                    if not (build_dir / ap).exists():
                        ok = False
                        break
            if ok:
                used_cache = True
                meta_out = meta_existing

        if not used_cache:
            # Prepare package directory and sources
            pkg_path = build_dir / package_name.replace(".", os.sep)
            pkg_path.mkdir(parents=True, exist_ok=True)
            (pkg_path / "__init__.py").write_text(
                "# Auto-generated by scikitplot.cython\n", encoding="utf-8"
            )

            reserved_pkg = {"__init__.py"}
            # Write support files / copy support paths into package dir
            _write_support_files(pkg_path, support_files, reserved_pkg)
            _copy_support_paths(pkg_path, support_paths, reserved_pkg)

            # Copy extra sources into build_dir (if provided) and compile from that location.
            # `_copy_extra_sources` returns absolute destination paths.
            extra_sources_paths = _copy_extra_sources(
                build_dir, extra_sources, reserved=set()
            )

            # Write module sources
            module_paths: list[tuple[str, Path]] = []
            for short_name, code in sorted(modules.items()):
                pyx = pkg_path / f"{short_name}.pyx"
                pyx.write_text(code, encoding="utf-8")
                module_paths.append((short_name, pyx))

            # Compile all extensions in one build invocation
            try:
                from Cython.Build import cythonize  # noqa: PLC0415
            except Exception as e:
                raise ImportError(
                    "Cython.Build.cythonize is required to compile extensions."
                ) from e

            try:
                from setuptools import Extension  # noqa: PLC0415
                from setuptools.dist import Distribution  # noqa: PLC0415
            except Exception as e:
                raise ImportError(
                    "setuptools is required to compile Cython extensions."
                ) from e

            _set_verbosity(verbose)

            ext_list = []
            for short_name, pyx in module_paths:
                full_name = f"{package_name}.{short_name}"
                sources = [str(pyx)] + [str(p) for p in extra_sources_paths]
                ext_list.append(
                    Extension(
                        name=full_name,
                        sources=sources,
                        include_dirs=list(include_dir_list),
                        library_dirs=[str(_to_path(p)) for p in (library_dirs or [])],
                        libraries=list(libraries or []),
                        define_macros=list(define_macros or []),
                        extra_compile_args=list(extra_compile_args or []),
                        extra_link_args=list(extra_link_args or []),
                        language=language,
                    )
                )

            try:
                try:
                    ext_modules = cythonize(
                        ext_list,
                        compiler_directives=dict(directives),
                        annotate=annotate,
                        quiet=(verbose < 0),
                    )
                except TypeError:
                    # Cython API drift: retry without the optional `quiet` kwarg.
                    ext_modules = cythonize(
                        ext_list,
                        compiler_directives=dict(directives),
                        annotate=annotate,
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Cythonize failed for package '{package_name}': {e}"
                ) from e

            dist = Distribution()
            dist.ext_modules = ext_modules

            cmd = dist.get_command_obj("build_ext")
            cmd.build_lib = str(build_dir)
            cmd.build_temp = str(build_dir / "build")
            cmd.inplace = False
            cmd.force = True

            try:
                dist.run_command("build_ext")
            except Exception as e:
                raise RuntimeError(
                    f"Compilation failed for package '{package_name}': {e}\n"
                    "Build prerequisites: a working C/C++ compiler toolchain and Python "
                    "development headers (e.g., python3-dev on Debian/Ubuntu)."
                ) from e

            # Discover artifacts per module
            module_artifacts: list[dict[str, Any]] = []
            for short_name, _ in module_paths:
                full_name = f"{package_name}.{short_name}"
                ext_path = _find_built_extension(pkg_path, short_name)
                if ext_path is None:
                    raise RuntimeError(
                        f"Build completed but extension file not found for '{full_name}' in {pkg_path}"
                    )
                full_name = f"{package_name}.{short_name}"
                rel_ext = str(ext_path.relative_to(build_dir).as_posix())

                html_rel: str | None = None
                if annotate:
                    html = pkg_path / f"{short_name}.html"
                    if html.exists():
                        html_rel = str(html.relative_to(build_dir).as_posix())

                module_artifacts.append(
                    {
                        "module_name": full_name,
                        "artifact": rel_ext,
                        "source_sha256": dict(module_digests).get(short_name),
                        "annotate_html": html_rel,
                    }
                )
            # Record per-module annotation HTML (if generated).
            # Cython writes <pyx_basename>.html next to the .pyx inside pkg_path.
            annotation_html_map: dict[str, str] = {}
            if annotate:
                for short_name, _ in module_paths:
                    html = pkg_path / f"{short_name}.html"
                    if html.exists():
                        annotation_html_map[f"{package_name}.{short_name}"] = str(
                            html.relative_to(build_dir).as_posix()
                        )

            # Deterministic "first" html (handy convenience)
            first_annotate_html: str | None = None
            if annotation_html_map:
                first_annotate_html = annotation_html_map[
                    sorted(annotation_html_map.keys())[0]
                ]
            # Write meta.json at build_dir root
            fingerprint = runtime_fingerprint(
                cython_version=Cython.__version__, numpy_version=np_version
            )
            meta_out = {
                "kind": "package",
                "key": key,
                "package_name": package_name,
                "modules": module_artifacts,
                "created_utc": _utc_now_iso(),
                "fingerprint": dict(fingerprint),
                # Canonical directives fields (both keys for compatibility)
                "directives": dict(directives),
                "compiler_directives": dict(directives),
                # Persist user-facing build options (introspection)
                "profile": profile,
                "annotate": bool(annotate),
                "view_annotate": bool(view_annotate),
                "extra_compile_args": list(extra_compile_args or []),
                "extra_link_args": list(extra_link_args or []),
                # Package-level annotation convenience
                "annotate_html": first_annotate_html,
                "annotation_html": dict(annotation_html_map),
                "include_dirs": list(include_dir_list),
                "support_files": _support_files_digest(support_files),
                "support_paths": _support_paths_digest(support_paths),
                "extra_sources": _support_paths_digest(extra_sources),
                "language": language,
            }
            write_meta(build_dir, meta_out)

        # Import modules (even if used_cache)
        # Ensure parent package modules exist for dotted imports.
        pkg_fs_dir = build_dir / package_name.replace(".", os.sep)
        _ensure_package(package_name, pkg_fs_dir)

        mods_meta = meta_out.get("modules") if isinstance(meta_out, dict) else None
        if not isinstance(mods_meta, list):
            raise RuntimeError("Invalid package metadata: missing modules list")

        for m in sorted(mods_meta, key=lambda d: str(d.get("module_name", ""))):
            if not isinstance(m, dict):
                continue
            full_name = m.get("module_name")
            ap = m.get("artifact")
            if not isinstance(full_name, str) or not isinstance(ap, str):
                continue
            ext_path = (build_dir / ap).resolve()
            module = import_extension(
                name=full_name, path=ext_path, key=key, build_dir=build_dir
            )
            # Build per-module BuildResult
            modules_out.append(
                BuildResult(
                    module=module,
                    key=key,
                    module_name=full_name,
                    build_dir=build_dir,
                    artifact_path=ext_path,
                    used_cache=used_cache,
                    created_utc=(
                        meta_out.get("created_utc")
                        if isinstance(meta_out, dict)
                        else None
                    ),
                    fingerprint=(
                        meta_out.get("fingerprint")
                        if isinstance(meta_out, dict)
                        else None
                    ),
                    source_sha256=(
                        m.get("source_sha256")
                        if isinstance(m.get("source_sha256"), str)
                        else None
                    ),
                    meta=meta_out if isinstance(meta_out, dict) else {},
                )
            )

        if view_annotate:
            # For packages, open the first module's annotation if present.
            first = (
                modules_out[0].module_name.split(".")[-1]
                if modules_out
                else package_name
            )
            html = _find_annotation(pkg_fs_dir, first)
            if html is not None:
                webbrowser.open(html.as_uri())

    return PackageBuildResult(
        package_name=package_name,
        key=key,
        build_dir=build_dir,
        results=tuple(modules_out),
        used_cache=used_cache,
        created_utc=meta_out.get("created_utc") if isinstance(meta_out, dict) else None,
        fingerprint=meta_out.get("fingerprint") if isinstance(meta_out, dict) else None,
        meta=meta_out if isinstance(meta_out, dict) else {},
    )


def build_extension_package_from_paths_result(  # noqa: D417
    modules: Mapping[str, str | Path],
    *,
    package_name: str,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    force_rebuild: bool = False,
    verbose: int = 0,
    profile: str | None = None,
    annotate: bool = False,
    view_annotate: bool = False,
    numpy_support: bool = True,
    numpy_required: bool = False,
    include_dirs: Sequence[str | Path] | None = None,
    library_dirs: Sequence[str | Path] | None = None,
    libraries: Sequence[str] | None = None,
    define_macros: Sequence[tuple[str, str | None]] | None = None,
    extra_compile_args: Sequence[str] | None = None,
    extra_link_args: Sequence[str] | None = None,
    compiler_directives: Mapping[str, Any] | None = None,
    extra_sources: Sequence[str | Path] | None = None,
    support_files: Mapping[str, str | bytes] | None = None,
    support_paths: Sequence[str | Path] | None = None,
    include_cwd: bool = True,
    lock_timeout_s: float = 60.0,
    language: str | None = None,
) -> PackageBuildResult:
    """
    Compile a *package* of multiple extension modules from existing ``.pyx`` files.

    Parameters
    ----------
    modules : Mapping[str, str | pathlib.Path]
        Mapping of ``{module_short_name: pyx_path}``.
    package_name : str
        Python package name.
    (other parameters)
        See :func:`build_extension_package_from_code_result`.

    Returns
    -------
    scikitplot.cython.PackageBuildResult
        Result object containing all loaded modules.
    """
    # Read sources and delegate to code builder to keep caching deterministic.
    code_map: dict[str, str] = {}
    for k, p in modules.items():
        pp = Path(p).expanduser().resolve()
        if not pp.exists() or not pp.is_file():
            raise FileNotFoundError(str(pp))
        code_map[k] = pp.read_text(encoding="utf-8")

    return build_extension_package_from_code_result(
        code_map,
        package_name=package_name,
        cache_dir=cache_dir,
        use_cache=use_cache,
        force_rebuild=force_rebuild,
        verbose=verbose,
        profile=profile,
        annotate=annotate,
        view_annotate=view_annotate,
        numpy_support=numpy_support,
        numpy_required=numpy_required,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        compiler_directives=compiler_directives,
        extra_sources=extra_sources,
        support_files=support_files,
        support_paths=support_paths,
        include_cwd=include_cwd,
        lock_timeout_s=lock_timeout_s,
        language=language,
    )


def _ensure_package(package_name: str, pkg_dir: Path) -> None:
    """
    Ensure package modules exist in ``sys.modules`` for dotted extension imports.

    Parameters
    ----------
    package_name : str
        Dotted package name (e.g., ``"mypkg"`` or ``"a.b.c"``).
    pkg_dir : pathlib.Path
        Filesystem directory for the *leaf* package (e.g., ``.../a/b/c``).

    Notes
    -----
    This does not import any Python code; it only prepares package containers so
    that importing an extension module as ``a.b.c.mod`` is valid even when the
    package is only present inside the build directory.
    """
    parts = package_name.split(".")
    n = len(parts)

    # Map each segment to its on-disk directory assuming pkg_dir is the leaf.
    seg_dirs: list[Path] = []
    for i in range(n):
        if i == n - 1:
            seg_dirs.append(pkg_dir)
        else:
            seg_dirs.append(pkg_dir.parents[n - i - 2])

    current = ""
    for i, part in enumerate(parts):
        current = part if not current else f"{current}.{part}"
        if current in sys.modules:
            continue
        mod = ModuleType(current)
        mod.__package__ = current
        mod.__path__ = [str(seg_dirs[i].resolve())]  # type: ignore[attr-defined]
        sys.modules[current] = mod
