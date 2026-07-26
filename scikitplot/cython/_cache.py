# scikitplot/cython/_cache.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Cache directory, cache-key, and cache registry utilities.

This module is strict:
- cache keys are 64-char hex digests
- cache roots are deterministic and user-overridable via env var
- metadata is stored in ``meta.json`` under each entry directory

It also provides a lightweight registry enabling re-import of compiled extension
modules after interpreter/kernel restarts.

Security:

Do not compile or import native code from untrusted sources.
"""

from __future__ import annotations

import json
import os
import platform
import re
import sys
import sysconfig
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

_ENV_CACHE_DIR = "SCIKITPLOT_CYTHON_CACHE_DIR"
_ENV_CACHE_DIR_SHORT = "SKPLT_CYTHON_CACHE_DIR"  # Short alias; takes priority when set.
_KEY_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CacheEntry",
    "PackageCacheEntry",
    "find_entries_by_name",
    "find_entry_by_key",
    "find_package_entry_by_key",
    "is_meta_schema_compatible",
    "is_valid_key",
    "iter_all_entry_dirs",
    "iter_cache_entries",
    "iter_package_entries",
    "make_cache_key",
    "meta_schema_version",
    "peek_cache_dir",
    "read_meta",
    "register_artifact_path",
    "resolve_cache_dir",
    "runtime_fingerprint",
    "source_digest",
    "write_meta",
]


#: Version of the ``meta.json`` cache-entry schema (CYTHON-SCH-001).  Bump this
#: whenever the set/meaning of persisted meta fields changes incompatibly.
#: Entries written by an incompatible schema version are treated as cache misses
#: and rebuilt rather than misread.  Version 0 denotes pre-versioning entries.
CACHE_SCHEMA_VERSION = 1

#: The meta key under which the schema version is stamped.
_SCHEMA_KEY = "meta_schema_version"


def meta_schema_version(meta: Mapping[str, Any] | None) -> int:
    """Return the schema version recorded in ``meta`` (0 if absent/legacy)."""
    if not meta:
        return 0
    v = meta.get(_SCHEMA_KEY, 0)
    return v if isinstance(v, int) and v >= 0 else 0


def is_meta_schema_compatible(meta: Mapping[str, Any] | None) -> bool:
    """Return whether ``meta`` can be trusted by the current reader.

    Compatible means the entry's schema version is a version this library knows
    how to read: any version from 1 up to :data:`CACHE_SCHEMA_VERSION`.  Legacy
    entries (version 0, pre-versioning) and entries from a *newer*, unknown
    schema are treated as incompatible so they are rebuilt rather than misread
    (CYTHON-SCH-001).
    """
    v = meta_schema_version(meta)
    return 1 <= v <= CACHE_SCHEMA_VERSION


def _env_cache_dir() -> str | None:
    """
    Return the env-var override for the cache directory, or None.

    Checks the short alias ``SKPLT_CYTHON_CACHE_DIR`` first, then the
    canonical ``SCIKITPLOT_CYTHON_CACHE_DIR``.  The short alias exists so
    that CI configurations and shell profiles can use a concise name.

    Returns
    -------
    str or None
        Non-empty env-var value, or ``None`` if neither variable is set.
    """
    return (
        os.environ.get(_ENV_CACHE_DIR_SHORT) or os.environ.get(_ENV_CACHE_DIR) or None
    )


def is_valid_key(key: str) -> bool:
    """
    Return True if ``key`` is a valid cache key.

    Parameters
    ----------
    key : str
        Candidate cache key.

    Returns
    -------
    bool
        True if ``key`` is a 64-character hex digest.
    """
    return isinstance(key, str) and _KEY_RE.fullmatch(key) is not None


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """
    A compiled *module* cache entry.

    Parameters
    ----------
    key : str
        Cache key (hex digest).
    build_dir : pathlib.Path
        Directory containing build artifacts for this key.
    module_name : str
        Python module name used to compile the extension.
    artifact_path : pathlib.Path
        Path to the compiled extension (e.g., ``.so`` or ``.pyd``).
    created_utc : str or None
        ISO timestamp (UTC) if available.
    fingerprint : Mapping[str, Any] or None
        Runtime fingerprint used when building this artifact (if available).
    """

    # NOTE: Defaults exist to satisfy documentation tooling that expects class
    # attributes to have defaults. In practice, cache entries returned by the
    # API are always fully populated.
    key: str = ""
    build_dir: Path = Path(".")
    module_name: str = ""
    artifact_path: Path = Path(".")
    created_utc: str | None = None
    fingerprint: Mapping[str, Any] | None = None

    def __repr__(self) -> str:  # pragma: no cover
        # NOTE: Explicit, all-fields repr to keep output stable and
        # introspectable (useful for debugging, logging, and Sphinx).
        fingerprint = dict(self.fingerprint) if self.fingerprint is not None else None
        return (
            "CacheEntry("
            f"key={self.key!r}, "
            f"build_dir={self.build_dir!r}, "
            f"module_name={self.module_name!r}, "
            f"artifact_path={self.artifact_path!r}, "
            f"created_utc={self.created_utc!r}, "
            f"fingerprint={fingerprint!r}"
            ")"
        )


@dataclass(frozen=True, slots=True)
class PackageCacheEntry:
    """
    A compiled *package* cache entry (multi-module build).

    Parameters
    ----------
    key : str
        Cache key (hex digest).
    build_dir : pathlib.Path
        Directory containing the package directory and artifacts.
    package_name : str
        Python package name (e.g., ``"mypkg"``).
    modules : tuple[str, ...]
        Full dotted module names included in the package build.
    artifacts : tuple[pathlib.Path, ...]
        Artifact paths for modules in the same order as ``modules``.
    created_utc : str or None
        ISO timestamp (UTC) if available.
    fingerprint : Mapping[str, Any] or None
        Runtime fingerprint used when building this artifact (if available).
    """

    # NOTE: Defaults exist to satisfy documentation tooling that expects class
    # attributes to have defaults. In practice, cache entries returned by the
    # API are always fully populated.
    key: str = ""
    build_dir: Path = Path(".")
    package_name: str = ""
    modules: tuple[str, ...] = ()
    artifacts: tuple[Path, ...] = ()
    created_utc: str | None = None
    fingerprint: Mapping[str, Any] | None = None

    def __repr__(self) -> str:  # pragma: no cover
        # NOTE: Explicit, all-fields repr to keep output stable and
        # introspectable (useful for debugging, logging, and Sphinx).
        fingerprint = dict(self.fingerprint) if self.fingerprint is not None else None
        return (
            "PackageCacheEntry("
            f"key={self.key!r}, "
            f"build_dir={self.build_dir!r}, "
            f"package_name={self.package_name!r}, "
            f"modules={self.modules!r}, "
            f"artifacts={self.artifacts!r}, "
            f"created_utc={self.created_utc!r}, "
            f"fingerprint={fingerprint!r}"
            ")"
        )


def resolve_cache_dir(cache_dir: str | Path | None) -> Path:
    """
    Resolve and create the cache directory.

    Parameters
    ----------
    cache_dir : str or pathlib.Path or None
        Requested cache dir. If None, use environment override or a default
        platform cache location.

    Returns
    -------
    pathlib.Path
        Resolved cache directory root (created if needed).

    Raises
    ------
    OSError
        If directory creation fails.

    Notes
    -----
    Environment override (if set) takes precedence:
    ``SCIKITPLOT_CYTHON_CACHE_DIR``.
    """
    env = _env_cache_dir()
    root = (
        Path(env)
        if env
        else (Path(cache_dir) if cache_dir is not None else _default_cache_dir())
    )
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def peek_cache_dir(cache_dir: str | Path | None) -> Path:
    """
    Resolve the cache directory path without creating it.

    Parameters
    ----------
    cache_dir : str or pathlib.Path or None
        Requested cache dir. If None, use environment override or a default
        platform cache location.

    Returns
    -------
    pathlib.Path
        Resolved cache directory root (may not exist).
    """
    env = _env_cache_dir()
    root = (
        Path(env)
        if env
        else (Path(cache_dir) if cache_dir is not None else _default_cache_dir())
    )
    return root.expanduser().resolve()


def _default_cache_dir() -> Path:
    """
    Set default cache directory.

    Returns
    -------
    pathlib.Path
        Platform-appropriate default cache path.
    """
    # Prefer XDG on POSIX, LOCALAPPDATA on Windows
    if os.name == "nt":
        base = (
            os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or str(Path.home())
        )
        return Path(base) / "scikitplot" / "cython_cache"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "scikitplot" / "cython"
    return Path.home() / ".cache" / "scikitplot" / "cython"


def make_cache_key(payload: Mapping[str, Any]) -> str:
    """
    Create a deterministic cache key from a JSON-serializable mapping.

    Parameters
    ----------
    payload : Mapping[str, Any]
        JSON-serializable mapping.

    Returns
    -------
    str
        64-character hex digest.
    """
    data = _json_dumps(payload).encode("utf-8")
    return sha256(data).hexdigest()


def _stable_repr(obj: Any) -> Any:
    """
    Convert objects to stable JSON-compatible representations.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, (list, tuple)):
        return [_stable_repr(x) for x in obj]
    if isinstance(obj, dict):
        return {
            str(k): _stable_repr(v)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
        }
    # Fallback: stable string representation
    return str(obj)


def _toolchain_fingerprint() -> dict[str, Any]:
    """Toolchain / ABI inputs that must invalidate the cache when they differ.

    Two artifacts built on the same host and Python but with a different C/C++
    compiler, a different CPython ABI (e.g. free-threaded), or a different
    pointer width are **not** interchangeable; reusing one for the other yields
    import failures or crashes.  These ``sysconfig`` values capture exactly that
    (CYTHON-CACHE-003).  Values are read without executing any subprocess.
    """
    get = sysconfig.get_config_var
    fp = {
        # Compiler identity/driver (GCC vs Clang vs MSVC, cross prefixes, etc.).
        "cc": get("CC") or "",
        "cxx": get("CXX") or "",
        # Extension ABI tag — the definitive CPython ABI the .so targets.
        "ext_suffix": get("EXT_SUFFIX") or "",
        "soabi": get("SOABI") or "",
        # Pointer width (32/64-bit) and free-threaded (GIL-disabled) build.
        "pointer_size": get("SIZEOF_VOID_P") or "",
        "gil_disabled": bool(get("Py_GIL_DISABLED") or 0),
        # Build platform tag (includes emscripten-wasm32 in browser builds).
        "sysconfig_platform": sysconfig.get_platform() or "",
    }
    # Effective compiler the build backend ACTUALLY selects, not a PATH/sysconfig
    # guess (CYTHON-PORT-001).  On Windows sysconfig's CC is often empty while
    # setuptools picks MSVC; keying from the resolved plan prevents reusing an
    # artifact built by a different toolchain.  Lazy import + never-raise.
    try:
        from ._profiles import resolved_toolchain  # noqa: PLC0415

        rt = resolved_toolchain()
        fp["resolved_compiler_type"] = rt.compiler_type
        fp["resolved_cc"] = rt.cc
        fp["resolved_cxx"] = rt.cxx
    except Exception:  # noqa: BLE001 - detection must never break caching
        fp["resolved_compiler_type"] = "unknown"
        fp["resolved_cc"] = ""
        fp["resolved_cxx"] = ""
    return fp


def runtime_fingerprint(
    *, cython_version: str, numpy_version: str | None
) -> Mapping[str, Any]:
    """
    Compute a runtime fingerprint for caching correctness.

    The fingerprint includes the interpreter, platform, and library versions
    **and** the toolchain / ABI inputs (compiler identity, extension ABI tag,
    pointer width, free-threaded flag) so that an artifact built with one
    compiler or ABI is never reused under an incompatible one
    (CYTHON-CACHE-003).

    Parameters
    ----------
    cython_version : str
        Cython version.
    numpy_version : str or None
        NumPy version (None if not used).

    Returns
    -------
    Mapping[str, Any]
        Fingerprint mapping.

    Notes
    -----
    Extending this mapping intentionally changes cache keys, so entries built by
    an older library version are treated as misses and rebuilt once — this is
    the correct behaviour, since those entries lacked toolchain/ABI safety.
    """
    fp: dict[str, Any] = {
        "python": platform.python_version(),
        "python_impl": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cython": cython_version,
        "numpy": numpy_version,
        "abi": getattr(sys, "abiflags", ""),
    }
    fp.update(_toolchain_fingerprint())
    return fp


def source_digest(data: bytes) -> str:
    """
    SHA-256 digest of source bytes.

    Parameters
    ----------
    data : bytes
        Source bytes.

    Returns
    -------
    str
        Hex digest.
    """
    return sha256(data).hexdigest()


def write_meta(build_dir: Path, meta: Mapping[str, Any]) -> None:
    """
    Write ``meta.json`` in the build directory atomically.

    Parameters
    ----------
    build_dir : pathlib.Path
        Cache entry directory.
    meta : Mapping[str, Any]
        Metadata mapping.

    Notes
    -----
    Uses a write-then-rename (atomic replace) pattern so that a crash during
    writing never leaves a partially-written ``meta.json``.  On POSIX this is
    an atomic operation; on Windows it uses ``replace()`` which is best-effort.
    """
    # Detect if caller passed a file path
    if build_dir.suffix == ".json":  # noqa: SIM108
        path = build_dir
    else:
        path = build_dir / "meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stamp the schema version on every write so future format changes are
    # detectable (CYTHON-SCH-001).  An explicit value in ``meta`` is preserved.
    stamped = dict(meta)
    stamped.setdefault(_SCHEMA_KEY, CACHE_SCHEMA_VERSION)
    tmp = path.with_suffix(path.suffix + ".tmp")  # meta.json.tmp
    tmp.write_text(_json_dumps(stamped) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_meta(build_dir: Path) -> Mapping[str, Any] | None:
    """
    Read ``meta.json`` from a build directory.

    Parameters
    ----------
    build_dir : pathlib.Path
        Cache entry directory.

    Returns
    -------
    Mapping[str, Any] or None
        Parsed metadata dict, or None if missing/invalid.
    """
    path = build_dir / "meta.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


def iter_all_entry_dirs(cache_root: str | Path | None) -> list[Path]:
    """
    Return all cache entry directories whose name is a valid cache key.

    Parameters
    ----------
    cache_root : str or pathlib.Path or None
        Cache root.

    Returns
    -------
    list[pathlib.Path]
        Entry directory paths, sorted by name for deterministic ordering.

    Notes
    -----
    A ``list`` is returned rather than a generator so that callers can iterate
    the result multiple times safely (e.g., once for stats and once for GC).
    A generator would be silently exhausted on the second pass, producing an
    empty sequence with no error.
    """
    root = peek_cache_dir(cache_root)
    if not root.exists():
        return []
    return [p for p in sorted(root.iterdir()) if p.is_dir() and is_valid_key(p.name)]


def iter_cache_entries(cache_dir: str | Path | None) -> list[CacheEntry]:
    """
    List *module* cache entries found under the cache directory.

    Package builds (``kind == 'package'``) are excluded; use
    :func:`iter_package_entries` for those.

    Parameters
    ----------
    cache_dir : str or pathlib.Path or None
        Cache root. If None, resolves to default.

    Returns
    -------
    list[CacheEntry]
        Entries with discovered artifacts.
    """
    root = peek_cache_dir(cache_dir)
    if not root.exists():
        return []

    entries: list[CacheEntry] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not is_valid_key(child.name):
            continue

        meta = read_meta(child)
        if meta is not None and meta.get("kind") == "package":
            continue

        artifact = _artifact_from_meta_or_guess(child, meta)
        if artifact is None:
            continue

        module_name = _module_name_from_meta_or_guess(meta, artifact)
        created_utc = meta.get("created_utc") if meta else None
        fingerprint = (
            meta.get("fingerprint")
            if meta and isinstance(meta.get("fingerprint"), dict)
            else None
        )

        entries.append(
            CacheEntry(
                key=child.name.lower(),
                build_dir=child,
                module_name=module_name,
                artifact_path=artifact,
                created_utc=created_utc,
                fingerprint=fingerprint,
            )
        )
    return entries


def iter_package_entries(cache_dir: str | Path | None) -> list[PackageCacheEntry]:
    """
    List *package* cache entries found under the cache directory.

    Parameters
    ----------
    cache_dir : str or pathlib.Path or None
        Cache root. If None, resolves to default.

    Returns
    -------
    list[PackageCacheEntry]
        Package entries with discovered artifacts.

    Notes
    -----
    Package entries are identified by ``meta.json`` containing ``kind == 'package'``.
    """
    root = peek_cache_dir(cache_dir)
    if not root.exists():
        return []

    out: list[PackageCacheEntry] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not is_valid_key(child.name):
            continue
        meta = read_meta(child)
        if meta is None or meta.get("kind") != "package":
            continue
        pkg = meta.get("package_name")
        mods = meta.get("modules")
        if not isinstance(pkg, str) or not pkg:
            continue
        if not isinstance(mods, list) or not mods:
            continue
        modules: list[str] = []
        artifacts: list[Path] = []
        for m in mods:
            if not isinstance(m, dict):
                continue
            mn = m.get("module_name")
            ap = m.get("artifact")
            if not isinstance(mn, str) or not mn:
                continue
            if not isinstance(ap, str) or not ap:
                continue
            p = (child / ap) if not os.path.isabs(ap) else Path(ap)
            if not p.exists():
                continue
            modules.append(mn)
            artifacts.append(p)
        if not modules or len(modules) != len(artifacts):
            continue
        created_utc = (
            meta.get("created_utc")
            if isinstance(meta.get("created_utc"), str)
            else None
        )
        fingerprint = (
            meta.get("fingerprint")
            if isinstance(meta.get("fingerprint"), dict)
            else None
        )
        out.append(
            PackageCacheEntry(
                key=child.name.lower(),
                build_dir=child,
                package_name=pkg,
                modules=tuple(modules),
                artifacts=tuple(artifacts),
                created_utc=created_utc,
                fingerprint=fingerprint,
            )
        )
    return out


def find_entry_by_key(cache_dir: str | Path | None, key: str) -> CacheEntry:
    """
    Find a single *module* cache entry by key.

    Parameters
    ----------
    cache_dir : str or pathlib.Path or None
        Cache root.
    key : str
        Cache key hex string.

    Returns
    -------
    CacheEntry
        The matching module entry.

    Raises
    ------
    FileNotFoundError
        If no matching module entry exists or no artifact is present.
    ValueError
        If key format is invalid or key corresponds to a package entry.
    """
    k = key.lower()
    if not is_valid_key(k):
        raise ValueError(f"Invalid cache key: {key!r}")

    root = peek_cache_dir(cache_dir)
    if not root.exists():
        raise FileNotFoundError(f"No cache directory: {root}")

    build_dir = root / k
    if not build_dir.exists():
        raise FileNotFoundError(f"No cache entry for key: {k}")

    meta = read_meta(build_dir)
    if meta is not None and meta.get("kind") == "package":
        raise ValueError(
            f"Key {k} refers to a package build. Use find_package_entry_by_key()."
        )

    artifact = _artifact_from_meta_or_guess(build_dir, meta)
    if artifact is None:
        raise FileNotFoundError(f"No compiled artifact found for key: {k}")

    module_name = _module_name_from_meta_or_guess(meta, artifact)
    created_utc = meta.get("created_utc") if meta else None
    fingerprint = (
        meta.get("fingerprint")
        if meta and isinstance(meta.get("fingerprint"), dict)
        else None
    )
    return CacheEntry(k, build_dir, module_name, artifact, created_utc, fingerprint)


def find_package_entry_by_key(
    cache_dir: str | Path | None, key: str
) -> PackageCacheEntry:
    """
    Find a single *package* cache entry by key.

    Parameters
    ----------
    cache_dir : str or pathlib.Path or None
        Cache root.
    key : str
        Cache key hex string.

    Returns
    -------
    PackageCacheEntry
        The matching package entry.

    Raises
    ------
    FileNotFoundError
        If no matching package entry exists or required artifacts are missing.
    ValueError
        If key format is invalid or key corresponds to a module entry.
    """
    k = key.lower()
    if not is_valid_key(k):
        raise ValueError(f"Invalid cache key: {key!r}")

    root = peek_cache_dir(cache_dir)
    if not root.exists():
        raise FileNotFoundError(f"No cache directory: {root}")

    build_dir = root / k
    if not build_dir.exists():
        raise FileNotFoundError(f"No cache entry for key: {k}")

    meta = read_meta(build_dir)
    if meta is None or meta.get("kind") != "package":
        raise ValueError(f"Key {k} does not refer to a package build.")

    # Parse this one entry directly instead of re-scanning the entire cache root.
    pkg = meta.get("package_name")
    mods_raw = meta.get("modules")
    if not isinstance(pkg, str) or not pkg:
        raise FileNotFoundError(f"Package entry is missing package_name for key: {k}")
    if not isinstance(mods_raw, list) or not mods_raw:
        raise FileNotFoundError(f"Package entry has no modules for key: {k}")

    modules: list[str] = []
    artifacts: list[Path] = []
    for m in mods_raw:
        if not isinstance(m, dict):
            continue
        mn = m.get("module_name")
        ap = m.get("artifact")
        if not isinstance(mn, str) or not mn:
            continue
        if not isinstance(ap, str) or not ap:
            continue
        p = (build_dir / ap) if not os.path.isabs(ap) else Path(ap)
        if not p.exists():
            continue
        modules.append(mn)
        artifacts.append(p)

    if not modules or len(modules) != len(artifacts):
        raise FileNotFoundError(f"Package entry is missing artifacts for key: {k}")

    created_utc = (
        meta.get("created_utc") if isinstance(meta.get("created_utc"), str) else None
    )
    fingerprint = (
        meta.get("fingerprint") if isinstance(meta.get("fingerprint"), dict) else None
    )
    return PackageCacheEntry(
        key=k,
        build_dir=build_dir,
        package_name=pkg,
        modules=tuple(modules),
        artifacts=tuple(artifacts),
        created_utc=created_utc,
        fingerprint=fingerprint,
    )


def find_entries_by_name(
    cache_dir: str | Path | None, module_name: str
) -> list[CacheEntry]:
    """
    Find module cache entries matching an exact module name.

    Parameters
    ----------
    cache_dir : str or pathlib.Path or None
        Cache root.
    module_name : str
        Module name (exact).

    Returns
    -------
    list[CacheEntry]
        Matching module entries.
    """
    return [e for e in iter_cache_entries(cache_dir) if e.module_name == module_name]


def register_artifact_path(
    cache_dir: str | Path | None,
    artifact_path: str | os.PathLike[str] | os.PathLike[bytes] | bytes,
    *,
    module_name: str,
    copy: bool = True,
) -> CacheEntry:
    """
    Register an existing compiled extension artifact into the cache registry.

    Parameters
    ----------
    cache_dir : str or pathlib.Path or None
        Cache root. If None, resolves to the default cache.
    artifact_path : path-like
        Path to a compiled extension artifact (``.so``/``.pyd``).
    module_name : str
        Module name the artifact was compiled for (init symbol name).
    copy : bool, default=True
        If True, copy the artifact into the cache entry directory.

    Returns
    -------
    CacheEntry
        The registered cache entry.

    Raises
    ------
    FileNotFoundError
        If the artifact does not exist.
    ValueError
        If the artifact does not have a valid extension suffix.
    OSError
        If writing metadata or copying fails.
    """
    from importlib.machinery import EXTENSION_SUFFIXES  # noqa: PLC0415

    root = resolve_cache_dir(cache_dir)

    ap = Path(os.fsdecode(os.fspath(artifact_path))).expanduser().resolve()
    if not ap.exists():
        raise FileNotFoundError(str(ap))

    if not any(ap.name.endswith(suf) for suf in EXTENSION_SUFFIXES):
        raise ValueError(f"Not a recognized extension artifact: {ap.name}")

    file_hash = _sha256_file(ap)
    key = make_cache_key(
        {
            "kind": "external",
            "module_name": module_name,
            "artifact_hash": file_hash,
            "artifact_name": ap.name,
        }
    )

    build_dir = root / key
    build_dir.mkdir(parents=True, exist_ok=True)

    if copy:
        dest = build_dir / ap.name
        if dest.exists():
            if _sha256_file(dest) != file_hash:
                raise OSError(f"Cache artifact collision for key {key}: {dest}")
        else:
            dest.write_bytes(ap.read_bytes())
        artifact = dest
        artifact_ref = ap.name
    else:
        artifact = ap
        artifact_ref = str(ap)

    meta = {
        "kind": "external",
        "key": key,
        "module_name": module_name,
        "artifact": artifact_ref,
        "external": True,
        "created_utc": _utc_iso(),
        "fingerprint": None,
    }
    write_meta(build_dir, meta)

    return CacheEntry(
        key=key,
        build_dir=build_dir,
        module_name=module_name,
        artifact_path=artifact,
        created_utc=meta["created_utc"],
        fingerprint=None,
    )


def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_iso() -> str:
    from datetime import datetime, timezone  # noqa: PLC0415

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _guess_artifact(build_dir: Path) -> Path | None:
    from importlib.machinery import EXTENSION_SUFFIXES  # noqa: PLC0415

    for suf in EXTENSION_SUFFIXES:
        # Prefer any extension artifact directly under build_dir
        for p in sorted(build_dir.glob(f"*{suf}")):
            if p.is_file():
                return p
    # Also allow artifacts under a single package directory (common for dotted names)
    for pkg in sorted(build_dir.iterdir()):
        if pkg.is_dir():
            for suf in EXTENSION_SUFFIXES:
                for p in sorted(pkg.glob(f"*{suf}")):
                    if p.is_file():
                        return p
    return None


def _guess_module_name(artifact: Path) -> str:
    # Best-effort: derive module stem. For package builds, this is insufficient and
    # meta.json is authoritative.
    stem = artifact.name
    for suf in (".so", ".pyd", ".dll", ".dylib"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    # Remove CPython ABI tags if present (best effort)
    return stem.split(".", 1)[0]


def _artifact_from_meta_or_guess(
    build_dir: Path, meta: Mapping[str, Any] | None
) -> Path | None:
    """Select the cache artifact, constrained to live inside ``build_dir``.

    The ``artifact`` name recorded in ``meta.json`` is treated as a **basename
    only**.  Any value that is absolute, contains a path separator, or resolves
    outside ``build_dir`` is rejected rather than imported (CYTHON-CACHE-002):
    a tampered ``meta.json`` must not be able to redirect the loader to an
    artifact outside the cache entry.  When ``meta`` records an
    ``artifact_sha256``, the selected file's content hash must match it.
    """
    artifact: Path | None = None
    if meta is not None:
        a = meta.get("artifact") or meta.get("artifact_filename")
        if isinstance(a, str) and a:
            # Reject absolute paths and any embedded directory component; only a
            # bare filename inside build_dir is acceptable.
            if os.path.isabs(a) or ("/" in a) or ("\\" in a) or (".." in Path(a).parts):
                artifact = None
            else:
                # Resolve without requiring existence (strict=False), then
                # confirm the artifact is a direct child of build_dir.
                try:
                    base = build_dir.resolve()
                    p = (build_dir / a).resolve()
                except OSError:
                    base = None
                    p = None
                if (
                    p is not None
                    and base is not None
                    and p.parent == base
                    and p.exists()
                    and p.is_file()
                ):
                    artifact = p
    selected = artifact or _guess_artifact(build_dir)
    if selected is not None and meta is not None:
        expected = meta.get("artifact_sha256")
        if isinstance(expected, str) and expected:
            actual = sha256(selected.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(
                    f"cache artifact integrity check failed for {selected.name} "
                    f"(recorded sha256 does not match on-disk content)"
                )
    return selected


def _module_name_from_meta_or_guess(
    meta: Mapping[str, Any] | None, artifact: Path
) -> str:
    if meta is not None:
        mn = meta.get("module_name")
        if isinstance(mn, str) and mn:
            return mn
    return _guess_module_name(artifact)


def _json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        _stable_repr(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
