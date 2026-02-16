# scikitplot/cython/_cache.py
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

import os
import platform
import re
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

_ENV_CACHE_DIR = "SCIKITPLOT_CYTHON_CACHE_DIR"
_KEY_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


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
    env = os.environ.get(_ENV_CACHE_DIR)
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
    env = os.environ.get(_ENV_CACHE_DIR)
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


def runtime_fingerprint(
    *, cython_version: str, numpy_version: str | None
) -> Mapping[str, Any]:
    """
    Compute a runtime fingerprint for caching correctness.

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
    """
    return {
        "python": platform.python_version(),
        "python_impl": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cython": cython_version,
        "numpy": numpy_version,
        "abi": getattr(sys, "abiflags", ""),
    }


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
    Write ``meta.json`` in the build directory.

    Parameters
    ----------
    build_dir : pathlib.Path
        Cache entry directory.
    meta : Mapping[str, Any]
        Metadata mapping.
    """
    path = build_dir / "meta.json"
    path.write_text(_json_dumps(meta) + "\n", encoding="utf-8")


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
        import json  # noqa: PLC0415

        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def iter_all_entry_dirs(cache_root: str | Path | None) -> Iterable[Path]:
    """
    Yield all cache entry directories whose name is a valid cache key.

    Parameters
    ----------
    cache_root : str or pathlib.Path or None
        Cache root.

    Yields
    ------
    pathlib.Path
        Entry directory paths.
    """
    root = peek_cache_dir(cache_root)
    if not root.exists():
        return []
    return (p for p in sorted(root.iterdir()) if p.is_dir() and is_valid_key(p.name))


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

    # Reuse parsing logic by iterating this one entry
    entries = iter_package_entries(root)
    for e in entries:
        if e.key == k:
            return e
    raise FileNotFoundError(f"Package entry is missing artifacts for key: {k}")


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
    artifact: Path | None = None
    if meta is not None:
        a = meta.get("artifact") or meta.get("artifact_filename")
        if isinstance(a, str) and a:
            p = (build_dir / a) if not os.path.isabs(a) else Path(a)
            if p.exists():
                artifact = p
    return artifact or _guess_artifact(build_dir)


def _module_name_from_meta_or_guess(
    meta: Mapping[str, Any] | None, artifact: Path
) -> str:
    if meta is not None:
        mn = meta.get("module_name")
        if isinstance(mn, str) and mn:
            return mn
    return _guess_module_name(artifact)


def _json_dumps(payload: Mapping[str, Any]) -> str:
    import json  # noqa: PLC0415

    return json.dumps(
        _stable_repr(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
