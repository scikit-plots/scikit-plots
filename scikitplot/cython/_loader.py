# scikitplot/cython/_loader.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Extension module loader.

Extension modules (``.so`` / ``.pyd``) must be imported with the *same* module
name they were compiled for, because the init symbol is name-dependent.

This module contains strict helper utilities used by the public API.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from types import ModuleType
from typing import Any

__all__ = [
    "import_extension",
    "import_extension_from_bytes",
    "import_extension_from_path",
]


def import_extension(
    *,
    name: str,
    path: Path,
    key: str | None = None,
    build_dir: Path | None = None,
) -> ModuleType:
    """
    Import an extension module from an explicit artifact path.

    Parameters
    ----------
    name : str
        Module name used at compilation time.
    path : pathlib.Path
        Compiled extension artifact path.
    key : str or None, default=None
        Cache key to attach to the loaded module.
    build_dir : pathlib.Path or None, default=None
        Cache entry directory to attach to the loaded module.

    Returns
    -------
    types.ModuleType
        Imported module.

    Raises
    ------
    ImportError
        If the module cannot be loaded.
    """
    importlib.invalidate_caches()

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        # Do not disturb any existing sys.modules[name] on a spec failure.
        raise ImportError(f"Could not create module spec for '{name}' from {path}")

    module = importlib.util.module_from_spec(spec)

    # Module-load transaction (CYTHON-LOAD-001): remember whatever is currently
    # registered so a failed (re)load never destroys a previously-working
    # module.  Per the import protocol, the new module is registered *before*
    # exec_module (so self-referential imports during initialisation resolve),
    # then rolled back to the prior state on any failure.
    _sentinel = object()
    prior = sys.modules.get(name, _sentinel)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except BaseException:
        # Roll back: restore the prior entry, or remove ours if there was none.
        if prior is _sentinel:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prior  # type: ignore[assignment]
        raise

    # Attach cache metadata for easy reuse after restart.
    try:
        if key is not None:
            setattr(module, "__scikitplot_cython_key__", key)  # noqa: B010
        if build_dir is not None:
            setattr(  # noqa: B010
                module,
                "__scikitplot_cython_build_dir__",
                str(build_dir),
            )
        setattr(module, "__scikitplot_cython_artifact__", str(path))  # noqa: B010
    except Exception:  # noqa: BLE001
        pass

    return module


def _read_meta_near_artifact(
    artifact: Path,
) -> tuple[dict[str, Any] | None, Path | None]:
    """
    Read meta.json from directories near an artifact.

    For module builds, meta.json is typically in the artifact's parent directory.
    For package builds, artifacts often live under ``<build_dir>/<package>/``, so
    meta.json is usually in ``artifact.parent.parent``.

    Returns
    -------
    (meta, build_dir)
        meta dict and the directory containing meta.json.
    """
    for d in (artifact.parent, artifact.parent.parent):
        m = d / "meta.json"
        if m.exists():
            try:
                data = json.loads(m.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data, d
            except Exception:  # noqa: BLE001
                return None, None
    return None, None


def import_extension_from_path(  # noqa: PLR0912
    artifact_path: str | os.PathLike[str] | os.PathLike[bytes] | bytes,
    *,
    module_name: str | None = None,
    key: str | None = None,
    build_dir: Path | None = None,
) -> ModuleType:
    """
    Import an extension module from a filesystem path.

    Parameters
    ----------
    artifact_path : path-like
        Path to a compiled extension artifact (``.so`` / ``.pyd``).
    module_name : str or None, default=None
        Module name the artifact was compiled for. If None, this function will
        attempt to read ``meta.json`` near the artifact to obtain the authoritative
        name (strict). For package builds, this may be ``meta.json`` one level above
        the package directory.
    key : str or None, default=None
        Optional cache key to attach to the loaded module (overrides meta.json).
    build_dir : pathlib.Path or None, default=None
        Optional build directory to attach to the loaded module (overrides meta.json).

    Returns
    -------
    types.ModuleType
        Imported extension module.

    Raises
    ------
    FileNotFoundError
        If the artifact does not exist.
    ValueError
        If the artifact suffix is invalid or module name cannot be determined.
    ImportError
        If the module cannot be loaded.
    """
    ap = Path(os.fsdecode(os.fspath(artifact_path))).expanduser().resolve()
    if not ap.exists():
        raise FileNotFoundError(str(ap))

    if not any(ap.name.endswith(suf) for suf in EXTENSION_SUFFIXES):
        raise ValueError(f"Not a recognized extension artifact: {ap.name}")

    name = module_name
    meta, meta_dir = _read_meta_near_artifact(ap)

    if name is None and meta is not None:
        kind = meta.get("kind")
        if kind == "package":
            modules = meta.get("modules")
            if isinstance(modules, list):
                # Match by artifact filename (relative paths stored in meta)
                for m in modules:
                    if not isinstance(m, dict):
                        continue
                    a = m.get("artifact")
                    mn = m.get("module_name")
                    if isinstance(a, str) and isinstance(mn, str) and a and mn:
                        # artifact in meta is relative to meta_dir/build_dir
                        cand = (meta_dir / a) if meta_dir is not None else None
                        if cand is not None and cand.resolve() == ap:
                            name = mn
                            break
        else:
            mn = meta.get("module_name")
            if isinstance(mn, str) and mn:
                name = mn

    if name is None:
        raise ValueError(
            f"module_name is required when meta.json is not available (artifact={ap})"
        )

    # Attach key/build_dir from metadata if caller did not override
    if key is None and meta is not None and isinstance(meta.get("key"), str):
        key = meta.get("key")
    if build_dir is None and meta_dir is not None:
        build_dir = meta_dir

    return import_extension(name=name, path=ap, key=key, build_dir=build_dir)


def import_extension_from_bytes(
    data: bytes,
    *,
    module_name: str,
    artifact_filename: str,
    temp_dir: str | os.PathLike[str] | None = None,
    key: str | None = None,
) -> ModuleType:
    """
    Import an extension module from raw artifact bytes.

    Notes
    -----
    Python extension modules cannot be imported directly from memory; the artifact
    must exist as a file on disk. This function writes the provided bytes to a
    deterministic location (by content hash) under ``temp_dir`` and imports it.

    Parameters
    ----------
    data : bytes
        Raw contents of a compiled extension artifact (``.so`` / ``.pyd``).
    module_name : str
        Module name the artifact was compiled for (init symbol name).
    artifact_filename : str
        Filename to use when writing the artifact (must end with a valid
        extension suffix). This must be a simple filename (no directories).
    temp_dir : str or os.PathLike or None, default=None
        Directory to place the hash-scoped artifact file. If None, a platform
        temp directory is used.
    key : str or None, default=None
        Optional cache key to attach to the loaded module.

    Returns
    -------
    types.ModuleType
        Imported extension module.

    Raises
    ------
    ValueError
        If ``artifact_filename`` is invalid.
    OSError
        If a conflicting artifact already exists at the deterministic path.
    ImportError
        If the module cannot be loaded.
    """
    if (
        not artifact_filename
        or ("/" in artifact_filename)
        or ("\\" in artifact_filename)
        or (not any(artifact_filename.endswith(suf) for suf in EXTENSION_SUFFIXES))
    ):
        raise ValueError(
            "artifact_filename must be a simple filename ending with a valid extension suffix"
        )

    td = Path(temp_dir) if temp_dir is not None else Path(tempfile.gettempdir())
    td = td.expanduser().resolve()

    from hashlib import sha256  # noqa: PLC0415

    h = sha256(data).hexdigest()
    parent = td / "scikitplot_cython_import"
    # Create the shared parent, then the content-scoped entry with private
    # (0700) permissions so other users cannot pre-create or tamper with the
    # artifact directory (CYTHON-LOAD-002).
    parent.mkdir(parents=True, exist_ok=True)
    out_dir = parent / h[:16]
    try:  # ruff:ignore[suppressible-exception]
        out_dir.mkdir(mode=0o700, exist_ok=True)
    except FileExistsError:
        pass
    out_path = out_dir / artifact_filename

    _stage_artifact_bytes_atomically(out_dir, out_path, data, expected_sha256=h)

    return import_extension(name=module_name, path=out_path, key=key, build_dir=out_dir)


def _stage_artifact_bytes_atomically(
    out_dir: Path, out_path: Path, data: bytes, *, expected_sha256: str
) -> None:
    """Atomically publish ``data`` at ``out_path`` without following symlinks.

    Security properties (CYTHON-LOAD-002):

    - If ``out_path`` already exists it must be a **regular file** (never a
      symlink) whose content hash matches ``expected_sha256``; otherwise an
      ``OSError`` is raised rather than importing an attacker-controlled or
      corrupted artifact.
    - New content is written to a unique temp file created with ``mkstemp``
      (``O_CREAT | O_EXCL``, no symlink follow) inside ``out_dir`` and then
      atomically ``os.replace``-d into place, so a concurrent importer never
      observes a partially written extension.

    Parameters
    ----------
    out_dir : pathlib.Path
        Private (0700) content-scoped directory that will hold the artifact.
    out_path : pathlib.Path
        Final artifact path.
    data : bytes
        Artifact bytes to publish.
    expected_sha256 : str
        Hex digest the published bytes must have.

    Raises
    ------
    OSError
        If an existing path is a symlink, is not a regular file, or holds
        different bytes (a genuine collision or tampering).
    """
    from hashlib import sha256  # noqa: PLC0415

    # Fast path: a correct artifact is already published.  Use lstat so a
    # symlink is detected instead of being transparently followed.
    if out_path.exists() or out_path.is_symlink():
        st = os.lstat(out_path)
        import stat as _stat  # noqa: PLC0415

        if not _stat.S_ISREG(st.st_mode):
            raise OSError(f"refusing to load non-regular artifact path: {out_path}")
        existing = out_path.read_bytes()
        if sha256(existing).hexdigest() != expected_sha256:
            raise OSError(f"Artifact collision at {out_path}")
        return

    # Write to a unique temp file (O_CREAT|O_EXCL, no follow) then atomic swap.
    fd, tmp_name = tempfile.mkstemp(prefix=".artifact-", dir=str(out_dir))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        # Atomic publish.  On POSIX os.replace overwrites silently; because the
        # path is content-addressed, any concurrent writer publishes identical
        # bytes, so last-writer-wins is safe.  We re-verify the final content
        # as defense in depth against tampering during the window.
        os.replace(tmp_path, out_path)
        published = out_path.read_bytes()
        if sha256(published).hexdigest() != expected_sha256:
            raise OSError(f"Artifact content mismatch after publish at {out_path}")
    finally:
        if tmp_path.exists():
            try:  # ruff:ignore[suppressible-exception]
                tmp_path.unlink()
            except FileNotFoundError:
                pass
