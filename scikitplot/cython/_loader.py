# scikitplot/cython/_loader.py
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
    sys.modules.pop(name, None)

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for '{name}' from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

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
    except Exception:
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
            except Exception:
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
    out_dir = td / "scikitplot_cython_import" / h[:16]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / artifact_filename

    if out_path.exists():
        existing = out_path.read_bytes()
        if existing != data:
            raise OSError(f"Artifact collision at {out_path}")
    else:
        out_path.write_bytes(data)

    return import_extension(name=module_name, path=out_path, key=key, build_dir=out_dir)
