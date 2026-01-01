# scikitplot/annoy/_utils.py
"""
Internal utilities shared across Annoy mixins.

This module is *not* part of the public API.

Notes
-----
- Helpers here are intentionally small, dependency-free, and deterministic.
- The Annoy backend is implemented as a C-extension type, so attribute access and
  mutation must be defensive (some builds disallow dynamic attributes).
"""

from __future__ import annotations

import os
import tempfile
import threading
from os import PathLike
from pathlib import Path  # noqa: F401
from typing import Any

# ----------------------------------------------------------------------
# Locks / backend access
# ----------------------------------------------------------------------

FALLBACK_LOCK = threading.RLock()
_RLOCK_TYPE = type(FALLBACK_LOCK)


def lock_for(obj: Any) -> threading.RLock:
    """
    Return a reusable re-entrant lock for *obj*.

    Preference order is deterministic:

    1) ``obj._get_lock()`` if present and returns an ``RLock``.
    2) ``obj._lock`` if present and is an ``RLock``.
    3) Attempt to set ``obj._lock`` to a new ``RLock``.
    4) Fallback to a module-level lock.

    The fallback lock is safe but may be more contended if many distinct objects
    are used concurrently.

    Parameters
    ----------
    obj
        Target object.

    Returns
    -------
    lock
        A re-entrant lock suitable for guarding short critical sections.
    """
    get_lock = getattr(obj, "_get_lock", None)
    if callable(get_lock):
        try:
            lock = get_lock()
            if isinstance(lock, _RLOCK_TYPE):
                return lock
        except Exception:
            # Never let lock acquisition become user-visible; fall back.
            pass

    try:
        lock = obj._lock
        if isinstance(lock, _RLOCK_TYPE):
            return lock
    except Exception:
        pass

    lock = threading.RLock()
    # Prefer object.__setattr__ to bypass custom __setattr__ implementations.
    try:
        object.__setattr__(obj, "_lock", lock)
        return lock
    except Exception:
        try:
            obj._lock = lock
            return lock
        except Exception:
            return FALLBACK_LOCK


# Backwards-compatible aliases (internal; may be referenced by older mixins)
_get_lock = lock_for
_lock_for = lock_for


def backend_for(obj: Any) -> Any:
    """
    Return the low-level Annoy backend for ``obj``.

    This helper supports both inheritance and composition styles:

    - **Inheritance**: the object *is* the backend.
    - **Composition**: the object stores the backend in ``obj._annoy``.

    If the object provides ``obj._backend`` (method or property), it is respected.

    Parameters
    ----------
    obj
        Target object.

    Returns
    -------
    backend
        The low-level backend instance.
    """
    # Fast-path: composition attribute, accessed defensively.
    try:
        return object.__getattribute__(obj, "_annoy")
    except Exception:
        pass

    backend_attr = getattr(obj, "_backend", None)
    if callable(backend_attr):
        try:
            return backend_attr()
        except TypeError:
            # Non-standard signatures are ignored deterministically.
            pass
    elif backend_attr is not None:
        # Property-style backend accessor.
        return backend_attr

    return obj


# Backwards-compatible alias
_backend = backend_for

# ----------------------------------------------------------------------
# Filesystem helpers
# ----------------------------------------------------------------------


def _ensure_parent_dir(path: str | PathLike[str]) -> None:
    """Create the parent directory for ``path`` if it does not exist."""
    # converts path-like objects to filesystem paths
    p = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(p))
    if parent:
        os.makedirs(parent, exist_ok=True)


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory for ``path`` if it does not exist."""
    # converts path-like objects to filesystem paths
    p = os.fspath(path)
    p = Path(p).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_replace(tmp_path: str, dst_path: str) -> None:
    """Replace ``dst_path`` with ``tmp_path`` using :func:`os.replace`."""
    os.replace(tmp_path, dst_path)


def atomic_write_text(
    path: str | PathLike[str],
    data: str,
    *,
    encoding: str = "utf-8",
    newline: str = "\n",
) -> None:
    """
    Atomically write text to ``path`` (best-effort cross-platform).

    The data is written to a unique temporary file in the same directory, flushed,
    fsynced, and then replaced into place via :func:`os.replace`.

    Parameters
    ----------
    path
        Destination path.
    data
        Text to write.
    encoding
        Text encoding.
    newline
        Newline policy passed to :func:`open`.

    Raises
    ------
    OSError
        If the temporary file cannot be written or replaced.
    """
    p = os.fspath(path)
    ensure_parent_dir(p)

    directory = os.path.dirname(os.path.abspath(p)) or "."
    base = os.path.basename(p)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{base}.",
        suffix=".tmp",
        dir=directory,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline=newline) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _atomic_replace(tmp, p)
    finally:
        # If replacement failed, clean up the temp file.
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass


def atomic_write_bytes(path: str | PathLike[str], data: bytes) -> None:
    """Atomically write bytes to ``path`` (best-effort cross-platform)."""
    p = os.fspath(path)
    ensure_parent_dir(p)

    directory = os.path.dirname(os.path.abspath(p)) or "."
    base = os.path.basename(p)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{base}.",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _atomic_replace(tmp, p)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass


def read_text(path: str | PathLike[str], *, encoding: str = "utf-8") -> str:
    """Read a UTF-8 text file."""
    with open(os.fspath(path), "r", encoding=encoding) as f:
        return f.read()


def read_bytes(path: str | PathLike[str]) -> bytes:
    """Read a binary file."""
    with open(os.fspath(path), "rb") as f:
        return f.read()
