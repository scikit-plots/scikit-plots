# scikitplot/annoy/_mixins/_io.py
"""
Persistence helpers for Annoy-backed indices.

This module provides **explicit, deterministic** I/O helpers on top of the
low-level Annoy backend.

There are two distinct persistence concepts:

1. **Annoy index persistence** (native, recommended)
   Writes/loads the actual forest via the low-level backend:

   - :meth:`save_index` / :meth:`load_index` wrap backend ``save`` / ``load``
   - :meth:`to_bytes` / :meth:`from_bytes` wrap backend ``serialize`` / ``deserialize``

2. **Python object persistence** (pickling)
   Pickling serializes the *Python object* and is handled by
   :class:`~scikitplot.annoy._mixins._pickle.PickleMixin`.
   (Pickle is unsafe on untrusted data; see that mixin's Notes.)

This file intentionally does **not** implement general-purpose ``pickle``.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from os import PathLike
from typing import Callable

from typing_extensions import Self

from .._utils import _backend, _get_lock, ensure_parent_dir

__all__ = ["IndexIOMixin"]


def _atomic_backend_write(target_path: str, write_fn: Callable[[str], None]) -> None:
    """
    Atomically write/replace a file produced by a backend writer.

    Parameters
    ----------
    target_path
        Final destination path.
    write_fn
        Function called as ``write_fn(tmp_path)`` that must write the full file.

    Notes
    -----
    - Uses a *unique* temporary file in the same directory as the target and
      replaces it into place via :func:`os.replace`.
    - Best-effort cleanup is attempted on failure.
    """
    ensure_parent_dir(target_path)
    directory = os.path.dirname(target_path) or "."
    base = os.path.basename(target_path)

    fd, tmp = tempfile.mkstemp(prefix=f".{base}.", suffix=".tmp", dir=directory)
    os.close(fd)

    try:
        write_fn(tmp)
        os.replace(tmp, target_path)
    finally:
        # If replacement failed, clean up the temp file.
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass


class IndexIOMixin:
    """
    Mixin adding explicit Annoy-native persistence helpers.

    The concrete class must provide low-level Annoy methods, typically from the
    C-extension backend:

    - ``save(path, prefault=...)``
    - ``load(path, prefault=...)``
    - ``serialize() -> bytes``
    - ``deserialize(data: bytes, prefault=...)``

    Notes
    -----
    - Methods in this mixin acquire a per-instance lock if one is available.
    - :meth:`save_index` defaults to atomic writes (write temp + :func:`os.replace`)
      to reduce the risk of partial/corrupt files on failure.
    - :meth:`save_bundle` / :meth:`load_bundle` require :meth:`to_json` /
      :meth:`from_json` (compose with :class:`~scikitplot.annoy._mixins._meta.MetaMixin`).
    """

    # --------
    # Disk I/O
    # --------
    def save_index(
        self,
        path: str | PathLike[str],
        *,
        prefault: bool | None = None,
        atomic: bool = True,
    ) -> None:
        """
        Persist the Annoy index to disk.

        Parameters
        ----------
        path : str or os.PathLike
            Destination path for the Annoy index file.
        prefault
            Forwarded to the backend. If ``None``, the backend default is used.
        atomic : bool, default=True
            If True, write to a unique temporary file and replace into place.

        Raises
        ------
        AttributeError
            If the backend does not provide ``save(path, prefault=...)``.
        OSError
            For filesystem-level failures.
        """
        p = os.fspath(path)
        ensure_parent_dir(p)

        backend = _backend(self)
        save = getattr(backend, "save", None)
        if not callable(save):
            raise AttributeError("Backend does not provide save(path, prefault=...)")

        lock = _get_lock(self)

        def _write(dst: str) -> None:
            if prefault is None:
                save(dst)
            else:
                save(dst, prefault=bool(prefault))

        with lock:
            if not atomic:
                _write(p)
            else:
                _atomic_backend_write(p, _write)

        # Best-effort: keep a stable 'on_disk_path' attribute in sync when possible.
        for attr in ("on_disk_path", "_on_disk_path"):
            with contextlib.suppress(Exception):
                setattr(self, attr, p)

    def load_index(
        self,
        path: str | PathLike[str],
        *,
        prefault: bool | None = None,
    ) -> None:
        """
        Load (mmap) an Annoy index file into this object.

        Parameters
        ----------
        path : str or os.PathLike
            Path to a file previously created by :meth:`save_index` or the
            backend ``save``.
        prefault
            Forwarded to the backend. If ``None``, the backend default is used.

        Raises
        ------
        AttributeError
            If the backend does not provide ``load(path, prefault=...)``.
        OSError
            If loading fails (backend or filesystem).
        """
        p = os.fspath(path)

        backend = _backend(self)
        load = getattr(backend, "load", None)
        if not callable(load):
            raise AttributeError("Backend does not provide load(path, prefault=...)")
        lock = _get_lock(self)

        with lock:
            if prefault is None:
                load(p)
            else:
                load(p, prefault=bool(prefault))

        for attr in ("on_disk_path", "_on_disk_path"):
            with contextlib.suppress(Exception):
                setattr(self, attr, p)

    # --------
    # Bytes I/O (serialize/deserialize)
    # --------
    def to_bytes(self) -> bytes:
        """
        Serialize the built index to bytes (backend ``serialize``).

        Returns
        -------
        data
            Serialized index bytes.

        Raises
        ------
        AttributeError
            If the backend does not provide ``serialize``.
        RuntimeError
            If serialization fails.
        TypeError
            If the backend returns non-bytes data.
        """
        backend = _backend(self)
        serialize = getattr(backend, "serialize", None)
        if not callable(serialize):
            raise AttributeError("Backend does not provide serialize() -> bytes")

        lock = _get_lock(self)
        with lock:
            data = serialize()
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("serialize() must return bytes")
        return bytes(data)

    @classmethod
    def from_bytes(
        cls: type[Self],
        data: bytes,
        *,
        f: int,
        metric: str,
        prefault: bool | None = None,
    ) -> Self:
        """
        Construct a new index and load it from serialized bytes.

        Parameters
        ----------
        data
            Bytes produced by :meth:`to_bytes` (backend ``serialize``).
        f
            Vector dimension for construction.
        metric
            Metric name for construction.
        prefault
            Forwarded to the backend ``deserialize`` if supported.

        Returns
        -------
        index
            Newly constructed index with the data loaded.

        Raises
        ------
        TypeError
            If ``data`` is not bytes-like.
        ValueError
            If ``f`` or ``metric`` is invalid.
        AttributeError
            If the backend does not provide ``deserialize``.
        """
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")
        if int(f) <= 0:
            raise ValueError("f must be a positive integer")
        if not isinstance(metric, str) or not metric:
            raise ValueError("metric must be a non-empty string")

        obj = cls(int(f), str(metric))
        backend = _backend(obj)
        deserialize = getattr(backend, "deserialize", None)
        if not callable(deserialize):
            raise AttributeError(
                "Backend does not provide deserialize(data, prefault=...)"
            )
        lock = _get_lock(obj)

        with lock:
            if prefault is None:
                deserialize(bytes(data))
            else:
                deserialize(bytes(data), prefault=bool(prefault))
        return obj

    # --------
    # Bundle I/O (manifest + index)
    # --------
    def save_bundle(
        self,
        directory: str | PathLike[str],
        *,
        index_filename: str = "index.ann",
        manifest_filename: str = "manifest.json",
        prefault: bool | None = None,
    ) -> None:
        """
        Save a *directory bundle* containing metadata + the index file.

        The bundle contains:
        - ``manifest.json``: metadata payload produced by :meth:`to_json`
        - ``index.ann``: Annoy index produced by :meth:`save_index`

        Parameters
        ----------
        directory
            Destination directory (created if missing).
        index_filename
            Filename for the Annoy index inside the directory.
        manifest_filename
            Filename for the metadata manifest inside the directory.
        prefault
            Forwarded to :meth:`save_index`.

        Raises
        ------
        AttributeError
            If :meth:`to_json` is not available (compose with :class:`~scikitplot.annoy._mixins._meta.MetaMixin`).
        OSError
            On filesystem failures.
        """
        # `MetaMixin` provides `to_json()`; keep the dependency explicit.
        to_json = getattr(self, "to_json", None)
        if not callable(to_json):
            raise AttributeError(
                "save_bundle requires to_json() (compose with MetaMixin)."
            )

        dir_s = os.fspath(directory)
        os.makedirs(dir_s, exist_ok=True)

        manifest_path = os.path.join(dir_s, manifest_filename)
        index_path = os.path.join(dir_s, index_filename)

        # Let MetaMixin handle atomic metadata writes.
        to_json(manifest_path)

        self.save_index(index_path, prefault=prefault, atomic=True)

    @classmethod
    def load_bundle(
        cls: type[Self],
        directory: str | PathLike[str],
        *,
        index_filename: str = "index.ann",
        manifest_filename: str = "manifest.json",
        prefault: bool | None = None,
    ) -> Self:
        """
        Load a directory bundle created by :meth:`save_bundle`.

        Parameters
        ----------
        directory
            Bundle directory.
        index_filename
            Filename for the Annoy index inside the directory.
        manifest_filename
            Filename for the metadata manifest inside the directory.
        prefault
            Forwarded to :meth:`load_index`.

        Returns
        -------
        index
            Newly constructed index.

        Raises
        ------
        AttributeError
            If :meth:`from_json` is not available (compose with :class:`~scikitplot.annoy._mixins._meta.MetaMixin`).
        TypeError
            If :meth:`from_json` returns an unexpected type.
        OSError
            On filesystem failures.
        """
        from_json = getattr(cls, "from_json", None)
        if not callable(from_json):
            raise AttributeError(
                "load_bundle requires from_json() (compose with MetaMixin)."
            )

        dir_s = os.fspath(directory)
        manifest_path = os.path.join(dir_s, manifest_filename)
        index_path = os.path.join(dir_s, index_filename)

        obj = from_json(manifest_path, load=False)
        if not isinstance(obj, cls):
            raise TypeError("from_json() returned an unexpected type")

        obj.load_index(index_path, prefault=prefault)
        return obj
