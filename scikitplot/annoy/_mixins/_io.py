# scikitplot/annoy/_mixins/_io.py
"""
Persistence helpers for Annoy-backed indices.

This module provides **explicit, deterministic** I/O helpers on top of the
low-level Annoy backend.

There are two distinct persistence concepts:

1. **Annoy index persistence** (native, recommended)
   Writes/loads the actual forest via the low-level backend:

   - :py:meth:`save_index` / :py:meth:`load_index` wrap backend ``save`` / ``load``
   - :py:meth:`to_bytes` / :py:meth:`from_bytes` wrap backend ``serialize`` / ``deserialize``

2. **Python object persistence** (pickling)
   Pickling serializes the *Python object* and is handled by
   :class:`~scikitplot.annoy._mixins._pickle.PickleMixin`.
   (Pickle is unsafe on untrusted data; see that mixin's Notes.)

This file intentionally does **not** implement general-purpose ``pickle``.
"""

from __future__ import annotations

import contextlib  # noqa: F401
import os
import shutil  # noqa: F401
import tempfile  # noqa: F401
from collections.abc import Callable  # noqa: F401
from os import PathLike
from pathlib import Path  # noqa: F401

# from typing import Callable
from typing_extensions import Self

from .._utils import backend_for, ensure_parent_dir, lock_for

__all__ = ["IndexIOMixin"]


class IndexIOMixin:
    """
    Mixin adding explicit Annoy-native persistence helpers.

    The concrete class must provide low-level Annoy methods, typically from the
    C-extension backend:

    - ``save(path, prefault=...)``
    - ``load(path, prefault=...)``
    - ``serialize() -> bytes-like``
    - ``deserialize(data: bytes-like, prefault=...)``

    Notes
    -----
    - Methods in this mixin acquire a per-instance lock if one is available.
    - :py:meth:`save_index` defaults to Annoy.save
    - :py:meth:`save_bundle` / :py:meth:`load_bundle` require :py:meth:`to_json` /
      :py:meth:`from_json` (compose with :class:`~scikitplot.annoy._mixins._meta.MetaMixin`).
    """

    # --------
    # Disk I/O
    # --------
    def save_index(
        self,
        path: str | PathLike[str],
        *,
        prefault: bool | None = None,
    ) -> Self:
        """
        Persist the Annoy index to disk.

        Parameters
        ----------
        path : str or os.PathLike
            Destination path for the Annoy index file.
        prefault
            Forwarded to the backend. If ``None``, the backend default is used.

        Raises
        ------
        AttributeError
            If the backend does not provide ``save(path, prefault=...)``.
        OSError
            For filesystem-level failures.
        """
        backend = backend_for(self)
        save = getattr(backend, "save", None)
        if not callable(save):
            raise AttributeError("Backend does not provide save(path, prefault=...)")

        p = os.fspath(path)
        ensure_parent_dir(p)

        lock = lock_for(self)
        with lock:
            if prefault is None:
                save(p)
            else:
                save(p, prefault=bool(prefault))
        return self

    @classmethod
    def load_index(
        cls: type[Self],
        f: int,
        metric: str,
        path: str | PathLike[str],
        *,
        prefault: bool | None = None,
    ) -> Self:
        """
        Load (mmap) an Annoy index file into this object.

        Parameters
        ----------
        f
            Vector dimension for construction.
        metric
            Metric name for construction.
        path : str or os.PathLike
            Path to a file previously created by :py:meth:`save_index` or the
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
        if int(f) <= 0:
            raise ValueError("f must be a positive integer")
        if not isinstance(metric, str) or not metric:
            raise ValueError("metric must be a non-empty string")

        obj = cls(int(f), metric)
        backend = backend_for(obj)
        load = getattr(backend, "load", None)
        if not callable(load):
            raise AttributeError("Backend does not provide load(path, prefault=...)")

        p = os.fspath(path)

        lock = lock_for(obj)
        with lock:
            if prefault is None:
                load(p)
            else:
                load(p, prefault=bool(prefault))
        return obj

    # --------
    # Bundle I/O (manifest + index)
    # --------
    def save_bundle(
        self,
        manifest_filename: str = "manifest.json",
        index_filename: str = "index.ann",
        *,
        prefault: bool | None = None,
    ) -> list[str]:
        """
        Save a *directory bundle* containing metadata + the index file.

        The bundle contains:
        - ``manifest.json``: metadata payload produced by :py:meth:`to_json`
        - ``index.ann``: Annoy index produced by :py:meth:`save_index`

        Parameters
        ----------
        manifest_filename
            Filename for the metadata manifest inside the directory.
        index_filename
            Filename for the Annoy index inside the directory.
        prefault
            Forwarded to :py:meth:`save_index`.

        Raises
        ------
        AttributeError
            If :py:meth:`to_json` is not available (compose with :class:`~scikitplot.annoy._mixins._meta.MetaMixin`).
        OSError
            On filesystem failures.
        """
        # `MetaMixin` provides `to_json()`; keep the dependency explicit.
        to_json = getattr(self, "to_json", None)
        if not callable(to_json):
            raise AttributeError(
                "save_bundle requires to_json() (compose with MetaMixin)."
            )

        manifest_path: str | PathLike[str] = os.fspath(manifest_filename)
        index_path: str | PathLike[str] = os.fspath(index_filename)

        lock = lock_for(self)
        with lock:
            self.save_index(index_path, prefault=prefault)
            # Must after save index
            to_json(manifest_path)
            return [manifest_path, index_path]

    @classmethod
    def load_bundle(
        cls: type[Self],
        manifest_filename: str = "manifest.json",
        index_filename: str = "index.ann",  # noqa: ARG003
        *,
        prefault: bool | None = None,  # noqa: ARG003
    ) -> Self:
        """
        Load a directory bundle created by :py:meth:`save_bundle`.

        Parameters
        ----------
        manifest_filename
            Filename for the metadata manifest inside the directory.
        index_filename
            Filename for the Annoy index inside the directory.
        prefault
            Forwarded to :py:meth:`load_index`.

        Returns
        -------
        index
            Newly constructed index.

        Raises
        ------
        AttributeError
            If :py:meth:`from_json` is not available (compose with :class:`~scikitplot.annoy._mixins._meta.MetaMixin`).
        TypeError
            If :py:meth:`from_json` returns an unexpected type.
        OSError
            On filesystem failures.
        """
        from_json = getattr(cls, "from_json", None)
        if not callable(from_json):
            raise AttributeError(
                "load_bundle requires from_json() (compose with MetaMixin)."
            )

        manifest_path: str | PathLike[str] = os.fspath(manifest_filename)
        obj = from_json(manifest_path, load=True)
        if not isinstance(obj, cls):
            raise TypeError("from_json() returned an unexpected type")

        index_path: str | PathLike[str]  # = os.fspath(index_filename)  # noqa: F842
        # obj.load_index(index_path, f=obj.f, metric=obj.metric, prefault=prefault)
        return obj

    # --------
    # Bytes I/O (serialize/deserialize)
    # --------
    def to_bytes(
        self,
        format=None,
    ) -> bytes:
        """
        Serialize the built index to bytes (backend ``serialize``).

        Parameters
        ----------
        format : {"native", "portable", "canonical"} or None, optional, default=None
            Serialization format. If ``None`` used ``"canonical"``

            * "native" (legacy): raw Annoy memory snapshot. Fastest, but
              only compatible when the ABI matches exactly.
            * "portable": prepend a small compatibility header (version,
              endianness, sizeof checks, metric, f) so deserialization fails
              loudly on mismatches.
            * "canonical": rebuildable wire format storing item vectors + build
              parameters. Portable across ABIs (within IEEE-754 float32) and
              restores by rebuilding trees deterministically.

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
            If the backend returns non-bytes-like data.

        Notes
        -----
        "Portable" blobs are the native snapshot with additional compatibility guards.
        They are not a cross-architecture wire format.

        "Canonical" blobs trade load time for portability: deserialization rebuilds
        the index with ``n_jobs=1`` for deterministic reconstruction.
        """
        backend = backend_for(self)
        serialize = getattr(backend, "serialize", None)
        if not callable(serialize):
            raise AttributeError("Backend does not provide serialize() -> bytes-like")

        lock = lock_for(self)
        with lock:
            data = serialize(format=format or "canonical")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("serialize() must return a bytes-like object")
        return bytes(data)

    @classmethod
    def from_bytes(
        cls: type[Self],
        data: bytes | bytearray | memoryview,
        *,
        f: int | None = None,
        metric: str | None = None,
        prefault: bool | None = None,
    ) -> Self:
        """
        Construct a new index and load it from serialized bytes.

        Parameters
        ----------
        data
            Bytes produced by :py:meth:`to_bytes` (backend ``serialize``).
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

        Notes
        -----
        Portable blobs add a small header (version, ABI sizes, endianness, metric, f)
        to ensure incompatible binaries fail loudly and safely. They are not a
        cross-architecture wire format; the payload remains Annoy's native snapshot.

        For ``data`` if fed :meth:`to_bytes(format='native') required params
        ``f``, ``metric``.
        """
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")
        # if int(f) <= 0:
        #     raise ValueError("f must be a positive integer")
        # if not isinstance(metric, str) or not metric:
        #     raise ValueError("metric must be a non-empty string")

        obj = cls(f, metric)
        backend = backend_for(obj)
        deserialize = getattr(backend, "deserialize", None)
        if not callable(deserialize):
            raise AttributeError(
                "Backend does not provide deserialize(data, prefault=...)"
            )

        lock = lock_for(obj)
        with lock:
            if prefault is None:
                deserialize(bytes(data))
            else:
                deserialize(bytes(data), prefault=bool(prefault))
        return obj
