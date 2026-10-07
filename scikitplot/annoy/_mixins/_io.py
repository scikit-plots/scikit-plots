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
import pathlib
import shutil  # noqa: F401
import tempfile  # noqa: F401
import uuid

# from typing import Callable
from collections.abc import Callable  # noqa: F401
from os import PathLike
from pathlib import Path  # noqa: F401

# Only imports when type checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
        TypeError
            If the backend does not provide ``save(path, prefault=...)``.
        OSError
            For filesystem-level failures.
        """
        backend = backend_for(self)
        save = getattr(backend, "save", None)
        if not callable(save):
            raise TypeError("Backend does not provide save(path, prefault=...)")

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
        TypeError
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
            raise TypeError("Backend does not provide load(path, prefault=...)")

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
        directory: str | os.PathLike[str],
        *,
        manifest_filename: str = "manifest.json",
        index_filename: str = "index.ann",
        prefault: bool | None = None,
    ) -> list[str]:
        """
        Publish a *directory bundle* holding the metadata and the index.

        Parameters
        ----------
        directory : str or path-like
            Bundle directory. Created if absent, replaced if present.
        manifest_filename, index_filename : str, optional
            Member names *inside* the bundle. They are never resolved against
            the process working directory.
        prefault : bool or None, optional
            Forwarded to :py:meth:`save_index`.

        Returns
        -------
        list of str
            Absolute paths of the published members, manifest first.

        Raises
        ------
        OSError
            If the bundle could not be written. Nothing is published in that
            case: a failed save leaves any previous bundle exactly as it was.

        Notes
        -----
        **User.** A bundle is self-contained and relocatable: move it, copy it,
        or publish it, and :py:meth:`load_bundle` reads it from wherever it is.

        After a successful save this index is backed by the published index
        file (``on_disk_path`` names it). After a failed save it holds the
        same vectors and trees as before, in memory.

        **Developer.** This previously took two filenames and no directory, so
        the defaults resolved against the process working directory and two
        callers using them overwrote each other. It also wrote the index and
        then the manifest in place, so a failure between the two left an index
        no loader could find. Both are fixed the same way the corpus artifact
        fixes them: build a candidate, then swap, so nothing is destroyed before
        a complete replacement exists.

        The backend's ``save`` memory-maps the file it wrote, so once the
        index member exists the index is backed by a file *inside the
        candidate*. POSIX lets a directory be renamed or removed with a mapped
        file in it; Windows does not (``PermissionError: [WinError 5]`` on the
        swap, seen in the Windows job of run 37555382019). The order below
        works on both, so there is one code path:

        1. write both members into the candidate;
        2. release the mapping (``unload``), swap the directories, map the
           published index (``load``);
        3. on any failure after step 1, take the index back into memory
           (``serialize``, ``unload``, ``deserialize``) *before* the candidate
           is removed, so the object keeps its contents and nothing of the
           candidate is in use when it is deleted.

        Not changed here, recorded in ``tasks/todo.md`` (ANNOY-BUNDLE-PATH):
        the manifest's ``on_disk_path`` is the path of the candidate, which no
        longer exists after publication. :py:meth:`load_bundle` does not read
        it.
        """
        backend = backend_for(self)
        needed = ("load", "unload", "serialize", "deserialize")
        missing = [
            name for name in needed if not callable(getattr(backend, name, None))
        ]
        if missing:
            raise TypeError(
                "Backend does not provide "
                + ", ".join(f"{name}()" for name in missing)
                + ", which save_bundle needs to publish a bundle and to keep "
                "this index usable if publishing fails"
            )

        target = pathlib.Path(os.fspath(directory)).resolve()
        candidate = target.parent / f".{target.name}.candidate-{uuid.uuid4().hex}"
        superseded = target.parent / f".{target.name}.superseded-{uuid.uuid4().hex}"
        candidate_index = os.fspath(candidate / index_filename)
        published_index = os.fspath(target / index_filename)

        def load(path: str) -> None:
            if prefault is None:
                backend.load(path)
            else:
                backend.load(path, prefault=bool(prefault))

        def back_into_memory(*, mapped: bool) -> None:
            # The only complete copy of the index is the candidate's member.
            if not mapped:
                load(candidate_index)
            snapshot = backend.serialize(format="native")
            backend.unload()
            backend.deserialize(snapshot)

        with lock_for(self):
            target.parent.mkdir(parents=True, exist_ok=True)
            candidate.mkdir(parents=True)
            try:
                self.save_index(candidate_index, prefault=prefault)
            except BaseException:
                # The backend switches to the file only after it is complete,
                # so a failed save leaves nothing of the candidate in use.
                shutil.rmtree(candidate, ignore_errors=True)
                raise

            mapped = True
            had_previous = False
            try:
                self.to_json(os.fspath(candidate / manifest_filename))
                backend.unload()
                mapped = False
                had_previous = target.exists()
                if had_previous:
                    os.replace(target, superseded)
                try:
                    os.replace(candidate, target)
                except BaseException:
                    if had_previous and superseded.exists():
                        os.replace(superseded, target)
                    raise
            except BaseException:
                try:
                    back_into_memory(mapped=mapped)
                finally:
                    shutil.rmtree(candidate, ignore_errors=True)
                raise
            load(published_index)
        if had_previous:
            shutil.rmtree(superseded, ignore_errors=True)
        return [
            os.fspath(target / manifest_filename),
            published_index,
        ]

    @classmethod
    def load_bundle(
        cls: type[Self],
        directory: str | os.PathLike[str],
        *,
        manifest_filename: str = "manifest.json",
        index_filename: str = "index.ann",
        prefault: bool | None = None,
    ) -> Self:
        """
        Load a bundle published by :py:meth:`save_bundle`.

        Parameters
        ----------
        directory : str or path-like
            Bundle directory.
        manifest_filename, index_filename : str, optional
            Member names inside the bundle.
        prefault : bool or None, optional
            Forwarded to :py:meth:`load_index`.

        Returns
        -------
        Self
            An index with its vectors loaded and ready to query.

        Raises
        ------
        OSError
            If a member is missing or unreadable.

        Notes
        -----
        **Developer.** This previously did not load the index at all: the
        ``load_index`` call was commented out and both ``index_filename`` and
        ``prefault`` were marked unused, so the vectors arrived only as a side
        effect of the manifest carrying an absolute path recorded at save time.
        A bundle that had been moved therefore failed, and naming a different
        index file had no effect whatsoever. Members are now resolved relative
        to the bundle, which is what makes it relocatable, and the index is
        loaded here rather than by accident.
        """
        root = pathlib.Path(os.fspath(directory)).resolve()
        # The manifest states the shape of the index; the index file supplies
        # its contents. load=False keeps from_json from chasing the absolute
        # path it recorded at save time, which is what made a moved bundle fail.
        described = cls.from_json(os.fspath(root / manifest_filename), load=False)
        return cls.load_index(
            described.f,
            described.metric,
            os.fspath(root / index_filename),
            prefault=prefault,
        )

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
        TypeError
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
            raise TypeError("Backend does not provide serialize() -> bytes-like")

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
        TypeError
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
            raise TypeError("Backend does not provide deserialize(data, prefault=...)")

        lock = lock_for(obj)
        with lock:
            if prefault is None:
                deserialize(bytes(data))
            else:
                deserialize(bytes(data), prefault=bool(prefault))
        return obj
