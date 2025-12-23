# scikitplot/annoy/_mixins/_io.py
"""
Index-file and object-file persistence helpers.

The Annoy stack has *two* distinct persistence layers:

1. **Index-file persistence** (Annoy-native): the low-level C-extension can
   write and memory-map the Annoy forest using :meth:`Annoy.save` /
   :meth:`Annoy.load`. This is the only way to persist the actual index
   structure efficiently.

2. **Python object persistence** (pickle family): serializes the *Python
   wrapper object* via ``pickle`` / ``cloudpickle`` / ``joblib``. This does
   **not** replace Annoy's native index persistence. It is only correct when
   the wrapped object implements a deterministic, Annoy-aware pickle contract
   (e.g., via :class:`~scikitplot.annoy._mixins._pickle.PickleMixin`).

This module keeps those concerns explicit:

* :class:`IndexIOMixin` wraps Annoy's native index-file I/O.
* :class:`PickleIOMixin` provides explicit Python-object serialization helpers
  for backwards compatibility and opt-in workflows.

Determinism:

* Annoy index files are deterministic for a fixed build configuration and
  identical inputs (dimension, metric, seed, build parameters) assuming the
  same Annoy implementation.
* Pickle-family formats are deterministic only within the same Python version
  and dependency versions. They are *not* a stable interchange format.

See Also
--------
scikitplot.annoy._mixins._manifest.ManifestMixin
    Exports/imports deterministic *metadata* (manifest) for reconstruction.
scikitplot.annoy._mixins._pickle.PickleMixin
    Provides deterministic, Annoy-specific pickling via either an on-disk
    index file or an in-memory serialized blob.
scikitplot.cexternals._annoy.annoylib.Annoy
    Low-level Annoy binding exposing :meth:`save` and :meth:`load`.

Notes
-----
Security: deserializing untrusted pickle/joblib data is unsafe.
"""

from __future__ import annotations

import os
from os import PathLike
from typing import Any, ClassVar, Literal, Self

SerializerBackend = Literal["pickle", "cloudpickle", "joblib"]

__all__ = [
    "IndexIOMixin",
    "PickleIOMixin",
    "SerializerBackend",
]


def _get_low_level(obj: Any) -> Any:
    """
    Return the low-level Annoy object.

    This helper centralizes the inheritance-vs-composition lookup used by the
    mixins in this module.

    Preference order is explicit and deterministic:

    1) ``obj._annoy`` when present (composition style)
    2) ``obj`` (inheritance style)
    """
    ll = getattr(obj, "_annoy", None)
    return ll if ll is not None else obj


class IndexIOMixin:
    """
    Mixin adding explicit Annoy-native index file persistence.

    The concrete class must provide low-level Annoy ``save``/``load`` methods.
    This is satisfied either by:

    - inheritance style: the class subclasses ``Annoy`` (``save``/``load`` on ``self``)
    - composition style: the class stores a low-level Annoy instance on ``self._annoy``

    Notes
    -----
    This mixin never pickles the Python object. It delegates persistence to
    Annoy's native binary index format.
    """

    # This mixin supports both inheritance-style (Index subclasses Annoy)
    # and composition-style (Index wraps a low-level Annoy instance).
    #
    # Concrete classes may optionally provide ``self._annoy``; if absent,
    # the low-level object is assumed to be ``self``.

    def _low_level(self) -> Any:
        """
        Return the low-level Annoy object.

        Preference order is explicit and deterministic:

        1) ``self._annoy`` when present (composition style)
        2) ``self`` (inheritance style)
        """
        ll = getattr(self, "_annoy", None)
        return ll if ll is not None else self

    def save_index(
        self, path: str | PathLike[str], *, prefault: bool | None = None
    ) -> None:
        """
        Persist the Annoy index to disk.

        Parameters
        ----------
        path : str or os.PathLike
            Destination path for the Annoy index file.
        prefault : bool or None, default=None
            If None, defer to the low-level object's configured prefault
            behavior.

        Raises
        ------
        OSError
            If the index file cannot be written.
        RuntimeError
            If the low-level index is not initialized or saving fails.

        See Also
        --------
        load_index
        """
        ll = self._low_level()
        p = os.fspath(path)
        fn = getattr(ll, "save", None)
        if fn is None:
            raise AttributeError("Low-level save() is not available")
        if prefault is None:
            fn(p)
        else:
            fn(p, prefault=prefault)

    def load_index(
        self, path: str | PathLike[str], *, prefault: bool | None = None
    ) -> None:
        """
        Load (mmap) an Annoy index file into this object.

        Parameters
        ----------
        path : str or os.PathLike
            Path to a file previously created by :meth:`save_index` or the
            low-level :meth:`Annoy.save`.
        prefault : bool or None, default=None
            If None, defer to the low-level object's configured prefault
            behavior.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        OSError
            If the file cannot be opened or mapped.
        RuntimeError
            If the index is incompatible with the current object.

        Notes
        -----
        Annoy requires that the in-memory object is compatible with the file
        being loaded (dimension and metric).
        """
        ll = self._low_level()
        p = os.fspath(path)
        fn = getattr(ll, "load", None)
        if fn is None:
            raise AttributeError("Low-level load() is not available")
        if prefault is None:
            fn(p)
        else:
            fn(p, prefault=prefault)

    def save_bundle(
        self,
        directory: str | PathLike[str],
        *,
        index_filename: str = "index.ann",
        manifest_filename: str = "manifest.json",
        prefault: bool | None = None,
    ) -> None:
        """
        Save a directory bundle containing both manifest and index file.

        This is a convenience for packaging a self-describing Annoy index
        artifact.

        Parameters
        ----------
        directory : str or os.PathLike
            Destination directory. Created if it does not exist.
        index_filename : str, default="index.ann"
            Filename for the Annoy index file inside ``directory``.
        manifest_filename : str, default="manifest.json"
            Filename for the manifest JSON inside ``directory``.
        prefault : bool or None, default=None
            Passed through to :meth:`save_index`.

        Raises
        ------
        AttributeError
            If the object does not provide :meth:`to_json` (from
            :class:`~scikitplot.annoy._mixins._manifest.ManifestMixin`).
        OSError
            If files cannot be written.

        Notes
        -----
        The manifest is written using an atomic replace (write temp + rename)
        to avoid partial writes.
        """
        # `ManifestMixin` provides `to_json()`; keep the dependency explicit.
        to_json = getattr(self, "to_json", None)
        if to_json is None:
            raise AttributeError(
                "save_bundle requires `to_json()` (compose with ManifestMixin)."
            )

        dir_s = os.fspath(directory)
        os.makedirs(dir_s, exist_ok=True)

        manifest_path = os.path.join(dir_s, manifest_filename)
        index_path = os.path.join(dir_s, index_filename)

        # 1) Write manifest atomically.
        payload: str = to_json(None)  # type: ignore[misc]
        tmp_manifest = manifest_path + ".tmp"
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_manifest, manifest_path)

        # 2) Write index atomically (best-effort).
        tmp_index = index_path + ".tmp"
        self.save_index(tmp_index, prefault=prefault)
        os.replace(tmp_index, index_path)

    @classmethod
    def load_bundle(
        cls,
        directory: str | PathLike[str],
        *,
        index_filename: str = "index.ann",
        manifest_filename: str = "manifest.json",
        prefault: bool | None = None,
    ) -> Self:
        """
        Load a directory bundle produced by :meth:`save_bundle`.

        Parameters
        ----------
        directory : str or os.PathLike
            Bundle directory containing ``manifest_filename`` and
            ``index_filename``.
        index_filename : str, default="index.ann"
            Index filename inside ``directory``.
        manifest_filename : str, default="manifest.json"
            Manifest filename inside ``directory``.
        prefault : bool or None, default=None
            Passed through to :meth:`load_index`.

        Returns
        -------
        obj : Self
            A newly constructed object with the index loaded.

        Raises
        ------
        AttributeError
            If the class does not provide :meth:`from_json` (from
            :class:`~scikitplot.annoy._mixins._manifest.ManifestMixin`).
        FileNotFoundError
            If either file is missing.
        OSError
            If files cannot be read or mapped.
        RuntimeError
            If the index is incompatible.

        See Also
        --------
        save_bundle
        """
        from_json = getattr(cls, "from_json", None)
        if from_json is None:
            raise AttributeError(
                "load_bundle requires `from_json()` (compose with ManifestMixin)."
            )

        dir_s = os.fspath(directory)
        manifest_path = os.path.join(dir_s, manifest_filename)
        index_path = os.path.join(dir_s, index_filename)

        obj = from_json(manifest_path, load=False)  # type: ignore[misc]
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected {cls.__name__!r} from {manifest_path!r}, got {type(obj).__name__!r}."
            )

        obj.load_index(index_path, prefault=prefault)
        return obj


class PickleIOMixin:
    """
    Mixin adding explicit Python-object to pickling dump/load helpers.

    Notes
    -----
    This mixin serializes the *Python object* using a pickle-family backend.
    It should not be used as a substitute for Annoy's native index
    persistence. Prefer composing with :class:`~scikitplot.annoy._mixins._pickle.PickleMixin`
    if you need Annoy-aware, deterministic object serialization.
    """  # noqa: D205, D415

    # Declared for static type checkers; real value is the concrete subclass.
    _io_expected_type: ClassVar[type[Self]]

    # This mixin supports both inheritance-style (Index subclasses Annoy)
    # and composition-style (Index wraps a low-level Annoy instance).
    #
    # Concrete classes may optionally provide ``self._annoy``; if absent,
    # the low-level object is assumed to be ``self``.

    def _low_level(self) -> Any:
        """
        Return the low-level Annoy object.

        Preference order is explicit and deterministic:

        1) ``self._annoy`` when present (composition style)
        2) ``self`` (inheritance style)
        """
        ll = getattr(self, "_annoy", None)
        return ll if ll is not None else self

    @classmethod
    def load_pickle(
        cls,
        path: str | PathLike[str],
        *,
        backend: SerializerBackend = "joblib",
        **kwargs: Any,
    ) -> Self:
        """
        Load an object from a binary file.

        Parameters
        ----------
        path : str or os.PathLike
            File path to read.
        backend : {'pickle', 'cloudpickle', 'joblib'}, default='joblib'
            Backend used for deserialization.
        **kwargs
            Extra keyword arguments forwarded to the backend loader.

            * ``joblib``: forwarded to :func:`joblib.load`
            * ``pickle``: currently unused (kept for forwards compatibility)
            * ``cloudpickle``: currently unused (kept for forwards compatibility)

        Returns
        -------
        obj : Self
            Deserialized object. Ensures the returned instance is of type
            ``cls``.

        Raises
        ------
        ValueError
            If ``backend`` is unsupported.
        TypeError
            If the deserialized object is not an instance of ``cls``.
        FileNotFoundError
            If the file does not exist.

        Notes
        -----
        All supported backends can execute arbitrary code during loading.
        Do not load untrusted data.
        """
        path_s = os.fspath(path)  # str(path)

        if backend == "pickle":
            import pickle  # noqa: PLC0415, S403

            with open(path_s, "rb") as f:
                obj = pickle.load(f)  # noqa: S301

        elif backend == "cloudpickle":
            import cloudpickle  # noqa: PLC0415, S403

            with open(path_s, "rb") as f:
                obj = cloudpickle.load(f)  # noqa: S301

        elif backend == "joblib":
            import joblib  # noqa: PLC0415

            obj = joblib.load(path_s, **kwargs)

        else:
            raise ValueError(
                f"Unsupported backend: {backend!r}. Use 'pickle', 'cloudpickle', or 'joblib'."
            )

        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected an instance of {cls.__name__!r} from {path_s!r}, got {type(obj).__name__!r}."
            )

        return obj

    def dump_pickle(
        self,
        path: str | PathLike[str],
        *,
        backend: SerializerBackend = "joblib",
        **kwargs: Any,
    ) -> None:
        """
        Serialize this object to a binary file.

        Parameters
        ----------
        path : str or os.PathLike
            Destination path.
        backend : {'pickle', 'cloudpickle', 'joblib'}, default='joblib'
            Backend used for serialization.
        **kwargs
            Extra keyword arguments forwarded to the backend dumper.

            * ``joblib``: forwarded to :func:`joblib.dump` (e.g. ``compress=3``)
            * ``pickle``: currently unused (kept for forwards compatibility)
            * ``cloudpickle``: currently unused (kept for forwards compatibility)
            * ``pickle`` / ``cloudpickle``:
              accepts ``protocol=int`` to control the pickle protocol.

        Raises
        ------
        ValueError
            If ``backend`` is unsupported.
        OSError
            If the file cannot be written.
        """
        path_s = os.fspath(path)  # str(path)

        if backend == "pickle":
            import pickle  # noqa: PLC0415, S403

            protocol = int(kwargs.pop("protocol", pickle.HIGHEST_PROTOCOL))
            with open(path_s, "wb") as f:
                pickle.dump(self, f, protocol=protocol)  # noqa: S301

        elif backend == "cloudpickle":
            import pickle  # noqa: PLC0415, S403

            import cloudpickle  # noqa: PLC0415, S403

            protocol = int(kwargs.pop("protocol", pickle.HIGHEST_PROTOCOL))
            with open(path_s, "wb") as f:
                cloudpickle.dump(self, f, protocol=protocol)  # noqa: S301

        elif backend == "joblib":
            import joblib  # noqa: PLC0415

            joblib.dump(self, path_s, **kwargs)

        else:
            raise ValueError(
                f"Unsupported backend: {backend!r}. Use 'pickle', 'cloudpickle', or 'joblib'."
            )
