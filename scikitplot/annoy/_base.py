# scikitplot/annoy/_base.py
"""
High-level public :class:`~scikitplot.annoy.Index`.

This module defines :class:`~scikitplot.annoy.Index` as a *direct subclass* of the
low-level Annoy backend (:class:`scikitplot.cexternals._annoy.Annoy`) and composes
higher-level behavior via mixins.

- The low-level Annoy backend is a C-extension type that implements the core
  operations (add/build/query/serialize/deserialize/save/load).
- The high-level :class:`~scikitplot.annoy.Index` keeps a stable Pythonic API by
  layering mixins on top of that backend (manifest, I/O, pickling, ndarray export,
  plotting helpers, etc.).
- Determinism: this class avoids implicit side effects and makes state/config
  explicit.

Notes
-----
Subclassing C-extension types can be subtle. The Annoy backend already provides
an instance attribute dictionary (``__dict__``) at the C level. Defining a pure
Python subclass *without* ``__slots__`` may cause CPython to create a second,
managed dict slot for the subclass, which can break introspection helpers like
``dir(obj)`` with errors such as::

    TypeError: this __dict__ descriptor does not support 'Index' objects

To keep behavior robust across CPython versions, :class:`~scikitplot.annoy.Index`
declares ``__slots__ = ()`` so the subclass does not add a second dict slot.

See Also
--------
scikitplot.cexternals._annoy.Annoy
    Low-level C-extension backend.
scikitplot.annoy._mixins._manifest.ManifestMixin
    Versioned manifest helpers.
scikitplot.annoy._mixins._io.IndexIOMixin
    Annoy-native index persistence helpers.
scikitplot.annoy._mixins._io.ObjectIOMixin
    Python-object serialization helpers (pickle/cloudpickle/joblib).
scikitplot.annoy._mixins._pickle.PickleMixin
    Pickle protocol for the high-level index (versioned state).
"""
#   .. seealso::
#     * :py:obj:`~scikitplot.annoy.Index.from_low_level`
#     * https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled

from __future__ import annotations

# import uuid  # f"annoy-{uuid.uuid4().hex}.annoy"
from typing import Any

from typing_extensions import Self

from ..cexternals._annoy import Annoy, AnnoyIndex  # noqa: F401
from ._mixins._io import IndexIOMixin, PickleIOMixin
from ._mixins._manifest import ManifestMixin
from ._mixins._ndarray import NDArrayExportMixin
from ._mixins._pickle import CompressMode, PickleMixin, PickleMode  # noqa: F401
from ._mixins._plotting import PlottingMixin
from ._mixins._vectors import VectorOpsMixin

__all__ = ["Index"]


class Index(
    Annoy,
    ManifestMixin,
    IndexIOMixin,
    PickleIOMixin,
    PickleMixin,
    VectorOpsMixin,
    NDArrayExportMixin,
    PlottingMixin,
):
    """
    High-level Annoy index composed from mixins.

    Parameters
    ----------
    f : int, default=0
        Vector dimension. If ``0``, Annoy can infer dimensionality lazily from
        the first added vector.
    metric : str or None, default=None
        Metric name (e.g. ``"angular"``, ``"euclidean"``, ``"manhattan"``,
        ``"dot"``, ``"hamming"``).
        If None, defaults to ``"angular"``.
    on_disk_path : str or os.PathLike or None, default=None
        Optional backing path used by Annoy's on-disk build and mmap load.
    prefault : bool, default=False
        Annoy prefault behavior (passed to low-level setter).
    schema_version : int or None, default=None
        Optional low-level schema marker (exposed by Annoy as ``schema_version``).
    seed : int or None, default=None
        Optional low-level seed.
    verbose : int or None, default=None
        Optional low-level verbosity level.
    pickle_mode : {"auto", "disk", "byte"}, default="auto"
        Pickle strategy used by :class:`~scikitplot.annoy._mixins._pickle.PickleMixin`.
    compress_mode : {"zlib", "gzip"} or None, default=None
        Optional compression used for ``pickle_mode="byte"``.

    Notes
    -----
    This class initializes the C-extension backend explicitly (``Annoy.__init__``)
    and then initializes mixin configuration explicitly. It does **not** rely on
    cooperative ``super().__init__`` across mixins.

    See Also
    --------
    Annoy
        Low-level backend type (this class subclasses it).
    ManifestMixin.to_manifest
        Construct a versioned manifest for reconstruction.
    IndexIOMixin.save_index
        Persist the index using Annoy's native format.
    PickleMixin.__getstate__
        Pickle contract for the high-level index.
    """

    # __slots__ = ()

    # # ------------------------------------------------------------------
    # # Robust introspection helpers
    # def __dir__(self) -> list[str]:
    #     """Return a robust attribute list.

    #     This overrides :meth:`object.__dir__` to avoid failures when CPython tries
    #     to access an incompatible ``__dict__`` descriptor for C-extension subtypes.

    #     Returns
    #     -------
    #     names : list of str
    #         Sorted attribute names.

    #     Notes
    #     -----
    #     This method is intentionally defensive and side-effect free.

    #     See Also
    #     --------
    #     object.__dir__
    #     """
    #     try:
    #         return sorted(set(object.__dir__(self)))
    #     except TypeError:
    #         names: set[str] = set(dir(type(self)))
    #         try:
    #             d = object.__getattribute__(self, "__dict__")
    #         except Exception:
    #             d = None
    #         if isinstance(d, dict):
    #             names.update(d.keys())
    #         return sorted(names)

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

    # ------------------------------------------------------------------
    # Low-level accessors
    @property
    def annoy(self) -> Annoy:
        """
        Return the low-level backend object.

        This high-level wrapper supports both:

        - **Inheritance style**: :class:`~scikitplot.annoy.Index` subclasses the
          C-extension backend, so the low-level object is ``self``.
        - **Composition style**: a concrete class may expose a private
          ``self._annoy`` attribute that holds the low-level backend instance.

        Returns
        -------
        backend : Annoy
            Low-level Annoy backend instance.

        Notes
        -----
        This accessor is deterministic and side-effect free.

        See Also
        --------
        Index._low_level
        """
        return self._low_level()

    @classmethod
    def from_low_level(cls, obj: Annoy, *, prefault: bool | None = None) -> Self:
        """
        Create a new :class:`~scikitplot.annoy.Index` from a low-level instance.

        This method round-trips through ``serialize``/``deserialize`` to avoid
        sharing internal low-level state between two Python objects.

        Parameters
        ----------
        obj : Annoy
            Low-level Annoy instance.
        prefault : bool or None, default=None
            Prefault override passed to :meth:`~.Annoy.deserialize`. If None, uses
            the destination object's configured prefault.

        Returns
        -------
        index : Index
            A new high-level index instance with the same contents as ``obj``.

        Raises
        ------
        RuntimeError
            If deserialization fails.

        Notes
        -----
        The implementation uses Annoy's native serialization. It does not attempt
        to copy internal pointers or C++ state directly.

        See Also
        --------
        Annoy.serialize
        Annoy.deserialize
        """
        payload = obj.serialize()
        inst = cls(obj.f, str(obj.metric))
        inst.deserialize(
            payload,
            prefault=prefault if prefault is not None else inst.prefault,
        )
        if obj.on_disk_path:
            inst.on_disk_path = str(obj.on_disk_path)

        return inst
