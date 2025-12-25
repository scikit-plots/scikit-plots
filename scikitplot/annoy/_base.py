# scikitplot/annoy/_base.py
"""
High-level public :class:`~scikitplot.annoy.Index`.

This module defines :class:`~scikitplot.annoy.Index` as a *direct subclass* of the
low-level ANNoy backend (:class:`~scikitplot.cexternals._annoy.Annoy`) and composes
higher-level behavior via mixins.

- The low-level Annoy backend is a C-extension type that implements the core
  operations (add/build/query/serialize/deserialize/save/load).
- The high-level :class:`~scikitplot.annoy.Index` keeps a stable Pythonic API by
  layering mixins on top of that backend (manifest, I/O, pickling, ndarray export,
  plotting helpers, etc.).

Notes
-----
This module avoids implicit side effects and keeps state/configuration explicit.
In particular:

- :class:`~scikitplot.annoy.Index` does **not** override the backend constructor.
  Initialization is provided by the C-extension type (:class:`~scikitplot.cexternals._annoy.Annoy`).
- Mixins used here are expected to be initialization-free (no required
  ``__init__``) or to provide safe defaults.

See Also
--------
scikitplot.cexternals._annoy.Annoy
    Low-level ANNoy backend.
scikitplot.annoy._mixins._manifest.ManifestMixin
    Versioned manifest helpers.
scikitplot.annoy._mixins._io.IndexIOMixin
    Annoy-native index persistence helpers.
scikitplot.annoy._mixins._io.PickleIOMixin
    High-level pickle helpers.
scikitplot.annoy._mixins._pickle.PickleMixin
    Pickle protocol for the high-level index (versioned state).
"""

#   .. seealso::
#     * :py:obj:`~scikitplot.annoy.Index.from_low_level`
#     * https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled

from __future__ import annotations

# import uuid  # f"annoy-{uuid.uuid4().hex}.annoy"
from typing_extensions import Self  # noqa: F401

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
    High-level ANNoy index composed from mixins.

    Parameters
    ----------
    f : int, default=0
        Vector dimensionality passed to the backend. If ``0``, the backend may
        infer dimensionality lazily from the first added vector.
    metric : str or None, default=None
        Metric name passed to the backend. Common values include
        ``"angular"``, ``"euclidean"``, ``"manhattan"``, ``"dot"``, and
        ``"hamming"`` (synonyms may also be accepted by the backend).
        If None, defaults to the backend default (typically ``"angular"``).
    on_disk_path : str or os.PathLike or None, default=None
        Optional backing path used by on-disk build and mmap load (backend-defined).
    prefault : bool, default=False
        Annoy prefault behavior (passed to low-level setter).
    schema_version : int or None, default=None
        Optional schema marker, when supported by the backend.
    seed : int or None, default=None
        Optional seed controlling backend randomness, when supported.
    verbose : int or None, default=None
        Optional verbosity level, when supported.

    Attributes
    ----------
    pickle_mode : PickleMode
        Pickle strategy used by :class:`~scikitplot.annoy._mixins._pickle.PickleMixin`.
    compress_mode : CompressMode or None
        Optional compression used by :class:`~scikitplot.annoy._mixins._pickle.PickleMixin`
        when serializing to bytes.

    Notes
    -----
    This class initializes the C-extension backend explicitly (``Annoy.__init__``)
    and then initializes mixin configuration explicitly. It does **not** rely on
    cooperative ``super().__init__`` across mixins.

    See Also
    --------
    scikitplot.cexternals._annoy.Annoy
    Index.from_low_level
    """

    # __slots__ = ()

    # ------------------------------------------------------------------
    # Low-level access
    # This wrapper supports both inheritance (self is the backend) and
    # composition (self._annoy holds the backend).

    # This mixin supports both inheritance-style (Index subclasses Annoy)
    # and composition-style (Index wraps a low-level Annoy instance).
    #
    # Concrete classes may optionally provide ``self._annoy``; if absent,
    # the low-level object is assumed to be ``self``.

    def _low_level(self) -> Annoy:
        """
        Return the low-level Annoy object.

        Preference order is explicit and deterministic:

        1) ``object._annoy`` when present (composition style)
        2) ``self`` (inheritance style)

        Notes
        -----
        This helper uses :func:`object.__getattribute__` to avoid triggering custom
        ``__getattr__`` / ``__getattribute__`` side effects during low-level access.
        """
        # ll = getattr(self, "_annoy", None)
        # return ll if ll is not None else obj
        try:
            return object.__getattribute__(self, "_annoy")
        except AttributeError:
            return self

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

    @annoy.setter
    def annoy(self, annoy: Annoy) -> None:
        self._annoy = annoy

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

        if obj.on_disk_path:
            inst.on_disk_path = str(obj.on_disk_path)

        inst.deserialize(
            payload,
            prefault=prefault if prefault is not None else inst.prefault,
        )
        return inst
