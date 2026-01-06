# scikitplot/annoy/_base.py
"""
High-level public :class:`~scikitplot.annoy` index.

This module defines :class:`~scikitplot.annoy.Index` as a *thin, explicit* Python
facade over the low-level Annoy backend implemented in the C-extension
(:class:`~scikitplot.cexternals._annoy.Annoy`).

Notes
-----
- **Deterministic**: no implicit guessing; conversions and fallbacks are explicit.
- **Stable invariants**: low-level semantics live in the C-extension; this layer
  focuses on ergonomics, documentation, and composition.
- **No cooperative init**: mixins must not depend on ``super().__init__`` being
  called across the MRO (the backend is a C type).
- Support inheritance-based and composition-based designs (mixin independence).

The mixins imported here provide higher-level capabilities (I/O, numpy
integration, pickling, plotting, metadata). This module is intentionally small:
it only contains glue needed to compose the public class.

See Also
--------
scikitplot.cexternals._annoy.Annoy
    Low-level backend (C-extension) providing the core ANN index operations.
"""

#   .. seealso::
#     * :py:meth:`~scikitplot.annoy.Index.from_low_level`
#     * https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled

from __future__ import annotations

import threading

# import uuid  # f"annoy-{uuid.uuid4().hex}.annoy"
from typing_extensions import Self

from ..cexternals._annoy import Annoy
from ._mixins._io import IndexIOMixin
from ._mixins._meta import MetaMixin
from ._mixins._ndarray import NDArrayMixin
from ._mixins._pickle import PickleMixin
from ._mixins._plotting import PlottingMixin
from ._mixins._vectors import VectorOpsMixin
from ._utils import FALLBACK_LOCK, lock_for

__all__ = [
    "Index",
]


# Type used for isinstance checks of threading.RLock() results.
_RLOCK_TYPE = type(FALLBACK_LOCK)

# ------------------------------------------------------------------
# Low-level access
# This wrapper supports both inheritance (self is the backend) and
# composition (self._annoy holds the backend).
#
# This mixin supports both inheritance-style (Index subclasses Annoy)
# and composition-style (Index wraps a low-level Annoy instance).
#
# Concrete classes may optionally provide ``self._annoy``; if absent,
# the low-level object is assumed to be ``self``.


class Index(
    Annoy,  # _base.py contract + invariants
    MetaMixin,  # metadata + sklearn integration helpers
    IndexIOMixin,  # save/load/bytes/bundles
    PickleMixin,  # pickle behavior (depends optionally on Meta/IO)
    VectorOpsMixin,  # query helpers
    NDArrayMixin,  # numpy/scipy/pandas adapters
    PlottingMixin,  # optional UI last
):
    """
    High-level ANNoy index composed from mixins.

    Parameters
    ----------
    f : int or None, optional, default=None
        Vector dimension. If ``0`` or ``None``, dimension may be inferred from the
        first vector passed to ``add_item`` (lazy mode).
        If None, treated as ``0`` (reset to default).
    metric : {"angular", "cosine", "euclidean", "l2", "lstsq", "manhattan", "l1", "cityblock", "taxicab", \
            "dot", "@", ".", "dotproduct", "inner", "innerproduct", "hamming"} or None, optional, default=None
        Distance metric (one of 'angular', 'euclidean', 'manhattan', 'dot', 'hamming').
        If omitted and ``f > 0``, defaults to ``'angular'`` (cosine-like).
        If omitted and ``f == 0``, metric may be set later before construction.
        If None, behavior depends on ``f``:

        * If ``f > 0``: defaults to ``'angular'`` (legacy behavior; may emit a
        :class:`FutureWarning`).
        * If ``f == 0``: leaves the metric unset (lazy). You may set
        :attr:`metric` later before construction, or it will default to
        ``'angular'`` on first :meth:`add_item`.
    n_neighbors : int, default=5
        Non-negative integer Number of neighbors to retrieve for each query.
    on_disk_path : str or None, optional, default=None
        If provided, configures the path for on-disk building. When the underlying
        index exists, this enables on-disk build mode (equivalent to calling
        :meth:`on_disk_build` with the same filename).

        Note: Annoy core truncates the target file when enabling on-disk build.
        This wrapper treats ``on_disk_path`` as strictly equivalent to calling
        :meth:`on_disk_build` with the same filename (truncate allowed).

        In lazy mode (``f==0`` and/or ``metric is None``), activation occurs once
        the underlying C++ index is created.
    prefault : bool or None, optional, default=None
        If True, request page-faulting index pages into memory when loading
        (when supported by the underlying platform/backing).
        If None, treated as ``False`` (reset to default).
    seed : int or None, optional, default=None
        Non-negative integer seed. If set before the index is constructed,
        the seed is stored and applied when the C++ index is created.
        Seed value ``0`` is treated as \"use Annoy's deterministic default seed\"
        (a :class:`UserWarning` is emitted when ``0`` is explicitly provided).
    verbose : int or None, optional, default=None
        Verbosity level. Values are clamped to the range ``[-2, 2]``.
        ``level >= 1`` enables Annoy's verbose logging; ``level <= 0`` disables it.
        Logging level inspired by gradient-boosting libraries:

        * ``<= 0`` : quiet (warnings only)
        * ``1``    : info (Annoy's ``verbose=True``)
        * ``>= 2`` : debug (currently same as info, reserved for future use)
    schema_version : int, optional, default=None
        Serialization/compatibility strategy marker.

        This does not change the Annoy on-disk format, but it *does* control
        how the index is snapshotted in pickles.

        * ``0`` or ``1``: pickle stores a ``portable-v1`` snapshot (fast restore,
        ABI-checked).
        * ``2``: pickle stores ``canonical-v1`` (portable across ABIs; restores by
        rebuilding deterministically).
        * ``>=3``: pickle stores both portable and canonical (canonical is used as
        a fallback if the ABI check fails).

        If None, treated as ``0`` (reset to default).

    Attributes
    ----------
    f : int, default=0
        Vector dimension. ``0`` means "unknown / lazy".
    metric : {'angular', 'euclidean', 'manhattan', 'dot', 'hamming'}, default="angular"
        Canonical metric name, or None if not configured yet (lazy).
    n_neighbors : int, default=5
        Non-negative integer Number of neighbors to retrieve for each query.
    on_disk_path : str or None, optional, default=None
        Configured on-disk build path. Setting this attribute enables on-disk
        build mode (equivalent to :meth:`on_disk_build`), with safety checks
        to avoid implicit truncation of existing files.
    seed, random_state : int or None, optional, default=None
        Non-negative integer seed.
    verbose : int or None, optional, default=None
        Verbosity level.
    prefault : bool, default=False
        Stored prefault flag (see :meth:`load`/`:meth:`save` prefault parameters).
    schema_version : int, default=0
        Reserved schema/version marker (stored; does not affect on-disk format).
    n_features, n_features_, n_features_in_ : int
        Alias of `f` (dimension), provided for scikit-learn naming parity.
    n_features_out_ : int
        Number of output features produced by transform.
    feature_names_in_ : list-like
        Input feature names seen during fit.
        Set only when explicitly provided via fit(..., feature_names=...).
    y : dict | None, optional, default=None
        If provided to fit(X, y), labels are stored here after a successful build.
        You may also set this property manually. When possible, the setter enforces
        that len(y) matches the current number of items (n_items).
    pickle_mode : PickleMode
        Pickle strategy used by :class:`~scikitplot.annoy._mixins._pickle.PickleMixin`.
    compress_mode : CompressMode or None
        Optional compression used by :class:`~scikitplot.annoy._mixins._pickle.PickleMixin`
        when serializing to bytes.

    Notes
    -----
    This class is a direct subclass of the C-extension backend. It does not
    override ``__new__`` and does not rely on cooperative initialization across
    mixins. Mixins must be written so that their methods work even if they
    define no ``__init__`` at all.

    See Also
    --------
    scikitplot.cexternals._annoy.Annoy
    Index.from_low_level
    """  # noqa: D301

    # __slots__ = ()
    # Re-entrant lock for deterministic, thread-safe reduce/rebuild paths.
    _lock: threading.RLock | None = None

    def _get_lock(self) -> threading.RLock:
        """
        Return a per-instance re-entrant lock.

        The lock is created lazily to avoid changing construction semantics of
        the C-extension backend. If the instance cannot store attributes, a
        module-level fallback lock is returned.

        Returns
        -------
        lock : threading.RLock
            Re-entrant lock guarding non-atomic multi-step operations.
        """
        # Prefer an instance lock when possible; fall back to a shared lock
        # when the C-extension type does not support dynamic attributes.
        try:
            lock = object.__getattribute__(self, "_lock")
        except Exception:
            lock = None
        if isinstance(lock, _RLOCK_TYPE):
            return lock

        new_lock = threading.RLock()
        try:
            object.__setattr__(self, "_lock", new_lock)
        except Exception:
            return FALLBACK_LOCK
        return new_lock

    def _backend(self) -> Annoy:
        """
        Return the low-level backend instance.

        This is a deterministic accessor that enables both inheritance and
        composition styles:

        1) If ``self._annoy`` exists, it is returned (composition).
        2) Otherwise, ``self`` is returned (inheritance).

        The implementation uses :func:`object.__getattribute__` to avoid
        triggering custom attribute lookup side effects.

        Returns
        -------
        backend : scikitplot.cexternals._annoy.Annoy
            Low-level Annoy backend.
        """
        try:
            return object.__getattribute__(self, "_annoy")
        except AttributeError:
            return self

    @property
    def backend(self) -> Annoy:
        """
        Public alias for :py:meth:`~scikitplot.annoy.Index._backend`.

        Returns
        -------
        backend : scikitplot.cexternals._annoy.Annoy
            Low-level Annoy backend instance.
        """
        return self._backend()

    @classmethod
    def from_low_level(cls, obj: Annoy, *, prefault: bool | None = None) -> Self:
        """
        Create a new :class:`~scikitplot.annoy.Index` from a low-level instance.

        The new object is rebuilt by round-tripping through Annoy's native
        ``serialize`` / ``deserialize`` to avoid sharing low-level state between
        two Python objects.

        Parameters
        ----------
        obj : scikitplot.cexternals._annoy.Annoy
            Low-level Annoy instance.
        prefault : bool or None, default=None
            Prefault override passed to :py:meth:`~.Annoy.deserialize`. If None, the
            value is taken from ``obj.get_params(deep=False)`` when available,
            otherwise it falls back to ``obj.prefault`` / destination defaults.

        Returns
        -------
        index : Index
            Newly constructed high-level index.

        Raises
        ------
        TypeError
            If ``obj`` is not an Annoy instance.
        RuntimeError
            If serialization or deserialization fails, or required configuration
            (e.g., ``f``) cannot be determined.

        Notes
        -----
        The implementation uses Annoy's native serialization. It does not attempt
        to copy internal pointers or C++ state directly.

        This method is deterministic. It always constructs a new index from the
        serialized payload; it does not share low-level state between objects.

        See Also
        --------
        Annoy.serialize
        Annoy.deserialize
        Annoy.get_params
        Annoy.set_params
        """
        if not isinstance(obj, Annoy):
            raise TypeError(f"obj must be an Annoy instance, got {type(obj)!r}")

        # Serialize + capture stable params under a lock to avoid torn reads if the
        # same low-level object is accessed concurrently.
        lock = lock_for(obj)
        with lock:
            payload = obj.serialize()
            if not isinstance(payload, (bytes, bytearray, memoryview)):
                raise RuntimeError("Annoy.serialize() must return a bytes-like object.")
            payload = bytes(payload)
            # Copy stable configuration (sklearn-style) before restoring the payload.
            try:
                params = dict(obj.get_params(deep=False))
            except Exception:
                params = {}

        # lazy default None
        f_raw = getattr(obj, "f", None)
        if f_raw is None:
            f_raw = params.get("f", None)  # noqa: SIM910
        if f_raw is None:
            raise RuntimeError("Cannot determine dimension 'f' from low-level object.")
        f = int(f_raw)
        metric = params.get("metric", getattr(obj, "metric", None))
        metric_s = str(metric) if metric is not None else None

        inst = cls(f) if metric_s is None else cls(f, metric_s)

        # Copy remaining params (strict in backend; unknown keys will raise).
        if params:
            rest = {k: v for k, v in params.items() if k not in {"f", "metric"}}
            if rest:
                inst.set_params(**rest)

        # Preserve explicit prefault override, if requested.
        if prefault is None:
            prefault_raw = params.get("prefault", getattr(obj, "prefault", None))
            if prefault_raw is None:
                prefault_raw = getattr(inst, "prefault", False)
            prefault = bool(prefault_raw)

        inst.deserialize(payload, prefault=bool(prefault))
        return inst
