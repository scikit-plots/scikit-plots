# scikitplot/annoy/_mixins/_pickle.py
"""
Annoy-aware pickling helpers for high-level wrappers.

Annoy has two persistence layers that must not be conflated:

1. **Annoy-native index persistence** writes/loads the *index structure* (forest)
   via the low-level C-extension (``save`` / ``load`` / ``serialize`` /
   ``deserialize``).

2. **Python object persistence** (pickling) serializes a *Python wrapper object*.

This module implements an **Annoy-aware pickle contract** for high-level
wrappers: it stores enough deterministic state to reconstruct the low-level
index via either:

- ``pickle_mode="disk"``: reload from an on-disk Annoy file.
- ``pickle_mode="byte"``: embed the serialized Annoy bytes into the pickle.

The ``"auto"`` mode is **rule-based and deterministic**: it selects
``"disk"`` when a backing on-disk path is known; otherwise it selects
``"byte"``.

This layer intentionally does **not** alter low-level semantics; it only
provides predictable Python glue.

See Also
--------
scikitplot.annoy._mixins._io.IndexIOMixin
    Explicit Annoy-native file I/O helpers (``save_index`` / ``load_index``).
scikitplot.annoy._mixins._manifest.ManifestMixin
    Versioned manifest export/import helpers (``to_manifest`` / ``from_manifest``).
scikitplot.annoy._mixins._io.PickleIOMixin
    Generic pickle-family dump/load helpers for Python objects.

Notes
-----
- Pickling a very large index in ``"byte"`` mode can create huge pickle payloads.
  Prefer ``"disk"`` mode if you can manage an Annoy index file alongside the
  pickle, or use explicit bundle I/O (manifest + index file).
- This mixin supports both common wrapper styles:
  *composition* (wrapping a low-level Annoy instance on ``self._annoy``) and
  *inheritance* (subclassing the low-level Annoy type). Helper methods define
  an explicit preference order.
"""

from __future__ import annotations

import gzip
import os
import threading
import zlib
from collections.abc import Mapping
from dataclasses import dataclass  # noqa: F401
from enum import Enum  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any, ClassVar, Literal, TypeAlias, cast

from typing_extensions import Self

# _T = TypeVar("_T")  # noqa: PYI018

CompressMode: TypeAlias = Literal["zlib", "gzip"] | None
"""
Compression used for ``"byte"`` pickling.

Compression is applied to the serialized Annoy bytes produced by the low-level
:meth:`serialize` API.
"""

PickleMode: TypeAlias = Literal["auto", "disk", "byte"]
"""
Persistence strategy used by :class:`~.PickleMixin`.

``"auto"`` is deterministic and selects:

- ``"disk"`` when a backing path is known (``on_disk_path`` or legacy fallback)
- otherwise ``"byte"``
"""

__all__ = [
    "CompressMode",
    "PickleMixin",
    "PickleMode",
]


def _get_noncallable_attr(obj: Any, name: str) -> Any:
    """Return attribute value only if it is present and non-callable."""
    if obj is None or not hasattr(obj, name):
        return None
    value = getattr(obj, name, None)
    return None if callable(value) else value


def _get_callable(obj: Any, name: str) -> Any:
    """Return attribute only if it is present and callable."""
    if obj is None:
        return None
    value = getattr(obj, name, None)
    return value if callable(value) else None


class PickleMixin:
    """
    Mixin implementing an Annoy-aware pickle contract.

    Parameters
    ----------
    f : int, default=0
        Vector dimension. Provided for compatibility with inheritance-style
        wrappers. Composition-style wrappers may ignore this parameter and
        configure the low-level object elsewhere.
    metric : str, default="angular"
        Distance metric. Provided for compatibility with inheritance-style
        wrappers.
    prefault : bool, default=False
        Default prefault behavior used during reconstruction.
    pickle_mode : {"auto", "disk", "byte"}, default="auto"
        Pickle strategy. See :data:`PickleMode`.
    compress_mode : {None, "zlib", "gzip"}, default=None
        Optional compression for ``"byte"`` mode.

    Notes
    -----
    - ``"byte"`` mode requires a built index (``get_n_trees() > 0``) because it
      embeds ``serialize()`` bytes in the pickle.
    - ``"disk"`` mode requires a known backing path (``on_disk_path``).
    """  # noqa: D205, D415

    # Re-entrant lock for deterministic, thread-safe reduce/rebuild paths.
    _lock: threading.RLock = threading.RLock()

    _compress_mode: CompressMode = None
    _pickle_mode: PickleMode = "auto"

    # Version for the pickle state dictionary produced by __reduce__.
    _PICKLE_STATE_VERSION: ClassVar[int] = 0

    # This mixin supports both inheritance-style (Index subclasses Annoy)
    # and composition-style (Index wraps a low-level Annoy instance).
    #
    # Concrete classes may optionally provide ``self._annoy``; if absent,
    # the low-level object is assumed to be ``self``.

    # ------------------------------------------------------------------
    # Low-level selection
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
    # Configuration properties
    @property
    def pickle_mode(self) -> PickleMode:
        return self._pickle_mode

    @pickle_mode.setter
    def pickle_mode(self, value: PickleMode) -> None:
        if value not in ("auto", "disk", "byte"):
            raise ValueError("pickle_mode must be one of: 'auto', 'disk', 'byte'")
        self._pickle_mode = value

    @property
    def compress_mode(self) -> CompressMode:
        return self._compress_mode

    @compress_mode.setter
    def compress_mode(self, value: CompressMode) -> None:
        if value not in (None, "zlib", "gzip"):
            raise ValueError("compress_mode must be one of: None, 'zlib', 'gzip'")
        self._compress_mode = value

    # ------------------------------------------------------------------
    # Low-level helpers
    def _ll_info(self) -> Mapping[str, Any] | None:
        """Return low-level ``info()`` mapping when available."""
        ll = self._low_level()
        info = getattr(ll, "info", None)
        if info is None:
            return None
        try:
            out = info()
        except Exception:
            return None
        return cast(Mapping[str, Any], out)

    def _config_f_metric(self) -> tuple[int, str]:
        """Return (f, metric) for reconstruction in a deterministic way."""
        info = self._ll_info()
        if info is not None:
            try:
                f = int(info.get("f", 0))
                metric = str(info.get("metric", ""))
                if f > 0 and metric:
                    return f, metric
            except Exception:
                pass

        f = int(getattr(self, "f", 0))
        metric = str(getattr(self, "metric", ""))
        return f, metric

    def _ll_on_disk_path(self) -> str | None:
        """Return the configured on-disk path if available."""
        ll = self._low_level()

        # Preferred: low-level path configuration (composition or inheritance).
        path = getattr(ll, "on_disk_path", None)
        if path:
            return os.fspath(path)

        # Fallback: some wrappers track the last known path explicitly.
        path = getattr(self, "_on_disk_path", None)
        if path:
            return os.fspath(path)
        return None

    def _ll_prefault(self) -> bool:
        """
        Return the low-level prefault flag when available.

        This method is used to persist a deterministic prefault setting in the
        pickle state. If the low-level backend does not expose ``prefault``, this
        falls back to ``self.prefault`` when present, otherwise ``False``.
        """
        ll = self._low_level()
        if ll is not None and hasattr(ll, "prefault"):
            val = getattr(ll, "prefault", None)
            if isinstance(val, bool):
                return val

        return getattr(self, "prefault", None)

    def _ll_get_n_trees(self) -> int:
        ll = self._low_level()
        fn = getattr(ll, "get_n_trees", None)
        if fn is None:
            return 0
        try:
            return int(fn())
        except Exception:
            return 0

    def _ll_serialize(self) -> bytes:
        ll = self._low_level()
        fn = getattr(ll, "serialize", None)
        if fn is None:
            raise AttributeError("Low-level object does not support serialize()")
        return cast(bytes, fn())

    def _ll_deserialize(self, data: bytes, *, prefault: bool | None) -> None:
        ll = self._low_level()
        fn = getattr(ll, "deserialize", None)
        if fn is None:
            raise AttributeError("Low-level object does not support deserialize()")

        # Preserve low-level call contract: pass prefault only when provided.
        if prefault is None:  # noqa: SIM108
            res = fn(data)
        else:
            res = fn(data, prefault=bool(prefault))

        if isinstance(res, bool) and not res:
            raise RuntimeError("deserialize failed")

    def _ll_load_index(self, path: str, *, prefault: bool | None) -> None:
        load_index = getattr(self, "load_index", None)
        if callable(load_index):
            if prefault is None:
                load_index(path)
            else:
                load_index(path, prefault=bool(prefault))
            return
        ll = self._low_level()
        fn = getattr(ll, "load", None)
        if fn is None:
            raise AttributeError("Neither load_index nor low-level load() is available")
        fn(path, prefault=prefault)

    # ------------------------------------------------------------------
    # Pickle protocol
    def __reduce__(self) -> object:  # noqa: PLR0912
        """
        Return a pickle reconstruction tuple.

        Returns
        -------
        object
            A reconstruction tuple ``(callable, (state,))`` where ``callable`` is
            :meth:`_rebuild` and ``state`` is a versioned mapping.

        Raises
        ------
        RuntimeError
            If the instance lacks a valid dimension/metric, or if ``byte`` mode
            is requested on an unbuilt index, or if ``disk`` mode is requested
            without a backing path.
        """
        with self._lock:
            f, metric = self._config_f_metric()
            prefault = self._ll_prefault()

            if f <= 0 or not metric:
                raise RuntimeError(
                    "Cannot pickle an Annoy index without defined dimension/metric. "
                    "Initialize with f>0 and a valid metric."
                )

            mode: PickleMode = self._pickle_mode
            if mode == "auto":
                mode = "disk" if self._ll_on_disk_path() else "byte"

            # Optional: include a manifest snapshot when available.
            manifest: Mapping[str, Any] | None = None
            to_manifest = getattr(self, "to_manifest", None)
            if callable(to_manifest):
                try:
                    manifest = cast(Mapping[str, Any], to_manifest())
                except Exception:
                    manifest = None

            if mode == "disk":
                path = self._ll_on_disk_path()
                if not path:
                    raise RuntimeError(
                        "Pickle mode 'disk' requires a known on-disk path. "
                        "Call save_index()/load_index() (or low-level save/load) first, "
                        "or use pickle_mode='byte'."
                    )

                state: dict[str, Any] = {
                    "pickle_state_version": self._PICKLE_STATE_VERSION,
                    "mode": "disk",
                    "f": f,
                    "metric": metric,
                    "prefault": prefault,
                    "path": str(path),
                }
                if manifest is not None:
                    state["manifest"] = dict(manifest)
                return (self.__class__._rebuild, (state,))

            if mode != "byte":
                raise RuntimeError("Invalid pickle mode")

            if self._ll_get_n_trees() <= 0:
                raise RuntimeError(
                    "Pickle mode 'byte' requires a built index. "
                    "Call build(...) first, or use pickle_mode='disk' after save/load."
                )

            data = self._ll_serialize()
            compress_mode = self._compress_mode

            if compress_mode == "zlib":
                data = zlib.compress(data)
            elif compress_mode == "gzip":
                data = gzip.compress(data)
            elif compress_mode is not None:
                raise RuntimeError("Invalid compression mode")

            state = {
                "pickle_state_version": self._PICKLE_STATE_VERSION,
                "mode": "byte",
                "f": f,
                "metric": metric,
                "prefault": prefault,
                "data": data,
                "compress_mode": compress_mode,
            }
            if manifest is not None:
                state["manifest"] = dict(manifest)
            return (self.__class__._rebuild, (state,))

    def __reduce_ex__(self, protocol: int) -> object:
        return self.__reduce__()

    @classmethod
    def _rebuild(cls: type[Self], state: Mapping[str, Any]) -> Self:  # noqa: PLR0912
        """Reconstruct an instance from the pickled state mapping."""
        mode = state.get("mode")
        if mode not in ("byte", "disk"):
            raise ValueError("Corrupted pickle state: invalid mode")

        version = int(state.get("pickle_state_version", 0))
        if version not in (0, cls._PICKLE_STATE_VERSION):
            raise ValueError("Corrupted pickle state: unsupported pickle_state_version")

        manifest = state.get("manifest", None)
        f = int(state.get("f", 0))
        metric = str(state.get("metric", ""))
        prefault = bool(state.get("prefault", False))

        if (f <= 0 or not metric) and isinstance(manifest, Mapping):
            try:
                f = int(manifest.get("f", f))
                metric = str(manifest.get("metric", metric))
            except Exception:
                pass

        if f <= 0:
            raise ValueError("Corrupted pickle state: invalid dimension f")
        if not metric:
            raise ValueError("Corrupted pickle state: invalid metric")

        # Prefer a manifest-aware constructor if available.
        from_manifest = getattr(cls, "from_manifest", None)
        if callable(from_manifest) and isinstance(manifest, Mapping):
            try:
                instance = cast(Self, from_manifest(dict(manifest), load=False))
            except Exception:
                instance = cls(f=f, metric=metric, prefault=prefault)  # type: ignore[call-arg]
        else:
            instance = cls(f=f, metric=metric, prefault=prefault)  # type: ignore[call-arg]

        # Ensure lock exists even if cls.__init__ did not initialize it.
        if not hasattr(instance, "_lock"):
            # setattr(instance, "_lock", threading.RLock())
            instance._lock = threading.RLock()

        # Keep wrapper-level configuration fields consistent.
        try:  # noqa: SIM105
            instance.prefault = prefault  # type: ignore[attr-defined]
        except AttributeError:
            # Prefault is optional on wrapper classes; low-level load/deserialize
            # will still receive the prefault flag via helper calls.
            pass

        instance._pickle_mode = "auto"
        instance._compress_mode = cast(CompressMode, state.get("compress_mode", None))

        if mode == "disk":
            path = state.get("path")
            if not path and isinstance(manifest, Mapping):
                path = manifest.get("on_disk_path")
            if not path:
                raise ValueError("Corrupted pickle state: missing path for disk mode")
            instance._ll_load_index(os.fspath(path), prefault=prefault)

            # Best-effort: record the path on wrapper and/or low-level if possible.
            try:  # noqa: SIM105
                instance._on_disk_path = str(path)
            except AttributeError:
                pass
            try:
                ll = instance._low_level()
                if hasattr(ll, "on_disk_path"):
                    ll.on_disk_path = os.fspath(path)
            except Exception:
                pass
            return instance

        # mode == "byte"
        data = cast(bytes, state.get("data", b""))

        compress_mode = state.get("compress_mode", None)
        if compress_mode == "zlib":
            data = zlib.decompress(data)
        elif compress_mode == "gzip":
            data = gzip.decompress(data)
        elif compress_mode is not None:
            raise ValueError("Corrupted pickle state: invalid compression mode")

        with instance._lock:
            instance._ll_deserialize(data, prefault=prefault)

        return instance
