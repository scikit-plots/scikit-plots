# scikitplot/annoy/_mixins/_pickle.py
"""
Pickling support for Annoy-backed indices.

This mixin provides deterministic ``pickle`` behavior for Annoy-backed indices
without changing the low-level Annoy semantics.

``pickle_mode`` controls how the index is represented in the pickle payload:

- ``"auto"`` (default): choose ``"disk"`` when an ``on_disk_path`` is known,
  otherwise choose ``"byte"``.
- ``"disk"``: store only the path and reload the mmap-backed index on unpickle.
- ``"byte"``: store serialized index bytes (requires a built index).

``compress_mode`` optionally compresses the serialized bytes in ``"byte"`` mode.

See Also
--------
scikitplot.annoy._mixins._io.IndexIOMixin
scikitplot.annoy._mixins._meta.MetaMixin

Notes
-----
Pickle is **unsafe on untrusted data**. Never unpickle data from an untrusted
source.
"""

from __future__ import annotations

import contextlib
import gzip
import os
import pickle
import zlib
from collections.abc import Mapping
from dataclasses import dataclass  # noqa: F401
from enum import Enum  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any, ClassVar, Literal, TypeAlias, cast

from typing_extensions import Self

from .._utils import _get_lock

CompressMode: TypeAlias = Literal["zlib", "gzip"] | None
"""\
Compression used for ``"byte"`` pickling.

Compression is applied to the serialized Annoy bytes produced by the low-level
:meth:`serialize` API.
"""

PickleMode: TypeAlias = Literal["auto", "disk", "byte"]
"""\
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


def _get_noncallable_attr(obj: Any, name: str) -> Any | None:
    """Return attribute value only if it is present and non-callable."""
    value = getattr(obj, name, None)
    return None if callable(value) else value


def _get_callable(obj: Any, name: str) -> Any | None:
    """Return attribute value only if it is present and callable."""
    value = getattr(obj, name, None)
    return value if callable(value) else None


def _get_int_param(
    obj: Any, *, methods: tuple[str, ...], attrs: tuple[str, ...], default: int
) -> int:
    """Resolve an int parameter deterministically from methods/attributes."""
    for m in methods:
        fn = _get_callable(obj, m)
        if fn is not None:
            return int(fn())
    for a in attrs:
        v = _get_noncallable_attr(obj, a)
        if v is not None:
            return int(v)
    return int(default)


def _get_str_param(
    obj: Any, *, methods: tuple[str, ...], attrs: tuple[str, ...], default: str
) -> str:
    """Resolve a str parameter deterministically from methods/attributes."""
    for m in methods:
        fn = _get_callable(obj, m)
        if fn is not None:
            return str(fn())
    for a in attrs:
        v = _get_noncallable_attr(obj, a)
        if v is not None:
            return str(v)
    return str(default)


def _get_bool_param(obj: Any, *, attrs: tuple[str, ...], default: bool) -> bool:
    """Resolve a bool parameter deterministically from attributes."""
    for a in attrs:
        v = _get_noncallable_attr(obj, a)
        if v is not None:
            return bool(v)
    return bool(default)


def _load_index_into(obj: Any, path: str, *, prefault: bool) -> None:
    """Load an on-disk index using the best available API (deterministic)."""
    # Prefer the Python mixin helper when present; fall back to the backend method.
    load_index = getattr(obj, "load_index", None)
    if callable(load_index):
        load_index(path, prefault=prefault)
        return

    load = getattr(obj, "load", None)
    if not callable(load):
        raise AttributeError("Backend does not provide load(path, prefault=...)")
    load(path, prefault=prefault)


def _serialize_index(obj: Any) -> bytes:
    """Serialize an index using the best available API (deterministic)."""
    to_bytes = getattr(obj, "to_bytes", None)
    if callable(to_bytes):
        data = to_bytes()
    else:
        serialize = getattr(obj, "serialize", None)
        if not callable(serialize):
            raise AttributeError("Backend does not provide serialize() -> bytes")
        data = serialize()

    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("serialize() must return bytes")
    return bytes(data)


def _deserialize_index(obj: Any, data: bytes, *, prefault: bool) -> None:
    """Deserialize an index using the best available API (deterministic)."""
    deserialize = getattr(obj, "deserialize", None)
    if not callable(deserialize):
        raise AttributeError("Backend does not provide deserialize(data, prefault=...)")
    deserialize(data, prefault=prefault)


class PickleMixin:
    """
    Mixin adding deterministic pickle support.

    Parameters
    ----------
    f : int, default=0
        Vector dimension (constructor argument for wrapper classes).
    metric : str, default="angular"
        Distance metric (constructor argument for wrapper classes).
    prefault : bool, default=False
        Default prefault flag used during reconstruction.
    compress_mode : {None, "zlib", "gzip"}, default=None
        Optional compression for ``"byte"`` mode.
    pickle_mode : {"auto", "disk", "byte"}, default="auto"
        Pickle strategy.

    Notes
    -----
    - ``"byte"`` mode requires a built index (``get_n_trees() > 0``).
    - ``"disk"`` mode requires an on-disk path (``on_disk_path`` or ``_on_disk_path``).
    """  # noqa: D205, D415

    _PICKLE_STATE_VERSION: ClassVar[int] = 1

    # Default knobs (wrapper classes may override in __init__).
    _compress_mode: CompressMode = None
    _pickle_mode: PickleMode = "auto"

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
    # Pickle protocol
    def __reduce__(self) -> object:
        # Delegate to __reduce_ex__ for consistent behavior.
        return self.__reduce_ex__(protocol=pickle.HIGHEST_PROTOCOL)

    def __reduce_ex__(self, protocol: int) -> object:  # noqa: ARG002, PLR0912
        """Return the pickle reduce tuple."""
        # Determine constructor essentials.
        # Determine constructor essentials deterministically.
        f = _get_int_param(self, methods=("get_f",), attrs=("f", "_f"), default=0)
        metric = _get_str_param(
            self, methods=("get_metric",), attrs=("metric", "_metric"), default=""
        )

        if f <= 0 or not metric:
            raise RuntimeError(
                "Cannot pickle an index without a valid (f, metric). "
                "Initialize with f>0 and a valid metric."
            )

        prefault = _get_bool_param(self, attrs=("prefault", "_prefault"), default=False)

        # Include metadata when available (deterministically).
        metadata: Mapping[str, Any] | None = None
        to_metadata = getattr(self, "to_metadata", None)
        if callable(to_metadata):
            with contextlib.suppress(Exception):
                meta_obj = to_metadata(include_info=False, strict=False)
                if isinstance(meta_obj, Mapping):
                    metadata = dict(meta_obj)

        # Decide mode.
        mode: PickleMode = self._pickle_mode
        if mode == "auto":
            path = _get_noncallable_attr(self, "on_disk_path") or _get_noncallable_attr(
                self, "_on_disk_path"
            )
            mode = "disk" if path else "byte"

        lock = _get_lock(self)
        with lock:
            if mode == "disk":
                path = _get_noncallable_attr(
                    self, "on_disk_path"
                ) or _get_noncallable_attr(self, "_on_disk_path")
                if not path:
                    raise RuntimeError(
                        "pickle_mode='disk' requires an on-disk path. "
                        "Call save_index()/load_index() (or low-level save/load) first, "
                        "or use pickle_mode='byte'."
                    )

                state: dict[str, Any] = {
                    "pickle_state_version": self._PICKLE_STATE_VERSION,
                    "mode": "disk",
                    "f": f,
                    "metric": metric,
                    "prefault": prefault,
                    "path": os.fspath(path),
                    "metadata": dict(metadata) if metadata is not None else None,
                }
                return (type(self)._rebuild, (state,))

            # mode == "byte"
            get_n_trees = getattr(self, "get_n_trees", None)
            if not callable(get_n_trees) or int(get_n_trees()) <= 0:
                raise RuntimeError(
                    "pickle_mode='byte' requires a built index. "
                    "Call build(...) first, or use pickle_mode='disk' after save/load."
                )

            data = _serialize_index(self)

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
                "compress_mode": compress_mode,
                "data": data,
                "metadata": dict(metadata) if metadata is not None else None,
            }
            return (type(self)._rebuild, (state,))

    @classmethod
    def _rebuild(cls: type[Self], state: Mapping[str, Any]) -> Self:  # noqa: PLR0912
        """Rebuild an instance from a pickle state mapping."""
        version = int(state.get("pickle_state_version", -1))
        if version != cls._PICKLE_STATE_VERSION:
            raise ValueError("Corrupted pickle state: unsupported pickle_state_version")

        mode = cast(PickleMode, state.get("mode", ""))
        metadata = state.get("metadata", None)
        f = int(state.get("f", 0))
        metric = str(state.get("metric", ""))
        prefault = bool(state.get("prefault", False))

        if f <= 0:
            raise ValueError("Corrupted pickle state: invalid dimension f")
        if not metric:
            raise ValueError("Corrupted pickle state: invalid metric")

        # Prefer reconstruction from metadata when available.
        if isinstance(metadata, Mapping):
            from_metadata = getattr(cls, "from_metadata", None)
            if callable(from_metadata):
                with contextlib.suppress(Exception):
                    instance = cast(Self, from_metadata(dict(metadata), load=False))
                if isinstance(instance, cls):
                    _get_lock(instance)
                    # keep wrapper-level config consistent
                    with contextlib.suppress(Exception):
                        instance.prefault = prefault  # type: ignore[attr-defined]
                    instance._pickle_mode = "auto"
                    # Continue to load below using the selected mode.
                else:
                    instance = cls(f, metric)
            else:
                instance = cls(f, metric)
        else:
            # Try keyword construction first; fall back to positional.
            try:
                instance = cls(f=f, metric=metric, prefault=prefault)  # type: ignore[call-arg]
            except Exception:
                instance = cls(f, metric)

        # Ensure lock exists even if cls.__init__ did not initialize it.
        _get_lock(instance)

        # Best-effort: keep wrapper-level configuration fields consistent.
        with contextlib.suppress(Exception):
            instance.prefault = prefault  # type: ignore[attr-defined]

        instance._pickle_mode = "auto"

        if mode == "disk":
            path = state.get("path", "")
            if not path and isinstance(metadata, Mapping):
                params = metadata.get("params")
                if isinstance(params, Mapping):
                    path = params.get("on_disk_path")
            if not path:
                raise ValueError("Corrupted pickle state: missing path for disk mode")

            _load_index_into(instance, os.fspath(path), prefault=prefault)

            # Best-effort: keep on_disk_path aligned.
            with contextlib.suppress(Exception):
                instance.on_disk_path = os.fspath(path)  # type: ignore[attr-defined]
            return instance

        # mode == "byte"
        if mode != "byte":
            raise ValueError("Corrupted pickle state: invalid mode")

        data = cast(bytes, state.get("data", b""))
        compress_mode = state.get("compress_mode", None)
        if compress_mode == "zlib":
            data = zlib.decompress(data)
        elif compress_mode == "gzip":
            data = gzip.decompress(data)
        elif compress_mode is not None:
            raise ValueError("Corrupted pickle state: invalid compress_mode")

        _deserialize_index(instance, bytes(data), prefault=prefault)

        return instance
