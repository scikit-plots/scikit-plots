"""
Pickling foundations for scikitplot.cexternals.annoy.

This layer does NOT modify the low-level C-API. It only provides:
- disk-path tracking on the Python side
- strict, robust pickling for pickle/joblib/cloudpickle

.. seealso::
    * :py:obj:`~scikitplot.annoy.Index.from_low_level`
    * https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled
"""

from __future__ import annotations

import gzip
import threading
import zlib
from typing import Any, Literal, Optional

from .. import Annoy

PickleMode = Literal["auto", "byte", "disk"]
CompressMode = Optional[Literal["zlib", "gzip"]]


class PathAwareAnnoy(Annoy):
    """
    Thin Python subclass that tracks the last known on-disk path.

    Pure metadata; core logic remains in the extension.

    .. seealso::
        * :py:obj:`~scikitplot.annoy.Index.from_low_level`
        * https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled
    """

    _on_disk_path: str | None = None

    def on_disk_build(self, path: str):
        out = super().on_disk_build(path)
        self._on_disk_path = str(path)
        return out

    def load(self, path: str, prefault: bool = False):
        out = super().load(path, prefault=prefault)
        self._on_disk_path = str(path)
        return out

    def save(self, path: str, prefault: bool = False):
        out = super().save(path, prefault=prefault)
        self._on_disk_path = str(path)
        return out


class PickleMixin(PathAwareAnnoy):
    """
    Adds strict persistence support.

    - 'byte' mode: stores serialized bytes, optionally compressed.
    - 'disk' mode: stores only the path (requires on_disk_build/load first).
    - 'auto' mode: deterministic policy:
        disk if on-disk path is known else byte.

    .. seealso::
        * :py:obj:`~scikitplot.annoy.Index.from_low_level`
        * https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled
    """

    def __init__(self, f: int = 0, metric: str = "angular"):
        super().__init__(f, metric)
        self._lock = threading.RLock()
        self._prefault: bool = False
        self._pickle_mode: PickleMode = "auto"
        self._compress_mode: CompressMode = None

    # -----------------------------
    # Configuration
    # -----------------------------
    @property
    def prefault(self) -> bool:
        return self._prefault

    @prefault.setter
    def prefault(self, value) -> None:
        self._prefault = bool(value)

    @property
    def on_disk_path(self) -> str | None:
        return getattr(self, "_on_disk_path", None)

    @on_disk_path.setter
    def on_disk_path(self, value: str | None) -> None:
        self._on_disk_path = str(value) if value is not None else None

    @property
    def pickle_mode(self) -> PickleMode:
        return self._pickle_mode

    @pickle_mode.setter
    def pickle_mode(self, value: PickleMode) -> None:
        if value not in ("auto", "byte", "disk"):
            raise ValueError("pickle_mode must be 'auto', 'byte', or 'disk'")
        self._pickle_mode = value

    @property
    def compress_mode(self) -> CompressMode:
        return self._compress_mode

    @compress_mode.setter
    def compress_mode(self, value: CompressMode) -> None:
        if value is not None and value not in ("zlib", "gzip"):
            raise ValueError("compress_mode must be None, 'zlib', or 'gzip'")
        self._compress_mode = value

    # -----------------------------
    # Pickling
    # -----------------------------
    def __reduce__(self):
        with self._lock:
            # Strict: prevent pickling incomplete/lazy instances
            if int(self.f) <= 0 or not str(self.metric):
                raise RuntimeError(
                    "Cannot pickle Annoy object with undefined dimension/metric. "
                    "Initialize with f>0 and a valid metric."
                )

            mode: PickleMode = self._pickle_mode
            if mode == "auto":
                mode = "disk" if self._on_disk_path else "byte"

            if mode == "disk":
                if not self._on_disk_path:
                    raise RuntimeError(
                        "Pickle mode 'disk' requires a known on-disk path. "
                        "Call on_disk_build(...) or load(...) first, "
                        "or use pickle_mode='byte'."
                    )

                state = {
                    "mode": "disk",
                    "f": int(self.f),
                    "metric": str(self.metric),
                    "prefault": bool(self._prefault),
                    "path": self._on_disk_path,
                }
                return (self.__class__._rebuild, (state,))

            if mode != "byte":
                raise RuntimeError("Invalid pickle mode")

            byte = self.serialize()
            compress_mode = self._compress_mode

            if compress_mode == "zlib":
                byte = zlib.compress(byte)
            elif compress_mode == "gzip":
                byte = gzip.compress(byte)
            elif compress_mode is not None:
                raise RuntimeError("Invalid compression mode")

            state = {
                "mode": "byte",
                "f": int(self.f),
                "metric": str(self.metric),
                "prefault": bool(self._prefault),
                "byte": byte,
                "compress_mode": compress_mode,
            }
            return (self.__class__._rebuild, (state,))

    @classmethod
    def _rebuild(cls, state: dict[str, Any]):
        mode = state.get("mode")
        if mode not in ("byte", "disk"):
            raise ValueError("Corrupted pickle state: invalid mode")

        f = int(state.get("f", 0))
        metric = str(state.get("metric", ""))
        prefault = bool(state.get("prefault", False))

        if f <= 0:
            raise ValueError("Corrupted pickle state: invalid dimension f")
        if not metric:
            raise ValueError("Corrupted pickle state: invalid metric")

        instance = cls(f, metric)

        if mode == "disk":
            path = state.get("path")
            if not path:
                raise ValueError("Corrupted pickle state: missing path for disk mode")

            instance._on_disk_path = str(path)
            instance.load(str(path), prefault=prefault)
            instance._pickle_mode = "auto"
            return instance

        # mode == "byte"
        byte = state.get("byte", b"")
        compress_mode = state.get("compress_mode", None)  # noqa: SIM910

        if compress_mode == "zlib":
            byte = zlib.decompress(byte)
        elif compress_mode == "gzip":
            byte = gzip.decompress(byte)
        elif compress_mode is not None:
            raise ValueError("Corrupted pickle state: invalid compression mode")

        with instance._lock:
            ok = instance.deserialize(byte, prefault=prefault)
            if not ok:
                raise RuntimeError("deserialize failed")

        instance._compress_mode = compress_mode
        instance._pickle_mode = "auto"
        instance._prefault = prefault
        return instance
