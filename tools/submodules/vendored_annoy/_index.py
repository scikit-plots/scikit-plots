# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Annoy index wrapper
===================

A safe, fully-serializable wrapper around :class:`AnnoyIndex`.

This module exists because the raw Annoy object holds an internal C++
pointer that cannot be pickled. We expose a clean Python API that:

* Enables **pickle / joblib** serialization across processes and machines.
* Adds optional transparent **compression** (``zlib`` or ``gzip``).
* Ensures **thread-safe** serialization/deserialization via ``RLock``.
* Restores Annoy indices from a **zero-copy binary buffer**.
* Preserves full C++ runtime speed; nothing changes in search performance.

This wrapper is intentionally minimal. It does not modify approximate
nearest-neighbor behavior, tree-building logic, or memory layout.
It only makes Annoy *safe* and *portable* in ML pipelines.

Examples
--------
>>> idx = Index(128, "angular")
>>> idx.add_item(0, [1]*128)
>>> idx.add_item(1, [2]*128)
>>> idx.build(50)
>>> idx.compress = True
>>> idx.save_to_file("index.joblib")

>>> idx2 = Index.load_from_file("index.joblib")
>>> # idx2.get_item_vector(0)
>>> idx2.get_nns_by_item(0, 10)

This class is suitable for very large indexes (multi-GB), including
up to billions of vectors, because Annoy exposes a contiguous binary
blob that can be serialized quickly.
"""

from __future__ import annotations

import gzip
import zlib
import joblib
import threading
from typing import Any, Dict
from .annoylib import Annoy as AnnoyIndex


class Index(AnnoyIndex):
    """
    A robust, pickle-safe, MLOps-ready wrapper around ``AnnoyIndex``.

    Provides:

    * **Pickling support** via a custom ``__reduce__`` implementation.
    * **Optional compression** (``zlib`` or ``gzip``).
    * **Thread-safe persistence** using a module-level ``RLock``.
    * **Zero-copy restore path** through Annoy's ``deserialize()`` buffer.

    Parameters
    ----------
    f : int
        Dimensionality of the vectors.
    metric : str
        Distance metric ("angular", "euclidean", "manhattan", ...).

    Notes
    -----
    Annoy stores all data in a compact contiguous block, so serialization
    is extremely fast even on multi-GB indexes. The wrapper is fully
    compatible with joblib multiprocessing and cloud model registries.
    """

    # ---------------------------------------------------------
    # Re-entrant lock for thread-safe pickle/restore operations
    # ---------------------------------------------------------
    _lock = threading.RLock()

    # ---------------------------------------------------------
    # Compression settings
    # ---------------------------------------------------------
    _compress: bool = False
    _compression_type: str = "zlib"  # {"zlib", "gzip"}

    # ------------------- Compression API ---------------------
    @property
    def compress(self) -> bool:
        """Whether the binary buffer is compressed before pickling."""
        return self._compress

    @compress.setter
    def compress(self, enabled: bool) -> None:
        self._compress = bool(enabled)

    @property
    def compression_type(self) -> str:
        """Compression algorithm to use when ``compress`` is True."""
        return self._compression_type

    @compression_type.setter
    def compression_type(self, value: str) -> None:
        if value not in ("zlib", "gzip"):
            raise ValueError("compression_type must be 'zlib' or 'gzip'")
        self._compression_type = value

    # ------------------- Pickling / Joblib --------------------
    def __reduce__(self):
        """
        Defines how ``pickle`` / ``joblib`` serialize this instance.

        Returns
        -------
        tuple
            ``(callable, args)`` used by pickle to reconstruct the object.

        Notes
        -----
        Only the exported Annoy binary buffer is serialized—not the C++
        pointer. This allows safe cross-process and cross-machine transfer.
        """
        with self._lock:
            buf = self.serialize()

            if self.compress:
                if self.compression_type == "zlib":
                    buf = zlib.compress(buf)
                else:
                    buf = gzip.compress(buf)

            state = {
                "f": self.f,
                "metric": self.metric,
                "data": buf,
                "compress": self.compress,
                "compression_type": self.compression_type,
            }

            return (self._rebuild, (state,))

    @classmethod
    def _rebuild(cls, state: Dict[str, Any]) -> "Index":
        """
        Internal factory used during unpickling.

        Parameters
        ----------
        state : dict
            Serialized metadata + (optionally compressed) byte buffer.

        Returns
        -------
        Index
            Fully reconstructed index.

        Raises
        ------
        RuntimeError
            If Annoy fails to deserialize the binary buffer.
        """
        instance = cls(state["f"], state["metric"])
        buf = state["data"]

        if state.get("compress", False):
            ctype = state.get("compression_type", "zlib")
            buf = zlib.decompress(buf) if ctype == "zlib" else gzip.decompress(buf)

        with instance._lock:
            if not instance.deserialize(buf):
                raise RuntimeError("Failed to deserialize Annoy index byte buffer")

        instance.compress = state.get("compress", False)
        instance.compression_type = state.get("compression_type", "zlib")
        return instance

    # ------------------- File Helpers -------------------------
    def save_to_file(self, path: str) -> None:
        """
        Persist the index to disk using joblib.

        Parameters
        ----------
        path : str
            Output ``.joblib`` file path.

        Notes
        -----
        Thread-safe. Locks during serialization.
        """
        with self._lock:
            joblib.dump(self, path)

    @classmethod
    def load_from_file(cls, path: str) -> "Index":
        """
        Load an index previously saved with :meth:`save_to_file`.

        Parameters
        ----------
        path : str

        Returns
        -------
        Index
            Restored index.
        """
        return joblib.load(path)
