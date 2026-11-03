# scikitplot/corpus/_search_cache.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
An exact search-result cache that cannot serve a stale or out-of-scope result.

Notes
-----
**User.** ``CachedSearch(index)`` answers ``search`` like the index does and
returns a stored response when *everything* that decides the response is the
same as before:

* the index's content generation (:attr:`RetrievalIndex.index_generation`
  fingerprint) -- a rebuilt index has a new one, so nothing from the old
  build is served;
* the index's security filter, by its canonical expression (not its label,
  which two scopes could share);
* the query text, every :class:`RetrievalConfig` field, the caller's filter
  in canonical form, and the query embedding's bytes, dtype and shape.

Only ``success`` and ``empty`` responses are kept: a ``degraded`` or
``partial`` one reflects a failure that may not recur. A config with
``probability_fusion`` is not cached -- its calibrators have no canonical
identity -- and is answered by the index every time (counted as
``uncacheable``). This is an exact cache; a "similar query" cache is not
offered, because a similar query is not the same question.

**Developer.** One wrapper per index, an LRU of ``max_entries`` under a lock,
so it is safe across threads. A hit returns a copy of the response with a new
hit list, so a caller that edits its result cannot change what the next caller
receives.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import threading
from typing import Any, Mapping

from ._filters import filter_to_dict, validate_filter
from ._retrieval import RetrievalStatus

__all__ = ["CachedSearch"]

_CACHEABLE = frozenset({RetrievalStatus.SUCCESS, RetrievalStatus.EMPTY})


def _canonical(value: Any) -> Any:
    """Return a JSON-able, order-independent form of a config field's value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, Mapping):
        return {
            str(k): _canonical(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    raise TypeError(f"no canonical form for a {type(value).__name__}")


def _embedding(query_embedding: Any) -> Any:
    if query_embedding is None:
        return None
    import numpy as np  # noqa: PLC0415 - only when an embedding is given

    array = np.ascontiguousarray(np.asarray(query_embedding))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


class CachedSearch:
    """
    Exact, bounded cache in front of one retrieval index.

    Parameters
    ----------
    index : RetrievalIndex
        The index to answer from.
    max_entries : int, default=1024
        Responses kept; the least recently used is evicted first.

    Raises
    ------
    ValueError
        If ``max_entries`` is not an int of at least 1.
    """

    def __init__(self, index: Any, *, max_entries: int = 1024) -> None:
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or max_entries < 1
        ):
            raise ValueError(f"max_entries must be an int >= 1, got {max_entries!r}")
        self.index = index
        self.max_entries = max_entries
        self._entries: collections.OrderedDict[str, Any] = collections.OrderedDict()
        self._lock = threading.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "stored": 0,
            "uncacheable": 0,
            "evicted": 0,
        }

    def key(
        self,
        query: str,
        config: Any = None,
        query_embedding: Any = None,
        filter: Any = None,  # noqa: A002
    ) -> str | None:
        """
        Return the cache key of a search, or ``None`` when it cannot be cached.

        Returns
        -------
        str or None
            64 hex digits; ``None`` for an index that is not built, or a
            config with ``probability_fusion``.
        """
        generation = self.index.index_generation
        if generation is None:
            return None
        cfg = self.index.config if config is None else config
        if getattr(cfg, "probability_fusion", None) is not None:
            return None
        security = self.index.security_filter
        payload = {
            "generation": generation.fingerprint,
            "scope": None if security is None else filter_to_dict(security.expression),
            "query": query,
            "config": {
                f.name: _canonical(getattr(cfg, f.name))
                for f in dataclasses.fields(cfg)
            },
            "filter": (
                None if filter is None else filter_to_dict(validate_filter(filter))
            ),
            "embedding": _embedding(query_embedding),
        }
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def search(
        self,
        query: str,
        *,
        config: Any = None,
        query_embedding: Any = None,
        filter: Any = None,  # noqa: A002
    ) -> Any:
        """
        Answer like :meth:`RetrievalIndex.search`, from the cache when possible.

        Returns
        -------
        RetrievalResponse
            Equal to what the index returns for the same arguments.
        """
        key = self.key(query, config, query_embedding, filter)
        if key is not None:
            with self._lock:
                stored = self._entries.get(key)
                if stored is not None:
                    self._entries.move_to_end(key)
                    self._stats["hits"] += 1
                    return dataclasses.replace(stored, hits=list(stored.hits))
        response = self.index.search(
            query, config=config, query_embedding=query_embedding, filter=filter
        )
        with self._lock:
            if key is None:
                self._stats["uncacheable"] += 1
                return response
            self._stats["misses"] += 1
            if response.status in _CACHEABLE:
                self._entries[key] = dataclasses.replace(
                    response,
                    hits=list(response.hits),
                )
                self._entries.move_to_end(key)
                self._stats["stored"] += 1
                while len(self._entries) > self.max_entries:
                    self._entries.popitem(last=False)
                    self._stats["evicted"] += 1
        return response

    def stats(self) -> dict[str, int]:
        """Count hits, misses, stored, uncacheable and evicted, and the size."""
        with self._lock:
            return {**self._stats, "size": len(self._entries)}

    def clear(self) -> None:
        """Drop every stored response."""
        with self._lock:
            self._entries.clear()
