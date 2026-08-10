# scikitplot/mcp/_hybrid.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Hybrid retrieval for :mod:`scikitplot.mcp`.

Combines several :class:`~scikitplot.mcp._core.DocsRetriever` legs into one,
fusing their ranked lists with **Reciprocal Rank Fusion (RRF)**. The three
intended legs, each already a plain ``DocsRetriever``:

* **Dense** — :class:`~scikitplot.mcp._corpus_annoy.CorpusAnnoyRetriever`
  (embeddings via ``scikitplot.corpus`` + ANN via ``scikitplot.annoy``); good at
  paraphrase / synonymy.
* **Sparse / lexical (BM25)** — :class:`Bm25Retriever` over
  ``scikitplot.corpus.SQLiteStorage`` FTS5, whose default ranking *is* BM25; good
  at exact terms, API names, error strings.
* **Graph** *(optional, designed)* — a retriever that expands seed hits along the
  documentation graph (cross-references, ``see also`` links, section hierarchy)
  and re-ranks; captures relationships neither dense nor lexical see. Its exact
  wiring depends on the relationship metadata the corpus exposes, so it is
  specified but not bound here (see ``DESIGN.md`` §Hybrid).

Why RRF
-------
Dense scores (cosine) and BM25 scores are not comparable, and min/max
normalisation is brittle. RRF ignores raw scores and fuses by *rank*:
``score(d) = Σ_legs weight_leg / (rrf_k + rank_leg(d))``. It is parameter-light
(``rrf_k≈60``), robust to one leg being miscalibrated, and a documented standard
for hybrid search. A document found by several legs is naturally boosted.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import replace
from typing import Any, Callable, Sequence

from ._core import DocsRetriever, RetrievedChunk, _coerce_finite_score

__all__ = [
    "DEFAULT_RRF_K",
    "Bm25Retriever",
    "HybridRetriever",
    "reciprocal_rank_fusion",
]

#: Standard RRF constant; dampens the contribution of lower ranks.
DEFAULT_RRF_K: int = 60

_MAX_RETRIEVAL_K: int = 50
logger = logging.getLogger(__name__)


def _chunk_key(chunk: RetrievedChunk) -> str:
    """Stable identity for dedup/fusion: doc_id, else (uri + text prefix)."""
    if chunk.doc_id:
        return chunk.doc_id
    return (chunk.source_uri or "") + "::" + (chunk.text or "")[:128]


def _deduplicate(items: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[str] = set()
    output: list[RetrievedChunk] = []
    for item in items:
        if not isinstance(item, RetrievedChunk):
            continue
        key = _chunk_key(item)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def reciprocal_rank_fusion(
    ranked_lists: Sequence[tuple[float, Sequence[Any]]],
    *,
    k: int = DEFAULT_RRF_K,
    key: Callable[[Any], str] = _chunk_key,
) -> dict[str, float]:
    """
    Fuse weighted ranked lists into a single ``key -> score`` map.

    Parameters
    ----------
    ranked_lists : sequence of (weight, items)
        Each leg's weight and its best-first ranked items.
    k : int, optional
        RRF constant (default :data:`DEFAULT_RRF_K`).
    key : callable, optional
        Maps an item to its fusion identity (default :func:`_chunk_key`).

    Returns
    -------
    dict
        ``identity -> fused score`` (higher is better).
    """
    k = int(k)
    if k < 1:
        raise ValueError("rrf k must be >= 1")

    fused: dict[str, float] = defaultdict(float)
    for raw_weight, items in ranked_lists:
        weight = _coerce_finite_score(raw_weight)
        if weight < 0:
            raise ValueError("RRF weights must be non-negative")
        if weight == 0:
            continue
        seen_in_leg: set[str] = set()
        for rank, item in enumerate(items or (), start=1):
            identity = key(item)
            if identity in seen_in_leg:
                continue
            seen_in_leg.add(identity)
            fused[identity] += weight / (k + rank)
    return dict(fused)


def _metadata_richness(chunk: RetrievedChunk) -> tuple[int, int, int, int]:
    return (
        int(bool(chunk.source_uri)),
        len(chunk.title or "") + len(chunk.anchor or ""),
        len(chunk.text or ""),
        len(chunk.extra or {}),
    )


class HybridRetriever(DocsRetriever):
    """
    Fuse several retrievers into one via Reciprocal Rank Fusion.

    Parameters
    ----------
    retrievers : sequence of DocsRetriever
        The legs to fuse (e.g. dense, BM25, graph).
    weights : sequence of float, optional
        Per-leg weight (default all ``1.0``). Length must match ``retrievers``.
    rrf_k : int, optional
        RRF constant (default :data:`DEFAULT_RRF_K`).
    fanout : int, optional
        Over-fetch factor: each leg is asked for ``fanout * k`` candidates so
        fusion has depth to work with (default ``4``).
    strict : bool, optional
        False.

    Notes
    -----
    Resilient: a leg that raises is skipped, not fatal — one broken backend must
    not take down retrieval. Read-only; results are sanitised downstream by
    :func:`~scikitplot.mcp._core.build_search_docs_result`.
    """

    def __init__(
        self,
        retrievers: Sequence[DocsRetriever],
        *,
        weights: Sequence[float] | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        fanout: int = 4,
        strict: bool = False,
    ) -> None:
        self._retrievers = list(retrievers)
        if not self._retrievers:
            raise ValueError("at least one retriever is required")

        if weights is None:
            weights = [1.0] * len(self._retrievers)
        if len(weights) != len(self._retrievers):
            raise ValueError("weights length must match retrievers length")

        self._weights = [
            _coerce_finite_score(weight, float("nan")) for weight in weights
        ]
        if any(not math.isfinite(weight) or weight < 0 for weight in self._weights):
            raise ValueError("weights must be finite and non-negative")
        if not any(weight > 0 for weight in self._weights):
            raise ValueError("at least one weight must be positive")

        self._rrf_k = int(rrf_k)
        if self._rrf_k < 1:
            raise ValueError("rrf_k must be >= 1")
        self._fanout = max(1, min(int(fanout), 20))
        self._strict = bool(strict)

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Query every leg, fuse by RRF, and return the top-``k`` fused chunks."""
        if not isinstance(query, str) or not query.strip():
            return []

        k = max(1, min(int(k), _MAX_RETRIEVAL_K))
        per_leg = min(_MAX_RETRIEVAL_K, max(k, self._fanout * k))

        ranked_lists: list[tuple[float, list[RetrievedChunk]]] = []
        best_chunk: dict[str, RetrievedChunk] = {}
        first_seen: dict[str, int] = {}
        sequence = 0

        for index, (retriever, weight) in enumerate(
            zip(self._retrievers, self._weights)
        ):
            if weight == 0:
                continue
            try:
                raw_hits = retriever.search(query, per_leg) or []
                if not isinstance(raw_hits, list):
                    raw_hits = list(raw_hits)
                invalid = [
                    hit for hit in raw_hits if not isinstance(hit, RetrievedChunk)
                ]
                if invalid:
                    raise TypeError(
                        f"retriever leg {index} returned {len(invalid)} non-RetrievedChunk item(s)"
                    )
                hits = _deduplicate(raw_hits)
            except Exception as exc:  # resilience boundary
                logger.warning(
                    "MCP retrieval leg %s (%s) failed and was skipped: %s",
                    index,
                    type(retriever).__name__,
                    exc,
                    exc_info=self._strict,
                )
                if self._strict:
                    raise
                continue

            ranked_lists.append((weight, hits))
            for hit in hits:
                identity = _chunk_key(hit)
                if identity not in first_seen:
                    first_seen[identity] = sequence
                    sequence += 1
                current = best_chunk.get(identity)
                if current is None or _metadata_richness(hit) > _metadata_richness(
                    current
                ):
                    best_chunk[identity] = hit

        fused = reciprocal_rank_fusion(ranked_lists, k=self._rrf_k)
        ordered = sorted(
            fused.items(),
            key=lambda item: (-item[1], first_seen.get(item[0], 10**12), item[0]),
        )[:k]
        return [
            replace(best_chunk[identity], score=score)
            for identity, score in ordered
            if identity in best_chunk
        ]


class Bm25Retriever(DocsRetriever):
    """
    Lexical retriever (FTS/BM25) leg backed by a full-text search seam.

    Parameters
    ----------
    fts_search : callable
        ``(query, k) -> list[(doc_id, score)]`` — BM25-ranked hits. In production
        this wraps ``scikitplot.corpus.SQLiteStorage`` FTS5 (whose default rank
        is BM25).
    doc_lookup : callable
        ``doc_id -> mapping`` with at least ``text`` and ``source_uri``.

    Notes
    -----
    BM25 complements dense retrieval on exact tokens — API symbols, flags, error
    messages — that embeddings often blur.
    """

    def __init__(
        self,
        fts_search: Callable[[str, int], list[tuple[str, float]]],
        doc_lookup: Callable[[str], dict[str, Any]],
        *,
        strict: bool = False,
    ) -> None:
        self._fts = fts_search
        self._lookup = doc_lookup
        self._strict = bool(strict)

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Run FTS5/BM25 and map hits to :class:`RetrievedChunk`."""
        if not isinstance(query, str) or not query.strip():
            return []
        k = max(1, min(int(k), _MAX_RETRIEVAL_K))

        try:
            hits = self._fts(query, k) or []
        except Exception as exc:
            logger.warning("BM25 search failed: %s", exc, exc_info=self._strict)
            if self._strict:
                raise
            return []

        output: list[RetrievedChunk] = []
        seen: set[str] = set()
        for item in hits:
            try:
                doc_id, score = item
                doc_id = str(doc_id)
                if not doc_id or doc_id in seen:
                    continue
                record = self._lookup(doc_id) or {}
                if not isinstance(record, dict):
                    raise TypeError("doc_lookup must return a mapping")
                text = str(record.get("text", ""))
                if not text:
                    continue
            except Exception as exc:
                logger.warning(
                    "BM25 hit mapping failed: %s", exc, exc_info=self._strict
                )
                if self._strict:
                    raise
                continue

            seen.add(doc_id)
            output.append(
                RetrievedChunk(
                    text=text,
                    source_uri=str(record.get("source_uri", "")),
                    score=_coerce_finite_score(score),
                    doc_id=doc_id,
                    title=str(record.get("title", "")),
                    anchor=str(record.get("anchor", "")),
                )
            )
        return output

    @classmethod
    def from_corpus_sqlite(
        cls,
        storage_path: str,
        *,
        strict: bool = False,
    ) -> Bm25Retriever:
        """
        Build from a corpus SQLite/FTS5 store (import-guarded).

        Wires ``scikitplot.corpus.SQLiteStorage`` FTS5 search. Raises
        :class:`RuntimeError` with an actionable message if corpus is absent.
        The exact ``query``/``StorageQuery`` field access is corpus-version
        specific — verify against the installed source (see DESIGN.md §Hybrid).
        """
        try:
            from scikitplot.corpus import SQLiteStorage, StorageQuery  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - optional integration path
            raise RuntimeError(
                "scikitplot.corpus is required for the BM25/FTS5 leg."
            ) from exc

        store = SQLiteStorage(storage_path)
        recent: dict[str, dict[str, Any]] = {}

        def _row_value(row: Any, *names: str) -> Any:
            for name in names:
                value = (
                    row.get(name) if isinstance(row, dict) else getattr(row, name, None)
                )
                if value not in (None, ""):
                    return value
            return ""

        def _record(row: Any) -> dict[str, Any]:
            return {
                "text": _row_value(row, "normalized_text", "text"),
                "source_uri": _row_value(row, "source_uri", "source", "input_path"),
                "title": _row_value(row, "title", "section_title", "section"),
                "anchor": _row_value(row, "anchor", "section_id", "heading_id"),
            }

        def _fts(query: str, k: int) -> list[tuple[str, float]]:
            result = store.query(StorageQuery(full_text=query, limit=k))
            rows = getattr(result, "documents", result) or []
            recent.clear()
            output: list[tuple[str, float]] = []
            for rank, row in enumerate(rows, start=1):
                doc_id = str(_row_value(row, "doc_id"))
                if not doc_id:
                    continue
                recent[doc_id] = _record(row)
                output.append((doc_id, 1.0 / rank))
            return output

        def _lookup(doc_id: str) -> dict[str, Any]:
            if doc_id in recent:
                return recent[doc_id]
            row = store.get(doc_id)
            return {} if row is None else _record(row)

        return cls(_fts, _lookup, strict=strict)
