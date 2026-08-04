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

from collections import defaultdict
from dataclasses import replace
from typing import Any, Callable, Sequence

from ._core import DocsRetriever, RetrievedChunk

__all__ = [
    "DEFAULT_RRF_K",
    "Bm25Retriever",
    "HybridRetriever",
    "reciprocal_rank_fusion",
]

#: Standard RRF constant; dampens the contribution of lower ranks.
DEFAULT_RRF_K: int = 60


def _chunk_key(c: RetrievedChunk) -> str:
    """Stable identity for dedup/fusion: doc_id, else (uri + text prefix)."""
    if c.doc_id:
        return c.doc_id
    return (c.source_uri or "") + "::" + (c.text or "")[:64]


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
    fused: dict[str, float] = defaultdict(float)
    for weight, items in ranked_lists:
        for rank, item in enumerate(items or [], start=1):
            fused[key(item)] += float(weight) / (k + rank)
    return dict(fused)


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
    ) -> None:
        self._retrievers = list(retrievers)
        if weights is None:
            weights = [1.0] * len(self._retrievers)
        if len(weights) != len(self._retrievers):
            raise ValueError("weights length must match retrievers length")
        self._weights = [float(w) for w in weights]
        self._rrf_k = int(rrf_k)
        self._fanout = max(1, int(fanout))

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Query every leg, fuse by RRF, and return the top-``k`` fused chunks."""
        if not isinstance(query, str) or not query.strip():
            return []
        k = max(1, int(k))
        per = max(k, self._fanout * k)

        ranked_lists: list[tuple[float, list[RetrievedChunk]]] = []
        best_chunk: dict[str, RetrievedChunk] = {}
        for retr, w in zip(self._retrievers, self._weights):
            try:
                hits = retr.search(query, per) or []
            except Exception:  # ruff: ignore[blind-except, try-except-continue]
                continue  # resilient: skip a failing leg
            ranked_lists.append((w, hits))
            for h in hits:
                key = _chunk_key(h)
                # keep the representative with the richest metadata
                cur = best_chunk.get(key)
                if cur is None or (len(h.title) + len(h.anchor)) > (
                    len(cur.title) + len(cur.anchor)
                ):
                    best_chunk[key] = h

        fused = reciprocal_rank_fusion(ranked_lists, k=self._rrf_k)
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            replace(best_chunk[key], score=score)
            for key, score in top
            if key in best_chunk
        ]


class Bm25Retriever(DocsRetriever):
    """
    Lexical (BM25) leg backed by a full-text search seam.

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
    ) -> None:
        self._fts = fts_search
        self._lookup = doc_lookup

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Run FTS5/BM25 and map hits to :class:`RetrievedChunk`."""
        if not isinstance(query, str) or not query.strip():
            return []
        k = max(1, min(int(k), 50))
        try:
            hits = self._fts(query, k) or []
        except Exception:  # ruff: ignore[blind-except]
            return []
        out: list[RetrievedChunk] = []
        for doc_id, score in hits:
            rec = {}
            try:
                rec = self._lookup(doc_id) or {}
            except Exception:  # ruff: ignore[blind-except, try-except-continue]
                continue
            out.append(
                RetrievedChunk(
                    text=str(rec.get("text", "")),
                    source_uri=str(rec.get("source_uri", "")),
                    score=float(score) if isinstance(score, (int, float)) else 0.0,
                    doc_id=str(doc_id),
                    title=str(rec.get("title", "")),
                    anchor=str(rec.get("anchor", "")),
                )
            )
        return out

    @classmethod
    def from_corpus_sqlite(cls, storage_path: str) -> Bm25Retriever:
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

        def _fts(query: str, k: int) -> list[tuple[str, float]]:
            # SQLiteStorage.query() returns a QueryResult whose ``documents``
            # are already ordered by BM25 when ``full_text`` is supplied.
            result = store.query(StorageQuery(full_text=query, limit=k))
            rows = getattr(result, "documents", result) or []
            out: list[tuple[str, float]] = []
            for rank, row in enumerate(rows, start=1):
                did = str(
                    getattr(row, "doc_id", "")
                    or (row.get("doc_id", "") if isinstance(row, dict) else "")
                )
                if not did:
                    continue
                # RRF consumes ordering rather than raw BM25 magnitudes. Use a
                # stable higher-is-better rank surrogate for RetrievedChunk.score.
                out.append((did, 1.0 / rank))
            return out

        def _lookup(doc_id: str) -> dict[str, Any]:
            row = store.get(doc_id)
            if row is None:
                return {}

            def g(name: str) -> Any:
                if isinstance(row, dict):
                    return row.get(name)
                return getattr(row, name, None)

            return {
                "text": g("normalized_text") or g("text") or "",
                "source_uri": g("source_uri") or g("source") or g("input_path") or "",
                "title": g("title") or g("section_title") or g("section") or "",
                "anchor": g("anchor") or g("section_id") or g("heading_id") or "",
            }

        return cls(_fts, _lookup)
