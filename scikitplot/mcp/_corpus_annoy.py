# scikitplot/mcp/_corpus_annoy.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Reference retriever composing :mod:`scikitplot.corpus` + :mod:`scikitplot.annoy`.

This is the flagship :class:`~scikitplot.mcp._core.DocsRetriever`: it retrieves
documentation passages using components scikit-plots already ships, so the MCP
server does not re-implement embedding, chunking, or vector search.

Composition (grounded in the public APIs of both modules)
--------------------------------------------------------
* **Ingest + embed (build time / server startup)** —
  ``scikitplot.corpus.CorpusBuilder(BuilderConfig(chunker='paragraph',
  normalize=True, enrich=True, embed=True, build_index=True)).build(docs_path)``
  reads RST / MyST / Markdown / built HTML, chunks, normalises, and embeds,
  yielding ``CorpusDocument`` objects that already carry source provenance
  (page / section) — i.e. the citation metadata.
* **Vector index** — the embedding vectors are indexed for ANN search by
  ``scikitplot.corpus.SimilarityIndex``, whose dense backend defaults to
  ``scikitplot.annoy`` (``add_item`` → ``build`` → ``get_nns_by_vector``,
  persistent and memory-mapped). The corpus index owns the backend selection
  and the cosine score contract, so this module does not re-implement vector
  search or distance-to-score maths. Hybrid keyword+vector fusion is available
  in-corpus via ``SimilarityIndex`` (BM25 + dense RRF).
* **Query time** — embed the query with the same ``EmbeddingEngine``, ask the
  index's ``query(vector, k)`` seam for top-k ``(doc_id, score)`` pairs, map
  each id back to its ``CorpusDocument`` and onto a :class:`RetrievedChunk`
  (text + source_uri + anchor + score).

To keep this testable without corpus/annoy installed, the class takes the three
seams by injection. :meth:`from_corpus_annoy` performs the real wiring behind an
import guard.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from ._core import DocsRetriever, RetrievedChunk

__all__ = ["CorpusAnnoyRetriever", "Embedder", "VectorIndex"]


class Embedder(Protocol):
    """Anything that turns a query string into a vector (e.g. corpus ``EmbeddingEngine``)."""

    def embed(self, text: str) -> Any: ...


class VectorIndex(Protocol):
    """
    Anything that returns ``(doc_id, score)`` pairs for a query vector.

    Satisfied by ``scikitplot.corpus.SimilarityIndex`` (via its ``query``
    seam), whose default dense backend is ``scikitplot.annoy``.
    """

    def query(self, vector: Any, k: int) -> list[tuple[str, float]]: ...


class CorpusAnnoyRetriever(DocsRetriever):
    """
    Docs retriever backed by an embedder + a vector index + a document lookup.

    Parameters
    ----------
    embedder : Embedder
        Query embedder (the *same* model used to embed the corpus).
    index : VectorIndex
        Vector index returning ``(doc_id, score)`` for a query vector.
    doc_lookup : callable
        ``doc_id -> mapping`` returning at least ``text`` and ``source_uri``,
        optionally ``title`` / ``anchor``. In production this reads the
        ``CorpusDocument`` from corpus storage.

    Notes
    -----
    Read-only. Retrieved text is untrusted and is sanitised downstream by
    :func:`~scikitplot.mcp._core.build_search_docs_result`.
    """

    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        doc_lookup: Callable[[str], dict[str, Any]],
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._lookup = doc_lookup

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Embed ``query``, ANN-search, and map hits to :class:`RetrievedChunk`."""
        if not isinstance(query, str) or not query.strip():
            return []
        k = max(1, min(int(k), 50))
        vector = self._embedder.embed(query)
        hits = self._index.query(vector, k) or []
        out: list[RetrievedChunk] = []
        for doc_id, score in hits:
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

    # ------------------------------------------------------------------
    @classmethod
    def from_corpus_annoy(
        cls,
        docs_path: str,
        *,
        metric: str = "angular",
        n_trees: int = 10,
        embedding_model: str = "all-MiniLM-L6-v2",
        backend: str = "annoy",
    ) -> CorpusAnnoyRetriever:
        """
        Build the real retriever from a docs directory (import-guarded).

        Wires :mod:`scikitplot.corpus` end to end. A single
        :class:`~scikitplot.corpus.CorpusBuilder` pass ingests, embeds, and
        builds an Annoy-backed :class:`~scikitplot.corpus.SimilarityIndex`
        (selected through ``index_kwargs``), and this retriever consumes that
        index's vector-level ``query`` seam directly. There is no second,
        ad-hoc Annoy index and no bespoke distance-to-score arithmetic — the
        corpus owns both the vector backend and the cosine score contract.

        Parameters
        ----------
        docs_path : str
            Directory of documentation sources (RST / MyST / Markdown / HTML).
        metric : str, optional
            Annoy metric (``'angular'`` for cosine-like on normalised vectors).
        n_trees : int, optional
            Annoy tree count (accuracy/size trade-off).
        embedding_model : str, optional
            Sentence-embedding model. The query embedder uses the *same* model,
            so query and corpus vectors share one space.
        backend : str, optional
            Dense ANN backend for the corpus index. ``'annoy'`` (default) or
            ``'auto'`` (Annoy first, then FAISS / Voyager / brute-force).

        Returns
        -------
        CorpusAnnoyRetriever

        Raises
        ------
        RuntimeError
            If :mod:`scikitplot.corpus` is unavailable, or the build produced
            no queryable semantic index (e.g. embeddings were unavailable).
        """
        try:
            from scikitplot.corpus import (  # noqa: PLC0415
                BuilderConfig,
                CorpusBuilder,
                EmbeddingEngine,
            )
        except Exception as exc:  # pragma: no cover - integration path
            raise RuntimeError(
                "scikitplot.corpus is required to build the retriever "
                "(pip install scikit-plots[corpus])."
            ) from exc

        # 1. One pass: ingest + embed + build the Annoy-backed similarity index.
        #    ``index_kwargs`` is forwarded verbatim into the corpus
        #    ``SearchConfig``, so the vector backend lives entirely in corpus.
        builder = CorpusBuilder(
            BuilderConfig(
                chunker="paragraph",
                normalize=True,
                enrich=True,
                embed=True,
                embedding_model=embedding_model,
                build_index=True,
                index_kwargs={
                    "match_mode": "semantic",
                    "backend": backend,
                    "annoy_metric": metric,
                    "annoy_n_trees": n_trees,
                },
            )
        )
        result = builder.build(docs_path)
        documents = getattr(result, "documents", result)
        index = getattr(result, "index", None)
        if index is None or not hasattr(index, "query"):
            raise RuntimeError(
                "corpus build produced no queryable semantic index; ensure "
                "embeddings are available so SimilarityIndex has a dense backend."
            )

        # 2. Query embedder — the SAME model that embedded the corpus, so query
        #    and document vectors live in one space.
        engine = EmbeddingEngine(model_name=embedding_model)

        # 3. Document lookup keyed by the doc_id the index returns.
        table: dict[str, dict[str, Any]] = {}
        for doc in documents:
            doc_id = getattr(doc, "doc_id", None)
            if doc_id is not None:
                table[str(doc_id)] = _doc_to_record(doc)

        return cls(
            _CorpusEmbedder(engine),
            _SimilarityVectorIndex(index),
            lambda did: table.get(str(did), {}),
        )


# ---- adapters over the corpus public API (kept private) ---------------------
class _CorpusEmbedder:
    """Embed one query string via a corpus ``EmbeddingEngine``.

    ``EmbeddingEngine.embed`` takes a ``list[str]`` and returns an
    ``(n, dim)`` array; a single query is row 0 of a one-element batch.
    (The previous implementation called a non-existent ``encode`` method and
    then passed a bare ``str`` to ``embed`` — both incorrect against the corpus
    ``EmbeddingEngine`` contract.)
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def embed(self, text: str) -> Any:
        vecs = self._engine.embed([text])
        return vecs[0]


class _SimilarityVectorIndex:
    """Adapt corpus ``SimilarityIndex.query`` to the :class:`VectorIndex` protocol.

    ``SimilarityIndex.query(vector, k)`` already returns ``(doc_id, score)``
    pairs with a unified cosine score, so this is a straight pass-through that
    keeps the retriever decoupled from the concrete index type.
    """

    def __init__(self, index: Any) -> None:
        self._index = index

    def query(self, vector: Any, k: int) -> list[tuple[str, float]]:
        return self._index.query(vector, k)


def _doc_to_record(doc: Any) -> dict[str, Any]:
    def g(*names):
        for n in names:
            v = getattr(doc, n, None)
            if v:
                return v
            if isinstance(doc, dict) and doc.get(n):
                return doc[n]
        return ""

    return {
        "text": g("normalized_text", "text", "content"),
        "source_uri": g("source_uri", "source", "url", "path"),
        "title": g("title", "section", "heading"),
        "anchor": g("anchor", "section_id", "fragment"),
    }
