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

import logging
from typing import Any, Callable, Protocol

from ._core import DocsRetriever, RetrievedChunk, _coerce_finite_score
from ._outcome import DEGRADED, EMPTY, FAILED, SUCCESS, LegRecord, RetrievalOutcome

__all__ = [
    "CorpusAnnoyRetriever",
    "Embedder",
    "VectorIndex",
]

logger = logging.getLogger(__name__)
_MAX_RETRIEVAL_K = 50
#: Evidence-path label reported in :class:`~scikitplot.mcp._outcome.LegRecord`.
_DENSE_LEG = "dense"
#: Cap on how many per-hit mapping errors are joined into one leg explanation,
#: so a pathological corpus cannot produce an unbounded diagnostic string.
_MAX_LEG_ERRORS = 3


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


# ---- adapters over the corpus public API (kept private) ---------------------
class _BatchQueryEmbedder:
    """Adapt a batch embedder/callable to the MCP single-query protocol."""

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder

    def embed(self, text: str) -> Any:
        batch = getattr(self._embedder, "embed", None)
        if batch is None:
            batch = self._embedder
        vectors = batch([text])
        if vectors is None or len(vectors) != 1:
            raise ValueError("embedder must return one vector for one query")
        return vectors[0]


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
        vectors = self._engine.embed([text])
        if vectors is None or len(vectors) != 1:
            raise ValueError("EmbeddingEngine.embed must return one vector")
        return vectors[0]


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


def _value(document: Any, name: str) -> Any:
    if isinstance(document, dict):
        return document.get(name)
    return getattr(document, name, None)


def _first(document: Any, *names: str) -> Any:
    for name in names:
        value = _value(document, name)
        if value not in (None, ""):
            return value
    return ""


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
        # "text": _first(doc, "normalized_text", "text", "content"),
        "text": g("normalized_text", "text", "content"),
        "source_uri": g("source_uri", "input_path", "source", "url", "path"),
        "title": g("source_title", "title", "section", "heading"),
        "anchor": g("anchor", "section_id", "fragment"),
    }


class CorpusAnnoyRetriever(DocsRetriever):
    """
    Docs dense retriever backed by an embedder + a vector index + a document lookup.

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
        *,
        strict: bool = False,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._lookup = doc_lookup
        self._strict = bool(strict)

    def get(self, doc_id: str) -> RetrievedChunk | None:
        """Return one indexed document by stable id for the MCP resource surface."""
        if not isinstance(doc_id, str) or not doc_id:
            return None
        record = self._lookup(doc_id) or {}
        if not isinstance(record, dict):
            if self._strict:
                raise TypeError("doc_lookup must return a mapping")
            return None
        text = str(record.get("text", ""))
        if not text:
            return None
        return RetrievedChunk(
            text=text,
            source_uri=str(record.get("source_uri", "")),
            score=0.0,
            doc_id=doc_id,
            title=str(record.get("title", "")),
            anchor=str(record.get("anchor", "")),
            extra=dict(record.get("extra", {}) or {}),
        )

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Embed ``query``, ANN-search, and map hits to :class:`RetrievedChunk`."""
        if not isinstance(query, str) or not query.strip():
            return RetrievalOutcome([], legs=[LegRecord(_DENSE_LEG, EMPTY)])
        k = max(1, min(int(k), _MAX_RETRIEVAL_K))

        try:
            vector = self._embedder.embed(query)
            if vector is None:
                raise ValueError("embedder returned no query vector")
            hits = self._index.query(vector, k) or []
        except Exception as exc:
            logger.warning("Dense retrieval failed: %s", exc, exc_info=self._strict)
            if self._strict:
                raise
            # M04: the dense leg did not run. Returning a bare [] here would be
            # indistinguishable from "the corpus holds no match".
            return RetrievalOutcome(
                [], legs=[LegRecord(_DENSE_LEG, FAILED, error=str(exc))]
            )

        output: list[RetrievedChunk] = []
        seen: set[str] = set()
        mapping_errors: list[str] = []
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
                    "Dense hit mapping failed: %s", exc, exc_info=self._strict
                )
                if self._strict:
                    raise
                # M04: a dropped hit is a partial loss, recorded so the caller
                # can tell a thin result from a complete one.
                mapping_errors.append(str(exc))
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
                    extra=dict(record.get("extra", {}) or {}),
                )
            )
        if mapping_errors:
            leg = LegRecord(
                _DENSE_LEG,
                DEGRADED,
                hit_count=len(output),
                error="; ".join(mapping_errors[:_MAX_LEG_ERRORS]),
            )
        else:
            leg = LegRecord(
                _DENSE_LEG,
                SUCCESS if output else EMPTY,
                hit_count=len(output),
            )
        return RetrievalOutcome(output, legs=[leg])

    # ------------------------------------------------------------------
    @classmethod
    def from_corpus_annoy(
        cls,
        docs_path: str,
        *,
        metric: str = "angular",
        n_trees: int = 10,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedder: Any | None = None,
        backend: str = "annoy",
        strict: bool = False,
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
            Sentence-embedding model used when *embedder* is ``None``. The query
            embedder uses the same model so query and corpus vectors share one
            space.
        embedder : callable or object with ``embed(list[str])``, optional
            Explicit local batch embedder. When provided, the corpus is ingested
            without model embeddings, this callable supplies document vectors,
            and the same callable embeds queries. This is suitable for
            deterministic/offline helpers such as
            :class:`scikitplot.corpus.HashEmbedder`.
        backend : str, optional
            Dense ANN backend for the corpus index. ``'annoy'`` (default) or
            ``'auto'`` (Annoy first, then FAISS / Voyager / brute-force).
        strict : bool, optional
            False.

        Returns
        -------
        CorpusAnnoyRetriever

        Raises
        ------
        RuntimeError
            If :mod:`scikitplot.corpus` is unavailable, or the build produced
            no queryable semantic index (e.g. embeddings were unavailable).
        ValueError
            Propagated from :class:`~scikitplot.corpus.CorpusBuilder` when
            *docs_path* contains no valid input sources.
        """
        try:
            from scikitplot.corpus import (  # noqa: PLC0415
                BuilderConfig,
                CorpusBuilder,
                EmbeddingEngine,
            )
        except ImportError as exc:  # pragma: no cover - integration path
            # MCP-D05: ImportError only. Catching Exception reported a BROKEN
            # corpus (installed but failing to import) as ABSENT, telling the
            # user to install a package that is already present.
            raise RuntimeError(
                "scikitplot.corpus is required to build the retriever; install "
                "the corpus/embedding extras (pip install scikit-plots[corpus])."
            ) from exc
        except Exception as exc:  # pragma: no cover - integration path
            # MCP-M00-08 / MCP-D05: BROKEN is not ABSENT.
            raise RuntimeError(
                "scikitplot.corpus is installed but failed to import; this is a "
                "broken installation, not a missing one. Original error: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if embedder is None:
            # Model-backed compatibility path. CorpusBuilder owns embedding and
            # index construction exactly as before.
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
                        "annoy_n_trees": int(n_trees),
                    },
                )
            )
            result = builder.build(docs_path)
            documents = list(getattr(result, "documents", result))
            index = getattr(result, "index", None)
            if index is None or not hasattr(index, "query"):
                raise RuntimeError(
                    "corpus build produced no queryable semantic index; ensure "
                    "embeddings are available so RetrievalIndex has a dense backend."
                )
            query_embedder: Embedder = _CorpusEmbedder(
                getattr(result, "embedding_engine", None)
                or getattr(builder, "embedding_engine", None)
                or EmbeddingEngine(model_name=embedding_model)
            )
        else:
            # Deterministic/local path. Ingest first, then apply the caller's
            # batch embedder and let Corpus own the selected vector backend.
            from scikitplot.corpus import (  # noqa: PLC0415
                RetrievalConfig,
                RetrievalIndex,
            )

            builder = CorpusBuilder(
                BuilderConfig(
                    chunker="paragraph",
                    normalize=True,
                    enrich=False,
                    embed=False,
                    build_index=False,
                )
            )
            result = builder.build(docs_path)
            raw_documents = list(getattr(result, "documents", result))
            if not raw_documents:
                raise RuntimeError("corpus build produced no retrievable documents")

            batch = getattr(embedder, "embed", None) or embedder
            texts = [
                str(getattr(doc, "normalized_text", None) or getattr(doc, "text", ""))
                for doc in raw_documents
            ]
            vectors = batch(texts)
            if vectors is None or len(vectors) != len(raw_documents):
                raise RuntimeError(
                    "custom embedder must return one vector per corpus document"
                )
            documents = [
                doc.replace(embedding=vectors[index])
                for index, doc in enumerate(raw_documents)
            ]
            backend_kwargs = (
                {"metric": metric, "n_trees": int(n_trees)}
                if backend in {"annoy", "auto"}
                else {}
            )
            index = RetrievalIndex(
                RetrievalConfig(
                    match_mode="semantic",
                    backend=backend,
                    index_kwargs=backend_kwargs,
                )
            )
            index.build(documents)
            if not index.has_embeddings or not hasattr(index, "query"):
                raise RuntimeError(
                    "corpus build produced no queryable semantic index; ensure "
                    "the requested vector backend is installed and usable."
                )
            query_embedder = _BatchQueryEmbedder(embedder)

        # Document lookup keyed by the doc_id the index returns.
        table: dict[str, dict[str, Any]] = {}
        for doc in documents:
            doc_id = _value(doc, "doc_id")
            if doc_id not in (None, ""):
                record = _doc_to_record(doc)
                if record["text"]:
                    table[str(doc_id)] = record

        if not table:
            raise RuntimeError("corpus build produced no retrievable documents")

        return cls(
            query_embedder,
            _SimilarityVectorIndex(index),
            lambda doc_id: table.get(str(doc_id), {}),
            strict=strict,
        )
