# scikitplot/mcp/_demo.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Small, deterministic, dependency-free BM25 backend for examples and tests.

This is intentionally not a replacement for ``scikitplot.corpus``.  It is a
fully runnable reference backend that demonstrates the MCP mechanism before a
production corpus/index is connected.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ._core import DocsRetriever, RetrievedChunk

__all__ = [
    "DemoDocument",
    "InMemoryBm25Retriever",
    "builtin_demo_retriever",
]

_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:+-]+", re.UNICODE)
_DOC_ID_RE = re.compile(r"\A[A-Za-z0-9._:-]{1,200}\Z")
_MAX_JSONL_BYTES = 16 * 1024 * 1024
_MAX_DOCUMENTS = 20_000
_MAX_DOCUMENT_CHARS = 100_000
_MAX_TOTAL_CHARS = 20_000_000
_MAX_RETRIEVAL_K = 50
_MAX_SOURCE_URI_CHARS = 2048
_MAX_TITLE_CHARS = 1000
_MAX_ANCHOR_CHARS = 200


@dataclass(frozen=True)
class DemoDocument:
    """One in-memory document used by :class:`InMemoryBm25Retriever`."""

    doc_id: str
    text: str
    source_uri: str = ""
    title: str = ""
    anchor: str = ""


def _tokens(text: str) -> list[str]:
    """Tokenize while ignoring separator punctuation at token boundaries.

    The permissive token alphabet intentionally keeps internal separators for
    identifiers, module names, URLs, and canary tokens. Boundary punctuation is
    stripped so ``TOKEN.`` in prose matches a query for ``TOKEN``.
    """
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0).casefold().strip("./:+-")
        if token:
            tokens.append(token)
    return tokens


def _bounded_int(value: Any, *, low: int, high: int) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        return low
    return max(low, min(converted, high))


class InMemoryBm25Retriever(DocsRetriever):
    """A compact BM25 implementation suitable for demos and small corpora.

    The index is immutable after construction, making concurrent reads safe.
    Search complexity is linear in the number of documents; production-scale
    deployments should use FTS5, a dedicated search engine, or the hybrid
    corpus/Annoy adapters instead.
    """

    def __init__(  # ruff: ignore[too-many-branches]
        self,
        documents: Iterable[DemoDocument],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        title_weight: float = 2.0,
    ) -> None:
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be a positive finite number")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be a finite number between 0 and 1")
        if not math.isfinite(title_weight) or title_weight < 0:
            raise ValueError("title_weight must be finite and non-negative")

        table: dict[str, DemoDocument] = {}
        term_frequencies: dict[str, dict[str, float]] = {}
        lengths: dict[str, float] = {}
        document_frequency: Counter[str] = Counter()
        total_chars = 0

        for document in documents:
            if not isinstance(document, DemoDocument):
                raise TypeError("documents must contain DemoDocument instances")
            doc_id = document.doc_id.strip()
            if not _DOC_ID_RE.fullmatch(doc_id):
                raise ValueError(
                    "doc_id must be 1-200 characters using letters, digits, dot, underscore, colon, or hyphen"
                )
            if doc_id in table:
                raise ValueError(f"duplicate doc_id: {doc_id!r}")
            if not document.text.strip():
                continue
            if len(table) >= _MAX_DOCUMENTS:
                raise ValueError(f"corpus exceeds {_MAX_DOCUMENTS} documents")
            if len(document.text) > _MAX_DOCUMENT_CHARS:
                raise ValueError(
                    f"document {doc_id!r} exceeds {_MAX_DOCUMENT_CHARS} text characters"
                )
            if len(document.source_uri) > _MAX_SOURCE_URI_CHARS:
                raise ValueError(f"document {doc_id!r} source_uri is too long")
            if len(document.title) > _MAX_TITLE_CHARS:
                raise ValueError(f"document {doc_id!r} title is too long")
            if len(document.anchor) > _MAX_ANCHOR_CHARS:
                raise ValueError(f"document {doc_id!r} anchor is too long")
            total_chars += (
                len(document.text)
                + len(document.source_uri)
                + len(document.title)
                + len(document.anchor)
            )
            if total_chars > _MAX_TOTAL_CHARS:
                raise ValueError(f"corpus exceeds {_MAX_TOTAL_CHARS} total characters")

            body = _tokens(document.text)
            title = _tokens(document.title)
            frequencies: dict[str, float] = {
                token: float(count) for token, count in Counter(body).items()
            }
            for token, count in Counter(title).items():
                frequencies[token] = frequencies.get(token, 0.0) + title_weight * count

            table[doc_id] = document
            term_frequencies[doc_id] = frequencies
            lengths[doc_id] = max(1.0, sum(frequencies.values()))
            document_frequency.update(
                term for term, value in frequencies.items() if value > 0
            )

        if not table:
            raise ValueError("at least one non-empty document is required")

        self._documents = table
        self._term_frequencies = term_frequencies
        self._lengths = lengths
        self._document_frequency = document_frequency
        self._document_count = len(table)
        self._average_length = sum(lengths.values()) / self._document_count
        self._k1 = float(k1)
        self._b = float(b)

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        if not isinstance(query, str) or not query.strip():
            return []
        query_terms = Counter(_tokens(query))
        if not query_terms:
            return []
        limit = _bounded_int(k, low=1, high=_MAX_RETRIEVAL_K)

        scored: list[tuple[float, str]] = []
        for doc_id, frequencies in self._term_frequencies.items():
            score = 0.0
            doc_len = self._lengths[doc_id]
            length_norm = 1.0 - self._b + self._b * (doc_len / self._average_length)
            for term, query_frequency in query_terms.items():
                term_frequency = frequencies.get(term, 0)
                if term_frequency <= 0:
                    continue
                df = self._document_frequency[term]
                inverse_document_frequency = math.log(
                    1.0 + (self._document_count - df + 0.5) / (df + 0.5)
                )
                numerator = term_frequency * (self._k1 + 1.0)
                denominator = term_frequency + self._k1 * length_norm
                score += (
                    inverse_document_frequency
                    * numerator
                    / denominator
                    * query_frequency
                )
            if score > 0 and math.isfinite(score):
                scored.append((score, doc_id))

        scored.sort(key=lambda item: (-item[0], item[1]))
        output: list[RetrievedChunk] = []
        for score, doc_id in scored[:limit]:
            document = self._documents[doc_id]
            output.append(
                RetrievedChunk(
                    text=document.text,
                    source_uri=document.source_uri,
                    score=score,
                    doc_id=document.doc_id,
                    title=document.title,
                    anchor=document.anchor,
                )
            )
        return output

    def get(self, doc_id: str) -> RetrievedChunk | None:
        """Return a document by stable identifier for the MCP resource layer."""
        document = self._documents.get(str(doc_id))
        if document is None:
            return None
        return RetrievedChunk(
            text=document.text,
            source_uri=document.source_uri,
            score=0.0,
            doc_id=document.doc_id,
            title=document.title,
            anchor=document.anchor,
        )

    @classmethod
    def from_jsonl(cls, path: str | os.PathLike[str]) -> InMemoryBm25Retriever:
        """Load bounded UTF-8 JSON Lines records.

        Each line must contain ``doc_id`` and ``text``. Optional fields are
        ``source_uri``, ``title``, and ``anchor``. The loader rejects duplicate
        IDs and oversized inputs instead of partially accepting ambiguous data.
        """
        source = Path(path)
        stat = source.stat()
        if not source.is_file():
            raise ValueError(f"not a regular file: {source}")
        if stat.st_size > _MAX_JSONL_BYTES:
            raise ValueError(f"JSONL file exceeds {_MAX_JSONL_BYTES} bytes")

        documents: list[DemoDocument] = []
        total_chars = 0
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                if len(documents) >= _MAX_DOCUMENTS:
                    raise ValueError(f"JSONL input exceeds {_MAX_DOCUMENTS} documents")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON on line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(  # ruff: ignore[type-check-without-type-error]
                        f"line {line_number} must be a JSON object"
                    )

                doc_id = str(record.get("doc_id", "")).strip()
                text = str(record.get("text", ""))
                if not doc_id or not text.strip():
                    raise ValueError(
                        f"line {line_number} requires non-empty doc_id and text"
                    )
                if len(text) > _MAX_DOCUMENT_CHARS:
                    raise ValueError(
                        f"line {line_number} text exceeds {_MAX_DOCUMENT_CHARS} characters"
                    )
                total_chars += len(text)
                if total_chars > _MAX_TOTAL_CHARS:
                    raise ValueError(
                        f"JSONL text exceeds {_MAX_TOTAL_CHARS} total characters"
                    )

                documents.append(
                    DemoDocument(
                        doc_id=doc_id,
                        text=text,
                        source_uri=str(record.get("source_uri", "")),
                        title=str(record.get("title", "")),
                        anchor=str(record.get("anchor", "")),
                    )
                )
        return cls(documents)


def builtin_demo_retriever() -> InMemoryBm25Retriever:
    """Return a tiny corpus that explains the sample's own mechanism."""
    return InMemoryBm25Retriever(
        [
            DemoDocument(
                doc_id="mechanism",
                title="MCP retrieval mechanism",
                source_uri="https://modelcontextprotocol.io/specification/2026-07-28/server/tools",
                anchor="overview",
                text=(
                    "The client discovers the search_docs tool, validates its JSON schema, "
                    "and calls it with a query and result limit. The MCP server validates "
                    "the input, invokes a DocsRetriever, converts RetrievedChunk objects "
                    "into bounded passages and citations, and returns structured output."
                ),
            ),
            DemoDocument(
                doc_id="security",
                title="Security boundary",
                source_uri="https://modelcontextprotocol.io/specification/2026-07-28/server/tools",
                anchor="security-considerations",
                text=(
                    "Retrieved documents are untrusted reference data. Character cleaning "
                    "and length caps prevent protocol corruption and prompt stuffing, but "
                    "they do not remove semantic prompt injection. Keep retrieved text in a "
                    "clearly labelled data field, validate links, rate limit calls, enforce "
                    "access control for remote servers, and audit tool use."
                ),
            ),
            DemoDocument(
                doc_id="transport",
                title="Transport choices",
                source_uri="https://modelcontextprotocol.io/specification/2026-07-28/basic/transports",
                anchor="overview",
                text=(
                    "Use stdio for a local client-spawned subprocess. Use Streamable HTTP "
                    "for remote or hosted deployment. Bind to localhost by default. Remote "
                    "deployment needs TLS, authentication, authorization, origin and host "
                    "validation, request size limits, quotas, and observability."
                ),
            ),
            DemoDocument(
                doc_id="backends",
                title="Backend evolution",
                source_uri="https://scikit-plots.github.io/dev/apis/scikitplot.corpus.html",
                text=(
                    "Start with the in-memory BM25 backend to verify MCP wiring. Then replace "
                    "it with Bm25Retriever over SQLite FTS5, CorpusAnnoyRetriever for dense "
                    "semantic retrieval, or HybridRetriever for reciprocal-rank fusion. The "
                    "server depends only on the DocsRetriever protocol, so transports do not "
                    "change when the backend evolves."
                ),
            ),
            DemoDocument(
                doc_id="scikitplot-canary-001",
                title="Scikit-plots MCP integration canary",
                source_uri="https://scikit-plots.github.io/dev/apis/scikitplot.corpus.html",
                text="The unique verification token is MCP_CANARY_7F3A91C2.",
                anchor="integration-canary",
            ),
        ]
    )
