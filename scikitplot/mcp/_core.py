# scikitplot/mcp/_core.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
SDK-agnostic core for :mod:`scikitplot.mcp`.

This is the part of the MCP server that does *not* depend on any particular MCP
transport or SDK: the retrieval contract, the retrieved-chunk type, and the
function that turns retrieval results into an MCP ``tools/call`` response with
**source citations** and **injection-safe** text.

Design rationale
----------------
The heavy pieces already exist elsewhere in scikit-plots and must NOT be
re-implemented here (DRY):

* **Retrieval** — :mod:`scikitplot.corpus` ingests / chunks / embeds documents
  and offers ``SimilarityIndex`` + ``SQLiteStorage`` (FTS5) search; and/or
  :mod:`scikitplot.annoy` provides an approximate-nearest-neighbour vector
  index. A concrete retriever composes those (see ``_corpus_annoy.py``).
* **MCP formatting** — :mod:`scikitplot.corpus` already exposes
  ``to_mcp_tool_result`` / ``to_mcp_resources``. When a real corpus result is
  in hand, prefer those. :func:`build_search_docs_result` here is the
  transport-neutral fallback used by the server layer and by tests, and it is
  the single place that enforces citation shape + text safety.

Keeping this layer SDK-agnostic means the same core is testable without the MCP
SDK installed, and portable across stdio / HTTP-SSE transports (wired in the
server layer, gated — see ``_maintenance/MCP_MODULE_DESIGN.md``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

__all__ = [
    "MAX_CHUNK_CHARS",
    "MAX_RESULTS",
    "DocsRetriever",
    "RetrievedChunk",
    "build_search_docs_result",
]

#: Hard cap on the characters of any single chunk placed into a tool result.
#: Retrieved document text is UNTRUSTED (it is corpus content, possibly
#: user-contributed); capping bounds prompt-stuffing and keeps responses within
#: client context limits.
MAX_CHUNK_CHARS: int = 4000

#: Hard cap on results returned in one tool call.
MAX_RESULTS: int = 20

#: Control characters to strip from untrusted chunk text before it enters a
#: JSON-RPC payload (keep normal whitespace: tab, LF, CR).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

#: URL schemes allowed in a citation link.
_SAFE_URL_SCHEMES = frozenset({"http", "https", ""})


@dataclass(frozen=True)
class RetrievedChunk:
    """
    One retrieved passage with the metadata needed to cite it.

    This is the boundary type between retrieval (corpus / annoy) and the MCP
    tool layer. A concrete retriever maps its native result (e.g. a
    ``scikitplot.corpus.SearchResult`` / ``CorpusDocument`` or a
    ``scikitplot.annoy`` neighbour) onto this shape.

    Parameters
    ----------
    text : str
        The passage text. Treated as untrusted; truncated and control-stripped
        before entering a tool result.
    source_uri : str
        Where the passage came from (page URL or path). Used to build the
        citation link; validated to an http(s)/relative scheme.
    score : float
        Retrieval score (higher = more relevant). Used for ordering only.
    doc_id : str, optional
        Stable identifier of the chunk (for ``resources/read`` follow-ups).
    title : str, optional
        Human-readable source title (e.g. page or section heading).
    anchor : str, optional
        In-page anchor / section id so the citation deep-links to the section.
    """

    text: str
    source_uri: str
    score: float = 0.0
    doc_id: str = ""
    title: str = ""
    anchor: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DocsRetriever(Protocol):
    """
    Structural contract every retrieval backend must satisfy.

    Implemented by the corpus+annoy adapter (``_corpus_annoy.py``) and by test
    doubles. Keeping it a :class:`typing.Protocol` means backends need not
    import this module or subclass anything.

    Methods
    -------
    search(query, k)
        Return up to ``k`` :class:`RetrievedChunk` for ``query``, best first.
    """

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]: ...


def _safe_uri(uri: str) -> str:
    """
    Return ``uri`` if its scheme is http(s) or relative, else ``''``.

    Prevents a poisoned corpus record from smuggling a ``javascript:`` /
    ``data:`` link into a citation. Mirrors the widget's ``_isSafeHref`` policy.
    """
    if not isinstance(uri, str) or not uri:
        return ""
    try:
        scheme = urlparse(uri).scheme.lower()
    except Exception:  # ruff: ignore[blind-except]
        return ""
    return uri if scheme in _SAFE_URL_SCHEMES else ""


def _clean_text(text: str, limit: int = MAX_CHUNK_CHARS) -> str:
    """Strip control chars and truncate untrusted chunk text."""
    if not isinstance(text, str):
        text = str(text)
    text = _CONTROL_RE.sub("", text)
    if len(text) > limit:
        text = text[:limit].rstrip() + "\u2026"
    return text


def build_search_docs_result(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    max_results: int = MAX_RESULTS,
) -> dict[str, Any]:
    """
    Format retrieval results as an MCP ``tools/call`` response with citations.

    Parameters
    ----------
    query : str
        The original query (echoed in structured output for traceability).
    chunks : list of RetrievedChunk
        Retrieval results, best first.
    max_results : int, optional
        Upper bound on results actually emitted (also capped by
        :data:`MAX_RESULTS`).

    Returns
    -------
    dict
        An MCP tool-result object::

            {
                "content": [{"type": "text", "text": ...}, ...],
                "structuredContent": {"query": ..., "citations": [...]},
                "isError": False,
            }

        The ``content`` blocks are human/model-readable, each carrying its
        citation marker; ``structuredContent.citations`` is the machine-usable
        list (source_uri already scheme-validated). Untrusted chunk text is
        control-stripped and length-capped.

    Notes
    -----
    * **Read-only.** ``search_docs`` needs no user confirmation. Any *sensitive*
      or write tool added later MUST require explicit confirmation per MCP
      guidance — see the design doc; this function is not that path.
    * Retrieved text is explicitly untrusted data: it is sanitised here, and the
      server layer marks it so the model treats it as context, not instructions.
    """
    safe = []
    for c in (chunks or [])[: max(0, min(max_results, MAX_RESULTS))]:
        uri = _safe_uri(c.source_uri)
        safe.append(
            {
                "text": _clean_text(c.text),
                "source_uri": uri,
                "title": _clean_text(c.title, 200),
                "anchor": _clean_text(c.anchor, 200),
                "doc_id": _clean_text(c.doc_id, 200),
                "score": float(c.score) if isinstance(c.score, (int, float)) else 0.0,
            }
        )

    if not safe:
        return {
            "content": [
                {
                    "type": "text",
                    "text": "No matching documentation was found for this query.",
                }
            ],
            "structuredContent": {"query": query, "citations": []},
            "isError": False,
        }

    content_blocks = []
    citations = []
    for i, s in enumerate(safe, start=1):
        link = s["source_uri"]
        if link and s["anchor"]:
            link = link + ("#" + s["anchor"] if "#" not in link else "")
        header = f"[{i}] {s['title'] or s['doc_id'] or 'source'}"
        cite_line = f"\u2014 {link}" if link else "\u2014 (no link)"
        content_blocks.append(
            {"type": "text", "text": f"{header}\n{s['text']}\n{cite_line}"}
        )
        citations.append(
            {
                "n": i,
                "source_uri": link,
                "title": s["title"],
                "anchor": s["anchor"],
                "doc_id": s["doc_id"],
                "score": s["score"],
            }
        )

    return {
        "content": content_blocks,
        "structuredContent": {"query": query, "citations": citations},
        "isError": False,
    }
