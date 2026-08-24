# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Read-only scikit-plots documentation toolkit for agent frameworks.

The core :class:`ScikitplotDocsToolkit` is framework-neutral and fully usable on
its own: it wraps :class:`~scikitplot.mcp.SearchCoordinator`, the Tier-L
orchestration, so it needs neither the MCP SDK, ``pydantic``, nor any agent
framework, and works on the Legacy Retrieval tier (Python 3.8+) where the
retriever's own dependencies permit. :func:`build_agno_toolkit`
adapts it to Agno (imported lazily).

Passages returned here are untrusted reference content, never instructions; callers
should cite ``source_uri`` values.
"""

from __future__ import annotations

from typing import Any


class ScikitplotDocsToolkit:
    """
    Framework-neutral, read-only documentation search.

    Parameters
    ----------
    retriever : DocsRetriever, optional
        Retrieval backend. Defaults to the dependency-free in-memory demo retriever.
    k_default : int, optional
        Default number of passages to return (default 5).

    Notes
    -----
    Backed by :class:`scikitplot.mcp.SearchCoordinator` (Legacy Retrieval tier):
    neither the MCP SDK nor ``pydantic`` is imported.
    """

    def __init__(self, retriever: Any | None = None, *, k_default: int = 5) -> None:
        from scikitplot.mcp._core import (  # ruff: ignore[import-outside-top-level]
            SearchCoordinator,
        )

        if retriever is None:
            from scikitplot.mcp._demo import (  # ruff: ignore[import-outside-top-level]
                builtin_demo_retriever,
            )

            retriever = builtin_demo_retriever()
        self._service = SearchCoordinator(retriever)
        self._k_default = int(k_default)

    def search_docs(self, query: str, k: int | None = None) -> dict:
        """
        Search scikit-plots documentation and return cited passages (read-only).

        Returns
        -------
        dict
            ``{query, count, passages, citations, message}`` with the invariant
            ``count == len(passages) == len(citations)``.
        """
        out = self._service.search(query, self._k_default if k is None else int(k))
        structured = out["structuredContent"]
        return {
            "query": structured["query"],
            "count": structured["count"],
            "passages": list(structured["passages"]),
            "citations": [dict(c) for c in structured["citations"]],
            "message": structured.get("message"),
        }


def build_agno_toolkit(retriever: Any | None = None, *, k_default: int = 5) -> Any:
    """
    Return an Agno ``Toolkit`` exposing read-only ``search_docs``.

    Requires the optional ``agno`` package; raises :class:`ImportError` with an
    install hint if it is absent. The underlying search is identical to
    :class:`ScikitplotDocsToolkit`.
    """
    try:
        from agno.tools import (  # ruff: ignore[import-outside-top-level] # type: ignore[]
            Toolkit,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The Agno integration requires the 'agno' package: pip install agno"
        ) from exc

    core = ScikitplotDocsToolkit(retriever, k_default=k_default)
    toolkit = Toolkit(name="scikitplot_docs")
    toolkit.register(core.search_docs)
    return toolkit
