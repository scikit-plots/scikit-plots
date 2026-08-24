# scikitplot/mcp/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Composable documentation retrieval for Model Context Protocol servers.

The default import surface is **independent of the MCP SDK, pydantic, and the
optional corpus/vector dependencies**. Importing :mod:`scikitplot.mcp` pulls only
the SDK-free retrieval core (contracts, fusion, the in-memory demo retriever) and
the pydantic-free capability/runtime-status helpers, so it works on the Legacy
Retrieval tier (Python 3.8+) with a base install.

The server layer (``SearchService``, ``create_server``, and the pydantic output
models ``SearchDocsOutput`` / ``CitationOutput``) is imported **lazily** on first
access. It requires the ``[mcp]`` extra (``pydantic`` always; ``mcp>=2.0.0,<3`` on
Python >= 3.10). No models, indexes, files, or network connections are opened at
import time.

.. seealso::
  * https://github.com/modelcontextprotocol/python-sdk
  * https://github.com/semantica-agi/semantica
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Eager: pydantic-free, dependency-light modules (Legacy Retrieval tier).
from . import _capabilities, _core, _corpus_annoy, _demo, _hybrid, _version
from ._capabilities import *  # noqa: F403
from ._core import *  # noqa: F403
from ._corpus_annoy import *  # noqa: F403
from ._demo import *  # noqa: F403
from ._hybrid import *  # noqa: F403
from ._version import *  # noqa: F403

# Names served lazily from ``_server`` (which imports pydantic, and — only inside
# ``create_server`` — the MCP SDK). Listed here so they appear in ``__all__`` and
# resolve via module ``__getattr__`` without importing pydantic at package import.
_SERVER_EXPORTS = (
    "CitationOutput",
    "SearchDocsOutput",
    "SearchService",
    "create_server",
)

__all__ = []
__all__ += _capabilities.__all__
__all__ += _core.__all__
__all__ += _corpus_annoy.__all__
__all__ += _demo.__all__
__all__ += _hybrid.__all__
__all__ += _version.__all__
__all__ += list(_SERVER_EXPORTS)

if TYPE_CHECKING:  # for type checkers/IDEs only; not executed at runtime
    from ._server import (  # noqa: F401
        CitationOutput,
        SearchDocsOutput,
        SearchService,
        create_server,
    )


def __getattr__(name: str) -> Any:
    """
    PEP 562 lazy access for the pydantic-backed server exports.

    Accessing e.g. ``scikitplot.mcp.SearchService`` imports ``_server`` (and thus
    ``pydantic``) only on first use; a base install can ``import scikitplot.mcp``
    and use the retrieval core without pydantic installed.
    """
    if name in _SERVER_EXPORTS:
        from . import _server  # ruff: ignore[import-outside-top-level]

        return getattr(_server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
