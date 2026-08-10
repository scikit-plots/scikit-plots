# scikitplot/mcp/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Composable documentation retrieval for Model Context Protocol servers.

The default import surface remains independent of the MCP SDK and optional
corpus/vector dependencies. Call :func:`create_server` only when the stable ``mcp>=2,<3``
server dependency is installed. No models, indexes, files, or network connections are
opened at import time.

.. seealso::
  * https://github.com/modelcontextprotocol/modelcontextprotocol
"""

from __future__ import annotations

from . import (
    _core,
    _corpus_annoy,
    _demo,
    _hybrid,
    _server,
    _version,
)
from ._core import *  # noqa: F403
from ._corpus_annoy import *  # noqa: F403
from ._demo import *  # noqa: F403
from ._hybrid import *  # noqa: F403
from ._server import *  # noqa: F403
from ._version import *  # noqa: F403

__all__ = []
__all__ += _core.__all__
__all__ += _corpus_annoy.__all__
__all__ += _demo.__all__
__all__ += _hybrid.__all__
__all__ += _server.__all__
__all__ += _version.__all__
