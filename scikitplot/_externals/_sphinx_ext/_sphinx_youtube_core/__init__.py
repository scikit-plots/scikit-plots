"""
Dependency-free YouTube provider primitives shared by Sphinx adapters.

This package owns provider grammar and leaf-player option validation only.
It intentionally imports neither Sphinx nor docutils, so catalog tooling,
the gallery adapter, and the standalone player can share one contract
without creating an extension dependency cycle.
"""

from __future__ import annotations

__all__: list[str] = []
