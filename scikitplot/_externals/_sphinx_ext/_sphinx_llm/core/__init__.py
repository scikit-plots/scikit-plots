# SPDX-License-Identifier: BSD-3-Clause
"""Canonical static-artifact contracts for ``_sphinx_llm``."""

from .artifacts import make_manifest, make_provenance
from .index import IndexPage, render_llms_index
from .routing import discovery_hrefs, html_path, markdown_routes

__all__ = [
    "IndexPage",
    "discovery_hrefs",
    "html_path",
    "make_manifest",
    "make_provenance",
    "markdown_routes",
    "render_llms_index",
]
