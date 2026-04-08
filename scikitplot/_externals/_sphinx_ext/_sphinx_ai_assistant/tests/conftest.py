# tests/_externals/_sphinx_ext/_sphinx_ai_assistant/conftest.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Pytest configuration and shared fixtures for
``scikitplot._externals._sphinx_ext._sphinx_ai_assistant`` tests.

All heavy dependencies (Sphinx, BeautifulSoup, markdownify) are mocked
at the session level so that the test suite can run in environments where
only the standard library is available.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Sphinx mock builder / app helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides: Any) -> MagicMock:
    """Return a minimal mock of a Sphinx Config object.

    Parameters
    ----------
    **overrides : Any
        Any attribute to set on the config mock.

    Returns
    -------
    unittest.mock.MagicMock
        A mock with sensible defaults for all config values the extension reads.
    """
    cfg = MagicMock()
    cfg.ai_assistant_enabled = True
    cfg.ai_assistant_position = "sidebar"
    cfg.ai_assistant_content_selector = "article"
    cfg.ai_assistant_content_selectors = [
        "article.bd-article",
        'div[role="main"]',
        'article[role="main"]',
    ]
    cfg.ai_assistant_generate_markdown = True
    cfg.ai_assistant_markdown_exclude_patterns = ["genindex", "search", "py-modindex"]
    cfg.ai_assistant_generate_llms_txt = True
    cfg.ai_assistant_base_url = ""
    cfg.ai_assistant_max_workers = 1
    cfg.ai_assistant_features = {
        "markdown_export": True,
        "view_markdown": True,
        "ai_chat": True,
        "mcp_integration": False,
    }
    cfg.ai_assistant_providers = {}
    cfg.ai_assistant_mcp_tools = {}
    cfg.html_baseurl = ""
    cfg.html_static_path = []
    cfg.project = "TestProject"
    for key, val in overrides.items():
        setattr(cfg, key, val)
    return cfg


def _make_builder(outdir: str) -> MagicMock:
    """Return a mock StandaloneHTMLBuilder.

    Parameters
    ----------
    outdir : str
        Path to the mock build output directory.

    Returns
    -------
    unittest.mock.MagicMock
    """
    from sphinx.builders.html import StandaloneHTMLBuilder  # real class for isinstance
    builder = MagicMock(spec=StandaloneHTMLBuilder)
    builder.outdir = outdir
    return builder


def _make_app(outdir: str, **config_overrides: Any) -> MagicMock:
    """Return a mock Sphinx application.

    Parameters
    ----------
    outdir : str
        Build output directory path.
    **config_overrides : Any
        Forwarded to :func:`_make_config`.

    Returns
    -------
    unittest.mock.MagicMock
    """
    app = MagicMock()
    app.config = _make_config(**config_overrides)
    app.builder = _make_builder(outdir)
    return app


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_html_tree(tmp_path: Path):
    """Create a minimal HTML output tree for integration-style tests.

    Returns
    -------
    pathlib.Path
        Root of the temporary output directory containing sample HTML files.
    """
    outdir = tmp_path / "html"
    outdir.mkdir()
    (outdir / "index.html").write_text(
        '<html><body><article role="main"><h1>Hello</h1><p>World</p></article></body></html>',
        encoding="utf-8",
    )
    (outdir / "genindex.html").write_text(
        '<html><body><div role="main">Index</div></body></html>',
        encoding="utf-8",
    )
    sub = outdir / "api"
    sub.mkdir()
    (sub / "module.html").write_text(
        '<html><body><article class="bd-article">API docs</article></body></html>',
        encoding="utf-8",
    )
    return outdir


@pytest.fixture()
def sphinx_app(tmp_html_tree: Path) -> MagicMock:
    """Mock Sphinx app wired to the tmp_html_tree output directory."""
    return _make_app(str(tmp_html_tree))


# Re-export helpers so tests can import from conftest
__all__ = [
    "_make_config",
    "_make_builder",
    "_make_app",
]
