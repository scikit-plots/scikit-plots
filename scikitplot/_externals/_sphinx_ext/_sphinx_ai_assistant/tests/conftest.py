# tests/_externals/_sphinx_ext/_sphinx_ai_assistant/conftest.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Pytest configuration and shared fixtures for
``scikitplot._externals._sphinx_ext._sphinx_ai_assistant`` tests.

Bootstrap
---------
All heavy dependencies (Sphinx, BeautifulSoup, markdownify) are available
when the full package is installed.  When running the submodule in isolation
(extracted ZIP, CI without the full scikitplot wheel), this conftest
registers the local ``__init__.py`` and ``_static/__init__.py`` under their
canonical dotted names so that all test imports work identically in both
environments.

Fixtures
--------
tmp_html_tree
    Minimal HTML directory tree used by integration-style tests.
sphinx_app
    Mock Sphinx application wired to ``tmp_html_tree``.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: make the submodule importable under its canonical dotted name
# even when running in isolation (full scikitplot package NOT installed).
# ---------------------------------------------------------------------------

_CANONICAL_NAME: str = "scikitplot._externals._sphinx_ext._sphinx_ai_assistant"
_STATIC_NAME: str = _CANONICAL_NAME + "._static"

_PARENT_PACKAGES: tuple[str, ...] = (
    "scikitplot",
    "scikitplot._externals",
    "scikitplot._externals._sphinx_ext",
)


def _bootstrap_submodule() -> None:
    """
    Register the local package files under the canonical dotted path.

    Safe to call multiple times — exits immediately if the module is already
    present in :data:`sys.modules`.

    Notes
    -----
    The function:

    1. Registers stub :class:`types.ModuleType` objects for every missing
       parent package so ``import scikitplot._externals…`` does not raise
       :exc:`ImportError`.
    2. Loads ``_static/__init__.py`` as ``…._sphinx_ai_assistant._static``.
    3. Loads the main ``__init__.py`` as ``…._sphinx_ai_assistant``.

    Attaching ``_static`` as an attribute of the parent module makes
    relative imports (``from ._static import …``) work inside the loaded
    code.
    """
    if _CANONICAL_NAME in sys.modules:
        return  # full package is installed — nothing to do

    # ---- Step 1: stub parent packages ----------------------------------------
    for pkg in _PARENT_PACKAGES:
        if pkg not in sys.modules:
            stub = types.ModuleType(pkg)
            stub.__path__ = []  # type: ignore[attr-defined]
            stub.__package__ = pkg
            sys.modules[pkg] = stub

    _root = Path(__file__).parent.parent  # …/_sphinx_ai_assistant/

    # ---- Step 2: load _static subpackage first --------------------------------
    _static_init = _root / "_static" / "__init__.py"
    if _STATIC_NAME not in sys.modules and _static_init.exists():
        static_spec = importlib.util.spec_from_file_location(
            _STATIC_NAME, str(_static_init)
        )
        static_mod = importlib.util.module_from_spec(static_spec)  # type: ignore[arg-type]
        static_mod.__package__ = _STATIC_NAME
        sys.modules[_STATIC_NAME] = static_mod
        static_spec.loader.exec_module(static_mod)  # type: ignore[union-attr]

    # ---- Step 3: load main __init__.py ----------------------------------------
    main_spec = importlib.util.spec_from_file_location(
        _CANONICAL_NAME,
        str(_root / "__init__.py"),
        submodule_search_locations=[str(_root)],
    )
    main_mod = importlib.util.module_from_spec(main_spec)  # type: ignore[arg-type]
    main_mod.__package__ = _CANONICAL_NAME
    sys.modules[_CANONICAL_NAME] = main_mod
    main_spec.loader.exec_module(main_mod)  # type: ignore[union-attr]

    # Attach _static as attribute so relative imports inside main module work
    if _STATIC_NAME in sys.modules:
        main_mod._static = sys.modules[_STATIC_NAME]  # type: ignore[attr-defined]
        sys.modules[_CANONICAL_NAME + "._static"] = sys.modules[_STATIC_NAME]


# _bootstrap_submodule()


# ---------------------------------------------------------------------------
# Sphinx mock builder / app helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> MagicMock:
    """
    Return a minimal mock of a Sphinx Config object.

    Parameters
    ----------
    **overrides : Any
        Keyword arguments set as attributes on the returned mock.

    Returns
    -------
    unittest.mock.MagicMock
        A mock with sensible defaults for all config values the extension
        reads.

    Notes
    -----
    Keep this function in sync with every ``app.add_config_value`` call in
    :func:`~scikitplot._externals._sphinx_ext._sphinx_ai_assistant.setup`.
    Missing keys will cause ``MagicMock`` auto-attribute creation, which
    returns a new ``MagicMock`` instance and can lead to subtle test
    failures — for example in :func:`_cfg_str` / :func:`_cfg_bool` guards
    that explicitly check ``isinstance(val, str)``.
    """
    cfg = MagicMock()
    # Core toggles
    cfg.ai_assistant_enabled = True
    cfg.ai_assistant_position = "sidebar"
    cfg.ai_assistant_content_selector = "article"
    # Content selectors
    cfg.ai_assistant_content_selectors = [
        "article.bd-article",
        'div[role="main"]',
        'article[role="main"]',
    ]
    cfg.ai_assistant_theme_preset = None
    # Markdown generation
    cfg.ai_assistant_generate_markdown = True
    cfg.ai_assistant_markdown_exclude_patterns = [
        "genindex", "search", "py-modindex",
    ]
    cfg.ai_assistant_strip_tags = ["script", "style", "nav", "footer"]
    cfg.ai_assistant_max_workers = 1
    # llms.txt generation
    cfg.ai_assistant_generate_llms_txt = True
    cfg.ai_assistant_base_url = ""
    cfg.ai_assistant_llms_txt_max_entries = None
    cfg.ai_assistant_llms_txt_full_content = False
    # Features
    cfg.ai_assistant_features = {
        "markdown_export": True,
        "view_markdown": True,
        "ai_chat": True,
        "mcp_integration": False,
        "pdf_export": True,
        "ai_panel": True,
    }
    cfg.ai_assistant_providers = {}
    cfg.ai_assistant_ollama_model = "llama3.2:latest"
    cfg.ai_assistant_mcp_tools = {}
    # Prompt customisation
    cfg.ai_assistant_intention = None
    cfg.ai_assistant_custom_context = None
    cfg.ai_assistant_custom_prompt_prefix = None
    cfg.ai_assistant_include_raw_image = False
    # PDF export
    cfg.ai_assistant_pdf_export_url = None
    # AI panel
    cfg.ai_assistant_panel_title = "AI Assistant"
    cfg.ai_assistant_panel_placeholder = "Ask a question about this page\u2026"
    cfg.ai_assistant_panel_api_enabled = False
    cfg.ai_assistant_panel_page_help = True

    # Modern AI-assistant defaults.  Keep these concrete: a missing attribute
    # on MagicMock becomes another MagicMock, which is intentionally rejected
    # by the extension's strict config validators and can create unrelated
    # warnings/errors in tests that are trying to assert a different signal.
    # This list mirrors the literal defaults registered by setup().
    cfg.ai_assistant_pdf_url_mode_toggle = True
    cfg.ai_assistant_copy_mode = "browser"
    cfg.ai_assistant_copy_mode_toggle = True
    cfg.ai_assistant_llms_txt_format = "structured"
    cfg.ai_assistant_llms_txt_summary = None
    cfg.ai_assistant_llms_txt_sections = None
    cfg.ai_assistant_strict = False

    cfg.ai_assistant_panel_quick_questions = []
    cfg.ai_assistant_panel_speak_banner = True
    cfg.ai_assistant_panel_trigger_label = "Ask AI"
    cfg.ai_assistant_panel_start_minimized = True
    cfg.ai_assistant_panel_trigger_toggle = True
    cfg.ai_assistant_panel_reasoning = False
    cfg.ai_assistant_panel_effort_levels = []
    cfg.ai_assistant_panel_effort_default = ""
    cfg.ai_assistant_panel_stub_models = True
    cfg.ai_assistant_panel_injection_notice = True
    cfg.ai_assistant_panel_model_editing = True
    cfg.ai_assistant_panel_persist = True
    cfg.ai_assistant_panel_remember_conversation = True
    cfg.ai_assistant_panel_shortcut = "Alt+Shift+A"
    cfg.ai_assistant_panel_api_url = ""
    cfg.ai_assistant_panel_api_model = ""
    cfg.ai_assistant_panel_api_models = []
    cfg.ai_assistant_panel_api_streaming = True
    cfg.ai_assistant_panel_inline_model_picker = True

    cfg.ai_assistant_panel_usage_policy = True
    cfg.ai_assistant_panel_usage_policy_title = "Usage Policy"
    cfg.ai_assistant_panel_usage_policy_html = ""
    cfg.ai_assistant_panel_terms = True
    cfg.ai_assistant_panel_terms_title = "Terms of Service"
    cfg.ai_assistant_panel_terms_link_text = "Terms of Service"
    cfg.ai_assistant_panel_terms_html = ""
    cfg.ai_assistant_panel_share = True
    cfg.ai_assistant_panel_share_label = "Share"
    cfg.ai_assistant_panel_share_targets = []
    cfg.ai_assistant_panel_links = True
    cfg.ai_assistant_panel_links_title = "Project Links"
    cfg.ai_assistant_panel_links_html = ""

    cfg.ai_assistant_panel_source = True
    cfg.ai_assistant_panel_source_url = ""
    cfg.ai_assistant_panel_source_label = "Source Repository"
    cfg.ai_assistant_panel_source_desc = ""
    cfg.ai_assistant_panel_source_btn_label = "Source"
    cfg.ai_assistant_panel_site = True
    cfg.ai_assistant_panel_site_url = ""
    cfg.ai_assistant_panel_site_label = "Project Website"
    cfg.ai_assistant_panel_site_desc = ""
    cfg.ai_assistant_panel_site_btn_label = "Website"
    cfg.ai_assistant_panel_dataset_repo = ""
    cfg.ai_assistant_panel_hf_space_url = ""
    cfg.ai_assistant_panel_hf_space_label = "HuggingFace Space"
    cfg.ai_assistant_panel_hf_dataset_url = ""
    cfg.ai_assistant_panel_hf_dataset_label = "HuggingFace Dataset"
    cfg.ai_assistant_panel_hf_endpoint_url = ""
    cfg.ai_assistant_panel_hf_endpoint_label = "Active Endpoint"

    cfg.ai_assistant_panel_hamburger = True
    cfg.ai_assistant_panel_feedback = True
    cfg.ai_assistant_panel_feedback_question = "Was this helpful?"
    cfg.ai_assistant_panel_feedback_options = []
    cfg.ai_assistant_panel_feedback_placeholder = ""
    cfg.ai_assistant_panel_feedback_submit = "Send feedback"
    cfg.ai_assistant_panel_feedback_thanks = "Thanks for your feedback!"
    cfg.ai_assistant_panel_feedback_log = False
    cfg.ai_assistant_panel_feedback_scale = "auto"

    cfg.ai_assistant_isolation_origin = ""
    cfg.ai_assistant_isolation_frame_path = "/ai-assistant-isolated.html"
    cfg.ai_assistant_isolation_context_max_chars = 200000
    cfg.ai_assistant_isolation_parent_origins = []
    cfg.ai_assistant_isolation_allow_microphone = False

    cfg.ai_assistant_panel_feedback_endpoint = ""
    cfg.ai_assistant_panel_feedback_token = ""
    cfg.ai_assistant_allow_runtime_tokens = False
    cfg.ai_assistant_allow_credentialed_fetch = False
    cfg.ai_assistant_global_share_endpoint = ""
    cfg.ai_assistant_global_share_token = ""
    cfg.ai_assistant_global_share_ttl_days = 30
    cfg.ai_assistant_training_endpoint = ""
    cfg.ai_assistant_endpoint_profiles = {}
    cfg.ai_assistant_endpoint_default_profile = ""

    cfg.ai_assistant_panel_privacy_title = "Privacy & Responsibility"
    cfg.ai_assistant_panel_privacy_link_text = "Privacy & Responsibility"
    cfg.ai_assistant_panel_privacy_html = ""
    cfg.ai_assistant_search_bar = False
    cfg.ai_assistant_search_bar_selector = ""
    cfg.ai_assistant_search_bar_position = "top"
    cfg.ai_assistant_search_bar_mini = False
    cfg.ai_assistant_panel_search_placeholder = "Ask AI about these docs…"

    # Standard Sphinx values
    cfg.html_baseurl = ""
    cfg.html_static_path = []
    cfg.project = "TestProject"
    for key, val in overrides.items():
        setattr(cfg, key, val)
    return cfg


def _make_builder(outdir: str) -> MagicMock:
    """
    Return a mock :class:`sphinx.builders.html.StandaloneHTMLBuilder`.

    Parameters
    ----------
    outdir : str
        Path to the mock build output directory.

    Returns
    -------
    unittest.mock.MagicMock
        Spec'd against the real ``StandaloneHTMLBuilder`` class so that
        ``isinstance`` checks pass.
    """
    from sphinx.builders.html import StandaloneHTMLBuilder  # real class for isinstance
    builder = MagicMock(spec=StandaloneHTMLBuilder)
    builder.outdir = outdir
    return builder


def _make_app(outdir: str, **config_overrides: Any) -> MagicMock:
    """
    Return a mock Sphinx application.

    Parameters
    ----------
    outdir : str
        Build output directory path.
    **config_overrides : Any
        Forwarded to :func:`_make_config`.

    Returns
    -------
    unittest.mock.MagicMock
        Mock app with ``.config`` and ``.builder`` attributes set.
    """
    app = MagicMock()
    app.config = _make_config(**config_overrides)
    app.builder = _make_builder(outdir)
    return app


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_html_tree(tmp_path: Path) -> Path:
    """
    Create a minimal HTML output tree for integration-style tests.

    Returns
    -------
    pathlib.Path
        Root of the temporary output directory.

    Notes
    -----
    Tree layout::

        html/
        ├── index.html          — article[role="main"]
        ├── genindex.html       — div[role="main"]  (excluded by default)
        └── api/
            └── module.html     — article.bd-article
    """
    outdir = tmp_path / "html"
    outdir.mkdir()
    (outdir / "index.html").write_text(
        '<html><body><article role="main"><h1>Hello</h1><p>World</p></article>'
        "</body></html>",
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
    """Mock Sphinx app wired to the ``tmp_html_tree`` output directory."""
    return _make_app(str(tmp_html_tree))


# Re-export helpers so test modules can do:
#   from conftest import _make_app
__all__ = [
    "_make_config",
    "_make_builder",
    "_make_app",
]
