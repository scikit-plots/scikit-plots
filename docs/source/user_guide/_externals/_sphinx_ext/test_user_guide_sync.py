"""
Static drift checks for the bundled Sphinx-extension user guides.

These checks intentionally avoid importing Sphinx or the extension packages.
They verify documentation coverage and selected public registrations directly
from source syntax so they can run in a source-only checkout.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _repository_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "scikitplot" / "_externals" / "_sphinx_ext"
        ).is_dir():
            return candidate
    raise AssertionError("could not locate repository root")


ROOT = _repository_root()
SOURCE = ROOT / "scikitplot" / "_externals" / "_sphinx_ext"
GUIDES = ROOT / "docs" / "source" / "user_guide" / "_externals" / "_sphinx_ext"


def _package_names() -> set[str]:
    return {
        path.name
        for path in SOURCE.iterdir()
        if path.is_dir() and path.name != "tests" and (path / "__init__.py").is_file()
    }


def _registered_literals(package: str, method: str) -> set[str]:
    values: set[str] = set()
    for source in (SOURCE / package).rglob("*.py"):
        if "tests" in source.parts:
            continue
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except SyntaxError as exc:  # pragma: no cover - source syntax is a harder failure
            raise AssertionError(f"cannot parse {source}: {exc}") from exc
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != method or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                values.add(first.value)
    return values


def test_every_bundled_child_package_has_a_user_guide() -> None:
    packages = _package_names()
    guides = {
        path.name
        for path in GUIDES.iterdir()
        if path.is_dir() and (path / "index.rst").is_file()
    }
    assert guides == packages


def test_root_index_reaches_every_child_guide() -> None:
    root_index = (GUIDES / "index.rst").read_text(encoding="utf-8")
    for package in sorted(_package_names()):
        assert f"<{package}/index>" in root_index


def test_child_guides_point_at_the_exact_runtime_package() -> None:
    for package in sorted(_package_names()):
        page = (GUIDES / package / "index.rst").read_text(encoding="utf-8")
        expected = f".. currentmodule:: scikitplot._externals._sphinx_ext.{package}"
        assert expected in page
        assert "This module contains functions related to" not in page


def test_documented_directives_exist_in_source_registration() -> None:
    documented = {
        "_pydata_component_list": {"component-list"},
        "_sphinx_ai_learn": {"ai-learn"},
        "_sphinx_feedback": {"feedback"},
        "_sphinx_gallery_grid": {"gallery-grid"},
        "_sphinx_llm": {"docref", "llms-ignore"},
        "_sphinx_youtube_gallery": {"youtube-gallery"},
        "_sphinxcontrib_youtube": {"youtube", "vimeo", "peertube"},
    }
    for package, expected in documented.items():
        assert expected <= _registered_literals(package, "add_directive")


def test_documented_core_config_names_exist_in_source_registration() -> None:
    documented = {
        "_sphinx_ai_assistant": {
            "ai_assistant_enabled",
            "ai_assistant_position",
            "ai_assistant_content_selector",
            "ai_assistant_theme_preset",
            "ai_assistant_generate_markdown",
            "ai_assistant_generate_llms_txt",
            "ai_assistant_llms_txt_full_content",
            "ai_assistant_panel_api_enabled",
            "ai_assistant_panel_api_url",
            "ai_assistant_panel_persist",
            "ai_assistant_panel_remember_conversation",
            "ai_assistant_isolation_origin",
        },
        "_sphinx_ai_learn": {
            "ai_learn_content_root",
            "ai_learn_site_id",
            "ai_learn_runtime",
            "ai_learn_explorer_search_variant",
            "ai_learn_media",
            "ai_learn_youtube_subscribe_url",
            "ai_learn_buttons_ratings",
        },
        "_sphinx_feedback": {
            "feedback_page_enabled",
            "feedback_position",
            "feedback_page_main",
            "feedback_quick_enabled",
            "feedback_detailed_enabled",
            "feedback_comment_enabled",
            "feedback_contributor_enabled",
            "feedback_counter_enabled",
            "feedback_counter_source",
            "feedback_endpoint",
            "feedback_site_id",
            "feedback_include",
            "feedback_exclude",
        },
        "_sphinx_gallery_grid": {"collection_search_variant"},
        "_sphinx_jinja_render": {"index_template_kwargs"},
        "_sphinx_llm": {
            "llms_txt_enabled",
            "llms_txt_discovery_links",
            "llms_txt_unknown_node_policy",
            "llms_txt_full_build",
            "llms_txt_html_fallback",
        },
        "_sphinx_youtube_gallery": {
            "youtube_catalog_path",
            "youtube_catalog_max_embeds",
        },
        "_sphinxcontrib_youtube": {
            "video_download_thumbnails",
            "video_download_limit",
            "video_download_max_bytes",
            "video_download_max_total_bytes",
        },
    }
    for package, expected in documented.items():
        assert expected <= _registered_literals(package, "add_config_value")
