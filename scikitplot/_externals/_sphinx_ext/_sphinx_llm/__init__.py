# scikitplot/_externals/_sphinx_ext/_sphinx_llm/__init__.py
#
# flake8: noqa: D213
#
# NVIDIA sphinx-llm compatibility core: Apache-2.0 (vendored under sphinx_llm/).
# scikit-plots integration/compatibility layer: BSD-3-Clause unless noted.
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Sphinx extension for static, machine-consumable documentation artifacts."""

from __future__ import annotations

from html import escape


def _inject_llms_discovery_links(app, pagename, templatename, context, doctree):
    """Advertise the canonical page Markdown and applicable llms.txt.

    The hrefs are relative to the current output page so version/language trees
    remain self-contained when deployed below a shared origin.
    """

    if not getattr(app.config, "llms_txt_enabled", True):
        return
    if not getattr(app.config, "llms_txt_discovery_links", True):
        return
    builder = getattr(app.builder, "name", "")
    if builder not in {"html", "dirhtml"}:
        return

    from .core.routing import discovery_hrefs  # ruff: ignore[import-outside-top-level]

    suffix_mode = getattr(app.config, "llms_txt_suffix_mode", "auto")
    try:
        markdown_href, llms_href = discovery_hrefs(
            pagename, builder=builder, suffix_mode=suffix_mode
        )
    except ValueError as exc:
        from sphinx.errors import (  # ruff: ignore[import-outside-top-level]
            ExtensionError,
        )

        raise ExtensionError(str(exc)) from exc

    metadata = (
        f'<link rel="alternate" type="text/markdown" href="{escape(markdown_href, quote=True)}" />\n'
        f'<link rel="describedby" href="{escape(llms_href, quote=True)}" />'
    )
    existing = context.get("metatags", "") or ""
    context["metatags"] = f"{existing}\n{metadata}" if existing else metadata


def setup(app):
    """Register the preserved NVIDIA core through downstream semantic layers."""

    from .adapters.sphinx import (  # ruff: ignore[import-outside-top-level]
        CanonicalMarkdownBuilder,
        register_semantic_support,
    )
    from .core.generator import (  # ruff: ignore[import-outside-top-level]
        CanonicalArtifactGenerator,
    )
    from .sphinx_llm.version import (  # ruff: ignore[import-outside-top-level]
        __version__,
    )

    summary_extension = (
        "scikitplot._externals._sphinx_ext._sphinx_llm.sphinx_llm.summary"
    )
    app.setup_extension(summary_extension)

    # The author-facing ignore directive behaves transparently in human output
    # and only suppresses its subtree in canonical LLM Markdown.
    register_semantic_support(app)

    if app.tags.has("sphinx_llm_markdown"):
        app.setup_extension("sphinx_markdown_builder")
        app.add_builder(CanonicalMarkdownBuilder)

    # NVIDIA-compatible baseline configuration.
    app.add_config_value("llms_txt_enabled", True, "")
    app.add_config_value("llms_txt_description", "", "env")
    app.add_config_value("llms_txt_build_parallel", True, "env")
    app.add_config_value("llms_txt_suffix_mode", "auto", "env")
    app.add_config_value("llms_txt_full_build", True, "env")
    app.add_config_value("llms_txt_exclude", [], "env")
    app.add_config_value("llms_txt_override_source", "", "env")

    # Downstream A03-A10 semantic/curation/artifact policy. Defaults preserve
    # existing output volume while making unsupported semantics observable.
    app.add_config_value("llms_txt_unknown_node_policy", "warn", "env")
    app.add_config_value("llms_txt_section_rules", [], "env")
    app.add_config_value("llms_txt_order", [], "env")
    app.add_config_value("llms_txt_index_max_bytes", None, "env")
    app.add_config_value("llms_txt_discovery_links", True, "html")
    app.add_config_value("llms_txt_full_max_bytes", None, "env")
    app.add_config_value("llms_txt_full_max_chars", None, "env")
    app.add_config_value("llms_txt_full_max_lines", None, "env")
    app.add_config_value("llms_txt_full_max_documents", None, "env")
    app.add_config_value("llms_txt_full_size_policy", "warn_keep", "env")
    app.add_config_value("llms_txt_code_files", [], "env")
    app.add_config_value("llms_txt_html_fallback", True, "env")

    generator = CanonicalArtifactGenerator(app)
    generator.setup()
    app.connect("html-page-context", _inject_llms_discovery_links)

    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
