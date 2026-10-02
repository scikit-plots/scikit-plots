"""Sphinx adapter for generic page feedback."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import ClassVar

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.errors import ConfigError
from sphinx.util.docutils import SphinxDirective

from . import __version__
from ._config import FeedbackConfigError, load_aggregate, page_enabled, validate_config

_ASSETS = Path(__file__).parent / "_static"
_ENV_VERSION = 2


class FeedbackMount(nodes.General, nodes.Element):
    """Explicit theme-independent feedback mount point."""


class FeedbackDirective(SphinxDirective):
    """Insert an explicit mount that shares the page's single controller."""

    has_content = False
    optional_arguments = 0
    option_spec: ClassVar = {"layout": directives.unchanged}

    def run(self):
        layout = str(self.options.get("layout", "compact") or "compact").strip().lower()
        if layout not in {"compact", "full"}:
            raise self.error("feedback: layout must be 'compact' or 'full'")
        node = FeedbackMount()
        node["layout"] = layout
        return [node]


def _visit_html(translator, node):
    layout = str(node.get("layout", "compact"))
    translator.body.append(
        '<div class="sphinx-feedback-mount" data-sphinx-feedback-mount '
        f'data-sphinx-feedback-layout="{layout}"></div>'
    )
    raise nodes.SkipNode


def _visit_other(translator, node):
    raise nodes.SkipNode


def _depart_noop(translator, node):
    return None


def _append_unique_config_path(config, name: str, path: Path) -> None:
    values = list(getattr(config, name, ()) or ())
    text = str(path)
    if text not in values:
        values.append(text)
    setattr(config, name, values)


def _configure(app, config) -> None:
    try:
        normalized = validate_config(config)
        aggregate = {}
        aggregate_meta = {"contract": "", "complete": False}
        if (
            normalized["page_enabled"]
            and normalized["counter_enabled"]
            and normalized["counter_source"] == "embedded"
        ):
            aggregate, aggregate_meta = load_aggregate(
                _ASSETS,
                config.feedback_aggregate_file,
                expected_site_id=normalized["site_id"],
                expected_page_revision=normalized["page_revision"],
                return_metadata=True,
            )
    except FeedbackConfigError as exc:
        raise ConfigError(str(exc)) from exc
    app._sphinx_feedback_config = normalized
    app._sphinx_feedback_aggregate = aggregate
    app._sphinx_feedback_aggregate_meta = aggregate_meta
    if normalized["page_enabled"]:
        _append_unique_config_path(config, "html_static_path", _ASSETS)


def _builder_inited(app) -> None:
    if getattr(app.builder, "format", "") != "html":
        return
    normalized = getattr(app, "_sphinx_feedback_config", None)
    if not normalized or not normalized["page_enabled"]:
        return
    app.add_css_file("sphinx-feedback.css", priority=690)
    app.add_js_file("sphinx-feedback.js", defer="defer", priority=691)


def _safe_script_json(payload: dict) -> str:
    return (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace("&", "\\u0026")
        .replace(">", "\\u003e")
    )


def _page_context(app, pagename, templatename, context, doctree) -> None:
    if doctree is None or getattr(app.builder, "format", "") != "html":
        return
    normalized = getattr(app, "_sphinx_feedback_config", None) or validate_config(
        app.config,
    )
    if not normalized["page_enabled"]:
        return
    explicit_mount = any(doctree.findall(FeedbackMount))
    if (
        normalized["position"] == "none"
        and not normalized["page_main"]
        and not explicit_mount
    ):
        return
    if not page_enabled(
        pagename,
        include=normalized["include"],
        exclude=normalized["exclude"],
    ):
        return
    aggregate = getattr(app, "_sphinx_feedback_aggregate", {}) or {}
    counter = None
    if normalized["counter_enabled"] and normalized["counter_source"] == "embedded":
        item = aggregate.get(pagename)
        aggregate_meta = getattr(app, "_sphinx_feedback_aggregate_meta", {}) or {}
        if item is None and aggregate_meta.get("complete") is True:
            # A complete V3 snapshot is the only safe basis for converting an
            # absent page row into an authoritative reviewed zero. Sparse
            # snapshots keep absence == unknown and therefore hide the counters.
            item = {
                "count": 0,
                "score": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
            }
        if item is not None:
            counter = {"count": int(item["count"]), "score": int(item["score"])}
            # V3 aggregates carry exact sign-distribution counts. Never infer
            # positive/negative counts from score+count because -5..+5 ratings
            # and neutral 0 make that mathematically ambiguous.
            for field in ("positive_count", "negative_count", "neutral_count"):
                if field in item:
                    counter[field] = int(item[field])
    page_revision = normalized["page_revision"]
    payload = {
        "contract": "page.feedback-config.v1",
        "enabled": True,
        "site_id": normalized["site_id"],
        "page_id": pagename,
        "page_revision": page_revision,
        "endpoint": normalized["endpoint"],
        "position": normalized["position"],
        "fallback": normalized["fallback"],
        "page_main": normalized["page_main"],
        "quick_enabled": normalized["quick_enabled"],
        "buttons_ratings": dict(normalized["buttons_ratings"]),
        "detailed_enabled": normalized["detailed_enabled"],
        "comment_enabled": normalized["comment_enabled"],
        "contributor_enabled": normalized["contributor_enabled"],
        "sidebar_selectors": list(normalized["sidebar_selectors"]),
        "main_selectors": list(normalized["main_selectors"]),
        "counter": counter,
    }
    script = (
        '<script type="application/json" id="sphinx-feedback-config">'
        + _safe_script_json(payload)
        + "</script>"
    )
    context["metatags"] = str(context.get("metatags", "")) + script
    return


def setup_extension(app):
    """Register the independent page-feedback extension."""
    if getattr(app, "_sphinx_feedback_registered", False):
        return {
            "version": __version__,
            "env_version": _ENV_VERSION,
            "parallel_read_safe": True,
            "parallel_write_safe": True,
        }
    root = __package__.rsplit(".", 1)[0]
    import_module(root + "._extension_setup").check_namespace(app, root)

    app.add_config_value("feedback_page_enabled", False, "html")
    app.add_config_value("feedback_position", "sidebar", "html")
    app.add_config_value("feedback_page_main", True, "html")
    app.add_config_value("feedback_position_fallback", "main-bottom", "html")
    app.add_config_value("feedback_quick_enabled", True, "html")
    app.add_config_value(
        "feedback_buttons_ratings",
        {"left_button_rating": "left", "right_button_rating": "right"},
        "html",
    )
    app.add_config_value("feedback_detailed_enabled", True, "html")
    app.add_config_value("feedback_comment_enabled", True, "html")
    app.add_config_value("feedback_contributor_enabled", True, "html")
    app.add_config_value("feedback_counter_enabled", True, "html")
    app.add_config_value("feedback_counter_source", "embedded", "html")
    app.add_config_value("feedback_endpoint", "", "html")
    app.add_config_value("feedback_site_id", "docs", "html")
    app.add_config_value("feedback_page_revision", "", "html")
    app.add_config_value("feedback_aggregate_file", "", "html")
    app.add_config_value("feedback_include", ["**"], "html")
    app.add_config_value(
        "feedback_exclude", ["search", "genindex", "py-modindex", "404"], "html"
    )
    app.add_config_value("feedback_sidebar_selectors", None, "html")
    app.add_config_value("feedback_main_selectors", None, "html")
    app.add_node(
        FeedbackMount,
        html=(_visit_html, _depart_noop),
        latex=(_visit_other, _depart_noop),
        text=(_visit_other, _depart_noop),
        man=(_visit_other, _depart_noop),
        texinfo=(_visit_other, _depart_noop),
    )
    app.add_directive("feedback", FeedbackDirective)
    app.connect("config-inited", _configure)
    app.connect("builder-inited", _builder_inited)
    app.connect("html-page-context", _page_context, priority=880)
    app._sphinx_feedback_registered = True
    return {
        "version": __version__,
        "env_version": _ENV_VERSION,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
