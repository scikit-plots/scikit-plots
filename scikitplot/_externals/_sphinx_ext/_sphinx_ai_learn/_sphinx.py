# scikitplot/_externals/_sphinx_ext/_sphinx_ai_learn/_sphinx.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Sphinx-only nodes, configuration, and application-local lifecycle hooks."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlsplit

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.errors import ConfigError
from sphinx.util import logging as sphinx_logging
from sphinx.util.docutils import SphinxDirective

from .._search_variant import SEARCH_VARIANTS
from . import __version__
from ._generation import feedback_outdated_documents
from ._materialize import materialize
from ._pages import TEMPLATES, setup_pages
from ._schema import (
    DOMAINS,
    ID_PATTERN,
    KINDS,
    SECTION_PRESETS,
    LearnValidationError,
)

_MAX_DOMAINS = 100
_MAX_SECTIONS = 32
_PRESET_FIELDS = 2
_MAX_TITLE = 200
_MAX_SUBSCRIBE_URL = 2048
_AI_LEARN_BUTTON_RATING_POSITIONS = frozenset({"left", "right"})
_AI_LEARN_BUTTONS_RATINGS_DEFAULT = {
    "left_button_rating": "left",
    "right_button_rating": "right",
}

_ASSETS = Path(__file__).parent / "_static"
_LOGGER = sphinx_logging.getLogger(__name__)
_ENV_SCHEMA_REVISION = "learn-json-materializer-v3"
_ENV_VERSION = 1


class LearnRoot(nodes.General, nodes.Element):
    """Owned mount with a complete static fallback as normal document nodes."""


def _visit_html(translator, node):
    payload = (
        json.dumps(node["payload"], ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace("&", "\\u0026")
    )
    translator.body.append('<div class="skplt-learn-ai" data-skplt-learn-ai-mount>')
    translator.body.append(
        '<script type="application/json" class="la-data">' + payload + "</script>"
    )


def _depart_html(translator, node):
    translator.body.append("</div>")


def _visit_other(translator, node):
    """Other builders visit the ordinary, static child nodes."""


class LearnDirective(SphinxDirective):
    """Render the independent catalog, optionally focused on one subject."""

    has_content = False
    option_spec: ClassVar = {
        "subject": directives.unchanged_required,
        "kind": lambda v: directives.choice(v, KINDS),
    }

    def run(self):  # ruff: ignore[too-many-branches]
        digest = getattr(self.env.app, "_ai_learn_content_digest", "")
        if self.env.temp_data.get("_ai_learn_dependencies_noted") != digest:
            for dependency in getattr(self.env.app, "_ai_learn_dependencies", ()):
                self.env.note_dependency(dependency)
            self.env.temp_data["_ai_learn_dependencies_noted"] = digest
        catalog = self.env.app._ai_learn_catalog
        initial = self.options.get("subject", "")
        if initial and initial not in {s["id"] for s in catalog["subjects"]}:
            raise self.error("ai-learn: unknown subject identifier")
        # Names are unique even with two widgets in a single document.
        prefix = "la-" + str(self.env.new_serialno("ai-learn"))
        root = LearnRoot()
        root["payload"] = {
            "catalog": catalog,
            "initial_subject": initial,
            "initial_kind": self.options.get("kind", "topic"),
            "site_id": self.config.ai_learn_site_id,
            "runtime": self.config.ai_learn_runtime,
            "sections": self.config.ai_learn_sections,
            "domains": self.config.ai_learn_domains,
        }
        fallback = nodes.container(classes=["la-static"])
        subjects = [
            s
            for s in catalog["subjects"]
            if (
                s["id"] == initial
                if initial
                else not self.options.get("kind") or s["kind"] == self.options["kind"]
            )
        ]
        lookup = {s["id"]: s for s in catalog["subjects"]}
        fallback += nodes.paragraph(text="Published snapshot: " + catalog["revision"])
        if not subjects:
            fallback += nodes.paragraph(
                text="No published topics yet. Enable JavaScript to prepare a local contribution."
            )
        for subject in subjects:
            section = nodes.section(ids=[prefix + "-" + subject["id"]])
            section += nodes.title(text=subject["title"])
            section += nodes.paragraph(text=subject["summary"])
            if "url" in subject:
                paragraph = nodes.paragraph()
                paragraph += nodes.reference(
                    text=(
                        "Original source"
                        if subject["kind"] == "source"
                        else "Open resource"
                    ),
                    refuri=subject["url"],
                )
                section += paragraph
            for content in subject["sections"]:
                child = nodes.section(
                    ids=[prefix + "-" + subject["id"] + "-" + content["id"]]
                )
                child += nodes.title(text=content["title"])
                for body in content["body"].split("\n\n"):
                    child += nodes.paragraph(text=body)
                for citation in content["citations"]:
                    source = lookup[citation["source_id"]]
                    para = nodes.paragraph(text=citation["locator"] + " — ")
                    para += nodes.reference(text=source["title"], refuri=source["url"])
                    child += para
                for link in content.get("links", []):
                    para = nodes.paragraph()
                    para += nodes.reference(text=link["title"], refuri=link["url"])
                    if link.get("meta"):
                        para += nodes.Text(" " + link["meta"])
                    child += para
                section += child
            if subject["related"]:
                section += nodes.paragraph(text="Related records")
                listing = nodes.bullet_list()
                for target in subject["related"]:
                    record = lookup[target]
                    paragraph = nodes.paragraph()
                    docname = _docname(self.env.app, record)
                    if docname and docname in self.env.found_docs:
                        ref = addnodes.pending_xref(
                            "",
                            refdomain="std",
                            reftype="doc",
                            reftarget="/" + docname,
                            refexplicit=True,
                            refdoc=self.env.docname,
                        )
                        ref += nodes.inline(text=record["title"])
                        paragraph += ref
                    else:
                        paragraph += nodes.Text(record["title"])
                    item = nodes.list_item()
                    item += paragraph
                    listing += item
                section += listing
            fallback += section
        root += fallback
        return [root]


def _docname(app, subject):
    root = app.config.ai_learn_content_root.strip("/")
    route = getattr(app, "_ai_learn_routes", {}).get(subject.get("id", ""), "")
    return root + "/" + route if root and route else ""


def _resolve_routes(app, doctree, docname):
    if app.builder.format != "html":
        return
    for root in doctree.findall(LearnRoot):
        root["payload"]["routes"] = {
            subject["id"]: app.builder.get_relative_uri(docname, target)
            for subject in root["payload"]["catalog"]["subjects"]
            if (target := _docname(app, subject)) and target in app.env.found_docs
        }


def _append_unique_config_path(config, name, path):
    """Append one Sphinx path exactly once, preserving author order."""
    values = list(getattr(config, name, ()) or ())
    value = str(path)
    if value not in values:
        values.append(value)
    setattr(config, name, values)


def _normalize_ai_learn_buttons_ratings(value):
    """Validate and normalize per-button community-count placement.

    ``left_button_rating`` addresses the thumbs-down quick action and
    ``right_button_rating`` addresses thumbs-up. Partial mappings inherit the
    balanced defaults. The normalized mapping is presentation-only: it never
    changes feedback values, aggregate authority, persistence, or request data.
    """
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ConfigError("ai_learn_buttons_ratings must be a dictionary")
    unknown = set(value) - set(_AI_LEARN_BUTTONS_RATINGS_DEFAULT)
    if unknown:
        labels = ", ".join(repr(item) for item in sorted(unknown, key=repr))
        raise ConfigError(
            "ai_learn_buttons_ratings contains unsupported key(s): " + labels
        )
    normalized = dict(_AI_LEARN_BUTTONS_RATINGS_DEFAULT)
    for key in _AI_LEARN_BUTTONS_RATINGS_DEFAULT:
        if key not in value:
            continue
        raw = value[key]
        if not isinstance(raw, str):
            raise ConfigError(
                f"ai_learn_buttons_ratings[{key!r}] must be 'left' or 'right'"
            )
        position = raw.strip().lower()
        if position not in _AI_LEARN_BUTTON_RATING_POSITIONS:
            raise ConfigError(
                f"ai_learn_buttons_ratings[{key!r}] must be 'left' or 'right'"
            )
        normalized[key] = position
    return normalized


def _configure(  # ruff: ignore[too-many-branches]
    app,
    config,
):
    """Validate runtime config, then materialize canonical JSON before discovery."""
    if config.ai_learn_runtime not in ("none", "assistant"):
        raise ConfigError("ai_learn_runtime must be 'none' or 'assistant'")
    if not isinstance(config.ai_learn_media, bool):
        raise ConfigError("ai_learn_media must be true or false")
    app._ai_learn_buttons_ratings = _normalize_ai_learn_buttons_ratings(
        getattr(config, "ai_learn_buttons_ratings", None)
    )
    if config.ai_learn_explorer_search_variant not in SEARCH_VARIANTS:
        raise ConfigError(
            "ai_learn_explorer_search_variant must be 'pill-overflow' or 'classic'"
        )

    if not isinstance(config.ai_learn_site_id, str) or not ID_PATTERN.fullmatch(
        config.ai_learn_site_id
    ):
        raise ConfigError("ai_learn_site_id must be a stable lowercase identifier")
    youtube_subscribe_url = config.ai_learn_youtube_subscribe_url
    if not isinstance(youtube_subscribe_url, str):
        raise ConfigError("ai_learn_youtube_subscribe_url must be a string")
    if len(youtube_subscribe_url) > _MAX_SUBSCRIBE_URL or any(
        ord(ch) < 32 or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
        for ch in youtube_subscribe_url  # lint
    ):
        raise ConfigError(
            "ai_learn_youtube_subscribe_url is too long or contains control characters"
        )
    if youtube_subscribe_url:
        try:
            parsed = urlsplit(youtube_subscribe_url)
            port = parsed.port
        except ValueError as exc:
            raise ConfigError(
                "ai_learn_youtube_subscribe_url must be a valid HTTPS youtube.com URL"
            ) from exc
        if (
            parsed.scheme != "https"
            or parsed.username
            or parsed.password
            or parsed.hostname
            not in {"youtube.com", "www.youtube.com", "m.youtube.com"}
            or port not in (None, 443)
        ):
            raise ConfigError(
                "ai_learn_youtube_subscribe_url must be an HTTPS youtube.com URL"
            )

    content_root = config.ai_learn_content_root
    if not isinstance(content_root, str) or not content_root.strip("/"):
        raise ConfigError(
            "ai_learn_content_root must be a relative document directory",
        )
    normalized_root = content_root.strip("/")
    if content_root != normalized_root or any(
        not ID_PATTERN.fullmatch(part) for part in normalized_root.split("/")
    ):
        raise ConfigError(
            "ai_learn_content_root must be a safe relative document directory",
        )

    if (
        not isinstance(config.ai_learn_domains, (list, tuple))
        or len(config.ai_learn_domains) > _MAX_DOMAINS
        or any(
            not isinstance(v, str) or not ID_PATTERN.fullmatch(v)
            for v in config.ai_learn_domains
        )
    ):
        raise ConfigError("ai_learn_domains must contain domain identifiers")
    if len(set(config.ai_learn_domains)) != len(config.ai_learn_domains):
        raise ConfigError("ai_learn_domains contains duplicate identifiers")
    presets = config.ai_learn_sections
    if not isinstance(presets, (list, tuple)) or not 1 <= len(presets) <= _MAX_SECTIONS:
        raise ConfigError("ai_learn_sections must contain 1 to 32 ID/title pairs")
    for row in presets:
        if (
            not isinstance(row, (list, tuple))
            or len(row) != _PRESET_FIELDS
            or not isinstance(row[0], str)
            or not ID_PATTERN.fullmatch(row[0])
            or not isinstance(row[1], str)
            or not row[1].strip()
            or len(row[1]) > _MAX_TITLE
        ):
            raise ConfigError("ai_learn_sections contains an invalid ID/title pair")
    if len({row[0] for row in presets}) != len(presets):
        raise ConfigError("ai_learn_sections contains duplicate IDs")

    root = Path(app.srcdir) / normalized_root
    try:
        tree, changed = materialize(root)
    except (OSError, LearnValidationError) as exc:
        raise ConfigError(
            "Unable to materialize AI Learn canonical JSON: " + str(exc),
        ) from exc

    app._ai_learn_catalog = tree.catalog
    app._ai_learn_routes = tree.routes
    app._ai_learn_prompts = tree.prompts
    app._ai_learn_skills = tree.skills
    app._ai_learn_generation_feedback = tree.generation_feedback
    app._ai_learn_dependencies = tree.dependencies
    app._ai_learn_feedback_dependencies = tree.feedback_dependencies
    app._ai_learn_feedback_digests = tree.feedback_digests
    app._ai_learn_content_digest = tree.content_digest
    app._ai_learn_tree_digest = tree.digest
    _LOGGER.info(
        "AI Learn: materialized %d canonical JSON files into RST (%d changed); revision %s",
        len(tree.source_digests),
        len(changed),
        tree.catalog["revision"],
    )
    _append_unique_config_path(config, "html_static_path", _ASSETS)
    _append_unique_config_path(config, "templates_path", TEMPLATES)


def _environment_signature(app):
    """Fingerprint parser semantics plus the normalized catalog snapshot."""
    return _ENV_SCHEMA_REVISION + ":" + getattr(app, "_ai_learn_content_digest", "")


def _outdated_learn_documents(app, env, added, changed, removed):
    """
    Invalidate stale Learn doctrees when catalog/parser semantics change.

    Docutils error nodes are persisted in Sphinx doctrees. Without this guard an
    incremental build can keep an old ``Unknown topic/catalog identifier`` node
    even after the catalog is configured correctly. The first build after this
    lifecycle revision therefore reparses the owned Learn tree automatically.
    """
    current_signature = _environment_signature(app)
    previous_signature = getattr(env, "_ai_learn_environment_signature", None)
    if previous_signature == current_signature:
        previous_feedback = getattr(env, "_ai_learn_feedback_digests", {}) or {}
        current_feedback = getattr(app, "_ai_learn_feedback_digests", {}) or {}
        return feedback_outdated_documents(
            root=app.config.ai_learn_content_root,
            routes=getattr(app, "_ai_learn_routes", {}),
            found_docs=env.found_docs,
            consumers=getattr(env, "_ai_learn_feedback_consumers", {}) or {},
            previous=previous_feedback,
            current=current_feedback,
        )
    root = app.config.ai_learn_content_root.strip("/")
    if not root:
        return []
    prefix = root + "/"
    return sorted(
        docname
        for docname in env.found_docs
        if docname == root or docname.startswith(prefix)
    )


def _remember_environment_signature(app, env):
    env._ai_learn_environment_signature = _environment_signature(app)
    env._ai_learn_feedback_digests = dict(
        getattr(app, "_ai_learn_feedback_digests", {}) or {}
    )


def _purge_feedback_consumer(app, env, docname):
    """Remove stale record->document feedback dependency registrations."""
    consumers = getattr(env, "_ai_learn_feedback_consumers", None)
    if not consumers:
        return
    empty = []
    for record_id, docnames in consumers.items():
        docnames.discard(docname)
        if not docnames:
            empty.append(record_id)
    for record_id in empty:
        consumers.pop(record_id, None)


def _merge_feedback_consumers(app, env, docnames, other):
    """Replace master feedback-consumer state for documents read by a worker.

    ``other`` can carry cached registrations for documents outside the worker's
    read set.  The master can also still carry registrations from the previous
    parse of a document.  First purge every document in ``docnames`` from the
    master, then merge only the worker's current registrations for that exact
    read set.  This gives parallel reads the same replace-on-reread semantics as
    serial ``env-purge-doc`` + parse, without importing unrelated worker cache.
    """
    worker_docs = set(docnames or ())
    if not worker_docs:
        return
    for docname in worker_docs:
        _purge_feedback_consumer(app, env, docname)
    incoming = getattr(other, "_ai_learn_feedback_consumers", None) or {}
    if not incoming:
        return
    consumers = getattr(env, "_ai_learn_feedback_consumers", None)
    if consumers is None:
        consumers = {}
        env._ai_learn_feedback_consumers = consumers
    for record_id, names in incoming.items():
        selected = set(names or ()) & worker_docs
        if selected:
            consumers.setdefault(record_id, set()).update(selected)


def _page_assets(app, pagename, templatename, context, doctree):
    if doctree is not None and any(doctree.findall(LearnRoot)):
        app.add_css_file("ai-learn.css")
        app.add_js_file("ai-learn.js", defer="defer", priority=600)


def setup_extension(app):
    """Wire public hooks without importing or starting the proxy application."""
    if getattr(app, "_ai_learn_registered", False):
        return {
            "version": __version__,
            "env_version": _ENV_VERSION,
            "parallel_read_safe": True,
            "parallel_write_safe": True,
        }
    root = __package__.rsplit(".", 1)[0]
    # Reuse the sibling namespace guard without importing a public Python library.

    import_module(root + "._extension_setup").check_namespace(app, root)
    # Canonical learn.page.v1 indexes may materialize typed sphinx-design
    # grid/card directives. Declare the dependency here rather than relying on
    # a project conf.py ordering accident; Sphinx setup_extension is idempotent.
    app.setup_extension("sphinx_design")
    app.add_config_value("ai_learn_content_root", "learn-ai", "env")
    app.add_config_value("ai_learn_site_id", "scikit-plots-learn", "env")
    app.add_config_value("ai_learn_runtime", "none", "env")
    app.add_config_value("ai_learn_explorer_search_variant", "pill-overflow", "env")
    app.add_config_value("ai_learn_media", False, "env")
    app.add_config_value("ai_learn_youtube_subscribe_url", "", "env")
    app.add_config_value(
        "ai_learn_buttons_ratings",
        dict(_AI_LEARN_BUTTONS_RATINGS_DEFAULT),
        "html",
    )
    media = app.config.ai_learn_media
    if not isinstance(media, bool):
        raise ConfigError("ai_learn_media must be true or false")
    if media:
        app.setup_extension(root + "._sphinx_gallery_grid")
        app.setup_extension(root + "._sphinxcontrib_youtube")
    app.add_config_value("ai_learn_domains", list(DOMAINS), "env")
    app.add_config_value("ai_learn_sections", list(SECTION_PRESETS), "env")
    # Config access honors both conf.py and command-line overrides.
    runtime = app.config.ai_learn_runtime
    if runtime not in ("none", "assistant"):
        raise ConfigError("ai_learn_runtime must be 'none' or 'assistant'")
    if runtime == "assistant":
        app.setup_extension(root + "._sphinx_ai_assistant")
    app.add_node(
        LearnRoot,
        html=(_visit_html, _depart_html),
        latex=(_visit_other, _visit_other),
        text=(_visit_other, _visit_other),
        man=(_visit_other, _visit_other),
        texinfo=(_visit_other, _visit_other),
    )
    setup_pages(app)
    app.add_directive("ai-learn", LearnDirective)
    app.connect("config-inited", _configure)
    app.connect("env-get-outdated", _outdated_learn_documents)
    app.connect("env-updated", _remember_environment_signature)
    app.connect("env-purge-doc", _purge_feedback_consumer)
    app.connect("env-merge-info", _merge_feedback_consumers)
    app.connect("html-page-context", _page_assets)
    app.connect("doctree-resolved", _resolve_routes)
    # Mark registration complete only after every config value, dependency,
    # node, directive and event hook succeeded.  A setup exception must never
    # make a later call look successfully registered on a half-configured app.
    app._ai_learn_registered = True
    return {
        "version": __version__,
        "env_version": _ENV_VERSION,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
