# scikitplot/_externals/_sphinx_ext/_sphinx_ai_learn/_pages.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Sphinx-native topic pages. RST owns headings; Jinja owns the controls.

All catalog prose stays ordinary text nodes. Only author-written directive
content is parsed as RST. No provider, retrieval or source-writing runs at build.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar
from urllib.parse import quote_plus, urlencode

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective

from .._search_variant import resolve_search_variant, search_variant_option
from ._generation import (
    compact_feedback_count,
    section_generation_feedback_stats,
    section_generation_id,
)
from ._materialize import DIRECTORIES, LABELS, page_size
from ._registry import (
    TOPIC_EMPTY_MESSAGES,
    canonical_detail_section_id,
    detail_sections,
    topic_sections,
)

TEMPLATES = Path(__file__).parent / "_templates"
DISPLAY_SIZES = (12, 25, 50, 75, 100, 125, 150)

# One renderer-owned registry defines the nine AI Learn creation studios.  The
# same entries power the shared studio navigation and generation-route lookup so
# visible labels cannot drift away from link targets.
STUDIO_DEFINITIONS = (
    ("topic", "Topics", "topics/new"),
    ("source", "Sources", "sources/new"),
    ("problem", "Open Problems", "open-problems/new"),
    ("whiteboard", "Whiteboards", "whiteboards/new"),
    ("video", "Videos", "videos/new"),
    ("audio", "Audio", "audios/new"),
    ("document", "Documents", "documents/new"),
    ("skill", "Skills", "skills/new"),
    ("topic-prompt", "Topic Prompts", "topic-prompts/new"),
)
_STUDIO_LEAF_BY_KIND = {kind: leaf for kind, _label, leaf in STUDIO_DEFINITIONS}


def _studio_navigation():
    """Return the safe relative navigation model used by every new.html studio."""
    return tuple(
        {"kind": kind, "label": label, "href": "../" + leaf + ".html"}
        for kind, label, leaf in STUDIO_DEFINITIONS
    )


def safe_json(value):
    return (
        json.dumps(value, ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace("&", "\\u0026")
    )


def render(translator, template, context):
    # Studio navigation and quick-feedback presentation are renderer policy, not
    # authored/canonical content. Inject both centrally so HTML-only config can
    # change without being frozen into persisted doctrees.
    app = getattr(translator.builder, "app", None)
    buttons_ratings = getattr(app, "_ai_learn_buttons_ratings", None) or {
        "left_button_rating": "left",
        "right_button_rating": "right",
    }
    merged = {
        **context,
        "studio_nav": _studio_navigation(),
        "ai_learn_buttons_ratings": dict(buttons_ratings),
    }
    return translator.builder.templates.render("learn/" + template, merged)


class PageMarker(nodes.General, nodes.Element):
    """One marker per materialized page, used by the Jinja context hook."""


class Component(nodes.General, nodes.Element):
    """Jinja wrapper around readable, searchable docutils child nodes."""


def visit_marker(translator, node):
    translator.body.append(
        render(translator, "page-data.html", {"payload": safe_json(node["payload"])})
    )
    raise nodes.SkipNode


def visit_component(translator, node):
    translator.body.append(render(translator, node["start"], node["context"]))
    if node.get("replace", False):
        raise nodes.SkipNode


def depart_component(translator, node):
    if node.get("end"):
        translator.body.append(render(translator, node["end"], node["context"]))


def passthrough(translator, node):
    pass


def skip(translator, node):
    raise nodes.SkipNode


def component(start, context, end=None, replace=False):
    result = Component()
    for key, value in {
        "start": start,
        "context": context,
        "end": end,
        "replace": replace,
    }.items():
        result[key] = value
    return result


def paragraphs(container, text):
    for paragraph in text.split("\n\n"):
        if paragraph.strip():
            container += nodes.paragraph(text=paragraph)


def _prompts(app):
    return getattr(app, "_ai_learn_prompts", ())


def _skills(app):
    return getattr(app, "_ai_learn_skills", ())


def lookup(directive):
    # The generated Learn tree is invalidated by a tree digest.  External pages
    # that embed ``ai-learn`` also need non-source JSON dependency tracking; note
    # the complete canonical set once per parsed document, not once per directive.
    digest = getattr(directive.env.app, "_ai_learn_content_digest", "")
    marker = directive.env.temp_data.get("_ai_learn_dependencies_noted")
    if marker != digest:
        for dependency in getattr(directive.env.app, "_ai_learn_dependencies", ()):
            directive.env.note_dependency(dependency)
        directive.env.temp_data["_ai_learn_dependencies_noted"] = digest
    return {s["id"]: s for s in directive.env.app._ai_learn_catalog["subjects"]}


def _note_feedback_dependencies(directive, record_id):
    """Track reviewed feedback only for documents that consume this record."""
    record_id = str(record_id or "").strip()
    if not record_id:
        return
    consumers = getattr(directive.env, "_ai_learn_feedback_consumers", None)
    if consumers is None:
        consumers = {}
        directive.env._ai_learn_feedback_consumers = consumers
    consumers.setdefault(record_id, set()).add(directive.env.docname)

    marker_key = "_ai_learn_feedback_dependencies_noted"
    marker = directive.env.temp_data.setdefault(marker_key, set())
    if record_id in marker:
        return
    for dependency in getattr(
        directive.env.app, "_ai_learn_feedback_dependencies", {}
    ).get(record_id, ()):
        directive.env.note_dependency(dependency)
    marker.add(record_id)


def find(directive, identity):
    result = lookup(directive).get(identity)
    if result is None:
        raise directive.error("Unknown topic/catalog identifier")
    return result


def sections(subject, app=None):
    return (
        topic_sections(
            subject,
            prompts=_prompts(app) if app is not None else None,
            skills=_skills(app) if app is not None else None,
        )
        if subject["kind"] == "topic"
        else detail_sections(subject)
    )


def related_records(subject, catalog):
    return [
        s
        for s in catalog.values()
        if s["id"] in subject["related"] or subject["id"] in s["related"]
    ]


def source_records(subject, catalog):
    identities = {c["source_id"] for s in subject["sections"] for c in s["citations"]}
    identities.update(
        s["id"] for s in related_records(subject, catalog) if s["kind"] == "source"
    )
    return [s for s in catalog.values() if s["id"] in identities]


def whiteboard_images(subject):
    """Return the ordered, schema-validated image set for one whiteboard."""
    if subject.get("kind") != "whiteboard":
        return []
    media = subject.get("media", {})
    result = []
    if media.get("image"):
        result.append(
            {
                "image": media["image"],
                "alt": media["alt"],
                "caption": media.get("caption", ""),
            }
        )
    result.extend(media.get("images", []))
    return result


def audio_media(subject):
    """Return validated published audio metadata for one Audio record."""
    if subject.get("kind") != "audio":
        return {}
    media = subject.get("media", {})
    if media.get("type") != "audio":
        return {}
    return dict(media)


def document_media(subject):
    """Return validated published document metadata for one Document record."""
    if subject.get("kind") != "document":
        return {}
    media = subject.get("media", {})
    if media.get("type") != "document":
        return {}
    return dict(media)


class AudioPlayerDirective(SphinxDirective):
    """Render one passive published Audio record without autoplay."""

    required_arguments = 1

    def run(self):
        subject = find(self, self.arguments[0])
        if subject.get("kind") != "audio":
            raise self.error("ai-audio-player requires an audio subject")
        media = audio_media(subject)
        if not media:
            return [
                nodes.paragraph(
                    text="No playable audio is attached to this record yet."
                )
            ]
        root = component(
            "audio-player.html",
            {"subject": subject, "media": media},
            replace=True,
        )
        root["audio_player"] = True
        return [root]


class DocumentViewerDirective(SphinxDirective):
    """Render one passive published Document record as a safe local link."""

    required_arguments = 1

    def run(self):
        subject = find(self, self.arguments[0])
        if subject.get("kind") != "document":
            raise self.error("ai-document-viewer requires a document subject")
        media = document_media(subject)
        if not media:
            return [
                nodes.paragraph(
                    text="No published document is attached to this record yet."
                )
            ]
        root = component(
            "document-viewer.html",
            {"subject": subject, "media": media},
            replace=True,
        )
        root["document_viewer"] = True
        fallback = nodes.paragraph()
        fallback += nodes.reference(text="Open document", refuri=media["src"])
        root += fallback
        return [root]


class PageDirective(SphinxDirective):
    required_arguments = 1

    def run(self):
        if any(self.state.document.findall(PageMarker)):
            raise self.error("Only one ai-topic-page is allowed in a document")
        subject = find(self, self.arguments[0])
        node = PageMarker()
        node["subject"] = subject
        catalog = lookup(self)
        evidence_sources = [
            {
                "id": source["id"],
                "title": source["title"],
                "url": source.get("url", ""),
                "summary": source.get("summary", ""),
                "publisher": source.get("publisher", ""),
                "format": source.get("format", ""),
                "sections": [
                    {
                        "id": row.get("id", ""),
                        "title": row.get("title", ""),
                        "body": row.get("body", "")[:6000],
                    }
                    for row in source.get("sections", [])[:8]
                ],
            }
            for source in source_records(subject, catalog)
        ]
        node["payload"] = {
            "subject": subject,
            "sections": sections(subject, self.env.app),
            "evidence_sources": evidence_sources,
            "site_id": self.config.ai_learn_site_id,
            "revision": self.env.app._ai_learn_catalog["revision"],
            "display_sizes": DISPLAY_SIZES,
            "page_chunk_size": page_size("library"),
        }
        return [node]


class OverviewDirective(SphinxDirective):
    required_arguments = 1

    def run(self):
        subject = find(self, self.arguments[0])
        catalog = lookup(self)
        root = component(
            "overview.html",
            {
                "subject": subject,
                "revision": self.env.app._ai_learn_catalog["revision"],
                "sources": source_records(subject, catalog),
                "related": related_records(subject, catalog),
                "citation_search_url": (
                    "https://www.semanticscholar.org/search?q="
                    + quote_plus(subject["title"])
                ),
                "bookmarks_href": "#",
                "collections_href": "#",
                "video_generation_href": "#",
                "audio_generation_href": "#",
                "document_generation_href": "#",
                "whiteboard_generation_href": "#",
            },
            replace=True,
        )
        root["overview_component"] = True
        paragraphs(root, subject["summary"])
        for source in source_records(subject, catalog):
            para = nodes.paragraph()
            para += nodes.reference(text=source["title"], refuri=source["url"])
            root += para
        return [root]


class MediaActionsDirective(SphinxDirective):
    """Render media-specific actions without assuming one mandatory parent kind."""

    required_arguments = 1

    def run(self):
        subject = find(self, self.arguments[0])
        if subject["kind"] not in {"video", "whiteboard", "audio", "document"}:
            raise self.error(
                "ai-media-actions requires a video, whiteboard, audio, or document subject"
            )
        catalog = lookup(self)
        related = related_records(subject, catalog)
        priority = ("topic", "source", "skill", "problem")
        context_target = next(
            (row for kind in priority for row in related if row["kind"] == kind),
            None,
        )
        label = {
            "topic": "View Topic",
            "source": "View Source",
            "skill": "View Skill",
            "problem": "View Open Problem",
        }.get(context_target["kind"] if context_target else "", "")
        create_label = {
            "video": "Create a Video",
            "audio": "Create an Audio",
            "whiteboard": "Create a Whiteboard",
            "document": "Create a Document",
        }[subject["kind"]]
        root = component(
            "media-actions.html",
            {
                "subject": subject,
                "sources": source_records(subject, catalog),
                "create_label": create_label,
                "context_target": context_target,
                "context_label": label,
                "bookmarks_href": "#",
                "collections_href": "#",
                "video_generation_href": "#",
                "audio_generation_href": "#",
                "document_generation_href": "#",
                "whiteboard_generation_href": "#",
                "revision": self.env.app._ai_learn_catalog["revision"],
            },
            replace=True,
        )
        root["media_actions"] = True
        if context_target:
            root += nodes.paragraph(text=f"{label}: {context_target['title']}")
        return [root]


class VideoGenerationDirective(SphinxDirective):
    """Render the video-generation shell without running providers at build time."""

    has_content = False

    def run(self):
        catalog = lookup(self)
        records = list(catalog.values())
        context = {
            "site_id": self.config.ai_learn_site_id,
            "revision": self.env.app._ai_learn_catalog["revision"],
            "runtime": self.config.ai_learn_runtime,
            "topics": sorted(
                (dict(row) for row in records if row["kind"] == "topic"),
                key=lambda row: row["title"].casefold(),
            ),
            "sources": sorted(
                (dict(row) for row in records if row["kind"] == "source"),
                key=lambda row: row["title"].casefold(),
            ),
            "videos": sorted(
                (dict(row) for row in records if row["kind"] == "video"),
                key=lambda row: row["title"].casefold(),
            ),
            "payload": "{}",
        }
        root = component("video-generation.html", context, replace=True)
        root["video_generator"] = True
        return [root]


class AudioGenerationDirective(SphinxDirective):
    """Render the Audio-generation shell without calling providers at build time."""

    has_content = False

    def run(self):
        catalog = lookup(self)
        records = list(catalog.values())
        context = {
            "site_id": self.config.ai_learn_site_id,
            "revision": self.env.app._ai_learn_catalog["revision"],
            "runtime": self.config.ai_learn_runtime,
            "topics": sorted(
                (dict(row) for row in records if row["kind"] == "topic"),
                key=lambda row: row["title"].casefold(),
            ),
            "sources": sorted(
                (dict(row) for row in records if row["kind"] == "source"),
                key=lambda row: row["title"].casefold(),
            ),
            "audio": sorted(
                (dict(row) for row in records if row["kind"] == "audio"),
                key=lambda row: row["title"].casefold(),
            ),
            "payload": "{}",
        }
        root = component("audio-generation.html", context, replace=True)
        root["audio_generator"] = True
        return [root]


class DocumentGenerationDirective(SphinxDirective):
    """Render the Document-generation shell without provider work at build time."""

    has_content = False

    def run(self):
        catalog = lookup(self)
        records = list(catalog.values())
        context = {
            "site_id": self.config.ai_learn_site_id,
            "revision": self.env.app._ai_learn_catalog["revision"],
            "runtime": self.config.ai_learn_runtime,
            "topics": sorted(
                (dict(row) for row in records if row["kind"] == "topic"),
                key=lambda row: row["title"].casefold(),
            ),
            "sources": sorted(
                (dict(row) for row in records if row["kind"] == "source"),
                key=lambda row: row["title"].casefold(),
            ),
            "documents": sorted(
                (dict(row) for row in records if row["kind"] == "document"),
                key=lambda row: row["title"].casefold(),
            ),
            "payload": "{}",
        }
        root = component("document-generation.html", context, replace=True)
        root["document_generator"] = True
        return [root]


class WhiteboardGenerationDirective(SphinxDirective):
    """Render the Whiteboard-generation shell using the image artifact runtime."""

    has_content = False

    def run(self):
        catalog = lookup(self)
        records = list(catalog.values())
        context = {
            "site_id": self.config.ai_learn_site_id,
            "revision": self.env.app._ai_learn_catalog["revision"],
            "runtime": self.config.ai_learn_runtime,
            "topics": sorted(
                (dict(row) for row in records if row["kind"] == "topic"),
                key=lambda row: row["title"].casefold(),
            ),
            "sources": sorted(
                (dict(row) for row in records if row["kind"] == "source"),
                key=lambda row: row["title"].casefold(),
            ),
            "whiteboards": sorted(
                (dict(row) for row in records if row["kind"] == "whiteboard"),
                key=lambda row: row["title"].casefold(),
            ),
            "payload": "{}",
        }
        root = component("whiteboard-generation.html", context, replace=True)
        root["whiteboard_generator"] = True
        return [root]


class RecordGenerationDirective(SphinxDirective):
    """Render one AI-assisted catalog-record creation studio."""

    required_arguments = 1
    has_content = False

    def run(self):
        kind = self.arguments[0]
        if kind not in {"topic", "source", "problem", "skill", "topic-prompt"}:
            raise self.error(
                "ai-record-generation requires topic, source, problem, skill, or topic-prompt"
            )
        catalog = lookup(self)
        context_rows = [
            {
                "id": row["id"],
                "kind": row["kind"],
                "title": row["title"],
                "summary": row.get("summary", ""),
                "url": row.get("url", ""),
                "domains": row.get("domains", []),
            }
            for row in catalog.values()
            if row["kind"] in {"topic", "source"}
        ]
        topics = sorted(
            (dict(row) for row in context_rows if row["kind"] == "topic"),
            key=lambda row: row["title"].casefold(),
        )
        sources = sorted(
            (dict(row) for row in context_rows if row["kind"] == "source"),
            key=lambda row: row["title"].casefold(),
        )
        context = {
            "creation_kind": kind,
            "site_id": self.config.ai_learn_site_id,
            "revision": self.env.app._ai_learn_catalog["revision"],
            "topics": topics,
            "sources": sources,
            "payload": safe_json(
                {
                    "contract": "learn.record-generation-page.v2",
                    "kind": kind,
                    "site_id": self.config.ai_learn_site_id,
                    "revision": self.env.app._ai_learn_catalog["revision"],
                    "topics": topics,
                    "sources": sources,
                }
            ),
        }
        root = component("record-generation.html", context, replace=True)
        root["record_generator"] = True
        return [root]


class SectionDirective(SphinxDirective):
    required_arguments = 1
    has_content = True
    option_spec: ClassVar = {"topic-id": directives.unchanged_required}

    def run(self):  # ruff: ignore[too-many-branches]
        subject = find(self, self.options.get("topic-id", ""))
        _note_feedback_dependencies(self, subject["id"])
        spec = next(
            (
                s
                for s in sections(subject, self.env.app)
                if s["id"] == self.arguments[0]
            ),
            None,
        )
        if not spec:
            raise self.error(
                "Unknown section identifier; add custom sections to the catalog first"
            )
        catalog = lookup(self)
        # Prefer an exact canonical section, then accept a legacy alias.
        content = next(
            (s for s in subject["sections"] if s["id"] == spec["id"]),
            None,
        )
        if content is None:
            content = next(
                (
                    s
                    for s in subject["sections"]
                    if canonical_detail_section_id(subject["kind"], s["id"])
                    == spec["id"]
                ),
                {"body": "", "citations": [], "links": []},
            )
        citations = [
            {
                **c,
                "title": catalog[c["source_id"]]["title"],
                "url": catalog[c["source_id"]]["url"],
            }
            for c in content["citations"]
        ]
        # Reference/evidence sections use the same validated Source records.
        is_reference_section = spec["id"] in {"evidence", "references"}
        if is_reference_section and not citations:
            citations = [
                {
                    "source_id": s["id"],
                    "title": s["title"],
                    "url": s["url"],
                    "locator": "Original source",
                }
                for s in source_records(subject, catalog)
            ]
        related_kind = {
            "video": "video",
            "audio": "audio",
            "document": "document",
            "whiteboard": "whiteboard",
            "open-problems": "problem",
            "related": "topic",
        }.get(spec["id"])
        related = [
            s for s in related_records(subject, catalog) if s["kind"] == related_kind
        ]
        filled = bool(
            content["body"]
            or self.content
            or related
            or content.get("links")
            or (is_reference_section and citations)
        )
        review = content.get("review")
        revision = self.env.app._ai_learn_catalog["revision"]
        evidence_review_state = (
            "stale"
            if review and review["revision"] != revision
            else (
                "reviewed" if review else ("pending" if citations else "not-applicable")
            )
        )
        # ``state`` remains the compatibility presentation state used by existing
        # CSS/JS. Publication and evidence review are independent authorities.
        state = (
            "stale"
            if evidence_review_state == "stale"
            else ("ready" if filled else "empty")
        )
        publication_state = "published" if filled else "unpublished"
        # Community feedback belongs only to immutable accepted section text.
        # Related media/evidence may make a section visually non-empty, but it
        # must not synthesize a feedback target for an empty canonical section.
        generation_id = (
            section_generation_id(subject, content)
            if content.get("body", "").strip()
            else ""
        )
        feedback_index = getattr(self.env.app, "_ai_learn_generation_feedback", {})
        sidecar_feedback = (
            feedback_index.get((subject["id"], content["id"], generation_id), ())
            if generation_id
            else ()
        )
        feedback_stats = section_generation_feedback_stats(content, sidecar_feedback)
        feedback_score = feedback_stats["score"]
        feedback_count = feedback_stats["count"]
        generation_history = []
        for generation in content.get("generations", []):
            rows = list(
                feedback_index.get(
                    (subject["id"], content["id"], generation["id"]),
                    (),
                )
            )
            generation_history.append(
                {
                    "id": generation["id"],
                    "created_at": generation["created_at"],
                    "contributors": generation.get("contributors", []),
                    "score": sum(int(row.get("rating", 0)) for row in rows),
                    "ratings": len(rows),
                    "active": generation["id"] == content.get("active_generation_id"),
                }
            )
        # Default presentation order is authority first, then community signal,
        # then recency.  Client-side controls may re-order the same immutable
        # accepted generations by rating or generated date; they never change
        # ``active_generation_id``.
        generation_history.sort(
            key=lambda row: (row["score"], row["created_at"], row["id"]),
            reverse=True,
        )
        generation_history.sort(key=lambda row: not row["active"])
        generation_mode = spec.get("generation", {}).get("mode", "none")
        state_label = (
            "Published"
            if filled
            else (
                "AI draft not generated"
                if generation_mode == "chat"
                else "Not available"
            )
        )
        root = component(
            "section-start.html",
            {
                "subject": subject,
                "spec": spec,
                "content": content,
                "filled": filled,
                "citations": citations,
                "review": review,
                "state": state,
                "publication_state": publication_state,
                "evidence_review_state": evidence_review_state,
                "generation_id": generation_id,
                "feedback_section_id": content.get("id", spec["id"]),
                "feedback_score": feedback_score,
                "feedback_count": feedback_count,
                "feedback_positive_count": feedback_stats["positive_count"],
                "feedback_negative_count": feedback_stats["negative_count"],
                "feedback_neutral_count": feedback_stats["neutral_count"],
                "feedback_count_display": compact_feedback_count(feedback_count),
                "feedback_positive_count_display": compact_feedback_count(
                    feedback_stats["positive_count"],
                ),
                "feedback_negative_count_display": compact_feedback_count(
                    feedback_stats["negative_count"],
                ),
                "generation_history": generation_history,
                "state_label": state_label,
                "empty_message": (
                    spec.get("empty_message")
                    or TOPIC_EMPTY_MESSAGES.get(
                        spec["id"],
                        spec["description"]
                        + " This section has not been generated yet.",
                    )
                ),
                "videos_href": "#",
                "video_generation_href": "#",
                "audio_href": "#",
                "audio_generation_href": "#",
                "document_href": "#",
                "document_generation_href": "#",
                "whiteboard_href": "#",
                "whiteboard_generation_href": "#",
                "source_creation_href": "#",
                "youtube_subscribe_url": self.config.ai_learn_youtube_subscribe_url,
            },
            "section-end.html",
        )
        root["section_component"] = True
        prose = nodes.container(classes=["learn-prose"])
        paragraphs(prose, content["body"])
        root += prose
        if spec["kind"] == "social" and content.get("links"):
            social = nodes.enumerated_list(classes=["learn-social-list"])
            for link in content["links"]:
                item = nodes.list_item()
                para = nodes.paragraph()
                para += nodes.reference(text=link["title"], refuri=link["url"])
                if link.get("meta"):
                    para += nodes.Text(" " + link["meta"])
                item += para
                social += item
            root += social
        if self.content:
            self.state.nested_parse(self.content, self.content_offset, root)
        if related:
            if spec["id"] == "video":
                embedded = add_inline_videos(self, root, related)
            elif spec["id"] == "audio":
                embedded = add_inline_audio(self, root, related)
            else:
                embedded = set()
            remaining = [record for record in related if record["id"] not in embedded]
            if remaining:
                add_cards(self, root, remaining)
        if not filled:
            root += nodes.paragraph(
                text=spec.get("empty_message")
                or TOPIC_EMPTY_MESSAGES.get(
                    spec["id"],
                    spec["description"] + " This section has not been generated yet.",
                ),
                classes=["learn-empty"],
            )
        # Non-HTML builders retain evidence links as semantic children.
        if citations:
            evidence = nodes.container(classes=["learn-print-evidence"])
            for citation in citations:
                para = nodes.paragraph(text=citation["locator"] + " — ")
                para += nodes.reference(text=citation["title"], refuri=citation["url"])
                evidence += para
            root += evidence
        return [root]


class PromptGroupDirective(SphinxDirective):
    """Render the shared prompt visibility controls on every topic page."""

    required_arguments = 1

    def run(self):
        subject = find(self, self.arguments[0])
        if subject["kind"] != "topic":
            raise self.error("ai-topic-prompts requires a topic subject")
        root = component(
            "prompt-group.html",
            {
                "subject": subject,
                "prompts": [
                    {**prompt, "href": "#"} for prompt in _prompts(self.env.app)
                ],
                "site_id": self.config.ai_learn_site_id,
                "prompts_href": "",
            },
            replace=True,
        )
        root["prompt_group"] = True
        return [root]


class PromptLibraryDirective(SphinxDirective):
    """Render the topic-prompt library with persistent local preferences."""

    has_content = False

    def run(self):
        prompts = [{**prompt, "href": "#"} for prompt in _prompts(self.env.app)]
        root = component(
            "prompt-library-start.html",
            {
                "prompts": prompts,
                "site_id": self.config.ai_learn_site_id,
                "display_sizes": DISPLAY_SIZES,
                "page_chunk_size": page_size("topic-prompt"),
            },
            "prompt-library-end.html",
        )
        root["prompt_library"] = True
        for prompt in prompts:
            card = component(
                "prompt-card-start.html",
                {"prompt": prompt, "site_id": self.config.ai_learn_site_id},
                "prompt-card-end.html",
            )
            card["prompt_card"] = True
            paragraphs(card, prompt["description"])
            root += card
        return [root]


class PromptDetailDirective(SphinxDirective):
    """Render one reusable topic prompt on its own document page."""

    required_arguments = 1
    has_content = False

    def run(self):
        prompt = next(
            (
                item
                for item in _prompts(self.env.app)
                if item["id"] == self.arguments[0]
            ),
            None,
        )
        if prompt is None:
            raise self.error("Unknown topic prompt identifier")
        root = component(
            "prompt-detail-start.html",
            {
                "prompt": {**prompt},
                "site_id": self.config.ai_learn_site_id,
                "library_href": "#",
            },
            "prompt-detail-end.html",
        )
        root["prompt_detail"] = True
        paragraphs(root, prompt["instruction"])
        return [root]


class SkillGroupDirective(SphinxDirective):
    """Render reusable skill visibility controls on every topic page."""

    required_arguments = 1

    def run(self):
        subject = find(self, self.arguments[0])
        if subject["kind"] != "topic":
            raise self.error("ai-topic-skills requires a topic subject")
        root = component(
            "skill-group.html",
            {
                "subject": subject,
                "skills": [{**skill, "href": "#"} for skill in _skills(self.env.app)],
                "site_id": self.config.ai_learn_site_id,
                "skills_href": "",
            },
            replace=True,
        )
        root["skill_group"] = True
        return [root]


class SkillLibraryDirective(SphinxDirective):
    """Render the reusable skill library with persistent local preferences."""

    has_content = False

    def run(self):
        skills = [{**skill, "href": "#"} for skill in _skills(self.env.app)]
        root = component(
            "skill-library-start.html",
            {
                "skills": skills,
                "site_id": self.config.ai_learn_site_id,
                "display_sizes": DISPLAY_SIZES,
                "page_chunk_size": page_size("skill"),
            },
            "skill-library-end.html",
        )
        root["skill_library"] = True
        for skill in skills:
            card = component(
                "skill-card-start.html",
                {"skill": skill, "site_id": self.config.ai_learn_site_id},
                "skill-card-end.html",
            )
            card["skill_card"] = True
            paragraphs(card, skill["description"])
            root += card
        return [root]


class SkillDetailDirective(SphinxDirective):
    """Render one reusable skill on its own document page."""

    required_arguments = 1
    has_content = False

    def run(self):
        skill = next(
            (item for item in _skills(self.env.app) if item["id"] == self.arguments[0]),
            None,
        )
        if skill is None:
            raise self.error("Unknown skill identifier")
        root = component(
            "skill-detail-start.html",
            {
                "skill": {**skill},
                "site_id": self.config.ai_learn_site_id,
                "library_href": "#",
            },
            "skill-detail-end.html",
        )
        root["skill_detail"] = True
        paragraphs(root, skill["instruction"])
        return [root]


class WhiteboardGalleryDirective(SphinxDirective):
    """Render semantic whiteboard images; JavaScript adds the modal viewer."""

    required_arguments = 1
    has_content = False

    def run(self):
        subject = find(self, self.arguments[0])
        if subject["kind"] != "whiteboard":
            raise self.error("ai-whiteboard-gallery requires a whiteboard subject")
        images = whiteboard_images(subject)
        if not images:
            return [nodes.paragraph(text="No whiteboard images have been added yet.")]
        root = component(
            "whiteboard-gallery-start.html",
            {"subject": subject, "count": len(images)},
            "whiteboard-gallery-end.html",
        )
        root["whiteboard_gallery"] = True
        for index, item in enumerate(images):
            figure = nodes.figure(classes=["learn-whiteboard-figure"])
            image = nodes.image(
                uri=item["image"],
                alt=item["alt"],
                classes=["learn-whiteboard-image"],
            )
            image["ids"].append(f"learn-whiteboard-{subject['id']}-{index + 1}")
            figure += image
            if item.get("caption"):
                figure += nodes.caption(text=item["caption"])
            root += figure
        return [root]


def record_docname(app, subject):
    prefix = app.config.ai_learn_content_root.strip("/")
    route = getattr(app, "_ai_learn_routes", {}).get(subject.get("id", ""), "")
    return prefix + "/" + route if prefix and route else ""


def record_href(app, docname, subject):
    target = record_docname(app, subject)
    if target and target in app.env.found_docs:
        return app.builder.get_relative_uri(docname, target)
    return subject.get("url", "#")


def add_inline_videos(directive, root, records):
    """
    Embed related YouTube videos directly inside a topic section.

    The catalog relationship remains the source of truth.  Only records with a
    validated ``media.youtube_id`` are embedded; records without a playable
    asset fall back to ordinary related cards instead of inventing media.
    """
    playable = [
        record for record in records if record.get("media", {}).get("youtube_id")
    ]
    if not playable:
        return set()
    listing = nodes.container(classes=["learn-topic-video-list"])
    root += listing
    for record in playable:
        block = nodes.container(classes=["learn-topic-video"])
        listing += block
        header = component(
            "inline-media-header.html",
            {"item": {**record, "href": "#"}, "label": "Video details"},
            replace=True,
        )
        header["inline_media_header"] = True
        block += header
        youtube_id = record["media"]["youtube_id"]
        lines = [
            f".. youtube:: {youtube_id}",
            "   :width: 100%",
            "   :aspect: 16:9",
            "   :privacy_mode:",
            "",
        ]
        directive.state.nested_parse(StringList(lines), 0, block)
    return {record["id"] for record in playable}


def add_inline_audio(directive, root, records):
    """Embed related published Audio records directly inside a topic section."""
    playable = [record for record in records if audio_media(record)]
    if not playable:
        return set()
    listing = nodes.container(classes=["learn-topic-audio-list"])
    for record in playable:
        block = nodes.container(classes=["learn-topic-audio"])
        header = component(
            "inline-media-header.html",
            {"record": record, "kind_label": "Audio"},
            replace=True,
        )
        header["inline_media_header"] = True
        block += header
        player = component(
            "audio-player.html",
            {"subject": record, "media": audio_media(record)},
            replace=True,
        )
        player["audio_player"] = True
        block += player
        listing += block
    root += listing
    return {record["id"] for record in playable}


def add_cards(directive, root, records):
    """Render catalog cards, promoting validated video media to native embeds."""
    listing = nodes.container(classes=["learn-items"])
    root += listing
    catalog = lookup(directive)
    for record in records:
        entry = nodes.container(classes=["learn-entry"])
        listing += entry
        item = {
            **record,
            "href": "#",
            "source_count": len(source_records(record, catalog)),
        }
        media = record.get("media", {})

        # Media indexes should lead with the validated artifact itself instead of
        # duplicating a prose summary. Video and Whiteboard therefore share one
        # semantic card hierarchy: metadata -> linked title -> native media.
        if record.get("kind") == "video" and media.get("youtube_id"):
            card = component(
                "media-card-start.html",
                {"item": item},
                "media-card-end.html",
            )
            card["media_card"] = True
            entry += card
            lines = [
                f".. youtube:: {media['youtube_id']}",
                "   :width: 100%",
                "   :aspect: 16:9",
                "   :privacy_mode:",
                "",
            ]
            directive.state.nested_parse(StringList(lines), 0, card)
            continue

        preview = (
            whiteboard_images(record)[0]
            if record.get("kind") == "whiteboard" and whiteboard_images(record)
            else None
        )
        if preview:
            card = component(
                "media-card-start.html",
                {"item": item},
                "media-card-end.html",
            )
            card["media_card"] = True
            entry += card
            image = nodes.image(
                uri=preview["image"],
                alt=preview["alt"],
                classes=["learn-whiteboard-image", "learn-whiteboard-index-image"],
            )
            image["width"] = "100%"
            card += image
            continue

        card = component("card.html", {"item": item}, replace=True)
        card["catalog_card"] = True
        entry += card


COLLECTION_MODES = {
    "want-to-read": "Want to Read",
    "currently-reading": "Currently Reading",
    "completed": "Completed",
}


class UserLibraryDirective(SphinxDirective):
    """Render browser-local bookmarks and reading-status collections."""

    required_arguments = 1

    def run(self):
        mode = self.arguments[0]
        if mode not in {"bookmarks", "collections", *COLLECTION_MODES}:
            raise self.error("Unknown user library mode")
        records = sorted(
            lookup(self).values(), key=lambda s: (s["kind"], s["title"].lower())
        )
        context = {
            "mode": mode,
            "collection_name": COLLECTION_MODES.get(mode, ""),
            "site_id": self.config.ai_learn_site_id,
            "revision": self.env.app._ai_learn_catalog["revision"],
            "records": [
                {
                    "id": record["id"],
                    "kind": record["kind"],
                    "title": record["title"],
                    "summary": record["summary"],
                    "domains": record["domains"],
                    "href": "#",
                }
                for record in records
            ],
            "collection_links": [
                {
                    "mode": slug,
                    "title": title,
                    "description": {
                        "want-to-read": (
                            "Learning records you plan to study or revisit."
                        ),
                        "currently-reading": (
                            "Learning records you are actively working through."
                        ),
                        "completed": "Learning records you have finished reviewing.",
                    }[slug],
                    "href": "#",
                }
                for slug, title in COLLECTION_MODES.items()
            ],
            "bookmarks_href": "#",
            "collections_href": "#",
            "payload": "",
        }
        context["payload"] = safe_json(
            {
                "mode": mode,
                "collection_name": context["collection_name"],
                "site_id": context["site_id"],
                "revision": context["revision"],
                "records": context["records"],
            }
        )
        root = component("user-library.html", context, replace=True)
        root["user_library"] = True
        paragraphs(
            root,
            "Bookmarks and reading-status collections are browser-local preferences. "
            "Open the HTML site to view or change them.",
        )
        return [root]


class IndexExplorerHeaderDirective(SphinxDirective):
    """Render the shared index explorer masthead and library shortcuts."""

    has_content = False
    option_spec: ClassVar = {
        "kicker": directives.unchanged_required,
        "title": directives.unchanged_required,
    }

    def run(self):
        root = component(
            "index-explorer-header.html",
            {
                "kicker": self.options["kicker"],
                "explorer_title": self.options["title"],
                "bookmarks_href": "#",
                "collections_href": "#",
            },
            replace=True,
        )
        root["index_explorer_header"] = True
        return [root]


class ExplorerDirective(SphinxDirective):
    required_arguments = 1
    option_spec: ClassVar = {
        "offset": directives.nonnegative_int,
        "search-variant": search_variant_option,
        "search_variant": search_variant_option,
    }

    def run(self):
        kind = self.arguments[0]
        if kind not in DIRECTORIES:
            raise self.error("Unknown explorer kind")
        records = sorted(
            (s for s in lookup(self).values() if s["kind"] == kind),
            key=lambda s: s["id"],
        )
        size = page_size(kind)
        offset = self.options.get("offset", 0)
        selected = records[offset : offset + size]
        page = offset // size
        try:
            search_control_variant = resolve_search_variant(
                self.options,
                self.config.ai_learn_explorer_search_variant,
                activation_keys=(),
            )
        except ValueError as exc:
            raise self.error(str(exc)) from exc
        context = {
            "kind": kind,
            "label": LABELS[kind],
            "total": len(records),
            "count": len(selected),
            "offset": offset,
            "page_chunk_size": size,
            "display_sizes": DISPLAY_SIZES,
            "tags": sorted({tag for s in selected for tag in s["domains"]}),
            "previous_page": ("index" if page == 1 else f"page-{page}") if page else "",
            "next_page": f"page-{page + 2}" if offset + size < len(records) else "",
            "previous": "",
            "next": "",
            "section_registry": safe_json(
                topic_sections(prompts=_prompts(self.env.app)),
            ),
            "search_control_variant": search_control_variant,
        }
        catalog = lookup(self)
        if kind == "topic":
            context.update(
                {
                    "sort_options": [
                        {"value": "created", "label": "Added"},
                        {"value": "title", "label": "Topic"},
                        {"value": "citations", "label": "Citations"},
                        {"value": "youtube", "label": "YouTube"},
                        {"value": "github", "label": "GitHub"},
                        {"value": "reddit", "label": "Reddit"},
                        {"value": "hackernews", "label": "Hacker News"},
                        {"value": "x", "label": "X"},
                    ],
                }
            )
            context["records"] = [
                {
                    **record,
                    "href": "#",
                    "source_count": len(source_records(record, catalog)),
                    "metrics": {
                        key: record.get("metrics", {}).get(key, 0)
                        for key in (
                            "citations",
                            "youtube",
                            "github",
                            "reddit",
                            "hackernews",
                            "x",
                        )
                    },
                }
                for record in selected
            ]
            root = component("topic-explorer.html", context, replace=True)
            root["explorer"] = True
            root["topic_table"] = True
            add_cards(self, root, selected)
            return [root]
        if kind in {"problem", "source", "skill"}:
            settings = {
                "problem": {
                    "item_label": "open problem",
                    "item_heading": "Open problem",
                    "explorer_title": "Open Problems Explorer",
                    "search_placeholder": "Problem, keyword, or status",
                    "numeric_sort": ["references"],
                    "sort_options": [
                        {"value": "created", "label": "Added"},
                        {"value": "title", "label": "Open problem"},
                        {"value": "status", "label": "Status"},
                        {"value": "references", "label": "Linked references"},
                    ],
                },
                "source": {
                    "item_label": "source",
                    "item_heading": "Source",
                    "explorer_title": "Source Library",
                    "search_placeholder": "Source, publisher, format, or keyword",
                    "numeric_sort": ["related"],
                    "sort_options": [
                        {"value": "created", "label": "Added"},
                        {"value": "title", "label": "Source"},
                        {"value": "publisher", "label": "Publisher"},
                        {"value": "format", "label": "Format"},
                        {"value": "related", "label": "Related records"},
                    ],
                },
                "skill": {
                    "item_label": "skill",
                    "item_heading": "Skill",
                    "explorer_title": "Skills Library",
                    "search_placeholder": "Skill, workflow, or keyword",
                    "numeric_sort": ["sections", "related"],
                    "sort_options": [
                        {"value": "created", "label": "Added"},
                        {"value": "title", "label": "Skill"},
                        {"value": "sections", "label": "Sections"},
                        {"value": "related", "label": "Related records"},
                    ],
                },
            }[kind]
            context.update(settings)
            enriched = []
            for record in selected:
                item = {
                    **record,
                    "href": "#",
                    "related_count": len(related_records(record, catalog)),
                    "section_count": len(record.get("sections", [])),
                }
                if kind == "problem":
                    item["status"] = record.get("status", "unverified")
                    item["source_count"] = len(source_records(record, catalog))
                elif kind == "source":
                    item["publisher"] = record.get("publisher", "")
                    item["format"] = record.get("format", "")
                enriched.append(item)
            context["records"] = enriched
            root = component("catalog-explorer.html", context, replace=True)
            root["explorer"] = True
            root["catalog_table"] = True
            add_cards(self, root, selected)
            return [root]
        card_settings = {
            "video": ("video", "videos", "Video, title, or keyword"),
            "audio": ("audio item", "audio items", "Audio, title, or keyword"),
            "document": ("document", "documents", "Document, title, or keyword"),
            "whiteboard": (
                "whiteboard",
                "whiteboards",
                "Whiteboard, title, or keyword",
            ),
        }
        item_label, item_plural, search_placeholder = card_settings.get(
            kind,
            (
                kind.replace("-", " "),
                kind.replace("-", " ") + "s",
                "Title, category, or keyword",
            ),
        )
        context.update(
            {
                "item_label": item_label,
                "item_plural": item_plural,
                "search_placeholder": search_placeholder,
                "filter_panel_id": f"learn-{kind}-filter-options",
                "search_input_id": f"learn-{kind}-search",
                "sort_options": [
                    {"value": "created", "label": "Added"},
                    {"value": "title", "label": "Title"},
                ],
            }
        )
        root = component("explorer-start.html", context, "explorer-end.html")
        root["explorer"] = True
        add_cards(self, root, selected)
        return [root]


def _generation_href(app, docname, kind, **params):
    """Resolve one allowlisted generation page with bounded query context."""
    leaf = _STUDIO_LEAF_BY_KIND.get(kind)
    if not leaf:
        return "#"
    target = app.config.ai_learn_content_root + "/" + leaf
    if target not in app.env.found_docs:
        return "#"
    href = app.builder.get_relative_uri(docname, target)
    query = {key: str(value) for key, value in params.items() if value}
    return href + ("?" + urlencode(query) if query else "")


def _video_generation_href(app, docname, **params):
    return _generation_href(app, docname, "video", **params)


def _audio_generation_href(app, docname, **params):
    return _generation_href(app, docname, "audio", **params)


def _document_generation_href(app, docname, **params):
    return _generation_href(app, docname, "document", **params)


def _whiteboard_generation_href(app, docname, **params):
    return _generation_href(app, docname, "whiteboard", **params)


def _source_creation_href(app, docname, **params):
    return _generation_href(app, docname, "source", **params)


def resolve(  # ruff: ignore[too-many-branches]
    app,
    doctree,
    docname,
):
    if app.builder.format != "html":
        return
    catalog = {s["id"]: s for s in app._ai_learn_catalog["subjects"]}
    for marker in doctree.findall(PageMarker):
        marker["payload"]["search"] = [
            {
                "id": s["id"],
                "title": s["title"],
                "kind": s["kind"],
                "summary": s["summary"],
                "url": s.get("url", ""),
                "publisher": s.get("publisher", ""),
                "domains": s["domains"],
                "href": record_href(app, docname, s),
            }
            for s in catalog.values()
        ]
    for item in doctree.findall(Component):
        ctx = item["context"]
        if "related" in ctx:
            ctx["related"] = [
                {**s, "href": record_href(app, docname, s)} for s in ctx["related"]
            ]
        if item.get("overview_component") or item.get("index_explorer_header"):
            for key, leaf in (
                ("bookmarks_href", "bookmarks/index"),
                ("collections_href", "collections/index"),
            ):
                target = app.config.ai_learn_content_root + "/" + leaf
                ctx[key] = (
                    app.builder.get_relative_uri(docname, target)
                    if target in app.env.found_docs
                    else "#"
                )
            if item.get("overview_component") and ctx.get("subject", {}).get(
                "kind"
            ) in {"topic", "source"}:
                subject = ctx["subject"]
                ctx["video_generation_href"] = _video_generation_href(
                    app, docname, mode=subject["kind"], id=subject["id"]
                )
                ctx["audio_generation_href"] = _audio_generation_href(
                    app, docname, mode=subject["kind"], id=subject["id"]
                )
                ctx["document_generation_href"] = _document_generation_href(
                    app, docname, mode=subject["kind"], id=subject["id"]
                )
                ctx["whiteboard_generation_href"] = _whiteboard_generation_href(
                    app, docname, mode=subject["kind"], id=subject["id"]
                )
        if item.get("topic_table") or item.get("catalog_table"):
            for record in ctx.get("records", []):
                record["href"] = record_href(app, docname, catalog[record["id"]])
        if item.get("media_actions"):
            if ctx.get("subject", {}).get("kind") == "video":
                ctx["video_generation_href"] = _video_generation_href(
                    app, docname, **{"from": "video", "id": ctx["subject"]["id"]}
                )
            elif ctx.get("subject", {}).get("kind") == "audio":
                ctx["audio_generation_href"] = _audio_generation_href(
                    app, docname, **{"from": "audio", "id": ctx["subject"]["id"]}
                )
            elif ctx.get("subject", {}).get("kind") == "document":
                ctx["document_generation_href"] = _document_generation_href(
                    app, docname, **{"from": "document", "id": ctx["subject"]["id"]}
                )
            elif ctx.get("subject", {}).get("kind") == "whiteboard":
                ctx["whiteboard_generation_href"] = _whiteboard_generation_href(
                    app, docname, **{"from": "whiteboard", "id": ctx["subject"]["id"]}
                )
            target = ctx.get("context_target")
            if target:
                ctx["context_target"] = {
                    **target,
                    "href": record_href(app, docname, target),
                }
            bookmarks_target = app.config.ai_learn_content_root + "/bookmarks/index"
            collections_target = app.config.ai_learn_content_root + "/collections/index"
            ctx["bookmarks_href"] = (
                app.builder.get_relative_uri(docname, bookmarks_target)
                if bookmarks_target in app.env.found_docs
                else "#"
            )
            ctx["collections_href"] = (
                app.builder.get_relative_uri(docname, collections_target)
                if collections_target in app.env.found_docs
                else "#"
            )
        if item.get("video_generator"):
            for group in ("topics", "sources", "videos"):
                ctx[group] = [
                    {**record, "href": record_href(app, docname, catalog[record["id"]])}
                    for record in ctx.get(group, [])
                ]
            ctx["payload"] = safe_json(
                {
                    "contract": "learn.video-generation-page.v1",
                    "site_id": ctx["site_id"],
                    "revision": ctx["revision"],
                    "runtime": ctx["runtime"],
                    "topics": [
                        {
                            "id": row["id"],
                            "title": row["title"],
                            "summary": row.get("summary", ""),
                            "domains": row.get("domains", []),
                            "related": row.get("related", []),
                            "href": row["href"],
                        }
                        for row in ctx["topics"]
                    ],
                    "sources": [
                        {
                            "id": row["id"],
                            "title": row["title"],
                            "summary": row.get("summary", ""),
                            "publisher": row.get("publisher", ""),
                            "format": row.get("format", ""),
                            "url": row.get("url", ""),
                            "href": row["href"],
                        }
                        for row in ctx["sources"]
                    ],
                    "videos": [
                        {
                            "id": row["id"],
                            "title": row["title"],
                            "summary": row.get("summary", ""),
                            "related": row.get("related", []),
                            "href": row["href"],
                        }
                        for row in ctx["videos"]
                    ],
                }
            )
        if item.get("audio_generator"):
            for group in ("topics", "sources", "audio"):
                ctx[group] = [
                    {**record, "href": record_href(app, docname, catalog[record["id"]])}
                    for record in ctx.get(group, [])
                ]
            ctx["payload"] = safe_json(
                {
                    "contract": "learn.audio-generation-page.v1",
                    "site_id": ctx["site_id"],
                    "revision": ctx["revision"],
                    "runtime": ctx["runtime"],
                    "topics": [
                        {
                            "id": row["id"],
                            "title": row["title"],
                            "summary": row.get("summary", ""),
                            "href": row["href"],
                        }
                        for row in ctx["topics"]
                    ],
                    "sources": [
                        {
                            "id": row["id"],
                            "title": row["title"],
                            "summary": row.get("summary", ""),
                            "href": row["href"],
                        }
                        for row in ctx["sources"]
                    ],
                    "audio": [
                        {
                            "id": row["id"],
                            "title": row["title"],
                            "summary": row.get("summary", ""),
                            "href": row["href"],
                        }
                        for row in ctx["audio"]
                    ],
                }
            )
        if item.get("document_generator"):
            for group in ("topics", "sources", "documents"):
                ctx[group] = [
                    {**record, "href": record_href(app, docname, catalog[record["id"]])}
                    for record in ctx.get(group, [])
                ]
            ctx["payload"] = safe_json(
                {
                    "contract": "learn.document-generation-page.v1",
                    "site_id": ctx["site_id"],
                    "revision": ctx["revision"],
                    "runtime": ctx["runtime"],
                    "topics": [
                        {k: row.get(k, "") for k in ("id", "title", "summary", "href")}
                        for row in ctx["topics"]
                    ],
                    "sources": [
                        {
                            k: row.get(k, "")
                            for k in ("id", "title", "summary", "url", "href")
                        }
                        for row in ctx["sources"]
                    ],
                    "documents": [
                        {k: row.get(k, "") for k in ("id", "title", "summary", "href")}
                        for row in ctx["documents"]
                    ],
                }
            )
        if item.get("whiteboard_generator"):
            for group in ("topics", "sources", "whiteboards"):
                ctx[group] = [
                    {**record, "href": record_href(app, docname, catalog[record["id"]])}
                    for record in ctx.get(group, [])
                ]
            ctx["payload"] = safe_json(
                {
                    "contract": "learn.whiteboard-generation-page.v1",
                    "site_id": ctx["site_id"],
                    "revision": ctx["revision"],
                    "runtime": ctx["runtime"],
                    "topics": [
                        {k: row.get(k, "") for k in ("id", "title", "summary", "href")}
                        for row in ctx["topics"]
                    ],
                    "sources": [
                        {
                            k: row.get(k, "")
                            for k in ("id", "title", "summary", "url", "href")
                        }
                        for row in ctx["sources"]
                    ],
                    "whiteboards": [
                        {k: row.get(k, "") for k in ("id", "title", "summary", "href")}
                        for row in ctx["whiteboards"]
                    ],
                }
            )
        if item.get("user_library"):
            for record in ctx.get("records", []):
                record["href"] = record_href(app, docname, catalog[record["id"]])
            collections_target = app.config.ai_learn_content_root + "/collections/index"
            bookmarks_target = app.config.ai_learn_content_root + "/bookmarks/index"
            ctx["collections_href"] = (
                app.builder.get_relative_uri(docname, collections_target)
                if collections_target in app.env.found_docs
                else "#"
            )
            ctx["bookmarks_href"] = (
                app.builder.get_relative_uri(docname, bookmarks_target)
                if bookmarks_target in app.env.found_docs
                else "#"
            )
            for link in ctx.get("collection_links", []):
                target = (
                    app.config.ai_learn_content_root
                    + "/collections/"
                    + link["mode"]
                    + "/index"
                )
                link["href"] = (
                    app.builder.get_relative_uri(docname, target)
                    if target in app.env.found_docs
                    else "#"
                )
            ctx["payload"] = safe_json(
                {
                    "mode": ctx["mode"],
                    "collection_name": ctx.get("collection_name", ""),
                    "site_id": ctx["site_id"],
                    "revision": ctx["revision"],
                    "records": ctx["records"],
                }
            )
        if "item" in ctx:
            ctx["item"]["href"] = record_href(app, docname, ctx["item"])
        if item.get("section_component"):
            ctx["source_creation_href"] = _source_creation_href(
                app,
                docname,
                subject=ctx.get("subject", {}).get("id", ""),
                section=ctx.get("spec", {}).get("id", ""),
            )
        if item.get("section_component") and ctx.get("spec", {}).get("id") == "video":
            target = app.config.ai_learn_content_root + "/videos/index"
            ctx["videos_href"] = (
                app.builder.get_relative_uri(docname, target)
                if target in app.env.found_docs
                else "#"
            )
            ctx["video_generation_href"] = _video_generation_href(
                app, docname, mode="topic", id=ctx["subject"]["id"]
            )
        if item.get("section_component") and ctx.get("spec", {}).get("id") == "audio":
            target = app.config.ai_learn_content_root + "/audios/index"
            ctx["audio_href"] = (
                app.builder.get_relative_uri(docname, target)
                if target in app.env.found_docs
                else "#"
            )
            ctx["audio_generation_href"] = _audio_generation_href(
                app, docname, mode="topic", id=ctx["subject"]["id"]
            )
        if (
            item.get("section_component")
            and ctx.get("spec", {}).get("id") == "document"
        ):
            target = app.config.ai_learn_content_root + "/documents/index"
            ctx["document_href"] = (
                app.builder.get_relative_uri(docname, target)
                if target in app.env.found_docs
                else "#"
            )
            ctx["document_generation_href"] = _document_generation_href(
                app, docname, mode="topic", id=ctx["subject"]["id"]
            )
        if (
            item.get("section_component")
            and ctx.get("spec", {}).get("id") == "whiteboard"
        ):
            target = app.config.ai_learn_content_root + "/whiteboards/index"
            ctx["whiteboard_href"] = (
                app.builder.get_relative_uri(docname, target)
                if target in app.env.found_docs
                else "#"
            )
            ctx["whiteboard_generation_href"] = _whiteboard_generation_href(
                app, docname, mode="topic", id=ctx["subject"]["id"]
            )
        if item.get("prompt_group") or item.get("prompt_library"):
            for prompt in ctx.get("prompts", []):
                target = (
                    app.config.ai_learn_content_root
                    + "/topic-prompts/"
                    + prompt["id"]
                    + "/index"
                )
                prompt["href"] = (
                    app.builder.get_relative_uri(docname, target)
                    if target in app.env.found_docs
                    else "#"
                )
        if item.get("prompt_card"):
            prompt = ctx["prompt"]
            target = (
                app.config.ai_learn_content_root
                + "/topic-prompts/"
                + prompt["id"]
                + "/index"
            )
            prompt["href"] = (
                app.builder.get_relative_uri(docname, target)
                if target in app.env.found_docs
                else "#"
            )
        if item.get("prompt_group"):
            target = app.config.ai_learn_content_root + "/topic-prompts/index"
            ctx["prompts_href"] = (
                app.builder.get_relative_uri(docname, target)
                if target in app.env.found_docs
                else "#"
            )
        if item.get("prompt_detail"):
            target = app.config.ai_learn_content_root + "/topic-prompts/index"
            ctx["library_href"] = (
                app.builder.get_relative_uri(docname, target)
                if target in app.env.found_docs
                else "#"
            )
        if item.get("skill_group") or item.get("skill_library"):
            for skill in ctx.get("skills", []):
                target = (
                    app.config.ai_learn_content_root
                    + "/skills/"
                    + skill["id"]
                    + "/index"
                )
                skill["href"] = (
                    app.builder.get_relative_uri(docname, target)
                    if target in app.env.found_docs
                    else "#"
                )
        if item.get("skill_card"):
            skill = ctx["skill"]
            target = (
                app.config.ai_learn_content_root + "/skills/" + skill["id"] + "/index"
            )
            skill["href"] = (
                app.builder.get_relative_uri(docname, target)
                if target in app.env.found_docs
                else "#"
            )
        if item.get("skill_group"):
            target = app.config.ai_learn_content_root + "/skills/index"
            ctx["skills_href"] = (
                app.builder.get_relative_uri(docname, target)
                if target in app.env.found_docs
                else "#"
            )
        if item.get("skill_detail"):
            target = app.config.ai_learn_content_root + "/skills/index"
            ctx["library_href"] = (
                app.builder.get_relative_uri(docname, target)
                if target in app.env.found_docs
                else "#"
            )
        if item.get("explorer"):
            parent = docname.rsplit("/", 1)[0]
            for key in ("next", "previous"):
                if ctx[key + "_page"]:
                    ctx[key] = app.builder.get_relative_uri(
                        docname, parent + "/" + ctx[key + "_page"]
                    )


def page_context(app, pagename, templatename, context, doctree):
    if doctree is None:
        return None
    markers = list(doctree.findall(PageMarker))
    components = any(doctree.findall(Component))
    if not markers and not components:
        return None
    if not markers:
        return None
    subject = markers[0]["subject"]
    specs = sections(subject, app)
    for spec in specs:
        spec["anchor"] = "learn-" + subject["id"] + "-" + spec["id"]
    # Only include real RST targets, allowing authors to use a subset manually.
    ids = {
        identity
        for node in doctree.findall(nodes.Element)
        for identity in node.get("ids", [])
    }
    specs = [s for s in specs if s["anchor"] in ids]
    catalog = {s["id"]: s for s in app._ai_learn_catalog["subjects"]}
    related_kind = "topic"
    related = [
        s for s in related_records(subject, catalog) if s["kind"] == related_kind
    ][:6]
    context.update(
        learn_subject=subject,
        learn_sections=specs,
        learn_revision=app._ai_learn_catalog["revision"],
        learn_site=app.config.ai_learn_site_id,
        learn_related=[{**s, "href": record_href(app, pagename, s)} for s in related],
    )
    if app.config.html_theme == "pydata_sphinx_theme":
        context["secondary_sidebar_items"] = ["learn/toc.html"]
    return "learn/page.html"


def setup_pages(app):
    # Register the scoped Learn page assets during extension setup, before any
    # HTML page is rendered. Registering from ``html-page-context`` is too late
    # for the current page in some Sphinx/theme combinations and makes behavior
    # depend on document build order (for example an index page may work while
    # the immediately preceding detail page has dead controls). The CSS/JS are
    # selector-scoped and no-op on pages without Learn markup.
    app.add_css_file("topic.css")
    app.add_js_file("generation-ui.js", defer="defer", priority=606)
    app.add_js_file("text-generation-ui.js", defer="defer", priority=607)
    app.add_js_file("evidence-review.js", defer="defer", priority=608)
    app.add_js_file("section-generation.js", defer="defer", priority=609)
    app.add_js_file("overview-generation.js", defer="defer", priority=610)
    app.add_js_file("topic.js", defer="defer", priority=611)
    app.add_js_file("generation-feedback.js", defer="defer", priority=612)
    app.add_js_file("generation-context.js", defer="defer", priority=613)
    app.add_js_file("video-generation.js", defer="defer", priority=615)
    app.add_js_file("audio-generation.js", defer="defer", priority=616)
    app.add_js_file("document-generation.js", defer="defer", priority=617)
    app.add_js_file("whiteboard-generation.js", defer="defer", priority=618)
    app.add_js_file("record-generation.js", defer="defer", priority=619)
    app.add_js_file("lens-selection.js", defer="defer", priority=620)
    app.add_node(
        PageMarker,
        html=(visit_marker, passthrough),
        text=(skip, passthrough),
        latex=(skip, passthrough),
        man=(skip, passthrough),
        texinfo=(skip, passthrough),
    )
    app.add_node(
        Component,
        html=(visit_component, depart_component),
        text=(passthrough, passthrough),
        latex=(passthrough, passthrough),
        man=(passthrough, passthrough),
        texinfo=(passthrough, passthrough),
    )
    app.add_directive("ai-topic-page", PageDirective)
    app.add_directive("ai-topic-overview", OverviewDirective)
    app.add_directive("ai-media-actions", MediaActionsDirective)
    app.add_directive("ai-video-generation", VideoGenerationDirective)
    app.add_directive("ai-audio-generation", AudioGenerationDirective)
    app.add_directive("ai-document-generation", DocumentGenerationDirective)
    app.add_directive("ai-whiteboard-generation", WhiteboardGenerationDirective)
    app.add_directive("ai-record-generation", RecordGenerationDirective)
    app.add_directive("ai-audio-player", AudioPlayerDirective)
    app.add_directive("ai-document-viewer", DocumentViewerDirective)
    app.add_directive("ai-topic-section", SectionDirective)
    app.add_directive("ai-topic-prompts", PromptGroupDirective)
    app.add_directive("ai-topic-prompt-library", PromptLibraryDirective)
    app.add_directive("ai-topic-prompt-detail", PromptDetailDirective)
    app.add_directive("ai-topic-skills", SkillGroupDirective)
    app.add_directive("ai-skill-library", SkillLibraryDirective)
    app.add_directive("ai-skill-detail", SkillDetailDirective)
    app.add_directive("ai-whiteboard-gallery", WhiteboardGalleryDirective)
    app.add_directive("ai-topic-user-library", UserLibraryDirective)
    app.add_directive("ai-index-explorer-header", IndexExplorerHeaderDirective)
    app.add_directive("ai-topic-explorer", ExplorerDirective)
    app.connect("doctree-resolved", resolve)
    app.connect("html-page-context", page_context, priority=900)
