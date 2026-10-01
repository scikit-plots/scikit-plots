# scikitplot/_externals/_sphinx_ext/_sphinx_ai_learn/_schema.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Bounded, Sphinx-free contracts for normalized learning data and drafts.

Canonical repository input is validated by the JSON artifact-tree materializer.
Bodies remain plain text and are never interpreted as RST or HTML.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

CATALOG_CONTRACT = "learn.catalog.v3"
CONTRIBUTION_CONTRACT = "learn.contribution.v1"
MAX_BYTES = 2 * 1024 * 1024
MAX_SUBJECTS = 1000
MAX_SUBJECT_SECTIONS = 256
MAX_SECTION_GENERATIONS = 32
MAX_GENERATION_FEEDBACK = 256
MAX_WHITEBOARD_IMAGES = 100
_ASCII_SPACE = 32
_ASCII_DELETE = 127
KINDS = (
    "topic",
    "problem",
    "source",
    "skill",
    "whiteboard",
    "video",
    "audio",
    "document",
)
ID_PATTERN = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
# Public reviewed-feedback event ids are current 192-bit browser CSPRNG nonces.
# Older event-id shapes are intentionally not accepted by the upgraded system.
FEEDBACK_EVENT_ID_PATTERN = re.compile(r"feedback-[0-9a-f]{48}\Z")
# Sidecar storage is append-only, so cap one generation independently of the
# global JSON-file ceiling.  This still supports four-digit community counts
# while preventing one hot generation from monopolizing build memory/files.
MAX_REVIEWED_FEEDBACK_EVENTS = 4096
from ._registry import TOPIC_SECTIONS_AFTER_INTERACTIONS, TOPIC_SECTIONS_BEFORE_PROMPTS

SECTION_PRESETS = tuple(
    (row[0], row[1])
    for row in (*TOPIC_SECTIONS_BEFORE_PROMPTS, *TOPIC_SECTIONS_AFTER_INTERACTIONS)
    if row[3] == "text"
)

DOMAINS = (
    "artificial-intelligence",
    "machine-learning",
    "deep-learning",
    "statistics",
    "data-science",
    "data-analytics",
    "data-visualization",
    "data-engineering",
)


class LearnValidationError(ValueError):
    """A stable contract error with a field path, without echoing input."""


def _object(value, allowed, required, path):
    if not isinstance(value, dict):
        raise LearnValidationError(f"{path}: expected object")
    if set(value) - set(allowed) or set(required) - set(value):
        raise LearnValidationError(f"{path}: unexpected or missing fields")
    return value


def _text(value, path, limit, *, empty=False, multiline=False):
    if not isinstance(value, str) or len(value) > limit:
        raise LearnValidationError(f"{path}: invalid text length or type")
    if any(
        (ord(c) < _ASCII_SPACE and not (multiline and c in "\n\t\r"))
        or ord(c) == _ASCII_DELETE
        for c in value
    ):
        raise LearnValidationError(f"{path}: control characters are not allowed")
    value = value.strip()
    if not empty and not value:
        raise LearnValidationError(f"{path}: text is required")
    return value


def _id(value, path):
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise LearnValidationError(f"{path}: invalid identifier")
    return value


def _list(value, path, maximum):
    if not isinstance(value, list) or len(value) > maximum:
        raise LearnValidationError(f"{path}: invalid list")
    return value


def _ids(value, path, maximum):
    rows = [_id(x, path) for x in _list(value, path, maximum)]
    if len(rows) != len(set(rows)):
        raise LearnValidationError(f"{path}: duplicate identifiers")
    return rows


def public_url(value, path="url"):
    """Validate a passive public link; this function performs no network I/O."""
    value = _text(value, path, 2048)
    try:
        parsed = urlsplit(value)
        port = parsed.port
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or "\\" in value
            or any(c.isspace() for c in value)
            or port not in (None, 443)
        ):
            raise ValueError
    except ValueError as exc:
        raise LearnValidationError(
            f"{path}: expected HTTPS URL without credentials"
        ) from exc
    return value


def _citations(value, path):
    result = []
    for row in _list(value, path, 32):
        _object(row, ("source_id", "locator"), ("source_id", "locator"), path)
        result.append(
            {
                "source_id": _id(row["source_id"], path + ".source_id"),
                "locator": _text(row["locator"], path + ".locator", 500),
            }
        )
    return result


def _links(value, path):
    """Validate passive, source-authored links shown inside social sections."""
    result = []
    for row in _list(value, path, 32):
        _object(row, ("title", "url", "meta"), ("title", "url"), path)
        item = {
            "title": _text(row["title"], path + ".title", 300),
            "url": public_url(row["url"], path + ".url"),
        }
        if "meta" in row:
            item["meta"] = _text(row["meta"], path + ".meta", 300, empty=True)
        result.append(item)
    return result


def _public_credits(value, path, maximum=32):
    """Validate bounded public display credits without implying identity."""
    credits = [_text(item, path, 80) for item in _list(value, path, maximum)]
    folded = [item.casefold() for item in credits]
    if len(folded) != len(set(folded)):
        raise LearnValidationError(f"{path}: duplicate display names")
    return credits


def _section_generations(value, path):
    """Validate immutable accepted section generations."""
    generations = []
    seen = set()
    for index, raw in enumerate(_list(value, path, MAX_SECTION_GENERATIONS)):
        where = f"{path}[{index}]"
        _object(
            raw,
            (
                "id",
                "created_at",
                "body",
                "citations",
                "links",
                "contributors",
                "review",
                "provenance",
            ),
            ("id", "created_at", "body", "contributors"),
            where,
        )
        generation_id = _id(raw["id"], where + ".id")
        if generation_id in seen:
            raise LearnValidationError(f"{path}: duplicate generation identifier")
        seen.add(generation_id)
        item = {
            "id": generation_id,
            "created_at": timestamp(raw["created_at"], where + ".created_at"),
            "body": _text(
                raw["body"], where + ".body", 50000, empty=True, multiline=True
            ),
            "citations": _citations(raw.get("citations", []), where + ".citations"),
            "links": _links(raw.get("links", []), where + ".links"),
            "contributors": _public_credits(
                raw.get("contributors", []), where + ".contributors", 32
            ),
        }
        if not item["contributors"]:
            raise LearnValidationError(
                f"{where}.contributors: expected at least one public credit"
            )
        if "review" in raw:
            review = raw["review"]
            _object(
                review,
                ("at", "by", "revision"),
                ("at", "by", "revision"),
                where + ".review",
            )
            if not item["citations"]:
                raise LearnValidationError(
                    f"{where}.review: evidence citations required"
                )
            item["review"] = {
                "at": timestamp(review["at"], where + ".review.at"),
                "by": _text(review["by"], where + ".review.by", 200),
                "revision": _text(review["revision"], where + ".review.revision", 128),
            }
        if "provenance" in raw:
            provenance = raw["provenance"]
            if not isinstance(provenance, dict):
                raise LearnValidationError(f"{where}.provenance: expected object")
            allowed = {
                "authorship",
                "model",
                "workflow_id",
                "agent",
                "skill",
                "request_id",
            }
            if set(provenance) - allowed:
                raise LearnValidationError(f"{where}.provenance: unexpected fields")
            clean = {}
            for key, limit in (
                ("authorship", 40),
                ("model", 200),
                ("workflow_id", 120),
                ("agent", 120),
                ("skill", 120),
                ("request_id", 200),
            ):
                if key in provenance:
                    clean[key] = _text(
                        provenance[key],
                        where + ".provenance." + key,
                        limit,
                        empty=True,
                    )
            if clean:
                item["provenance"] = clean
        generations.append(item)
    if not generations:
        raise LearnValidationError(f"{path}: expected at least one generation")
    return generations


def _metrics(value, path="subject.metrics"):
    allowed = ("citations", "youtube", "github", "reddit", "hackernews", "x")
    _object(value, allowed, (), path)
    result = {}
    for key, raw in value.items():
        if (
            not isinstance(raw, int)
            or isinstance(raw, bool)
            or raw < 0
            or raw > 2_000_000_000  # ruff: ignore[magic-value-comparison]
        ):
            raise LearnValidationError(f"{path}.{key}: expected non-negative integer")
        result[key] = raw
    return result


def _local_media_image(value, path):
    """Validate one passive, local image used by a whiteboard gallery."""
    _object(value, ("image", "alt", "caption"), ("image", "alt"), path)
    image = _text(value["image"], path + ".image", 500)
    if (
        not re.fullmatch(r"/_static/[a-zA-Z0-9_./-]+\.(png|jpg|jpeg|webp|svg)", image)
        or ".." in Path(image).parts
    ):
        raise LearnValidationError(f"{path}.image: expected a local /_static/ asset")
    result = {
        "image": image,
        "alt": _text(value["alt"], path + ".alt", 1000),
    }
    if "caption" in value:
        result["caption"] = _text(value["caption"], path + ".caption", 1000, empty=True)
    return result


def _local_media_audio(value, path="media"):
    """Validate one passive, durable local audio asset for published Learn pages."""
    _object(
        value,
        ("type", "src", "mime_type", "duration_seconds"),
        ("type", "src", "mime_type"),
        path,
    )
    if value["type"] != "audio":
        raise LearnValidationError(f"{path}.type: expected audio")
    src = _text(value["src"], path + ".src", 500)
    if (
        not re.fullmatch(r"/_static/[a-zA-Z0-9_./-]+\.(mp3|wav)", src)
        or ".." in Path(src).parts
    ):
        raise LearnValidationError(
            f"{path}.src: expected a local /_static/ .mp3 or .wav asset"
        )
    mime = _text(value["mime_type"], path + ".mime_type", 100)
    expected = "audio/mpeg" if src.lower().endswith(".mp3") else "audio/wav"
    if mime != expected:
        raise LearnValidationError(
            f"{path}.mime_type: does not match the local audio asset"
        )
    result = {"type": "audio", "src": src, "mime_type": mime}
    if "duration_seconds" in value:
        duration = value["duration_seconds"]
        if (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or duration <= 0
            or duration > 86_400  # ruff: ignore[magic-value-comparison]
        ):
            raise LearnValidationError(
                f"{path}.duration_seconds: expected a positive duration"
            )
        result["duration_seconds"] = float(duration)
    return result


def _local_media_document(value, path="media"):
    """Validate one passive local document asset; active HTML is not accepted."""
    _object(value, ("type", "src", "mime_type"), ("type", "src", "mime_type"), path)
    if value["type"] != "document":
        raise LearnValidationError(f"{path}.type: expected document")
    src = _text(value["src"], path + ".src", 500)
    match = re.fullmatch(r"/_static/[a-zA-Z0-9_./-]+\.(md|rst|txt|pdf)", src)
    if not match or ".." in Path(src).parts:
        raise LearnValidationError(
            f"{path}.src: expected a local /_static/ .md, .rst, .txt, or .pdf asset"
        )
    expected = {
        "md": "text/markdown",
        "rst": "text/x-rst",
        "txt": "text/plain",
        "pdf": "application/pdf",
    }[match.group(1).lower()]
    mime = _text(value["mime_type"], path + ".mime_type", 100)
    if mime != expected:
        raise LearnValidationError(
            f"{path}.mime_type: does not match the local document asset"
        )
    return {"type": "document", "src": src, "mime_type": mime}


def timestamp(value, path="created_at"):
    """Require a real UTC second, suitable for a stable document filename."""
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value
    ):
        raise LearnValidationError(f"{path}: expected UTC timestamp")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise LearnValidationError(f"{path}: invalid UTC timestamp") from exc
    return value


def interactions(value):
    """Local provenance claims, not verification or a full conversation log."""
    result = []
    for event in _list(value, "interactions", 32):
        _object(
            event,
            ("id", "at", "action", "section_id", "request_id", "workflow_id"),
            ("id", "at", "action", "section_id", "request_id", "workflow_id"),
            "interaction",
        )
        if event["action"] not in ("prompt-copied", "ai-result-selected"):
            raise LearnValidationError("interaction.action: unsupported action")
        if event["workflow_id"] != "learn.explanation.v1":
            raise LearnValidationError("interaction.workflow_id: unsupported workflow")
        result.append(
            {
                "id": _id(event["id"], "interaction.id"),
                "at": timestamp(event["at"], "interaction.at"),
                "action": event["action"],
                "section_id": _id(event["section_id"], "interaction.section_id"),
                "request_id": _id(event["request_id"], "interaction.request_id"),
                "workflow_id": event["workflow_id"],
            }
        )
    if len({event["id"] for event in result}) != len(result):
        raise LearnValidationError("interactions: duplicate identifier")
    return result


def validate_subject(  # ruff: ignore[too-many-branches]
    value,
):  # noqa: PLR0912 -- explicit bounded field validation
    """Return a fresh normalized subject with stable identifiers."""
    _object(
        value,
        (
            "id",
            "kind",
            "title",
            "summary",
            "domains",
            "related",
            "sections",
            "url",
            "created_at",
            "format",
            "publisher",
            "status",
            "media",
            "authors",
            "metrics",
        ),
        ("id", "kind", "title"),
        "subject",
    )
    if value["kind"] not in KINDS:
        raise LearnValidationError("subject.kind: unknown kind")
    result = {
        "id": _id(value["id"], "subject.id"),
        "kind": value["kind"],
        "title": _text(value["title"], "subject.title", 200),
        "summary": _text(
            value.get("summary", ""),
            "subject.summary",
            2000,
            empty=True,
            multiline=True,
        ),
        "domains": _ids(value.get("domains", []), "subject.domains", 20),
        "related": _ids(value.get("related", []), "subject.related", 100),
        "sections": [],
    }
    if "authors" in value:
        result["authors"] = _public_credits(value["authors"], "subject.authors", 32)
    if "metrics" in value:
        result["metrics"] = _metrics(value["metrics"])
    seen = set()
    for section in _list(
        value.get("sections", []),
        "subject.sections",
        MAX_SUBJECT_SECTIONS,
    ):
        _object(
            section,
            (
                "id",
                "title",
                "body",
                "citations",
                "links",
                "instructions",
                "expanded",
                "review",
                "contributors",
                "active_generation_id",
                "generations",
            ),
            ("id", "title", "body"),
            "section",
        )
        section_id = _id(section["id"], "section.id")
        if section_id in seen:
            raise LearnValidationError("section.id: duplicate identifier")
        seen.add(section_id)
        result["sections"].append(
            {
                "id": section_id,
                "title": _text(section["title"], "section.title", 200),
                "body": _text(
                    section["body"], "section.body", 50000, empty=True, multiline=True
                ),
                "citations": _citations(
                    section.get("citations", []), "section.citations"
                ),
                "links": _links(section.get("links", []), "section.links"),
            }
        )
        normalized = result["sections"][-1]
        if "instructions" in section:
            normalized["instructions"] = _text(
                section["instructions"],
                "section.instructions",
                2000,
                empty=True,
                multiline=True,
            )
        if "expanded" in section:
            if not isinstance(section["expanded"], bool):
                raise LearnValidationError("section.expanded: expected boolean")
            normalized["expanded"] = section["expanded"]
        if "contributors" in section:
            normalized["contributors"] = _public_credits(
                section["contributors"], "section.contributors", 32
            )
        if "review" in section:
            review = section["review"]
            _object(
                review,
                ("at", "by", "revision"),
                ("at", "by", "revision"),
                "section.review",
            )
            if not normalized["citations"]:
                raise LearnValidationError(
                    "section.review: evidence citations required"
                )
            normalized["review"] = {
                "at": timestamp(review["at"]),
                "by": _text(review["by"], "review.by", 200),
                "revision": _text(review["revision"], "review.revision", 128),
            }
        if "generations" in section or "active_generation_id" in section:
            if not {"generations", "active_generation_id"}.issubset(section):
                raise LearnValidationError(
                    "section.generations: active_generation_id and generations are required together"
                )
            generations = _section_generations(
                section["generations"], "section.generations"
            )
            active_generation_id = _id(
                section["active_generation_id"], "section.active_generation_id"
            )
            active = next(
                (row for row in generations if row["id"] == active_generation_id),
                None,
            )
            if active is None:
                raise LearnValidationError(
                    "section.active_generation_id: unknown generation"
                )
            # The flattened fields remain the compatibility projection consumed by
            # existing directives/templates. They must be byte-semantically equal
            # to the explicitly active accepted generation so there is one source
            # of truth rather than two independently editable copies.
            for key in ("body", "citations", "links", "contributors"):
                if normalized.get(key, [] if key != "body" else "") != active.get(
                    key, [] if key != "body" else ""
                ):
                    raise LearnValidationError(
                        f"section.{key}: must match active generation"
                    )
            if normalized.get("review") != active.get("review"):
                raise LearnValidationError(
                    "section.review: must match active generation"
                )
            normalized["active_generation_id"] = active_generation_id
            normalized["generations"] = generations
    for field in ("format", "publisher"):
        if field in value:
            result[field] = _text(value[field], "subject." + field, 200)
    if "status" in value:
        if value["status"] not in (
            "unverified",
            "open",
            "partial",
            "resolved",
            "disputed",
            "stale",
        ):
            raise LearnValidationError("subject.status: unknown status")
        result["status"] = value["status"]
    if "media" in value:
        media = value["media"]
        if value["kind"] == "audio":
            result["media"] = _local_media_audio(media, "subject.media")
        elif value["kind"] == "document":
            result["media"] = _local_media_document(media, "subject.media")
        else:
            _object(
                media,
                ("youtube_id", "image", "alt", "caption", "images"),
                (),
                "subject.media",
            )
            if value["kind"] not in ("video", "whiteboard"):
                raise LearnValidationError("subject.media: expected a media subject")
            result["media"] = {}
            if "youtube_id" in media:
                if (
                    value["kind"] != "video"
                    or not isinstance(media["youtube_id"], str)
                    or not re.fullmatch(r"[A-Za-z0-9_-]{11}", media["youtube_id"])
                ):
                    raise LearnValidationError(
                        "media.youtube_id: invalid video identifier"
                    )
                result["media"]["youtube_id"] = media["youtube_id"]
            if "image" in media:
                primary_value = {"image": media["image"], "alt": media.get("alt")}
                if "caption" in media:
                    primary_value["caption"] = media["caption"]
                primary = _local_media_image(primary_value, "media")
                result["media"].update(primary)
            elif "alt" in media or "caption" in media:
                raise LearnValidationError("media.alt/caption: requires media.image")
            if "images" in media:
                if value["kind"] != "whiteboard":
                    raise LearnValidationError(
                        "media.images: additional images are only supported for whiteboards"
                    )
                images = [
                    _local_media_image(row, f"media.images[{index}]")
                    for index, row in enumerate(
                        _list(media["images"], "media.images", MAX_WHITEBOARD_IMAGES)
                    )
                ]
                paths = (
                    [result["media"]["image"]] if "image" in result["media"] else []
                ) + [row["image"] for row in images]
                if len(paths) > MAX_WHITEBOARD_IMAGES:
                    raise LearnValidationError(
                        "media.images: whiteboard image limit exceeded"
                    )
                if len(paths) != len(set(paths)):
                    raise LearnValidationError("media.images: duplicate image assets")
                result["media"]["images"] = images
    elif value["kind"] == "audio":
        raise LearnValidationError(
            "subject.media: audio records require local audio media"
        )
    elif value["kind"] == "document":
        raise LearnValidationError(
            "subject.media: document records require local document media"
        )
    if "created_at" in value:
        result["created_at"] = timestamp(value["created_at"])
    if value["kind"] in ("source", "whiteboard", "video"):
        result["url"] = public_url(value.get("url"))
    elif "url" in value:
        raise LearnValidationError(
            "subject.url: only source, whiteboard, and video records have a resource URL"
        )
    return result


def validate_catalog(value):
    """Validate the entire graph, including citation targets and duplicates."""
    _object(
        value,
        ("contract", "revision", "subjects"),
        ("contract", "revision", "subjects"),
        "catalog",
    )
    if value["contract"] != CATALOG_CONTRACT:
        raise LearnValidationError("catalog.contract: unsupported version")
    subjects = [
        validate_subject(x)
        for x in _list(value["subjects"], "catalog.subjects", MAX_SUBJECTS)
    ]
    by_id = {s["id"]: s for s in subjects}
    if len(by_id) != len(subjects):
        raise LearnValidationError("catalog.subjects: duplicate subject identifier")
    graph_errors = []
    for subject in subjects:
        graph_errors.extend(
            f"subject.related: {subject['id']} -> unknown target {target_id}"
            for target_id in subject["related"]
            if target_id not in by_id
        )
        for section in subject["sections"]:
            for citation in section["citations"]:
                target = by_id.get(citation["source_id"])
                if target is None:
                    graph_errors.append(
                        "section.citations: "
                        f"{subject['id']}#{section['id']} -> unknown source "
                        f"{citation['source_id']}"
                    )
                elif target["kind"] != "source":
                    graph_errors.append(
                        "section.citations: "
                        f"{subject['id']}#{section['id']} -> target "
                        f"{citation['source_id']} has kind {target['kind']}, expected source"
                    )
    if graph_errors:
        raise LearnValidationError(
            "catalog: invalid graph references:\n"
            + "\n".join(f"  - {message}" for message in graph_errors)
        )
    result = {
        "contract": CATALOG_CONTRACT,
        "revision": _text(value["revision"], "catalog.revision", 128),
        "subjects": subjects,
    }
    if len(canonical_bytes(result)) > MAX_BYTES:
        raise LearnValidationError("catalog: encoded size exceeds limit")
    return result


def canonical_bytes(value):
    """Encode deterministic UTF-8 JSON for fingerprints and exports."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def validate_contribution(value):
    """Validate a local draft export; never grant publication or training rights."""
    _object(
        value,
        (
            "contract",
            "draft_id",
            "base_revision",
            "subject",
            "section",
            "authorship",
            "interactions",
        ),
        ("contract", "draft_id", "base_revision", "subject", "section", "authorship"),
        "contribution",
    )
    if value["contract"] != CONTRIBUTION_CONTRACT:
        raise LearnValidationError("contribution.contract: unsupported version")
    subject = validate_subject(value["subject"])
    if subject["sections"]:
        raise LearnValidationError(
            "contribution.subject: section revisions belong in section"
        )
    section = value["section"]
    probe = dict(subject, sections=[section])
    normalized = validate_subject(probe)["sections"][0]
    if value["authorship"] not in ("human", "ai-assisted"):
        raise LearnValidationError("contribution.authorship: unsupported value")
    result = {
        "contract": CONTRIBUTION_CONTRACT,
        "draft_id": _id(value["draft_id"], "contribution.draft_id"),
        "base_revision": _text(
            value["base_revision"], "contribution.base_revision", 128
        ),
        "subject": subject,
        "section": normalized,
        "authorship": value["authorship"],
    }
    if "interactions" in value:
        result["interactions"] = interactions(value["interactions"])
        if any(
            event["section_id"] != normalized["id"] for event in result["interactions"]
        ):
            raise LearnValidationError("interactions: section mismatch")
    return result
