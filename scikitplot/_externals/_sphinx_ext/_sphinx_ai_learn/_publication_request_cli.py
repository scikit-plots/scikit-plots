# scikitplot/_externals/_sphinx_ext/_sphinx_ai_learn/_publication_request_cli.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Validate one transported reviewed draft and build a JSON-only review bundle.

This CLI is the repository trust boundary used by the GitHub Actions publication
workflow.  It deliberately revalidates the browser/proxy envelope instead of
trusting transport-side validation.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys  # ruff: ignore[unused-import]
from pathlib import Path

from ._materialize import load_content_tree
from ._publication import (
    PAGE_OVERVIEW_DRAFT_CONTRACT,
    PUBLICATION_CONTRACT,
    RECORD_DRAFT_CONTRACT,
    SKILL_DRAFT_CONTRACT,
    TOPIC_PROMPT_DRAFT_CONTRACT,
    publication_plan,
    suggested_artifact_id,
)
from ._publication_cli import _write_bundle
from ._schema import FEEDBACK_EVENT_ID_PATTERN, LearnValidationError, timestamp

logger = logging.getLogger(__name__)

REQUEST_CONTRACT = "learn.publication-request.v1"
SECTION_DRAFT_CONTRACT = "learn.section-draft.v2"
_MAX_REQUEST_BYTES = 60_000


def _load_request(path):
    raw = Path(path).read_bytes()
    if not raw or len(raw) > _MAX_REQUEST_BYTES:
        raise LearnValidationError("publication request: encoded size exceeds limit")

    def duplicates(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise LearnValidationError(
                    f"publication request: duplicate JSON key {key!r}"
                )
            out[key] = value
        return out

    def nonfinite(value):
        raise LearnValidationError(
            f"publication request: non-finite number {value!r} is not allowed"
        )

    try:
        value = json.loads(
            raw,
            object_pairs_hook=duplicates,
            parse_constant=nonfinite,
        )
    except LearnValidationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LearnValidationError("publication request: invalid JSON") from exc
    if not isinstance(value, dict):
        raise LearnValidationError("publication request: expected object")
    return value


def _text(value, *, path, limit, empty=False):
    if not isinstance(value, str):
        raise LearnValidationError(f"{path}: expected string")
    value = value.strip()
    if not empty and not value:
        raise LearnValidationError(f"{path}: expected non-empty string")
    if len(value) > limit:
        raise LearnValidationError(f"{path}: exceeds limit")
    return value


def _public_display_name(value, *, path="contributor.display_name"):
    value = _text(value, path=path, limit=80, empty=True)
    if any(
        ord(ch) < 32 or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
        for ch in value
    ):
        raise LearnValidationError(f"{path}: control characters are not allowed")
    return value or "Anonymous"


def _contributor_display_name(request):
    """Return bounded public credit without treating it as verified identity."""
    if "contributor" in request:
        value = request.get("contributor")
        if not isinstance(value, dict) or set(value) - {"display_name"}:
            raise LearnValidationError("contributor: expected {display_name}")
        return _public_display_name(value.get("display_name", ""))
    if "author" in request:
        return _text(request.get("author"), path="author", limit=80)
    return "Anonymous"


def _generation_created_at(draft, subject):
    """Return deterministic accepted-generation time from reviewed draft metadata."""
    candidates = []
    provenance = draft.get("provenance") if isinstance(draft, dict) else None
    if isinstance(provenance, dict):
        candidates.append(provenance.get("generated_at"))
    if isinstance(draft, dict):
        candidates.append(draft.get("generated_at"))
    for raw in candidates:
        if not isinstance(raw, str):
            continue
        raw = raw.strip()  # ruff: ignore[redefined-loop-name]
        raw = re.sub(  # ruff: ignore[redefined-loop-name]
            r"\.\d{1,6}Z$",
            "Z",
            raw,
        )
        try:
            return timestamp(raw, "generation.created_at")
        except LearnValidationError:
            continue
    return timestamp(subject.get("created_at"), "subject.created_at")


def _generation_provenance(draft):
    """Project only bounded public model/workflow provenance into canonical JSON."""
    source = draft.get("provenance") if isinstance(draft, dict) else None
    if not isinstance(source, dict):
        source = draft if isinstance(draft, dict) else {}
    result = {}
    for key, limit in (
        ("authorship", 40),
        ("model", 200),
        ("workflow_id", 120),
        ("agent", 120),
        ("skill", 120),
        ("request_id", 200),
    ):
        if isinstance(source.get(key), str):
            result[key] = _text(
                source[key],
                path="generation.provenance." + key,
                limit=limit,
                empty=True,
            )
    return result


def _request_operation(root, request):  # noqa: PLR0912
    if request.get("contract") != REQUEST_CONTRACT:
        raise LearnValidationError("publication request: unsupported contract/action")
    action = request.get("action")
    if action not in {"publish", "feedback"}:
        raise LearnValidationError("publication request: unsupported contract/action")

    feedback_allowed = {
        "contract",
        "action",
        "base_revision",
        "subject_id",
        "section_id",
        "generation_id",
        "feedback_id",
        "rating",
        "comment",
        "contributor",
        "feedback_mode",
    }
    publish_allowed = {
        "contract",
        "action",
        "draft",
        "base_revision",
        "created_at",
        "metadata_reviewed",
        "artifact_id",
        "author",
        "contributor",
        "order",
        "default_enabled",
        "subject_id",
        "section_id",
        "section_title",
    }
    allowed = feedback_allowed if action == "feedback" else publish_allowed
    if set(request) - allowed:
        raise LearnValidationError(
            f"{action} request: unexpected fields: "
            + ", ".join(sorted(set(request) - allowed))
        )
    if "base_revision" not in request:
        raise LearnValidationError("publication request: missing required fields")
    contributor = _contributor_display_name(request)
    base_revision = _text(
        request.get("base_revision"),
        path="publication request.base_revision",
        limit=128,
    )
    if action == "feedback" and not re.fullmatch(r"tree-[0-9a-f]{16}", base_revision):
        raise LearnValidationError(
            "publication request.base_revision: invalid tree revision"
        )
    tree = load_content_tree(root)
    # Publishing/replacing authored content is revision-bound because the draft
    # may depend on the exact repository state it reviewed.  Feedback is
    # different: it targets an immutable generation identifier and is an
    # append-only event.  Allow a stale page revision for feedback so unrelated
    # merges (or another accepted feedback event) do not make the static page
    # unusable; the generation lookup below remains the authority and fails
    # closed if that target no longer exists.
    if action != "feedback" and tree.catalog["revision"] != base_revision:
        raise LearnValidationError(
            "publication request.base_revision: catalog changed; regenerate/review the draft"
        )

    if action == "feedback":
        required = {
            "subject_id",
            "section_id",
            "generation_id",
            "feedback_id",
            "rating",
        }
        if not required.issubset(request):
            raise LearnValidationError("feedback request: missing required fields")
        rating = request.get("rating")
        if (
            isinstance(rating, bool)
            or not isinstance(rating, int)
            or not -5 <= rating <= 5  # ruff: ignore[magic-value-comparison]
        ):
            raise LearnValidationError("rating: expected integer from -5 to 5")
        feedback_id = _text(request["feedback_id"], path="feedback_id", limit=64)
        if not FEEDBACK_EVENT_ID_PATTERN.fullmatch(feedback_id):
            raise LearnValidationError(
                "feedback_id: expected opaque reviewed-feedback event nonce"
            )
        op = {
            "op": "rate-section-generation",
            "subject_id": _text(request["subject_id"], path="subject_id", limit=64),
            "section_id": _text(request["section_id"], path="section_id", limit=64),
            "generation_id": _text(
                request["generation_id"], path="generation_id", limit=64
            ),
            "feedback_id": feedback_id,
            "rating": rating,
            "contributor": contributor,
        }
        if "comment" in request:
            comment = _text(
                request.get("comment", ""), path="comment", limit=2000, empty=True
            )
            if any(
                (
                    ord(ch) < 32  # ruff: ignore[magic-value-comparison]
                    and ch not in "\t\n\r"
                )
                or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
                for ch in comment
            ):
                raise LearnValidationError(
                    "comment: contains unsupported control characters",
                )
            if comment:
                op["comment"] = comment
        if "feedback_mode" in request:
            mode = _text(request.get("feedback_mode"), path="feedback_mode", limit=16)
            if mode not in {"quick", "detailed"}:
                raise LearnValidationError(
                    "feedback_mode: expected quick or detailed",
                )
            if mode == "quick" and rating not in {-1, 1}:
                raise LearnValidationError(
                    "rating: quick feedback must be -1 or 1",
                )
            op["feedback_mode"] = mode
        return tree, op

    if "draft" not in request:
        raise LearnValidationError(
            "publication request: missing required fields",
        )
    draft = request.get("draft")
    if not isinstance(draft, dict):
        raise LearnValidationError(
            "publication request.draft: expected object",
        )
    provenance = draft.get("provenance")
    embedded_revision = ""
    if isinstance(provenance, dict) and isinstance(
        provenance.get("base_revision"),
        str,
    ):
        embedded_revision = provenance["base_revision"].strip()
    elif isinstance(draft.get("base_revision"), str):
        embedded_revision = draft["base_revision"].strip()
    if embedded_revision and embedded_revision != base_revision:
        raise LearnValidationError(
            "publication request: draft/base revision mismatch",
        )

    contract = draft.get("contract")
    if contract in {
        RECORD_DRAFT_CONTRACT,
        TOPIC_PROMPT_DRAFT_CONTRACT,
        SKILL_DRAFT_CONTRACT,
    }:
        existing_ids = [subject["id"] for subject in tree.catalog["subjects"]]
        existing_ids.extend(prompt["id"] for prompt in tree.prompts)
        existing_ids.extend(skill["id"] for skill in tree.skills)
        artifact_id = request.get("artifact_id", "auto")
        if artifact_id == "auto":
            artifact_id = suggested_artifact_id(draft, existing_ids=existing_ids)
        else:
            artifact_id = _text(artifact_id, path="artifact_id", limit=64)

        if contract == TOPIC_PROMPT_DRAFT_CONTRACT:
            order = request.get("order")
            if order is None:
                order = (
                    max((prompt["order"] for prompt in tree.prompts), default=0) + 10
                )
            return tree, {
                "op": "create-topic-prompt",
                "prompt_id": artifact_id,
                "author": contributor,
                "order": order,
                "default_enabled": bool(request.get("default_enabled", False)),
                "draft": draft,
            }
        if contract == SKILL_DRAFT_CONTRACT:
            order = request.get("order")
            if order is None:
                order = max((skill["order"] for skill in tree.skills), default=0) + 10
            return tree, {
                "op": "create-skill",
                "skill_id": artifact_id,
                "author": contributor,
                "order": order,
                "default_enabled": bool(request.get("default_enabled", False)),
                "draft": draft,
            }
        created_at = timestamp(request.get("created_at"), "created_at")
        op = {
            "op": "create-record",
            "record_id": artifact_id,
            "created_at": created_at,
            "contributor": contributor,
            "draft": draft,
        }
        if draft.get("kind") == "source":
            if request.get("metadata_reviewed") is not True:
                raise LearnValidationError(
                    "metadata_reviewed: explicit Source metadata review is required"
                )
            op["metadata_reviewed"] = True
        return tree, op

    if contract == PAGE_OVERVIEW_DRAFT_CONTRACT:
        subject_id = _text(
            request.get("subject_id"),
            path="subject_id",
            limit=128,
        )
        section_id = _text(
            request.get("section_id", "summary"),
            path="section_id",
            limit=128,
        )
        title = _text(
            request.get("section_title", "Summary"),
            path="section_title",
            limit=200,
        )
        body = _text(
            draft.get("body"),
            path="draft.body",
            limit=50_000,
            empty=True,
        )
        subject = next(
            (row for row in tree.catalog["subjects"] if row["id"] == subject_id), None
        )
        if subject is None:
            raise LearnValidationError("subject_id: target subject does not exist")
        return tree, {
            "op": "upsert-section",
            "subject_id": subject_id,
            "section_id": section_id,
            "title": title,
            "body": body,
            "contributor": contributor,
            "created_at": _generation_created_at(draft, subject),
            "provenance": _generation_provenance(draft),
        }

    if contract == SECTION_DRAFT_CONTRACT:
        subject_id = _text(request.get("subject_id"), path="subject_id", limit=128)
        section_id = _text(request.get("section_id"), path="section_id", limit=128)
        title = _text(
            request.get("section_title", draft.get("title")),
            path="section_title",
            limit=200,
        )
        body = _text(draft.get("body"), path="draft.body", limit=50_000, empty=True)
        subject = next(
            (row for row in tree.catalog["subjects"] if row["id"] == subject_id), None
        )
        if subject is None:
            raise LearnValidationError("subject_id: target subject does not exist")
        return tree, {
            "op": "upsert-section",
            "subject_id": subject_id,
            "section_id": section_id,
            "title": title,
            "body": body,
            "contributor": contributor,
            "created_at": _generation_created_at(draft, subject),
            "provenance": _generation_provenance(draft),
        }

    raise LearnValidationError("publication request.draft.contract: unsupported")


def plan_publication_request(content_root, request):
    tree, operation = _request_operation(content_root, request)
    publication = {
        "contract": PUBLICATION_CONTRACT,
        "base_revision": tree.catalog["revision"],
        "operations": [operation],
    }
    return publication_plan(content_root, publication)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("content_root")
    parser.add_argument("request")
    parser.add_argument("destination")
    parser.add_argument("--repo-prefix", default="docs/source/learn-ai")
    args = parser.parse_args(argv)
    request = _load_request(args.request)
    plan = plan_publication_request(Path(args.content_root), request)
    manifest = _write_bundle(plan, args.destination, args.repo_prefix)
    logger.info(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
