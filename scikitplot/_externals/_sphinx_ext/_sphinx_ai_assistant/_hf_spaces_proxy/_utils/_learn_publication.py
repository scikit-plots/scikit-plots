# scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_utils/_learn_publication.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Strict transport policy for reviewed AI Learn publication requests.

This module deliberately knows nothing about canonical AI Learn repository
projection.  The browser/proxy boundary validates and transports a reviewed
private draft; the repository-local workflow reruns ``_sphinx_ai_learn``'s
publication planner against the current canonical JSON tree.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

PUBLICATION_REQUEST_CONTRACT = "learn.publication-request.v1"
PUBLICATION_RECEIPT_CONTRACT = "learn.publication-receipt.v1"
PUBLICATION_CAPABILITY_CONTRACT = "learn.publication-capability.v1"
SUPPORTED_PUBLICATION_MODES = frozenset({"disabled", "stub", "github"})
SUPPORTED_DRAFT_CONTRACTS = frozenset(
    {
        "learn.record-creation-draft.v1",
        "learn.topic-prompt-draft.v1",
        "learn.skill-draft.v1",
        "learn.page-overview-draft.v1",
        "learn.section-draft.v2",
    }
)
_REQUEST_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_REPO_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,98}[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,98}[A-Za-z0-9])?$"
)
_SAFE_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
# Browser feedback ids are opaque event nonces, never semantic/user identifiers.
# V70 emitted 128-bit (32 hex) ids; V71 emits 192-bit (48 hex) ids.
_FEEDBACK_EVENT_ID_RE = re.compile(r"^feedback-(?:[0-9a-f]{32}|[0-9a-f]{48})$")
_TREE_REVISION_RE = re.compile(r"^tree-[0-9a-f]{16}$")
_UTC_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_SAFE_SUBJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_SAFE_SECTION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class LearnPublicationTransportError(ValueError):
    """Raised when a browser publication envelope is unsafe or malformed."""


@dataclass(frozen=True)
class LearnPublicationPolicy:
    """Server-owned publication destination and transport authority."""

    mode: str
    repository: str
    default_branch: str
    canonical_prefix: str
    workflow: str
    max_request_bytes: int


def _strict_json_loads(raw: bytes, *, max_bytes: int) -> dict[str, Any]:
    if not isinstance(raw, (bytes, bytearray)):
        raise LearnPublicationTransportError("publication request: expected bytes")
    if not raw or len(raw) > max_bytes:
        raise LearnPublicationTransportError(
            "publication request: encoded size exceeds limit"
        )

    def reject_duplicates(pairs):
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise LearnPublicationTransportError(
                    f"publication request: duplicate JSON key {key!r}"
                )
            out[key] = value
        return out

    def reject_constant(value):
        raise LearnPublicationTransportError(
            f"publication request: non-finite number {value!r} is not allowed"
        )

    try:
        value = json.loads(
            bytes(raw),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except LearnPublicationTransportError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise LearnPublicationTransportError(
            "publication request: invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise LearnPublicationTransportError("publication request: expected object")
    return value


def _bounded_text(value: Any, *, name: str, limit: int, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise LearnPublicationTransportError(f"{name}: expected string")
    text = value.strip()
    if not empty and not text:
        raise LearnPublicationTransportError(f"{name}: expected non-empty string")
    if len(text) > limit or any(  # lint
        (ord(ch) < 32 and ch not in "\t\n\r")  # ruff: ignore[magic-value-comparison]
        or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
        for ch in text
    ):
        raise LearnPublicationTransportError(f"{name}: invalid text")
    return text


def _public_display_name(value: Any, *, name: str = "contributor.display_name") -> str:
    """Validate public credit without allowing layout/control spoofing."""
    text = _bounded_text(value, name=name, limit=80, empty=True)
    if any(
        ord(ch) < 32 or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
        for ch in text
    ):
        raise LearnPublicationTransportError(
            f"{name}: control characters are not allowed"
        )
    return text or "Anonymous"


def _normalized_prefix(value: str) -> str:
    text = _bounded_text(value, name="canonical prefix", limit=240)
    if "\\" in text or text.startswith("/") or text.endswith("/"):
        raise LearnPublicationTransportError("canonical prefix: unsafe path")
    path = PurePosixPath(text)
    if any(part in {"", ".", ".."} for part in path.parts) or path.as_posix() != text:
        raise LearnPublicationTransportError("canonical prefix: unsafe path")
    return text


def _normalized_branch(value: str) -> str:
    text = _bounded_text(value, name="default branch", limit=160)
    if (
        text.startswith("-")
        or text.endswith(("/", ".", ".lock"))
        or ".." in text
        or "@{" in text
        or "\\" in text
        or any(  # lint
            ch.isspace()
            or ord(ch) < 32  # ruff: ignore[magic-value-comparison]
            or ch in "~^:?*["
            for ch in text
        )
    ):
        raise LearnPublicationTransportError("default branch: unsafe git ref")
    return text


def _normalized_workflow(value: str) -> str:
    text = _bounded_text(value, name="workflow", limit=120)
    if "/" in text or "\\" in text or text in {".", ".."}:
        raise LearnPublicationTransportError("workflow: expected a workflow file name")
    if not text.endswith((".yml", ".yaml")):
        raise LearnPublicationTransportError("workflow: expected .yml or .yaml")
    return text


def build_publication_policy(
    *,
    mode: str,
    repository: str,
    default_branch: str,
    canonical_prefix: str,
    workflow: str,
    max_request_bytes: int,
) -> LearnPublicationPolicy:
    selected = str(mode or "disabled").strip().lower() or "disabled"
    if selected not in SUPPORTED_PUBLICATION_MODES:
        selected = "disabled"
    repo = _bounded_text(repository, name="repository", limit=200)
    if not _REPO_RE.fullmatch(repo):
        raise LearnPublicationTransportError("repository: expected owner/repo")
    branch = _normalized_branch(default_branch)
    prefix = _normalized_prefix(canonical_prefix)
    workflow_name = _normalized_workflow(workflow)
    try:
        limit = int(max_request_bytes)
    except (TypeError, ValueError) as exc:
        raise LearnPublicationTransportError(
            "max request bytes: expected integer"
        ) from exc
    # Keep generous envelope room below workflow_dispatch's 65,535-char input
    # ceiling. The repository workflow revalidates the same request again.
    if not 4096 <= limit <= 60_000:  # ruff: ignore[magic-value-comparison]
        raise LearnPublicationTransportError("max request bytes: out of range")
    return LearnPublicationPolicy(
        mode=selected,
        repository=repo,
        default_branch=branch,
        canonical_prefix=prefix,
        workflow=workflow_name,
        max_request_bytes=limit,
    )


def capability_document(
    policy: LearnPublicationPolicy,
    *,
    credential_ready: bool,
) -> dict[str, Any]:
    ready = policy.mode == "stub" or (policy.mode == "github" and credential_ready)
    return {
        "contract": PUBLICATION_CAPABILITY_CONTRACT,
        "mode": policy.mode,
        "ready": bool(ready),
        "repository": policy.repository,
        "default_branch": policy.default_branch,
        "canonical_prefix": policy.canonical_prefix,
        "workflow": policy.workflow,
        "max_request_bytes": policy.max_request_bytes,
        "review_required": True,
        "browser_repository_override": False,
        "browser_credentials": False,
    }


def parse_publication_request(  # ruff: ignore[too-many-branches]
    raw: bytes,
    policy: LearnPublicationPolicy,
) -> dict[str, Any]:
    value = _strict_json_loads(raw, max_bytes=policy.max_request_bytes)
    action = value.get("action")
    if value.get("contract") != PUBLICATION_REQUEST_CONTRACT:
        raise LearnPublicationTransportError(
            "publication request: unsupported contract",
        )
    if action == "test":
        if set(value) != {"contract", "action"}:
            raise LearnPublicationTransportError("publication test: unexpected fields")
        return {"contract": PUBLICATION_REQUEST_CONTRACT, "action": "test"}
    if action not in {"publish", "feedback"}:
        raise LearnPublicationTransportError("publication request: unsupported action")

    if action == "feedback":
        allowed = {
            "contract",
            "action",
            "base_revision",
            "subject_id",
            "section_id",
            "generation_id",
            "feedback_id",
            "created_at",
            "rating",
            "comment",
            "contributor",
            "feedback_mode",
        }
        if set(value) - allowed:
            raise LearnPublicationTransportError("feedback request: unexpected fields")
        required = allowed - {"comment", "contributor", "created_at", "feedback_mode"}
        if not required.issubset(value):
            raise LearnPublicationTransportError(
                "feedback request: missing required fields"
            )
        base_revision = _bounded_text(
            value["base_revision"], name="base_revision", limit=128
        )
        if not _TREE_REVISION_RE.fullmatch(base_revision):
            raise LearnPublicationTransportError("base_revision: invalid tree revision")
        out: dict[str, Any] = {
            "contract": PUBLICATION_REQUEST_CONTRACT,
            "action": "feedback",
            "base_revision": base_revision,
        }
        for key, regex in (
            ("subject_id", _SAFE_SUBJECT_ID_RE),
            ("section_id", _SAFE_SECTION_ID_RE),
            ("generation_id", _SAFE_ID_RE),
            ("feedback_id", _FEEDBACK_EVENT_ID_RE),
        ):
            item = _bounded_text(
                value[key],
                name=key,
                limit=128 if key in {"subject_id", "section_id"} else 64,
            )
            if not regex.fullmatch(item):
                raise LearnPublicationTransportError(f"{key}: invalid identifier")
            out[key] = item
        rating = value["rating"]
        if (
            isinstance(rating, bool)
            or not isinstance(rating, int)
            or not -5 <= rating <= 5  # ruff: ignore[magic-value-comparison]
        ):
            raise LearnPublicationTransportError(
                "rating: expected integer from -5 to 5"
            )
        out["rating"] = rating
        if "comment" in value:
            comment = _bounded_text(
                value["comment"], name="comment", limit=2000, empty=True
            )
            if comment:
                out["comment"] = comment
        if "feedback_mode" in value:
            mode = _bounded_text(value["feedback_mode"], name="feedback_mode", limit=16)
            if mode not in {"quick", "detailed"}:
                raise LearnPublicationTransportError(
                    "feedback_mode: expected quick or detailed"
                )
            if mode == "quick" and rating not in {-1, 1}:
                raise LearnPublicationTransportError(
                    "rating: quick feedback must be -1 or 1"
                )
            out["feedback_mode"] = mode
        if "created_at" in value:
            # Backward-compatible transport field only. The repository planner
            # validates it but does not persist browser-authored time as
            # canonical feedback metadata.
            created_at = _bounded_text(
                value["created_at"],
                name="created_at",
                limit=64,
            )
            if not _UTC_TIMESTAMP_RE.fullmatch(created_at):
                raise LearnPublicationTransportError(
                    "created_at: expected UTC timestamp",
                )
            try:
                datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError as exc:
                raise LearnPublicationTransportError(
                    "created_at: invalid UTC timestamp"
                ) from exc
            out["created_at"] = created_at
        contributor = value.get("contributor", {"display_name": "Anonymous"})
        if not isinstance(contributor, dict) or set(contributor) - {"display_name"}:
            raise LearnPublicationTransportError("contributor: expected {display_name}")
        display_name = _public_display_name(contributor.get("display_name", ""))
        out["contributor"] = {"display_name": display_name}
        return out

    allowed = {
        "contract",
        "action",
        "draft",
        "base_revision",
        "created_at",
        "metadata_reviewed",
        "artifact_id",
        "author",
        "order",
        "default_enabled",
        "subject_id",
        "section_id",
        "section_title",
        "contributor",
    }
    if set(value) - allowed:
        raise LearnPublicationTransportError("publication request: unexpected fields")
    required = {"contract", "action", "draft", "base_revision"}
    if not required.issubset(value):
        raise LearnPublicationTransportError(
            "publication request: missing required fields"
        )
    draft = value.get("draft")
    if not isinstance(draft, dict):
        raise LearnPublicationTransportError(
            "publication request draft: expected object"
        )
    draft_contract = draft.get("contract")
    if draft_contract not in SUPPORTED_DRAFT_CONTRACTS:
        raise LearnPublicationTransportError(
            "publication request draft: unsupported contract"
        )
    base_revision = _bounded_text(
        value.get("base_revision"), name="base_revision", limit=128
    )
    provenance = draft.get("provenance")
    draft_revision = ""
    if isinstance(provenance, dict) and isinstance(
        provenance.get("base_revision"), str
    ):
        draft_revision = provenance["base_revision"].strip()
    elif isinstance(draft.get("base_revision"), str):
        draft_revision = draft["base_revision"].strip()
    if draft_revision and draft_revision != base_revision:
        raise LearnPublicationTransportError(
            "publication request: draft/base revision mismatch"
        )

    out: dict[str, Any] = {
        "contract": PUBLICATION_REQUEST_CONTRACT,
        "action": "publish",
        "draft": draft,
        "base_revision": base_revision,
        "artifact_id": "auto",
        "author": "community",
        "default_enabled": False,
        "metadata_reviewed": False,
    }
    if "created_at" in value:
        out["created_at"] = _bounded_text(
            value["created_at"], name="created_at", limit=64
        )
    if "metadata_reviewed" in value:
        if not isinstance(value["metadata_reviewed"], bool):
            raise LearnPublicationTransportError("metadata_reviewed: expected boolean")
        out["metadata_reviewed"] = value["metadata_reviewed"]
    if "default_enabled" in value:
        if not isinstance(value["default_enabled"], bool):
            raise LearnPublicationTransportError("default_enabled: expected boolean")
        out["default_enabled"] = value["default_enabled"]
    if "artifact_id" in value:
        artifact_id = _bounded_text(value["artifact_id"], name="artifact_id", limit=64)
        if artifact_id != "auto" and not _SAFE_ID_RE.fullmatch(artifact_id):
            raise LearnPublicationTransportError("artifact_id: invalid identifier")
        out["artifact_id"] = artifact_id
    if "author" in value:
        out["author"] = _bounded_text(value["author"], name="author", limit=200)
    if "contributor" in value:
        contributor = value["contributor"]
        if not isinstance(contributor, dict) or set(contributor) - {"display_name"}:
            raise LearnPublicationTransportError("contributor: expected {display_name}")
        display_name = _public_display_name(contributor.get("display_name", ""))
        out["contributor"] = {"display_name": display_name}
    if "order" in value:
        if isinstance(value["order"], bool) or not isinstance(value["order"], int):
            raise LearnPublicationTransportError("order: expected integer")
        if not 0 <= value["order"] <= 10_000:  # ruff: ignore[magic-value-comparison]
            raise LearnPublicationTransportError("order: out of range")
        out["order"] = value["order"]
    for key, regex in (
        ("subject_id", _SAFE_SUBJECT_ID_RE),
        ("section_id", _SAFE_SECTION_ID_RE),
    ):
        if key in value:
            item = _bounded_text(value[key], name=key, limit=128)
            if not regex.fullmatch(item):
                raise LearnPublicationTransportError(f"{key}: invalid identifier")
            out[key] = item
    if "section_title" in value:
        out["section_title"] = _bounded_text(
            value["section_title"], name="section_title", limit=200
        )

    # Contract-specific routing metadata. Canonical validation is repeated in
    # the repository workflow, but fail obvious transport mistakes early.
    if draft_contract == "learn.record-creation-draft.v1" and "created_at" not in out:
        raise LearnPublicationTransportError(
            "created_at: required for record publication",
        )
    if draft_contract == "learn.page-overview-draft.v1":
        if "subject_id" not in out:
            raise LearnPublicationTransportError(
                "subject_id: required for overview publication",
            )
        out.setdefault("section_id", "summary")
        out.setdefault("section_title", "Summary")
    if draft_contract == "learn.section-draft.v2":
        if "subject_id" not in out or "section_id" not in out:
            raise LearnPublicationTransportError(
                "subject_id and section_id: required for section publication"
            )
        if "section_title" not in out:
            title = draft.get("title")
            if not isinstance(title, str) or not title.strip():
                raise LearnPublicationTransportError(
                    "section_title: required when section draft has no title"
                )
            out["section_title"] = title.strip()[:200]
    return out


def canonical_request_json(request: dict[str, Any]) -> str:
    try:
        text = json.dumps(
            request,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise LearnPublicationTransportError(
            "publication request: cannot canonicalize JSON"
        ) from exc
    return text


def publication_request_id(request: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_request_json(request).encode("utf-8")).hexdigest()


def validate_request_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not _REQUEST_ID_RE.fullmatch(text):
        raise LearnPublicationTransportError("request_id: invalid")
    return text


def workflow_dispatch_body(
    policy: LearnPublicationPolicy,
    request: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    request_json = canonical_request_json(request)
    request_id = publication_request_id(request)
    payload = {
        "ref": policy.default_branch,
        "inputs": {
            "operation": "publish",
            "request_id": request_id,
            "request_json": request_json,
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) > 65_535:  # ruff: ignore[magic-value-comparison]
        raise LearnPublicationTransportError(
            "publication request: GitHub workflow input payload exceeds limit"
        )
    return request_id, payload
