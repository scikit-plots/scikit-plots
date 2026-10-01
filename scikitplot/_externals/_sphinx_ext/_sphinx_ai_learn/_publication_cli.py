# scikitplot/_externals/_sphinx_ext/_sphinx_ai_learn/_publication_cli.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare a canonical-JSON repository review bundle from one private AI Learn draft."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path, PurePosixPath

from ._materialize import load_content_tree
from ._publication import (
    PUBLICATION_CONTRACT,
    SKILL_DRAFT_CONTRACT,
    TOPIC_PROMPT_DRAFT_CONTRACT,
    publication_plan,
    suggested_artifact_id,
)
from ._schema import LearnValidationError

_MAX_DRAFT_BYTES = 512 * 1024
logger = logging.getLogger(__name__)


def _load_draft(path):
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) > _MAX_DRAFT_BYTES:
        raise LearnValidationError("draft: encoded size exceeds limit")

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise LearnValidationError(f"draft: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except LearnValidationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LearnValidationError("draft: invalid JSON") from exc
    if not isinstance(value, dict):
        raise LearnValidationError("draft: expected object")
    return value


def _safe_prefix(value):
    value = str(value or "")
    if "\\" in value or value.startswith("/") or value.endswith("/"):
        raise LearnValidationError("repo-prefix: expected a relative normalized path")
    path = PurePosixPath(value) if value else PurePosixPath()
    if any(part in {"", ".", ".."} for part in path.parts) or (
        value and path.as_posix() != value
    ):
        raise LearnValidationError("repo-prefix: unsafe path")
    return path


def _write_bundle(plan, destination, repo_prefix):
    destination = Path(destination)
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise LearnValidationError("destination: expected a real directory")
    if destination.exists() and any(destination.iterdir()):
        raise LearnValidationError("destination: expected an empty directory")
    destination.mkdir(parents=True, exist_ok=True)
    prefix = _safe_prefix(repo_prefix)
    hashes = {}
    for relative, raw in sorted(plan["files"].items()):
        source_path = PurePosixPath(relative)
        if source_path.is_absolute() or ".." in source_path.parts:
            raise LearnValidationError("publication plan: unsafe file path")
        if source_path.suffix != ".json":
            raise LearnValidationError(
                "publication plan: only canonical JSON may be bundled",
            )
        repo_path = prefix / source_path if prefix.parts else source_path
        target = destination.joinpath(*repo_path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        hashes[str(repo_path)] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    deleted = [
        str(prefix / PurePosixPath(name)) if prefix.parts else name
        for name in plan["deleted"]
    ]
    manifest = {
        "contract": "learn.publication-review-bundle.v2",
        "base_revision": plan["base_revision"],
        "revision": plan["revision"],
        "affected_ids": plan["affected_ids"],
        "files": hashes,
        "deleted": deleted,
        "routes": plan["routes"],
    }
    (destination / "publication-plan.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("content_root", help="Canonical docs/source/learn-ai directory")
    parser.add_argument(
        "draft",
        help="Private record, Topic Prompt, or Skill draft JSON",
    )
    parser.add_argument(
        "destination",
        help="Empty directory for the review bundle",
    )
    parser.add_argument(
        "--created-at",
        help="UTC second for record creation, e.g. 2026-09-24T00:00:00Z",
    )
    parser.add_argument(
        "--id",
        default="auto",
        help="Stable record/prompt ID or 'auto'",
    )
    parser.add_argument(
        "--metadata-reviewed",
        action="store_true",
        help="Required before a Source draft can be prepared for publication",
    )
    parser.add_argument(
        "--author",
        default="community",
        help="Reviewed author label for Topic Prompts and Skills",
    )
    parser.add_argument(
        "--order",
        type=int,
        help="Explicit Topic Prompt/Skill ordering value; defaults after its current registry",
    )
    parser.add_argument(
        "--default-enabled",
        action="store_true",
        help="Enable a newly published Topic Prompt or Skill by default",
    )
    parser.add_argument(
        "--repo-prefix",
        default="docs/source/learn-ai",
        help="Repository path containing the canonical AI Learn JSON tree",
    )
    args = parser.parse_args(argv)

    tree = load_content_tree(args.content_root)
    draft = _load_draft(args.draft)
    existing_ids = [subject["id"] for subject in tree.catalog["subjects"]]
    existing_ids.extend(prompt["id"] for prompt in tree.prompts)
    existing_ids.extend(skill["id"] for skill in tree.skills)
    artifact_id = (
        suggested_artifact_id(draft, existing_ids=existing_ids)
        if args.id == "auto"
        else args.id
    )

    if draft.get("contract") == TOPIC_PROMPT_DRAFT_CONTRACT:
        order = args.order
        if order is None:
            order = max((prompt["order"] for prompt in tree.prompts), default=0) + 10
        operation = {
            "op": "create-topic-prompt",
            "prompt_id": artifact_id,
            "author": args.author,
            "order": order,
            "default_enabled": args.default_enabled,
            "draft": draft,
        }
    elif draft.get("contract") == SKILL_DRAFT_CONTRACT:
        order = args.order
        if order is None:
            order = max((skill["order"] for skill in tree.skills), default=0) + 10
        operation = {
            "op": "create-skill",
            "skill_id": artifact_id,
            "author": args.author,
            "order": order,
            "default_enabled": args.default_enabled,
            "draft": draft,
        }
    else:
        if not args.created_at:
            parser.error("--created-at is required for record drafts")
        operation = {
            "op": "create-record",
            "record_id": artifact_id,
            "created_at": args.created_at,
            "draft": draft,
            **({"metadata_reviewed": True} if args.metadata_reviewed else {}),
        }

    publication = {
        "contract": PUBLICATION_CONTRACT,
        "base_revision": tree.catalog["revision"],
        "operations": [operation],
    }
    plan = publication_plan(args.content_root, publication)
    logger.info(
        "Writing publication review bundle to %s (%d files)",
        args.destination,
        len(plan["files"]),
    )
    manifest = _write_bundle(plan, args.destination, args.repo_prefix)
    logger.info(
        "Publication manifest: %s",
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
    )


if __name__ == "__main__":
    main()
