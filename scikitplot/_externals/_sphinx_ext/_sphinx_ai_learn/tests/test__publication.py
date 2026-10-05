"""Review-first publication mutates canonical JSON only."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

import _learn_site

from _sphinx_ext._sphinx_ai_learn._materialize import load_content_tree
from _sphinx_ext._sphinx_ai_learn._publication import (
    PUBLICATION_CONTRACT,
    _project_json_tree,
    apply_publication,
    prompt_from_draft,
    skill_from_draft,
    publication_plan,
    record_from_draft,
    section_generation_id,
    suggested_artifact_id,
)
from _sphinx_ext._sphinx_ai_learn._publication_cli import main as publication_cli
from _sphinx_ext._sphinx_ai_learn._schema import LearnValidationError



def _feedback_id(index=1):
    """Return the current 192-bit reviewed-feedback event identifier shape."""
    return "feedback-" + f"{int(index):048x}"[-48:]


def _json_only_copy(tmp_path):
    root = tmp_path / "learn-ai"
    for source in _learn_site.content_root().rglob("*.json"):
        target = root / source.relative_to(_learn_site.content_root())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return root


def _apply_plan(root, plan):
    for relative in plan["deleted"]:
        (root / relative).unlink(missing_ok=True)
    for relative, raw in plan["files"].items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)


def topic_draft():
    return {
        "contract": "learn.record-creation-draft.v1",
        "kind": "topic",
        "title": "Introduction to Supervised Learning",
        "summary": "A bounded draft.",
        "domains": ["machine-learning"],
        "sections": [
            {"id": "summary", "title": "Summary", "body": "Draft summary."}
        ],
        "evidence_gaps": ["Need a benchmark."],
        "related_questions": ["How does regularization change the result?"],
    }


def prompt_draft():
    return {
        "contract": "learn.topic-prompt-draft.v1",
        "kind": "topic-prompt",
        "title": "Counterfactual Check",
        "description": "Test claims against plausible counterfactuals.",
        "instruction": (
            "Use only available evidence to identify counterfactual checks and "
            "state unsupported gaps explicitly."
        ),
    }


def skill_draft():
    return {
        "contract": "learn.skill-draft.v1",
        "kind": "skill",
        "title": "Source Triangulation",
        "description": "Compare claims across multiple available sources.",
        "instruction": (
            "Compare independently available sources, preserve disagreements, "
            "and state when the available evidence is insufficient."
        ),
    }


def test_publication_projection_preserves_sidebar_layout_authority(tmp_path):
    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    record_rel, record = next(
        (rel, row)
        for rel, row in tree.records.items()
        if any(ref["id"] == "summary" for ref in row["section_refs"])
    )
    summary_ref = next(ref for ref in record["section_refs"] if ref["id"] == "summary")
    summary_rel = record_rel.parent / summary_ref["source"]

    for rel in (record_rel, summary_rel):
        path = root / rel
        data = json.loads(path.read_text(encoding="utf-8"))
        data["hide_secondary_sidebar"] = False
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    tree = load_content_tree(root)
    projected = _project_json_tree(
        tree, tree.catalog, tree.prompts, tree.skills
    )
    assert json.loads(projected[record_rel])["hide_secondary_sidebar"] is False
    assert json.loads(projected[summary_rel])["hide_secondary_sidebar"] is False


def test_record_draft_maps_auxiliary_topic_fields_without_rst_interpretation():
    record = record_from_draft(
        topic_draft(),
        record_id="topic-supervised-learning-a1b2c3d4",
        created_at="2026-09-24T00:00:00Z",
    )
    by_id = {row["id"]: row for row in record["sections"]}
    assert by_id["knowledge-gaps"]["body"] == "- Need a benchmark."
    assert "regularization" in by_id["continue-learning"]["body"]
    assert all(row["citations"] == [] for row in record["sections"])


def test_historical_numeric_section_ids_are_normalized_at_draft_boundary():
    draft = topic_draft()
    draft["sections"] = [
        {"id": "1", "title": "First", "body": "One"},
        {"id": "2", "title": "Second", "body": "Two"},
    ]
    record = record_from_draft(
        draft,
        record_id="topic-numeric-sections-a1b2c3d4",
        created_at="2026-09-24T00:00:00Z",
    )
    assert [row["id"] for row in record["sections"][:2]] == [
        "section-1",
        "section-2",
    ]


def test_record_draft_rejects_unknown_fields_and_source_review_is_explicit():
    draft = topic_draft()
    draft["browser_selected_repo_path"] = "../../unexpected.rst"
    with pytest.raises(LearnValidationError, match="unexpected fields"):
        record_from_draft(
            draft,
            record_id="topic-safe-boundary",
            created_at="2026-09-24T00:00:00Z",
        )

    source = {
        "contract": "learn.record-creation-draft.v1",
        "kind": "source",
        "title": "Source",
        "summary": "Summary",
        "domains": ["machine-learning"],
        "sections": [],
        "url": "https://example.com/paper",
        "metadata_questions": ["Confirm publication date"],
        "requires_metadata_review": True,
    }
    with pytest.raises(LearnValidationError, match="reviewed"):
        record_from_draft(
            source,
            record_id="source-paper-a1b2c3d4",
            created_at="2026-09-24T00:00:00Z",
        )
    record = record_from_draft(
        source,
        record_id="source-paper-a1b2c3d4",
        created_at="2026-09-24T00:00:00Z",
        metadata_reviewed=True,
    )
    assert record["sections"][-1]["id"] == "metadata-review"


def test_suggested_identity_ignores_mutable_draft_content_and_handles_prompts():
    first = topic_draft()
    second = copy.deepcopy(first)
    second["summary"] = "Regenerated prose must not silently change identity."
    assert suggested_artifact_id(first) == "topic-introduction-to-supervised-learning"
    assert suggested_artifact_id(second) == suggested_artifact_id(first)
    assert suggested_artifact_id(
        first,
        existing_ids={
            "topic-introduction-to-supervised-learning",
            "topic-introduction-to-supervised-learning-2",
        },
    ) == "topic-introduction-to-supervised-learning-3"
    assert suggested_artifact_id(prompt_draft()) == "counterfactual-check"
    assert suggested_artifact_id(skill_draft()) == "skill-source-triangulation"


def test_prompt_draft_requires_review_owned_registry_metadata():
    prompt = prompt_from_draft(
        prompt_draft(),
        prompt_id="counterfactual-check",
        author="community",
        order=140,
        default_enabled=False,
    )
    assert prompt["id"] == "counterfactual-check"
    assert prompt["order"] == 140
    assert prompt["default_enabled"] is False
    assert "Generation contract:" not in prompt["instruction"]


def test_skill_draft_requires_review_owned_registry_metadata():
    skill = skill_from_draft(
        skill_draft(),
        skill_id="skill-source-triangulation",
        author="community",
        order=60,
        default_enabled=False,
    )
    assert skill["id"] == "skill-source-triangulation"
    assert skill["order"] == 60
    assert skill["default_enabled"] is False
    assert skill["domains"] == []
    assert skill["related"] == []
    assert "Generation contract:" not in skill["instruction"]


def test_create_record_plan_is_json_only_deterministic_and_exact(tmp_path):
    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    draft = topic_draft()
    record_id = suggested_artifact_id(
        draft, existing_ids=(row["id"] for row in tree.catalog["subjects"])
    )
    publication = {
        "contract": PUBLICATION_CONTRACT,
        "base_revision": tree.catalog["revision"],
        "operations": [
            {
                "op": "create-record",
                "record_id": record_id,
                "created_at": "2026-09-24T00:00:00Z",
                "draft": draft,
            }
        ],
    }
    first = publication_plan(root, publication)
    second = publication_plan(root, copy.deepcopy(publication))
    assert first == second
    assert first["contract"] == "learn.publication-plan.v2"
    assert first["deleted"] == []
    assert first["files"]
    assert all(name.endswith(".json") for name in first["files"])
    assert not any(name.endswith(".rst") for name in first["files"])
    assert any(name.endswith("/index.json") for name in first["files"])

    _apply_plan(root, first)
    loaded = load_content_tree(root)
    assert loaded.catalog["revision"] == first["revision"]
    assert record_id in loaded.routes


def test_new_topic_with_custom_sections_and_interaction_slots_projects_valid_tree(tmp_path):
    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    draft = topic_draft()
    draft["sections"] = [
        {"id": str(index), "title": f"Custom {index}", "body": f"Body {index}"}
        for index in range(1, 6)
    ]
    record_id = "topic-custom-capacity"
    publication = {
        "contract": PUBLICATION_CONTRACT,
        "base_revision": tree.catalog["revision"],
        "operations": [
            {
                "op": "create-record",
                "record_id": record_id,
                "created_at": "2026-09-24T03:30:00Z",
                "draft": draft,
            }
        ],
    }
    plan = publication_plan(root, publication)
    assert len(plan["files"]) == 36
    _apply_plan(root, plan)
    projected = load_content_tree(root)
    assert projected.catalog["revision"] == plan["revision"]
    subject = next(row for row in projected.catalog["subjects"] if row["id"] == record_id)
    assert len(subject["sections"]) > 32
    assert any(row["id"] == "skill-check-reference" for row in subject["sections"])
    assert any(row["id"] == "eli14" for row in subject["sections"])


def test_section_patch_and_evidence_attachment_change_only_section_json(tmp_path):
    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    topic = next(row for row in tree.catalog["subjects"] if row["kind"] == "topic")
    source = next(row for row in tree.catalog["subjects"] if row["kind"] == "source")
    publication = {
        "contract": PUBLICATION_CONTRACT,
        "base_revision": tree.catalog["revision"],
        "operations": [
            {
                "op": "upsert-section",
                "subject_id": topic["id"],
                "section_id": "summary",
                "title": "Summary",
                "body": "A reviewed replacement summary.",
            },
            {
                "op": "attach-source",
                "subject_id": topic["id"],
                "section_id": "summary",
                "section_title": "Summary",
                "source_id": source["id"],
                "locator": "Reviewed locator",
            },
        ],
    }
    plan = publication_plan(root, publication)
    assert len(plan["files"]) == 1
    relative = next(iter(plan["files"]))
    assert relative.endswith("/summary.json")
    payload = json.loads(plan["files"][relative])
    assert payload["contract"] == "learn.section.v2"
    active = next(
        row
        for row in payload["section"]["generations"]
        if row["id"] == payload["section"]["active_generation_id"]
    )
    assert active["body"] == "A reviewed replacement summary."
    assert {"source_id": source["id"], "locator": "Reviewed locator"} in active[
        "citations"
    ]


def test_new_topic_prompt_fans_out_atomically_to_every_topic(tmp_path):
    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    topics = [row for row in tree.catalog["subjects"] if row["kind"] == "topic"]
    publication = {
        "contract": PUBLICATION_CONTRACT,
        "base_revision": tree.catalog["revision"],
        "operations": [
            {
                "op": "create-topic-prompt",
                "prompt_id": "counterfactual-check",
                "author": "community",
                "order": 140,
                "default_enabled": False,
                "draft": prompt_draft(),
            }
        ],
    }
    plan = publication_plan(root, publication)
    assert len(plan["files"]) == 1 + 2 * len(topics)
    assert "topic-prompts/counterfactual-check/index.json" in plan["files"]
    assert sum(name.endswith("/topic-prompts/counterfactual-check.json") for name in plan["files"]) == len(topics)
    assert sum(name.endswith("/index.json") and name.startswith("topics/") for name in plan["files"]) == len(topics)

    _apply_plan(root, plan)
    loaded = load_content_tree(root)
    assert loaded.catalog["revision"] == plan["revision"]
    assert loaded.prompts[-1]["id"] == "counterfactual-check"


def test_new_skill_fans_out_atomically_to_every_topic(tmp_path):
    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    topics = [row for row in tree.catalog["subjects"] if row["kind"] == "topic"]
    publication = {
        "contract": PUBLICATION_CONTRACT,
        "base_revision": tree.catalog["revision"],
        "operations": [
            {
                "op": "create-skill",
                "skill_id": "skill-source-triangulation",
                "author": "community",
                "order": 60,
                "default_enabled": False,
                "draft": skill_draft(),
            }
        ],
    }
    plan = publication_plan(root, publication)
    assert len(plan["files"]) == 1 + 2 * len(topics)
    assert "skills/skill-source-triangulation/index.json" in plan["files"]
    assert sum(
        name.endswith("/skills/skill-source-triangulation.json")
        for name in plan["files"]
    ) == len(topics)
    assert sum(
        name.endswith("/index.json") and name.startswith("topics/")
        for name in plan["files"]
    ) == len(topics)
    assert all(name.endswith(".json") for name in plan["files"])

    _apply_plan(root, plan)
    loaded = load_content_tree(root)
    assert loaded.catalog["revision"] == plan["revision"]
    assert loaded.skills[-1]["id"] == "skill-source-triangulation"
    assert loaded.routes["skill-source-triangulation"] == "skills/skill-source-triangulation/index"


def test_publication_rejects_cross_registry_and_structural_interaction_collisions(tmp_path):
    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    existing_skill = tree.skills[0]
    cross = {
        "contract": PUBLICATION_CONTRACT,
        "base_revision": tree.catalog["revision"],
        "operations": [
            {
                "op": "create-topic-prompt",
                "prompt_id": existing_skill["id"],
                "author": "community",
                "order": 140,
                "default_enabled": False,
                "draft": prompt_draft(),
            }
        ],
    }
    with pytest.raises(LearnValidationError, match="must be disjoint"):
        publication_plan(root, cross)

    structural = copy.deepcopy(cross)
    structural["operations"][0]["prompt_id"] = "summary"
    with pytest.raises(LearnValidationError, match="structural section"):
        publication_plan(root, structural)


def test_publication_refuses_stale_base_revision_and_bad_source_target(tmp_path):
    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    topic = next(row for row in tree.catalog["subjects"] if row["kind"] == "topic")
    stale = {
        "contract": PUBLICATION_CONTRACT,
        "base_revision": "stale",
        "operations": [
            {
                "op": "upsert-section",
                "subject_id": topic["id"],
                "section_id": "summary",
                "title": "Summary",
                "body": "New",
            }
        ],
    }
    with pytest.raises(LearnValidationError, match="canonical tree changed"):
        publication_plan(root, stale)

    bad = copy.deepcopy(stale)
    bad["base_revision"] = tree.catalog["revision"]
    bad["operations"] = [
        {
            "op": "attach-source",
            "subject_id": topic["id"],
            "section_id": "summary",
            "section_title": "Summary",
            "source_id": topic["id"],
            "locator": "Example",
        }
    ]
    with pytest.raises(LearnValidationError, match="Source"):
        publication_plan(root, bad)


def test_review_bundle_cli_contains_json_only(tmp_path):
    root = _json_only_copy(tmp_path)
    draft_path = tmp_path / "draft.json"
    bundle = tmp_path / "bundle"
    draft_path.write_text(json.dumps(topic_draft()), encoding="utf-8")
    publication_cli(
        [
            str(root),
            str(draft_path),
            str(bundle),
            "--created-at",
            "2026-09-24T00:00:00Z",
        ]
    )
    manifest = json.loads((bundle / "publication-plan.json").read_text())
    assert manifest["contract"] == "learn.publication-review-bundle.v2"
    assert manifest["files"]
    assert all(name.endswith(".json") for name in manifest["files"])
    assert all(name.startswith("docs/source/learn-ai/") for name in manifest["files"])


def test_review_bundle_cli_supports_skill_without_record_timestamp(tmp_path):
    root = _json_only_copy(tmp_path)
    draft_path = tmp_path / "skill-draft.json"
    bundle = tmp_path / "skill-bundle"
    draft_path.write_text(json.dumps(skill_draft()), encoding="utf-8")
    publication_cli([str(root), str(draft_path), str(bundle)])
    manifest = json.loads((bundle / "publication-plan.json").read_text())
    assert manifest["files"]
    assert all(name.endswith(".json") for name in manifest["files"])
    assert any(
        name.endswith("skills/skill-source-triangulation/index.json")
        for name in manifest["files"]
    )


def test_review_bundle_cli_rejects_duplicate_json_keys(tmp_path):
    root = _json_only_copy(tmp_path)
    draft = tmp_path / "draft.json"
    draft.write_text(
        '{"contract":"learn.record-creation-draft.v1","kind":"topic",'
        '"title":"One","title":"Two"}',
        encoding="utf-8",
    )
    with pytest.raises(LearnValidationError, match="duplicate JSON key"):
        publication_cli(
            [
                str(root),
                str(draft),
                str(tmp_path / "bundle"),
                "--created-at",
                "2026-09-24T00:00:00Z",
            ]
        )


def test_reviewed_transport_request_reuses_canonical_publication_planner(tmp_path):
    from _sphinx_ext._sphinx_ai_learn._publication_request_cli import plan_publication_request

    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    draft = topic_draft()
    draft["provenance"] = {"base_revision": tree.catalog["revision"]}
    request = {
        "contract": "learn.publication-request.v1",
        "action": "publish",
        "draft": draft,
        "base_revision": tree.catalog["revision"],
        "created_at": "2026-09-25T01:00:00Z",
        "artifact_id": "auto",
        "author": "community",
        "default_enabled": False,
    }
    plan = plan_publication_request(root, request)
    assert plan["base_revision"] == tree.catalog["revision"]
    assert plan["files"]
    assert all(path.endswith(".json") for path in plan["files"])
    assert not plan["deleted"]




def test_section_publication_credit_is_public_metadata_and_preserves_history(tmp_path):
    from _sphinx_ext._sphinx_ai_learn._publication_request_cli import plan_publication_request

    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    topic = next(row for row in tree.catalog["subjects"] if row["kind"] == "topic")
    section_id = "summary"
    request = {
        "contract": "learn.publication-request.v1",
        "action": "publish",
        "draft": {
            "contract": "learn.section-draft.v2",
            "title": "Summary",
            "body": "Reviewed body one.",
            "provenance": {"base_revision": tree.catalog["revision"]},
        },
        "base_revision": tree.catalog["revision"],
        "subject_id": topic["id"],
        "section_id": section_id,
        "section_title": "Summary",
        "contributor": {"display_name": "DataFox"},
    }
    first = plan_publication_request(root, request)
    relative = next(name for name in first["files"] if name.endswith("/summary.json"))
    payload = json.loads(first["files"][relative])
    active = next(
        row
        for row in payload["section"]["generations"]
        if row["id"] == payload["section"]["active_generation_id"]
    )
    assert active["contributors"] == ["DataFox"]
    _apply_plan(root, first)

    tree2 = load_content_tree(root)
    request["base_revision"] = tree2.catalog["revision"]
    request["draft"]["body"] = "Reviewed body two."
    request["draft"]["provenance"]["base_revision"] = tree2.catalog["revision"]
    request["contributor"] = {"display_name": "Ada"}
    second = plan_publication_request(root, request)
    relative2 = next(name for name in second["files"] if name.endswith("/summary.json"))
    payload2 = json.loads(second["files"][relative2])
    active2 = next(
        row
        for row in payload2["section"]["generations"]
        if row["id"] == payload2["section"]["active_generation_id"]
    )
    assert active2["contributors"] == ["Ada"]
    assert [row["contributors"] for row in payload2["section"]["generations"]][-2:] == [
        ["DataFox"],
        ["Ada"],
    ]


def test_publication_credit_defaults_to_anonymous_and_rejects_extra_identity_fields(tmp_path):
    from _sphinx_ext._sphinx_ai_learn._publication_request_cli import plan_publication_request

    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    topic = next(row for row in tree.catalog["subjects"] if row["kind"] == "topic")
    request = {
        "contract": "learn.publication-request.v1",
        "action": "publish",
        "draft": {
            "contract": "learn.section-draft.v2",
            "title": "Summary",
            "body": "Anonymous body.",
            "provenance": {"base_revision": tree.catalog["revision"]},
        },
        "base_revision": tree.catalog["revision"],
        "subject_id": topic["id"],
        "section_id": "summary",
        "section_title": "Summary",
    }
    plan = plan_publication_request(root, request)
    relative = next(name for name in plan["files"] if name.endswith("/summary.json"))
    payload = json.loads(plan["files"][relative])
    active = next(
        row
        for row in payload["section"]["generations"]
        if row["id"] == payload["section"]["active_generation_id"]
    )
    assert active["contributors"][-1] == "Anonymous"

    request["contributor"] = {"display_name": "Ada", "email": "private@example.org"}
    with pytest.raises(LearnValidationError, match=r"expected \{display_name\}"):
        plan_publication_request(root, request)

    request["contributor"] = {"display_name": "Ada\nInjected"}
    with pytest.raises(LearnValidationError, match="control characters"):
        plan_publication_request(root, request)


def test_publication_credit_maps_to_native_record_prompt_and_skill_authorship(tmp_path):
    from _sphinx_ext._sphinx_ai_learn._publication_request_cli import plan_publication_request

    root = _json_only_copy(tmp_path)
    revision = load_content_tree(root).catalog["revision"]

    prompt = {
        "contract": "learn.publication-request.v1",
        "action": "publish",
        "base_revision": revision,
        "artifact_id": "credit-prompt-test",
        "contributor": {"display_name": "DataFox"},
        "draft": {
            "contract": "learn.topic-prompt-draft.v1",
            "kind": "topic-prompt",
            "title": "Credit Prompt Test",
            "description": "Description",
            "instruction": "Instruction",
        },
    }
    prompt_plan = plan_publication_request(root, prompt)
    prompt_payload = json.loads(prompt_plan["files"]["topic-prompts/credit-prompt-test/index.json"])
    assert prompt_payload["author"] == "DataFox"
    assert prompt_payload["authors"] == ["DataFox"]

    skill = {
        "contract": "learn.publication-request.v1",
        "action": "publish",
        "base_revision": revision,
        "artifact_id": "credit-skill-test",
        "contributor": {"display_name": ""},
        "draft": {
            "contract": "learn.skill-draft.v1",
            "kind": "skill",
            "title": "Credit Skill Test",
            "description": "Description",
            "instruction": "Instruction",
        },
    }
    skill_plan = plan_publication_request(root, skill)
    skill_payload = json.loads(skill_plan["files"]["skills/credit-skill-test/index.json"])
    assert skill_payload["author"] == "Anonymous"
    assert skill_payload["authors"] == ["Anonymous"]

    record = {
        "contract": "learn.publication-request.v1",
        "action": "publish",
        "base_revision": revision,
        "artifact_id": "topic-credit-record-test",
        "created_at": "2026-09-27T00:00:00Z",
        "draft": {
            "contract": "learn.record-creation-draft.v1",
            "kind": "topic",
            "title": "Credit Record Test",
            "summary": "Summary",
            "domains": ["statistics"],
            "sections": [{"id": "summary", "title": "Summary", "body": "Body"}],
        },
    }
    record_plan = plan_publication_request(root, record)
    record_payload = next(
        json.loads(raw)
        for raw in record_plan["files"].values()
        if json.loads(raw).get("contract") == "learn.record.v2"
        and json.loads(raw).get("record", {}).get("title") == "Credit Record Test"
    )
    assert record_payload["record"]["authors"] == ["Anonymous"]
    record_sections = [
        json.loads(raw)["section"]
        for name, raw in record_plan["files"].items()
        if name.startswith("topics/")
        and name.endswith(".json")
        and json.loads(raw).get("contract") in {"learn.section.v1", "learn.section.v2"}
    ]
    contributed_sections = [
        section
        for section in record_sections
        if section.get("body") or section.get("citations") or section.get("links")
    ]
    assert contributed_sections
    assert all(
        section.get("contributors") == ["Anonymous"]
        for section in contributed_sections
    )


def test_request_actions_reject_cross_action_fields(tmp_path):
    from _sphinx_ext._sphinx_ai_learn._publication_request_cli import plan_publication_request

    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    topic = next(
        row
        for row in tree.catalog["subjects"]
        if row["kind"] == "topic"
        and any(section["id"] == "summary" and section["body"] for section in row["sections"])
    )
    section = next(row for row in topic["sections"] if row["id"] == "summary")
    feedback = {
        "contract": "learn.publication-request.v1",
        "action": "feedback",
        "base_revision": tree.catalog["revision"],
        "subject_id": topic["id"],
        "section_id": "summary",
        "generation_id": section_generation_id(topic, section),
        "feedback_id": _feedback_id(1),
        "created_at": "2026-09-27T04:00:00Z",
        "rating": 1,
        "draft": {"body": "must not be ignored"},
    }
    with pytest.raises(LearnValidationError, match="feedback request: unexpected fields"):
        plan_publication_request(root, feedback)

    publish = {
        "contract": "learn.publication-request.v1",
        "action": "publish",
        "base_revision": tree.catalog["revision"],
        "subject_id": topic["id"],
        "section_id": "summary",
        "section_title": "Summary",
        "draft": {
            "contract": "learn.section-draft.v2",
            "title": "Summary",
            "body": "Reviewed body.",
            "provenance": {"base_revision": tree.catalog["revision"]},
        },
        "rating": 5,
    }
    with pytest.raises(LearnValidationError, match="publish request: unexpected fields"):
        plan_publication_request(root, publish)




def test_feedback_rejects_browser_authored_created_at(tmp_path):
    """Feedback chronology belongs to repository review history, not the browser."""
    from _sphinx_ext._sphinx_ai_learn._publication_request_cli import plan_publication_request

    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    topic = next(
        row
        for row in tree.catalog["subjects"]
        if row["kind"] == "topic"
        and any(section["id"] == "summary" and section["body"] for section in row["sections"])
    )
    section = next(row for row in topic["sections"] if row["id"] == "summary")
    generation_id = section_generation_id(topic, section)

    request = {
        "contract": "learn.publication-request.v1",
        "action": "feedback",
        "base_revision": tree.catalog["revision"],
        "subject_id": topic["id"],
        "section_id": "summary",
        "generation_id": generation_id,
        "feedback_id": _feedback_id(90),
        "rating": 1,
        "feedback_mode": "quick",
        "created_at": "2026-09-27T04:00:00Z",
    }
    with pytest.raises(LearnValidationError, match="feedback request: unexpected fields: created_at"):
        plan_publication_request(root, request)

    direct = {
        "contract": PUBLICATION_CONTRACT,
        "base_revision": tree.catalog["revision"],
        "operations": [
            {
                "op": "rate-section-generation",
                "subject_id": topic["id"],
                "section_id": "summary",
                "generation_id": generation_id,
                "feedback_id": _feedback_id(91),
                "rating": 1,
                "contributor": "Anonymous",
                "feedback_mode": "quick",
                "created_at": "2026-09-27T04:00:00Z",
            }
        ],
    }
    with pytest.raises(LearnValidationError, match="unexpected or missing fields"):
        publication_plan(root, direct)



def test_republishing_same_generation_adds_participant_without_duplication(tmp_path):
    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    topic = next(row for row in tree.catalog["subjects"] if row["kind"] == "topic")
    operation = {
        "op": "upsert-section",
        "subject_id": topic["id"],
        "section_id": "summary",
        "title": "Summary",
        "body": "Same accepted generation.",
        "created_at": "2026-09-27T06:00:00Z",
        "provenance": {"model": "stub/model", "workflow_id": "wf-1"},
        "contributor": "DataFox",
    }
    first = publication_plan(
        root,
        {
            "contract": PUBLICATION_CONTRACT,
            "base_revision": tree.catalog["revision"],
            "operations": [operation],
        },
    )
    _apply_plan(root, first)
    current = load_content_tree(root)
    replay = copy.deepcopy(operation)
    replay["contributor"] = "Ada"
    second = publication_plan(
        root,
        {
            "contract": PUBLICATION_CONTRACT,
            "base_revision": current.catalog["revision"],
            "operations": [replay],
        },
    )
    relative = next(name for name in second["files"] if name.endswith("/summary.json"))
    payload = json.loads(second["files"][relative])
    assert len(payload["section"]["generations"]) >= 1
    active = next(
        row
        for row in payload["section"]["generations"]
        if row["id"] == payload["section"]["active_generation_id"]
    )
    assert active["body"] == "Same accepted generation."
    assert active["contributors"] == ["DataFox", "Ada"]
    ids = [row["id"] for row in payload["section"]["generations"]]
    assert len(ids) == len(set(ids))


def test_first_publication_into_empty_placeholder_does_not_create_fake_history(tmp_path):
    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    target = None
    for subject in tree.catalog["subjects"]:
        if subject["kind"] == "skill":
            continue
        for section in subject["sections"]:
            if not section.get("body", "").strip() and not section.get("generations"):
                target = (subject, section)
                break
        if target:
            break
    assert target is not None
    subject, section = target
    plan = publication_plan(
        root,
        {
            "contract": PUBLICATION_CONTRACT,
            "base_revision": tree.catalog["revision"],
            "operations": [
                {
                    "op": "upsert-section",
                    "subject_id": subject["id"],
                    "section_id": section["id"],
                    "title": section["title"],
                    "body": "First accepted content.",
                    "created_at": "2026-09-27T06:10:00Z",
                    "contributor": "DataFox",
                }
            ],
        },
    )
    relative = next(
        name
        for name in plan["files"]
        if name.endswith(f"/{section['id']}.json")
    )
    payload = json.loads(plan["files"][relative])
    assert payload["contract"] == "learn.section.v2"
    assert len(payload["section"]["generations"]) == 1
    assert payload["section"]["generations"][0]["body"] == "First accepted content."

def test_reviewed_feedback_uses_immutable_sidecar_and_is_replay_safe(tmp_path):
    from _sphinx_ext._sphinx_ai_learn._publication_request_cli import plan_publication_request

    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    topic = next(
        row
        for row in tree.catalog["subjects"]
        if row["kind"] == "topic"
        and any(section["id"] == "summary" and section["body"] for section in row["sections"])
    )
    section = next(row for row in topic["sections"] if row["id"] == "summary")
    generation_id = section_generation_id(topic, section)
    request = {
        "contract": "learn.publication-request.v1",
        "action": "feedback",
        "base_revision": tree.catalog["revision"],
        "subject_id": topic["id"],
        "section_id": "summary",
        "generation_id": generation_id,
        "feedback_id": _feedback_id(1),
        "rating": 1,
        "feedback_mode": "quick",
        "contributor": {"display_name": "DataFox"},
    }
    plan = plan_publication_request(root, request)
    assert not any(name.endswith("/summary.json") for name in plan["files"])
    relative = next(name for name in plan["files"] if "/feedback/" in name)
    payload = json.loads(plan["files"][relative])
    assert payload == {
        "contract": "learn.generation-feedback.v1",
        "feedback": {
            "contributor": "DataFox",
            "id": _feedback_id(1),
            "mode": "quick",
            "rating": 1,
        },
        "generation_id": generation_id,
        "record_id": topic["id"],
        "section_id": "summary",
    }
    _apply_plan(root, plan)

    tree2 = load_content_tree(root)
    assert tree2.generation_feedback[(topic["id"], "summary", generation_id)][0]["rating"] == 1
    # Rating a current v1 accepted section must not rewrite/promote its section JSON.
    final_section = next(
        row
        for row in next(s for s in tree2.catalog["subjects"] if s["id"] == topic["id"])["sections"]
        if row["id"] == "summary"
    )
    assert "generations" not in final_section
    request["base_revision"] = tree2.catalog["revision"]
    replay = plan_publication_request(root, request)
    assert replay["files"] == {}
    assert replay["deleted"] == []


def test_v1_section_feedback_remains_attached_after_later_section_promotion(tmp_path):
    """A current v1 section's projected generation survives v2 promotion."""
    from _sphinx_ext._sphinx_ai_learn._publication_request_cli import plan_publication_request

    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    topic = next(
        row
        for row in tree.catalog["subjects"]
        if row["kind"] == "topic"
        and any(section["id"] == "summary" and section["body"] for section in row["sections"])
    )
    section = next(row for row in topic["sections"] if row["id"] == "summary")
    assert "generations" not in section
    projected_id = section_generation_id(topic, section)

    feedback = {
        "contract": "learn.publication-request.v1",
        "action": "feedback",
        "base_revision": tree.catalog["revision"],
        "subject_id": topic["id"],
        "section_id": "summary",
        "generation_id": projected_id,
        "feedback_id": _feedback_id(2),
        "rating": 1,
        "feedback_mode": "quick",
    }
    _apply_plan(root, plan_publication_request(root, feedback))

    current = load_content_tree(root)
    publish = {
        "contract": PUBLICATION_CONTRACT,
        "base_revision": current.catalog["revision"],
        "operations": [
            {
                "op": "upsert-section",
                "subject_id": topic["id"],
                "section_id": "summary",
                "title": "Summary",
                "body": "A later accepted summary generation.",
                "created_at": "2026-09-27T18:00:00Z",
                "contributor": "LaterAuthor",
            }
        ],
    }
    _apply_plan(root, publication_plan(root, publish))

    promoted = load_content_tree(root)
    promoted_topic = next(row for row in promoted.catalog["subjects"] if row["id"] == topic["id"])
    promoted_section = next(row for row in promoted_topic["sections"] if row["id"] == "summary")
    ids = [row["id"] for row in promoted_section["generations"]]
    assert projected_id in ids
    assert promoted_section["active_generation_id"] != projected_id
    assert promoted.generation_feedback[(topic["id"], "summary", projected_id)] == (
        {
            "id": _feedback_id(2),
            "rating": 1,
            "contributor": "Anonymous",
            "mode": "quick",
        },
    )


def test_generation_replay_preserves_later_feedback_and_review_metadata(tmp_path):
    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    topic = next(row for row in tree.catalog["subjects"] if row["kind"] == "topic")
    publication = {
        "contract": PUBLICATION_CONTRACT,
        "base_revision": tree.catalog["revision"],
        "operations": [
            {
                "op": "upsert-section",
                "subject_id": topic["id"],
                "section_id": "summary",
                "title": "Summary",
                "body": "Stable accepted generation.",
                "contributor": "DataFox",
                "created_at": "2026-09-27T04:10:00Z",
                "generation_id": "generation-replay-preserve",
            }
        ],
    }
    first = publication_plan(root, publication)
    _apply_plan(root, first)

    current = load_content_tree(root)
    rate = {
        "contract": PUBLICATION_CONTRACT,
        "base_revision": current.catalog["revision"],
        "operations": [
            {
                "op": "rate-section-generation",
                "subject_id": topic["id"],
                "section_id": "summary",
                "generation_id": "generation-replay-preserve",
                "feedback_id": _feedback_id(3),
                "rating": 5,
                "contributor": "Ada",
            }
        ],
    }
    rated = publication_plan(root, rate)
    _apply_plan(root, rated)

    after = load_content_tree(root)
    replay = copy.deepcopy(publication)
    replay["base_revision"] = after.catalog["revision"]
    plan = publication_plan(root, replay)
    # Re-activating the same immutable generation is a no-op; later feedback
    # remains canonical instead of causing an identifier collision.
    assert plan["files"] == {}
    assert plan["deleted"] == []


def test_detailed_feedback_accepts_full_eleven_point_scale_and_rejects_collision(tmp_path):
    from _sphinx_ext._sphinx_ai_learn._publication_request_cli import plan_publication_request

    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    topic = next(
        row
        for row in tree.catalog["subjects"]
        if row["kind"] == "topic"
        and any(section["id"] == "summary" and section["body"] for section in row["sections"])
    )
    section = next(row for row in topic["sections"] if row["id"] == "summary")
    generation_id = section_generation_id(topic, section)
    for index, rating in enumerate(range(-5, 6)):
        current = load_content_tree(root)
        request = {
            "contract": "learn.publication-request.v1",
            "action": "feedback",
            "base_revision": current.catalog["revision"],
            "subject_id": topic["id"],
            "section_id": "summary",
            "generation_id": generation_id,
            "feedback_id": _feedback_id(100 + index),
            "rating": rating,
            "feedback_mode": "detailed",
            "comment": "" if rating == 0 else f"rating {rating}",
            "contributor": {"display_name": ""},
        }
        plan = plan_publication_request(root, request)
        assert len(plan["files"]) == 1
        assert "/feedback/" in next(iter(plan["files"]))
        _apply_plan(root, plan)
    final = load_content_tree(root)
    rows = final.generation_feedback[(topic["id"], "summary", generation_id)]
    assert [row["rating"] for row in rows] == list(range(-5, 6))
    assert sum(row["rating"] for row in rows) == 0
    assert len(rows) == 11
    assert all(row["mode"] == "detailed" for row in rows)
    assert all("created_at" not in row for row in rows)

    current = load_content_tree(root)
    collision = {
        "contract": "learn.publication-request.v1",
        "action": "feedback",
        "base_revision": current.catalog["revision"],
        "subject_id": topic["id"],
        "section_id": "summary",
        "generation_id": generation_id,
        "feedback_id": _feedback_id(100),
        "rating": 5,
    }
    with pytest.raises(LearnValidationError, match="feedback identifier collision"):
        plan_publication_request(root, collision)



    invalid_quick = dict(collision)
    invalid_quick["feedback_id"] = _feedback_id(101)
    invalid_quick["feedback_mode"] = "quick"
    invalid_quick["rating"] = 5
    with pytest.raises(LearnValidationError, match="quick feedback"):
        plan_publication_request(root, invalid_quick)



def test_concurrent_feedback_plans_touch_distinct_sidecars_only(tmp_path):
    from _sphinx_ext._sphinx_ai_learn._publication_request_cli import plan_publication_request

    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    topic = next(
        row
        for row in tree.catalog["subjects"]
        if row["kind"] == "topic"
        and any(section["id"] == "summary" and section["body"] for section in row["sections"])
    )
    section = next(row for row in topic["sections"] if row["id"] == "summary")
    generation_id = section_generation_id(topic, section)
    base = {
        "contract": "learn.publication-request.v1",
        "action": "feedback",
        "base_revision": tree.catalog["revision"],
        "subject_id": topic["id"],
        "section_id": "summary",
        "generation_id": generation_id,
        "rating": 1,
        "feedback_mode": "quick",
    }
    first = plan_publication_request(
        root, {**base, "feedback_id": _feedback_id(201)}
    )
    second = plan_publication_request(
        root, {**base, "feedback_id": _feedback_id(202), "rating": -1}
    )
    assert len(first["files"]) == len(second["files"]) == 1
    path1 = next(iter(first["files"]))
    path2 = next(iter(second["files"]))
    assert path1 != path2
    assert "/feedback/" in path1 and "/feedback/" in path2
    assert not path1.endswith("/summary.json") and not path2.endswith("/summary.json")

    # Simulate two independently reviewed branches landing sequentially. Since
    # each event owns a distinct file, the second tree remains valid without a
    # hot-file merge/rebase step.
    _apply_plan(root, first)
    _apply_plan(root, second)
    loaded = load_content_tree(root)
    rows = loaded.generation_feedback[(topic["id"], "summary", generation_id)]
    assert sorted(row["rating"] for row in rows) == [-1, 1]

def test_reviewed_source_request_requires_explicit_metadata_review(tmp_path):
    from _sphinx_ext._sphinx_ai_learn._publication_request_cli import plan_publication_request

    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    draft = {
        "contract": "learn.record-creation-draft.v1",
        "kind": "source",
        "title": "Reviewed source",
        "summary": "Summary",
        "domains": ["statistics"],
        "sections": [{"id": "overview", "title": "Overview", "body": "Notes"}],
        "url": "https://example.org/source",
        "metadata_questions": [],
        "requires_metadata_review": True,
        "provenance": {"base_revision": tree.catalog["revision"]},
    }
    request = {
        "contract": "learn.publication-request.v1",
        "action": "publish",
        "draft": draft,
        "base_revision": tree.catalog["revision"],
        "created_at": "2026-09-25T01:00:00Z",
        "artifact_id": "auto",
    }
    with pytest.raises(LearnValidationError, match="metadata_reviewed"):
        plan_publication_request(root, request)
    request["metadata_reviewed"] = True
    plan = plan_publication_request(root, request)
    assert plan["files"]
    assert all(path.endswith(".json") for path in plan["files"])


def test_reviewed_publication_workflow_separates_dry_run_and_json_only_write_authority():
    root = _learn_site.site_repository()
    workflow_path = root / ".github" / "workflows" / "ai-learn-publish.yml"
    text = workflow_path.read_text()
    assert "transport-test:" in text
    assert "permissions:\n      contents: read" in text
    assert "validate-plan-and-open-pr:" in text
    assert "contents: write" in text
    assert "pull-requests: write" in text
    assert "inputs.operation == 'test' &&" in text
    assert "inputs.operation == 'publish' &&" in text
    assert text.count("github.repository == 'scikit-plots/learn' &&") == 2
    assert text.count("github.ref_name == github.event.repository.default_branch") == 2
    assert "AI_LEARN_BASE_BRANCH: ${{ github.event.repository.default_branch }}" in text
    assert "PYTHONDONTWRITEBYTECODE: '1'" in text
    assert "persist-credentials: false" in text
    assert "persist-credentials: true" not in text
    assert "Publication changed forbidden repository paths" in text
    assert text.count("'git', 'ls-files', '--others', '--exclude-standard'") >= 2
    assert "planned_untracked" in text
    assert "['git', 'show', f'{remote_ref}:{name}']" in text
    assert "current reviewed JSON plan paths" in text
    assert "current reviewed JSON plan content" in text
    assert "Non-JSON staged path detected" in text
    assert "docs/source/learn-ai" in text
    assert "install -m 700 .github/scripts/ai-learn-git-askpass.sh" in text
    assert "GIT_ASKPASS: ${{ runner.temp }}/ai-learn-git-askpass.sh" in text
    assert text.count("GH_TOKEN: ${{ github.token }}") >= 3
    assert text.count("GH_REPO: ${{ github.repository }}") >= 2
    assert "GIT_TERMINAL_PROMPT: '0'" in text
    assert "git push origin HEAD:\"$BRANCH\"" in text
    assert "branch_reusable=true" in text
    assert "--state closed" in text
    assert "--state all" in text
    assert "Recognize terminal deterministic replay" in text
    assert "Already merged AI Learn review" in text
    assert "steps.terminal.outputs.replayed != 'true'" in text
    assert "Refusing to create a second review for the same deterministic request id." in text
    assert "refusing to treat a Git transport/authentication failure as branch absence" in text
    assert "Reserved publication branch contains forbidden paths" in text
    assert "Reserved publication branch does not match the current reviewed JSON plan" in text
    assert "Recovered matching AI Learn publication branch; PR creation will be retried." in text
    assert "Feedback {rating_label} · {subject}#{section}" in text
    assert r"- Review reference: \`${SHORT_ID}\`" in text
    assert r"- Request: \`${REQUEST_ID}\`" not in text
    assert "gh pr create" in text
    assert "https://x-access-token" not in text
    proxy = (root / "docs/source/scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/app.py").read_text()
    assert "def _learn_publication_rate_identity(request: Request, *, scope: str)" in proxy
    assert "_learn_publication_local_identity_secret: bytes = secrets.token_bytes(32)" in proxy
    assert "hmac.new(" in proxy
    assert proxy.count("_learn_publication_rate_identity(") == 3
    assert 'scope="learn-publication-test"' in proxy
    assert 'scope="learn-publication"' in proxy
    assert 'message = f"ai-learn-publication:{scope}\\0{identity}"' in proxy
    assert "\n          PY\n          PY\n" not in text




def test_reviewed_publication_workflow_run_blocks_are_bash_syntax_valid():
    root = _learn_site.site_repository()
    workflow = (root / ".github" / "workflows" / "ai-learn-publish.yml").read_text()
    lines = workflow.splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        match = re.match(r"^(\s*)run:\s*\|\s*$", lines[index])
        if match is None:
            index += 1
            continue
        base = len(match.group(1))
        index += 1
        body = []
        while index < len(lines):
            line = lines[index]
            indent = len(line) - len(line.lstrip())
            if line.strip() and indent <= base:
                break
            body.append(line[base + 2 :] if len(line) >= base + 2 else "")
            index += 1
        blocks.append("\n".join(body))
    assert len(blocks) >= 10
    for block_index, script in enumerate(blocks):
        script = re.sub(r"\$\{\{.*?\}\}", "GITHUB_EXPR", script)
        result = subprocess.run(
            ["bash", "-n"],
            input=script,
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
        assert result.returncode == 0, (
            f"workflow run block {block_index}: {result.stderr}"
        )

def test_reviewed_publication_git_askpass_is_prompt_scoped_and_fail_closed():
    root = _learn_site.site_repository()
    helper = root / ".github" / "scripts" / "ai-learn-git-askpass.sh"
    assert helper.is_file()

    env = os.environ.copy()
    env["GH_TOKEN"] = "test-publication-token"

    username = subprocess.run(
        ["/bin/sh", str(helper), "Username for 'https://github.com':"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=5,
    )
    assert username.stdout.strip() == "x-access-token"
    assert "test-publication-token" not in username.stdout

    password = subprocess.run(
        ["/bin/sh", str(helper), "Password for 'https://x-access-token@github.com':"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=5,
    )
    assert password.stdout.strip() == "test-publication-token"

    unknown = subprocess.run(
        ["/bin/sh", str(helper), "Unexpected prompt"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=5,
    )
    assert unknown.returncode != 0
    assert unknown.stdout == ""

    missing_env = env.copy()
    missing_env.pop("GH_TOKEN", None)
    missing = subprocess.run(
        ["/bin/sh", str(helper), "Password for 'https://github.com':"],
        check=False,
        capture_output=True,
        text=True,
        env=missing_env,
        timeout=5,
    )
    assert missing.returncode != 0
    assert missing.stdout == ""


def test_feedback_new_writes_require_opaque_event_nonce_at_repository_boundary(tmp_path):
    from _sphinx_ext._sphinx_ai_learn._publication_request_cli import plan_publication_request

    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    topic = next(
        row
        for row in tree.catalog["subjects"]
        if row["kind"] == "topic"
        and any(section["id"] == "summary" and section["body"] for section in row["sections"])
    )
    section = next(row for row in topic["sections"] if row["id"] == "summary")
    generation_id = section_generation_id(topic, section)
    request = {
        "contract": "learn.publication-request.v1",
        "action": "feedback",
        "base_revision": tree.catalog["revision"],
        "subject_id": topic["id"],
        "section_id": "summary",
        "generation_id": generation_id,
        "feedback_id": "feedback-user-or-email-derived-label",
        "rating": 1,
        "feedback_mode": "quick",
    }
    with pytest.raises(LearnValidationError, match="opaque reviewed-feedback event nonce"):
        plan_publication_request(root, request)

    invalid_revision = dict(request)
    invalid_revision.update(
        feedback_id=_feedback_id(300),
        base_revision="private@example.org",
    )
    with pytest.raises(LearnValidationError, match="invalid tree revision"):
        plan_publication_request(root, invalid_revision)

    direct = {
        "contract": PUBLICATION_CONTRACT,
        "base_revision": tree.catalog["revision"],
        "operations": [
            {
                "op": "rate-section-generation",
                "subject_id": topic["id"],
                "section_id": "summary",
                "generation_id": generation_id,
                "feedback_id": "feedback-semantic-direct-write",
                "rating": 1,
                "contributor": "Anonymous",
                "feedback_mode": "quick",
            }
        ],
    }
    with pytest.raises(LearnValidationError, match="opaque reviewed-feedback event nonce"):
        publication_plan(root, direct)

def test_feedback_accepts_stale_page_revision_when_generation_identity_still_exists(tmp_path):
    """Independent sidecars survive unrelated/recent repository merges."""
    from _sphinx_ext._sphinx_ai_learn._publication_request_cli import plan_publication_request

    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    topic = next(
        row
        for row in tree.catalog["subjects"]
        if row["kind"] == "topic"
        and any(section["id"] == "summary" and section["body"] for section in row["sections"])
    )
    section = next(row for row in topic["sections"] if row["id"] == "summary")
    generation_id = section_generation_id(topic, section)
    stale_revision = tree.catalog["revision"]

    first = {
        "contract": "learn.publication-request.v1",
        "action": "feedback",
        "base_revision": stale_revision,
        "subject_id": topic["id"],
        "section_id": "summary",
        "generation_id": generation_id,
        "feedback_id": _feedback_id(301),
        "rating": 1,
        "feedback_mode": "quick",
    }
    plan = plan_publication_request(root, first)
    _apply_plan(root, plan)
    # Feedback-only metadata must not invalidate authored drafts or evidence
    # reviews.  The full tree digest changes, but the content revision does not.
    after_feedback = load_content_tree(root)
    assert after_feedback.catalog["revision"] == stale_revision

    # Make an unrelated authored content change so the page revision is truly
    # stale. Feedback still succeeds because it targets immutable generation
    # identity rather than mutable page revision.
    other = next(row for row in after_feedback.catalog["subjects"] if row["kind"] == "topic" and row["id"] != topic["id"])
    authored = publication_plan(
        root,
        {
            "contract": PUBLICATION_CONTRACT,
            "base_revision": after_feedback.catalog["revision"],
            "operations": [
                {
                    "op": "upsert-section",
                    "subject_id": other["id"],
                    "section_id": "summary",
                    "title": "Summary",
                    "body": "Unrelated accepted content change.",
                    "contributor": "Audit",
                    "created_at": "2026-09-27T12:00:00Z",
                }
            ],
        },
    )
    _apply_plan(root, authored)
    assert load_content_tree(root).catalog["revision"] != stale_revision

    second = dict(first)
    second.update(
        feedback_id=_feedback_id(302),
        rating=-1,
    )
    plan2 = plan_publication_request(root, second)
    assert len(plan2["files"]) == 1
    assert "/feedback/" in next(iter(plan2["files"]))
    _apply_plan(root, plan2)
    rows = load_content_tree(root).generation_feedback[(topic["id"], "summary", generation_id)]
    assert [row["rating"] for row in rows][-2:] == [1, -1]

    missing = dict(second)
    missing["feedback_id"] = _feedback_id(303)
    missing["generation_id"] = "generation-does-not-exist"
    with pytest.raises(LearnValidationError, match="generation does not exist"):
        plan_publication_request(root, missing)
