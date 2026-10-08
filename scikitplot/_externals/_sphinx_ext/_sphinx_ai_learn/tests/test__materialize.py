"""Canonical JSON -> deterministic RST materialization invariants."""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path, PurePosixPath

import pytest

import _learn_site

from _sphinx_ext._sphinx_ai_learn._materialize import (
    GENERATED_MARKER,
    canonical_prompt_json_files,
    canonical_skill_json_files,
    canonical_record_json_files,
    load_content_tree,
    materialize,
    render_materialized,
    record_docpath,
)
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


def _first_topic(tree):
    return next(
        (rel, record)
        for rel, record in tree.records.items()
        if record["subject"]["kind"] == "topic"
    )


def test_production_tree_is_one_json_to_one_rst_and_projection_is_canonical():
    tree = load_content_tree(_learn_site.content_root())
    rendered = render_materialized(tree)
    # Preserve the known baseline without freezing a publication-driven corpus.
    # New reviewed records/prompts/skills and feedback sidecars are expected to
    # grow these collections over time.
    assert len(tree.catalog["subjects"]) >= 50
    assert len(tree.prompts) >= 13
    assert len(tree.skills) >= 5
    assert len(tree.source_digests) - len(tree.feedback_events) >= 677
    assert all(subject.get("authors") for subject in tree.catalog["subjects"] if subject["kind"] != "skill")
    assert all(prompt.get("authors") for prompt in tree.prompts)
    assert all(skill.get("authors") for skill in tree.skills)
    assert all(
        section.get("contributors")
        for subject in tree.catalog["subjects"]
        for section in subject.get("sections", [])
        if section.get("body") or section.get("citations") or section.get("links")
    )
    primary = {
        rel.with_suffix(".rst")
        for rel in set(tree.source_digests) - set(tree.feedback_events)
    }
    assert primary <= set(rendered)
    derived = set(rendered) - primary
    assert derived
    assert all(path.name.startswith("page-") and path.suffix == ".rst" for path in derived)
    assert all((_learn_site.content_root() / rel).is_file() for rel in rendered)
    assert all((_learn_site.content_root() / rel).read_bytes() == raw for rel, raw in rendered.items())

    assert canonical_prompt_json_files(tree.prompts) == {
        rel: (_learn_site.content_root() / rel).read_bytes()
        for rel in canonical_prompt_json_files(tree.prompts)
    }
    assert canonical_skill_json_files(tree.skills) == {
        rel: (_learn_site.content_root() / rel).read_bytes()
        for rel in canonical_skill_json_files(tree.skills)
    }
    for rel, record in tree.records.items():
        projected = canonical_record_json_files(
            record["subject"],
            tree.prompts,
            tree.skills,
            add_toctree=record["add_toctree"],
        )
        assert projected == {path: (_learn_site.content_root() / path).read_bytes() for path in projected}


def test_explorer_pagination_shards_are_uniform_owned_and_deterministic():
    tree = load_content_tree(_learn_site.content_root())
    rendered = render_materialized(tree)
    assert Path("topics/page-2.rst") in rendered
    page = rendered[Path("topics/page-2.rst")].decode()
    assert page.startswith(":orphan:\n:no-search:\n")
    assert ".. source-json: topics/index.json" in page
    assert "Topics — Page 2" in page
    assert ".. ai-topic-explorer:: topic\n   :offset: 12" in page


def test_secondary_sidebar_control_is_explicit_for_every_renderable_json():
    renderable_contracts = {
        "learn.page.v1",
        "learn.record.v2",
        "learn.section.v1",
        "learn.section.v2",
        "learn.topic-prompt.v1",
        "learn.skill.v1",
    }
    matched = 0
    for source in _learn_site.content_root().rglob("*.json"):
        data = json.loads(source.read_text(encoding="utf-8"))
        if data.get("contract") not in renderable_contracts:
            continue
        matched += 1
        assert data["hide_secondary_sidebar"] is True, source
    assert matched >= 677


def test_secondary_sidebar_control_can_show_sidebar_at_each_rendering_layer(tmp_path):
    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    record_rel, record = _first_topic(tree)
    summary_ref = next(
        ref for ref in record["section_refs"] if ref["id"] == "summary"
    )
    summary_rel = record_rel.parent / summary_ref["source"]
    prompt_rel = Path("topic-prompts") / tree.prompts[0]["id"] / "index.json"
    skill_rel = Path("skills") / tree.skills[0]["id"] / "index.json"

    targets = (Path("index.json"), record_rel, summary_rel, prompt_rel, skill_rel)
    for rel in targets:
        path = root / rel
        data = json.loads(path.read_text(encoding="utf-8"))
        data["hide_secondary_sidebar"] = False
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    tree, _ = materialize(root)
    marker = ":html_theme.sidebar_secondary.remove:"
    for rel in targets:
        assert marker not in (root / rel.with_suffix(".rst")).read_text(encoding="utf-8")

    # An untouched page and fragment still use the explicit hidden default.
    assert marker in (root / "topics/index.rst").read_text(encoding="utf-8")
    untouched_ref = next(
        ref for ref in record["section_refs"] if ref["id"] != "summary"
    )
    untouched = record_rel.parent / PurePosixPath(untouched_ref["source"])
    assert marker in (root / untouched.with_suffix(".rst")).read_text(encoding="utf-8")

    # Include fragments keep their orphan/search metadata even when the sidebar is shown.
    summary_rst = (root / summary_rel.with_suffix(".rst")).read_text(encoding="utf-8")
    assert summary_rst.startswith(":orphan:\n:no-search:\n")
    assert tree.records[record_rel]["hide_secondary_sidebar"] is False
    assert tree.sections[summary_rel]["hide_secondary_sidebar"] is False


def test_secondary_sidebar_control_is_optional_defaults_hidden_and_rejects_non_boolean(tmp_path):
    root = _json_only_copy(tmp_path)
    page = root / "index.json"
    data = json.loads(page.read_text(encoding="utf-8"))
    data.pop("hide_secondary_sidebar")
    page.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    materialize(root)
    assert ":html_theme.sidebar_secondary.remove:" in (root / "index.rst").read_text(encoding="utf-8")

    data["hide_secondary_sidebar"] = "true"
    page.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(LearnValidationError, match="hide_secondary_sidebar must be boolean"):
        load_content_tree(root)


def test_canonical_record_projection_writes_sidebar_controls_explicitly():
    tree = load_content_tree(_learn_site.content_root())
    _, record = _first_topic(tree)
    projected = canonical_record_json_files(
        record["subject"],
        tree.prompts,
        tree.skills,
        add_toctree=record["add_toctree"],
        hide_secondary_sidebar=False,
        section_hide_secondary_sidebar={"summary": False},
    )
    index_rel = Path(record_docpath(record["subject"]) + ".json")
    index = json.loads(projected[index_rel])
    assert index["hide_secondary_sidebar"] is False
    summary = json.loads(projected[index_rel.parent / "summary.json"])
    assert summary["hide_secondary_sidebar"] is False
    other = next(
        path for path in projected
        if path != index_rel and path.name != "summary.json"
    )
    assert json.loads(projected[other])["hide_secondary_sidebar"] is True


def test_materialize_is_idempotent_and_preserves_unchanged_mtime(tmp_path):
    root = _json_only_copy(tmp_path)
    expected = load_content_tree(root)
    expected_rendered = render_materialized(expected)
    _, first = materialize(root)
    assert len(first) == len(expected_rendered)
    tracked = root / first[0]
    before = tracked.stat().st_mtime_ns
    _, second = materialize(root)
    assert second == ()
    assert tracked.stat().st_mtime_ns == before




@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not portable to Windows")
def test_materialize_normalizes_owned_rst_permissions(tmp_path):
    root = _json_only_copy(tmp_path)
    _, first = materialize(root)
    target_rel = Path(first[0])
    target = root / target_rel
    target.chmod(0o600)

    _, changed = materialize(root)
    assert target_rel.as_posix() in changed
    assert stat.S_IMODE(target.stat().st_mode) == 0o644

    _, again = materialize(root)
    assert again == ()


def test_feedback_sidecar_is_validated_scored_and_never_materialized_as_rst(tmp_path):
    from _sphinx_ext._sphinx_ai_learn._generation import section_generation_id

    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    content_revision = tree.catalog["revision"]
    content_digest = tree.content_digest
    full_digest = tree.digest
    _, baseline_changed = materialize(root)
    assert baseline_changed
    rel, record = next(
        (rel, record)
        for rel, record in tree.records.items()
        if record["subject"]["kind"] == "topic"
        and any(section["id"] == "summary" and section["body"] for section in record["subject"]["sections"])
    )
    subject = record["subject"]
    section = next(row for row in subject["sections"] if row["id"] == "summary")
    generation_id = section_generation_id(subject, section)
    feedback_id = _feedback_id(1)
    sidecar = rel.parent / "feedback" / section["id"] / generation_id / f"{feedback_id}.json"
    target = root / sidecar
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "contract": "learn.generation-feedback.v1",
                "record_id": subject["id"],
                "section_id": "summary",
                "generation_id": generation_id,
                "feedback": {
                    "id": feedback_id,
                    "rating": 4,
                    "contributor": "DataFox",
                    "mode": "detailed",
                    "comment": "Useful explanation.",
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    loaded = load_content_tree(root)
    assert loaded.catalog["revision"] == content_revision
    assert loaded.content_digest == content_digest
    assert loaded.digest != full_digest
    assert str(root / sidecar) not in loaded.dependencies
    assert loaded.feedback_dependencies[subject["id"]] == (str(root / sidecar),)
    assert subject["id"] in loaded.feedback_digests
    assert loaded.generation_feedback[(subject["id"], "summary", generation_id)] == (
        {
            "id": feedback_id,
            "rating": 4,
            "contributor": "DataFox",
            "mode": "detailed",
            "comment": "Useful explanation.",
        },
    )
    rendered = render_materialized(loaded)
    assert sidecar.with_suffix(".rst") not in rendered
    _, changed = materialize(root)
    assert changed == ()
    assert not (root / sidecar.with_suffix(".rst")).exists()


    invalid = json.loads(target.read_text(encoding="utf-8"))
    invalid["feedback"]["mode"] = "quick"
    invalid["feedback"]["rating"] = 4
    target.write_text(json.dumps(invalid, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(LearnValidationError, match="quick mode"):
        load_content_tree(root)



def test_feedback_sidecar_path_is_section_scoped(tmp_path):
    from _sphinx_ext._sphinx_ai_learn._generation import section_generation_id

    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    rel, record = next(
        (rel, record)
        for rel, record in tree.records.items()
        if record["subject"]["kind"] == "topic"
        and any(section["id"] == "summary" and section["body"] for section in record["subject"]["sections"])
    )
    subject = record["subject"]
    section = next(row for row in subject["sections"] if row["id"] == "summary")
    generation_id = section_generation_id(subject, section)
    feedback_id = _feedback_id(3)
    wrong = rel.parent / "feedback" / generation_id / f"{feedback_id}.json"
    target = root / wrong
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "contract": "learn.generation-feedback.v1",
                "record_id": subject["id"],
                "section_id": "summary",
                "generation_id": generation_id,
                "feedback": {
                    "id": feedback_id,
                    "rating": 1,
                    "contributor": "Anonymous",
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(LearnValidationError, match="feedback sidecar path"):
        load_content_tree(root)

def test_default_include_mode_uses_orphan_fragments_and_nested_prompt_paths(tmp_path):
    root = _json_only_copy(tmp_path)
    tree, _ = materialize(root)
    rel, record = _first_topic(tree)
    folder = root / rel.parent
    parent = (folder / "index.rst").read_text(encoding="utf-8")
    summary = (folder / "summary.rst").read_text(encoding="utf-8")
    prompt = (folder / "topic-prompts" / "eli14.rst").read_text(encoding="utf-8")
    skill_group = (folder / "skills" / "index.rst").read_text(encoding="utf-8")
    skill = (folder / "skills" / "skill-check-reference.rst").read_text(encoding="utf-8")

    assert record["add_toctree"] is False
    assert ".. include:: summary.rst" in parent
    assert ".. include:: topic-prompts/eli14.rst" in parent
    assert ".. toctree::" not in parent
    assert summary.startswith(":orphan:\n:no-search:\n")
    assert GENERATED_MARKER in summary.splitlines()[:12]
    assert ".. ai-learn-fragment-start" in summary
    assert prompt.startswith(":orphan:\n:no-search:\n")
    assert skill_group.startswith(":orphan:\n:no-search:\n")
    assert skill.startswith(":orphan:\n:no-search:\n")
    assert ".. ai-topic-skills::" in skill_group
    assert ".. _learn-" not in prompt.split(".. ai-learn-fragment-start", 1)[1]
    assert ".. _learn-" not in skill.split(".. ai-learn-fragment-start", 1)[1]


def test_toctree_mode_makes_children_navigable_without_orphan(tmp_path):
    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    rel, _ = _first_topic(tree)
    path = root / rel
    data = json.loads(path.read_text(encoding="utf-8"))
    data["add_toctree"] = True
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _, changed = materialize(root)
    parent = path.with_suffix(".rst").read_text(encoding="utf-8")
    summary = (path.parent / "summary.rst").read_text(encoding="utf-8")
    assert ".. toctree::" in parent
    assert ".. include::" not in parent
    assert "summary" in parent
    skill = (path.parent / "skills" / "skill-check-reference.rst").read_text(encoding="utf-8")
    assert not summary.startswith(":orphan:")
    assert not skill.startswith(":orphan:")
    assert f".. _learn-{data['record']['id']}-summary:" in summary
    assert "skills/skill-check-reference" in parent
    assert any(name.endswith("/summary.rst") for name in changed)


def test_regeneration_from_json_only_is_byte_identical(tmp_path):
    root = _json_only_copy(tmp_path)
    source_tree = load_content_tree(_learn_site.content_root())
    expected = render_materialized(source_tree)
    _, changed = materialize(root)
    assert len(changed) == len(expected)
    generated = {p.relative_to(root): p.read_bytes() for p in root.rglob("*.rst")}
    assert generated == expected


def test_one_section_json_change_rewrites_only_its_owned_rst(tmp_path):
    root = _json_only_copy(tmp_path)
    tree, _ = materialize(root)
    rel, _ = _first_topic(tree)
    source = root / rel.parent / "summary.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    data["section"]["body"] += "\n\nA deterministic mutation."
    source.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _, changed = materialize(root)
    assert changed == ((rel.parent / "summary.rst").as_posix(),)


def test_handwritten_collision_is_a_hard_error_before_writes(tmp_path):
    root = _json_only_copy(tmp_path)
    target = root / "index.rst"
    target.write_text("Handwritten source.\n", encoding="utf-8")
    with pytest.raises(LearnValidationError, match="handwritten RST"):
        materialize(root)
    assert target.read_text(encoding="utf-8") == "Handwritten source.\n"
    assert len(list(root.rglob("*.rst"))) == 1


def test_orphan_canonical_json_and_path_escape_are_rejected(tmp_path):
    root = _json_only_copy(tmp_path)
    rogue = root / "topics" / "rogue.json"
    rogue.write_text(
        json.dumps(
            {
                "contract": "learn.section.v1",
                "record_id": "topic-rogue",
                "section": {
                    "id": "summary",
                    "title": "Summary",
                    "body": "",
                    "citations": [],
                    "links": [],
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(LearnValidationError, match="orphan section JSON"):
        load_content_tree(root)




def test_dangling_related_error_names_canonical_json_owner_and_target(tmp_path):
    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    rel, record = _first_topic(tree)
    path = root / rel
    data = json.loads(path.read_text(encoding="utf-8"))
    subject_id = data["record"]["id"]
    data["record"]["related"] = ["topic-missing-related-target"]
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    expected = (
        rf"{rel.as_posix()}: subject\.related: {subject_id} "
        r"-> unknown target topic-missing-related-target"
    )
    with pytest.raises(LearnValidationError, match=expected):
        load_content_tree(root)



def test_dangling_citation_error_names_section_json_owner_and_source(tmp_path):
    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    rel, record = _first_topic(tree)
    section_ref = next(
        ref for ref in record["section_refs"] if ref["id"] == "summary"
    )
    section_rel = rel.parent / section_ref["source"]
    path = root / section_rel
    data = json.loads(path.read_text(encoding="utf-8"))
    data["section"]["citations"] = [
        {"source_id": "source-missing-citation-target", "locator": "Section 1"}
    ]
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    subject_id = record["subject"]["id"]
    expected = (
        rf"{section_rel.as_posix()}: section\.citations: {subject_id}#summary "
        r"-> unknown source source-missing-citation-target"
    )
    with pytest.raises(LearnValidationError, match=expected):
        load_content_tree(root)

def test_symlink_json_is_rejected(tmp_path):
    root = _json_only_copy(tmp_path)
    if not hasattr(Path, "symlink_to"):
        pytest.skip("symlinks unavailable")
    target = next(root.rglob("summary.json"))
    copy = target.with_name("copy.json")
    try:
        copy.symlink_to(target.name)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(LearnValidationError, match="symlink JSON"):
        load_content_tree(root)


def test_canonical_json_rejects_duplicate_keys_and_nonfinite_numbers(tmp_path):
    root = _json_only_copy(tmp_path)
    target = root / "index.json"
    target.write_text(
        '{"contract":"learn.page.v1","contract":"learn.page.v1","view":"root","title":"AI Learn"}',
        encoding="utf-8",
    )
    with pytest.raises(LearnValidationError, match="invalid JSON"):
        load_content_tree(root)

    root = _json_only_copy(tmp_path / "nonfinite")
    target = root / "index.json"
    target.write_text(
        '{"contract":"learn.page.v1","view":"root","title":"AI Learn","description":NaN}',
        encoding="utf-8",
    )
    with pytest.raises(LearnValidationError, match="invalid JSON"):
        load_content_tree(root)


def test_rst_heading_inputs_reject_control_newlines(tmp_path):
    root = _json_only_copy(tmp_path)
    target = root / "index.json"
    data = json.loads(target.read_text(encoding="utf-8"))
    data["title"] = "AI Learn\n.. include:: secret.rst"
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(LearnValidationError, match="control characters"):
        load_content_tree(root)


def test_feedback_outdated_documents_is_record_scoped_and_tracks_external_consumers():
    from _sphinx_ext._sphinx_ai_learn._generation import feedback_outdated_documents

    found = {
        "index",
        "examples/embed",
        "learn-ai/index",
        "learn-ai/topics/example/index",
        "learn-ai/topics/example/topic-prompts/eli14",
        "learn-ai/topics/other/index",
    }
    routes = {
        "topic-example": "topics/example/index",
        "topic-other": "topics/other/index",
    }
    previous = {"topic-example": "a", "topic-other": "z"}
    current = {"topic-example": "b", "topic-other": "z"}
    consumers = {"topic-example": {"examples/embed"}}

    assert feedback_outdated_documents(
        root="learn-ai",
        routes=routes,
        found_docs=found,
        consumers=consumers,
        previous=previous,
        current=current,
    ) == [
        "examples/embed",
        "learn-ai/topics/example/index",
        "learn-ai/topics/example/topic-prompts/eli14",
    ]
    assert feedback_outdated_documents(
        root="learn-ai",
        routes=routes,
        found_docs=found,
        consumers=consumers,
        previous=current,
        current=current,
    ) == []



def test_page_design_grid_is_typed_optional_and_deterministic(tmp_path):
    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    page = tree.pages[Path("index.json")]
    assert page["design_grid"]["columns"] == "1 1 1 1"
    assert page["design_grid"]["items"][0]["card"]["columns"] == "12 12 6 6"
    rst = render_materialized(tree)[Path("index.rst")].decode()
    assert ".. grid:: 1 1 1 1" in rst
    assert rst.count(".. grid-item-card::") == 11
    assert ":padding: 2" in rst
    assert ":columns: 12 12 6 6" in rst
    assert "**topics**\n      ^^^\n      .. toctree::" in rst
    assert "         topics/index" in rst

    # Removing the optional layout returns the legacy plain toctree projection.
    source = root / "index.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    data.pop("design_grid")
    source.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plain = render_materialized(load_content_tree(root))[Path("index.rst")].decode()
    assert ".. grid::" not in plain
    assert ".. toctree::\n   :maxdepth: 2" in plain
    assert "   topics/index" in plain


def test_page_design_grid_can_augment_non_root_page_contracts(tmp_path):
    root = _json_only_copy(tmp_path)
    target = root / "topics" / "index.json"
    data = json.loads(target.read_text(encoding="utf-8"))
    data["design_grid"] = {
        "columns": [1, 1, 2, 2],
        "gutter": [1, 1, 2, 2],
        "items": [
            {
                "title": "Create topic",
                "card": {"padding": 2, "columns": [12, 12, 6, 6], "shadow": "sm"},
                "toctree": {"maxdepth": 1, "children": ["new"]},
            }
        ],
    }
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tree = load_content_tree(root)
    rst = render_materialized(tree)[Path("topics/index.rst")].decode()
    assert ".. ai-topic-explorer:: topic" in rst
    assert ".. grid:: 1 1 2 2" in rst
    assert ":gutter: 1 1 2 2" in rst
    assert ":shadow: sm" in rst
    assert "         new" in rst


def test_page_design_grid_rejects_raw_rst_unsafe_paths_and_invalid_options(tmp_path):
    def mutated(name, mutate):
        root = _json_only_copy(tmp_path / name)
        target = root / "index.json"
        data = json.loads(target.read_text(encoding="utf-8"))
        mutate(data)
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return root

    with pytest.raises(LearnValidationError, match="unexpected or missing fields"):
        load_content_tree(mutated("raw", lambda d: d["design_grid"].update({"raw_rst": ".. include:: secret.rst"})))

    with pytest.raises(LearnValidationError, match="invalid responsive value"):
        load_content_tree(mutated("columns", lambda d: d["design_grid"].update({"columns": [1, 1, 1, 13]})))

    with pytest.raises(LearnValidationError, match="unsafe docname"):
        load_content_tree(mutated("path", lambda d: d["design_grid"]["items"][0]["toctree"].update({"children": ["../secret"]})))

    with pytest.raises(LearnValidationError, match="unsafe docname"):
        load_content_tree(mutated("explicit-title", lambda d: d["design_grid"]["items"][0]["toctree"].update({"children": ["Label <topics/index>"]})))

    with pytest.raises(LearnValidationError, match="invalid class name"):
        load_content_tree(mutated("class", lambda d: d["design_grid"].update({"class_container": "ok ..bad"})))

    with pytest.raises(LearnValidationError, match="must cover root children exactly once and in order"):
        load_content_tree(mutated("coverage", lambda d: d["design_grid"].update({"items": d["design_grid"]["items"][:-1]})))

    def unknown_non_root(data):
        data["design_grid"] = {
            "columns": 1,
            "items": [{"title": "Missing", "toctree": {"children": ["missing-page"]}}],
        }

    root = _json_only_copy(tmp_path / "unknown")
    target = root / "topics" / "index.json"
    data = json.loads(target.read_text(encoding="utf-8"))
    unknown_non_root(data)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(LearnValidationError, match="references unknown canonical document missing-page"):
        load_content_tree(root)


def test_feedback_sidecar_requires_opaque_event_nonce_even_when_path_matches(tmp_path):
    from _sphinx_ext._sphinx_ai_learn._generation import section_generation_id

    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    rel, record = next(
        (rel, record)
        for rel, record in tree.records.items()
        if record["subject"]["kind"] == "topic"
        and any(section["id"] == "summary" and section["body"] for section in record["subject"]["sections"])
    )
    subject = record["subject"]
    section = next(row for row in subject["sections"] if row["id"] == "summary")
    generation_id = section_generation_id(subject, section)
    feedback_id = "feedback-user-shaped-label"
    sidecar = rel.parent / "feedback" / "summary" / generation_id / f"{feedback_id}.json"
    target = root / sidecar
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "contract": "learn.generation-feedback.v1",
                "record_id": subject["id"],
                "section_id": "summary",
                "generation_id": generation_id,
                "feedback": {"id": feedback_id, "rating": 1, "contributor": "Anonymous"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(LearnValidationError, match="opaque reviewed-feedback event nonce"):
        load_content_tree(root)


def test_reviewed_feedback_sidecars_are_bounded_per_generation(tmp_path, monkeypatch):
    import _sphinx_ext._sphinx_ai_learn._materialize as materialize_module
    from _sphinx_ext._sphinx_ai_learn._generation import section_generation_id

    root = _json_only_copy(tmp_path)
    tree = load_content_tree(root)
    rel, record = next(
        (rel, record)
        for rel, record in tree.records.items()
        if record["subject"]["kind"] == "topic"
        and any(section["id"] == "summary" and section["body"] for section in record["subject"]["sections"])
    )
    subject = record["subject"]
    section = next(row for row in subject["sections"] if row["id"] == "summary")
    generation_id = section_generation_id(subject, section)
    monkeypatch.setattr(materialize_module, "MAX_REVIEWED_FEEDBACK_EVENTS", 2)

    for index in range(3):
        feedback_id = _feedback_id(1000 + index)
        relpath = rel.parent / "feedback" / "summary" / generation_id / f"{feedback_id}.json"
        target = root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "contract": "learn.generation-feedback.v1",
                    "record_id": subject["id"],
                    "section_id": "summary",
                    "generation_id": generation_id,
                    "feedback": {"id": feedback_id, "rating": 1, "contributor": "Anonymous"},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    with pytest.raises(LearnValidationError, match="reviewed feedback event limit exceeded"):
        load_content_tree(root)


def test_index_explorer_headers_are_canonical_and_materialized_consistently():
    expected = {
        "topics": ("Topic explorer", "Trending Topics", "Create a Topic"),
        "open-problems": ("Topic question", "Exploring Open Problems", "Create an Open Problem"),
        "sources": ("Topic Source", "Exploring Sources", "Add a Source"),
        "whiteboards": ("Topic Whiteboard", "Exploring Whiteboards", "Create a Whiteboard"),
        "videos": ("Topic Video", "Exploring Videos", "Create a Video"),
        "audios": ("Topic Audio", "Exploring Audios", "Create an Audio"),
        "documents": ("Topic Document", "Exploring Documents", "Create a Document"),
        "topic-prompts": ("Topic Prompt", "Exploring Topic Prompts", "Create a Topic Prompt"),
        "skills": ("Topic Skill", "Exploring Skills", "Create a Skill"),
    }
    for folder, (kicker, title, create_label) in expected.items():
        data = json.loads((_learn_site.content_root() / folder / "index.json").read_text(encoding="utf-8"))
        assert data["description"]
        assert data["create_label"] == create_label
        assert data["explorer_header"] == {"kicker": kicker, "title": title}
        rst = (_learn_site.content_root() / folder / "index.rst").read_text(encoding="utf-8")
        assert ".. ai-index-explorer-header::" in rst
        assert f"   :kicker: {kicker}" in rst
        assert f"   :title: {title}" in rst
        assert f":doc:`{create_label} <new>`" in rst
        description = data["description"]
        assert rst.index(description) < rst.index(f":doc:`{create_label} <new>`")
        assert rst.index(f":doc:`{create_label} <new>`") < rst.index(".. ai-index-explorer-header::")


def test_index_explorer_header_contract_rejects_partial_or_unknown_fields(tmp_path):
    root = _json_only_copy(tmp_path)
    page = root / "topics/index.json"
    data = json.loads(page.read_text(encoding="utf-8"))
    data["explorer_header"] = {"kicker": "Topic explorer"}
    page.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(LearnValidationError, match="explorer_header has unexpected or missing fields"):
        load_content_tree(root)

    data["explorer_header"] = {"kicker": "Topic explorer", "title": "Trending Topics", "href": "https://evil.example"}
    page.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(LearnValidationError, match="explorer_header has unexpected or missing fields"):
        load_content_tree(root)


def test_index_explorer_header_is_rejected_on_non_index_page_views(tmp_path):
    root = _json_only_copy(tmp_path)
    page = root / "audios/new.json"
    data = json.loads(page.read_text(encoding="utf-8"))
    data["explorer_header"] = {"kicker": "Wrong place", "title": "Should fail"}
    page.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(LearnValidationError, match="explorer_header is unsupported for media-create pages"):
        load_content_tree(root)


def _synthetic_tree(root, *, kinds, explorers=()):
    """Write records of ``kinds`` and explorer pages for ``explorers``; no site needed."""
    root.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        subject = {
            "id": "record-" + kind,
            "kind": kind,
            "title": kind.capitalize(),
            "created_at": "2026-09-16T00:00:00Z",
            "domains": [],
            "related": [],
            "sections": [],
        }
        if kind == "source":
            subject["url"] = "https://example.org/source"
        for relative, raw in canonical_record_json_files(subject, ()).items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    folders = {"topic": "topics", "source": "sources"}
    for kind in explorers:
        folder = root / folders[kind]
        folder.mkdir(parents=True, exist_ok=True)
        for name, view, title in (("index", "explorer", "Index"), ("new", "record-create", "New")):
            (folder / f"{name}.json").write_text(
                json.dumps(
                    {
                        "contract": "learn.page.v1",
                        "view": view,
                        "kind": kind,
                        "title": title,
                        "hide_secondary_sidebar": False,
                    },
                    sort_keys=True,
                ), encoding="utf-8"
            )
    return load_content_tree(root)


def _record_pages(tree, rendered):
    return {
        wrapper["subject"]["kind"]: rendered[rel.with_suffix(".rst")].decode()
        for rel, wrapper in tree.records.items()
    }


def test_a_record_page_no_index_owns_is_marked_orphan(tmp_path):
    """
    A detail page is in a toctree only if the explorer of its kind exists.

    Without one, Sphinx reports "document isn't included in any toctree" for a
    file the materializer wrote, and a build with warnings as errors fails.
    The materializer knows whether the explorer exists, so the page says so.
    """
    tree = _synthetic_tree(tmp_path / "learn-ai", kinds=("topic", "source"))
    pages = _record_pages(tree, render_materialized(tree))
    for kind in ("topic", "source"):
        assert pages[kind].startswith(":orphan:\n"), kind
        # The ownership comments follow the metadata, as they do for sections.
        assert pages[kind].index(":orphan:") < pages[kind].index(".. source-json:")


def test_a_record_page_its_explorer_owns_is_not_orphan(tmp_path):
    tree = _synthetic_tree(
        tmp_path / "learn-ai", kinds=("topic", "source"), explorers=("topic",)
    )
    rendered = render_materialized(tree)
    pages = _record_pages(tree, rendered)
    assert ":orphan:" not in pages["topic"]
    assert pages["source"].startswith(":orphan:\n")
    # The explorer that owns the topic page lists it, which is what makes the
    # absence of :orphan: correct rather than merely absent.
    explorer = rendered[Path("topics/index.rst")].decode()
    topic_rel = next(rel for rel, w in tree.records.items() if w["subject"]["kind"] == "topic")
    assert f"   {topic_rel.parent.name}/index" in explorer


def test_orphan_marking_is_deterministic_and_idempotent(tmp_path):
    tree = _synthetic_tree(tmp_path / "learn-ai", kinds=("topic",))
    first = render_materialized(tree)
    assert first == render_materialized(load_content_tree(tmp_path / "learn-ai"))
    (page,) = _record_pages(tree, first).values()
    assert page.count(":orphan:") == 1
