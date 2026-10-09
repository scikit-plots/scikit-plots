# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
import io
import sys
import types
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "generate_apis_reference" / "__init__.py"
SPEC = importlib.util.spec_from_file_location("_generate_apis_reference_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
api_ref = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = api_ref
SPEC.loader.exec_module(api_ref)


def _reference(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "docs" / "source" / "apis_reference.py"
    path.parent.mkdir(parents=True)
    path.write_text(body, encoding="utf-8")
    return path


def _fake_module(monkeypatch, name: str, exports: dict[str, object], all_names=None):
    module = types.ModuleType(name)
    for symbol, value in exports.items():
        if callable(value) and hasattr(value, "__module__"):
            value.__module__ = name
        setattr(module, symbol, value)
    if all_names is not None:
        module.__all__ = list(all_names)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_load_reference_does_not_execute_document(tmp_path):
    path = _reference(
        tmp_path,
        "raise RuntimeError('must never execute')\n"
        "APIS_REFERENCE = {'fake.mod': {'sections': ["
        "{'title': 'One', 'autosummary': ['ok']} ]}}\n",
    )
    model = api_ref.load_reference(path)
    assert list(model.modules) == ["fake.mod"]
    assert model.modules["fake.mod"].documented == ["ok"]


def test_inspection_detects_stale_and_assigns_new_symbol(tmp_path, monkeypatch):
    def keep():
        pass

    def added():
        pass

    _fake_module(
        monkeypatch,
        "fake.mod",
        {"keep": keep, "added": added},
        all_names=["keep", "added"],
    )
    path = _reference(
        tmp_path,
        "APIS_REFERENCE = {\n"
        "    'fake.mod': {\n"
        "        'sections': [\n"
        "            {'title': 'One', 'autosummary': [\n"
        "                'keep',\n"
        "                'gone',\n"
        "            ]},\n"
        "        ],\n"
        "    },\n"
        "}\n",
    )
    model = api_ref.load_reference(path)
    report = api_ref.inspect_reference(
        model,
        api_ref.Policy(distribution="unused", require_distribution=False),
        require_distribution=False,
    )
    result = report.modules["fake.mod"]
    assert result.stale == ["gone"]
    assert result.missing == ["added"]
    assert result.assigned == {"added": 0}

    rendered = api_ref.render_plan(model, report)
    assert "'gone'" not in rendered.source
    assert "'added'" in rendered.source
    assert rendered.changed


def test_comment_guard_blocks_automatic_stale_prune(tmp_path, monkeypatch):
    def keep():
        pass

    _fake_module(monkeypatch, "fake.mod", {"keep": keep}, all_names=["keep"])
    path = _reference(
        tmp_path,
        "APIS_REFERENCE = {\n"
        "    'fake.mod': {'sections': [\n"
        "        {'title': 'One', 'autosummary': [\n"
        "            'keep',\n"
        "            # special compatibility alias\n"
        "            'gone',\n"
        "        ]},\n"
        "    ]},\n"
        "}\n",
    )
    model = api_ref.load_reference(path)
    report = api_ref.inspect_reference(
        model,
        api_ref.Policy(distribution="unused", require_distribution=False),
        require_distribution=False,
    )
    rendered = api_ref.render_plan(model, report)
    assert "'gone'" in rendered.source
    assert rendered.blocked_removals


def test_explicit_sources_disambiguate_missing_symbol(tmp_path, monkeypatch):
    package = types.ModuleType("fake.mod")

    def left():
        pass

    def right():
        pass

    def new_right():
        pass

    left.__module__ = "fake.mod.left_impl"
    right.__module__ = "fake.mod.right_impl"
    new_right.__module__ = "fake.mod.right_impl"
    package.left = left
    package.right = right
    package.new_right = new_right
    package.__all__ = ["left", "right", "new_right"]
    monkeypatch.setitem(sys.modules, "fake.mod", package)

    path = _reference(
        tmp_path,
        "APIS_REFERENCE = {\n"
        " 'fake.mod': {'sections': [\n"
        "   {'title': 'Left', 'sources': ['fake.mod.left_impl'], 'autosummary': ['left']},\n"
        "   {'title': 'Right', 'sources': ['fake.mod.right_impl'], 'autosummary': ['right']},\n"
        " ]},\n"
        "}\n",
    )
    model = api_ref.load_reference(path)
    report = api_ref.inspect_reference(
        model,
        api_ref.Policy(distribution="unused", require_distribution=False),
        require_distribution=False,
    )
    result = report.modules["fake.mod"]
    assert result.assigned == {"new_right": 1}
    assert not result.ambiguous


def test_same_origin_in_multiple_sections_remains_ambiguous(tmp_path, monkeypatch):
    def a():
        pass

    def b():
        pass

    def c():
        pass

    _fake_module(monkeypatch, "fake.mod", {"a": a, "b": b, "c": c}, all_names=["a", "b", "c"])
    path = _reference(
        tmp_path,
        "APIS_REFERENCE = {\n"
        " 'fake.mod': {'sections': [\n"
        "   {'title': 'A', 'autosummary': ['a']},\n"
        "   {'title': 'B', 'autosummary': ['b']},\n"
        " ]},\n"
        "}\n",
    )
    model = api_ref.load_reference(path)
    report = api_ref.inspect_reference(
        model,
        api_ref.Policy(distribution="unused", require_distribution=False),
        require_distribution=False,
    )
    assert report.modules["fake.mod"].ambiguous == {"c": [0, 1]}


def test_cli_generate_is_dry_run_until_apply(tmp_path, monkeypatch):
    def keep():
        pass

    def added():
        pass

    _fake_module(monkeypatch, "fake.mod", {"keep": keep, "added": added}, all_names=["keep", "added"])
    (tmp_path / "pyproject.toml").write_text(
        "[tool.scikitplot.maintenance.api_reference]\n"
        "reference_file = 'docs/source/apis_reference.py'\n"
        "distribution = 'unused'\n"
        "package_root = 'fake'\n"
        "public_source = 'auto'\n"
        "require_distribution = false\n"
        "include_module_objects = false\n",
        encoding="utf-8",
    )
    path = _reference(
        tmp_path,
        "APIS_REFERENCE = {'fake.mod': {'sections': ["
        "{'title': 'One', 'autosummary': ['keep']} ]}}\n",
    )
    before = path.read_text(encoding="utf-8")
    out = io.StringIO()
    err = io.StringIO()
    status = api_ref.main(
        ["--repo-root", str(tmp_path), "generate"],
        stdout=out,
        stderr=err,
    )
    assert status == 0
    assert path.read_text(encoding="utf-8") == before
    assert "+" in out.getvalue() and "added" in out.getvalue()

    out = io.StringIO()
    status = api_ref.main(
        ["--repo-root", str(tmp_path), "generate", "--apply"],
        stdout=out,
        stderr=err,
    )
    assert status == 0
    assert "'added'" in path.read_text(encoding="utf-8")


def test_module_import_has_no_cli_side_effects():
    assert callable(api_ref.main)
    assert callable(api_ref.load_reference)
    assert callable(api_ref.inspect_reference)


def test_stale_class_is_pruned_with_matching_autosummary(tmp_path, monkeypatch):
    class Keep:
        pass

    _fake_module(monkeypatch, "fake.mod", {"Keep": Keep}, all_names=["Keep"])
    path = _reference(
        tmp_path,
        "APIS_REFERENCE = {\n"
        " 'fake.mod': {'sections': [\n"
        "   {'title': 'Classes', 'autosummary': [\n"
        "      'Keep',\n"
        "      'Gone',\n"
        "   ], 'classes': [\n"
        "      'Keep',\n"
        "      'Gone',\n"
        "   ]},\n"
        " ]},\n"
        "}\n",
    )
    model = api_ref.load_reference(path)
    report = api_ref.inspect_reference(
        model,
        api_ref.Policy(distribution="unused", require_distribution=False),
        require_distribution=False,
    )
    assert report.modules["fake.mod"].stale == ["Gone"]
    assert report.modules["fake.mod"].stale_classes == ["Gone"]
    rendered = api_ref.render_plan(model, report)
    assert "'Gone'" not in rendered.source
    assert any("[classes]" in item for item in rendered.removed)


def test_dotted_optional_import_failure_is_unverified_not_stale(tmp_path, monkeypatch):
    package = types.ModuleType("fake.mod")
    package.__path__ = []
    package.__all__ = []
    monkeypatch.setitem(sys.modules, "fake.mod", package)
    path = _reference(
        tmp_path,
        "APIS_REFERENCE = {\n"
        " 'fake.mod': {'sections': [\n"
        "   {'title': 'Optional', 'autosummary': ['optional.Feature']},\n"
        " ]},\n"
        "}\n",
    )
    model = api_ref.load_reference(path)
    report = api_ref.inspect_reference(
        model,
        api_ref.Policy(distribution="unused", require_distribution=False),
        require_distribution=False,
    )
    result = report.modules["fake.mod"]
    assert result.stale == []
    assert result.unverified and result.unverified[0].startswith("optional.Feature:")


def test_cli_refuses_reference_outside_repository(tmp_path):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    outside = tmp_path.parent / "outside_api_reference.py"
    outside.write_text("APIS_REFERENCE = {}\n", encoding="utf-8")
    out = io.StringIO()
    err = io.StringIO()
    status = api_ref.main(
        [
            "--repo-root",
            str(tmp_path),
            "--reference-file",
            str(outside),
            "--allow-uninstalled",
            "plan",
        ],
        stdout=out,
        stderr=err,
    )
    assert status == 2
    assert "inside the repository root" in err.getvalue()


def test_full_rebuild_source_round_trips_canonical_template(tmp_path):
    source = api_ref.build_reference_source()
    blueprint = api_ref.reference_blueprint_copy()
    rebuilt = _reference(tmp_path, source)
    model = api_ref.load_reference(rebuilt)
    assert list(model.modules) == list(blueprint["api_reference"])
    assert sum(len(item.sections) for item in model.modules.values()) == sum(
        len(item["sections"]) for item in blueprint["api_reference"].values()
    )
    assert source == api_ref.default_template_path().read_text(encoding="utf-8")
    assert "apis_reference.py.in" in source


def test_full_rebuild_can_recreate_deleted_reference(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.scikitplot.maintenance.api_reference]\n"
        "reference_file = 'docs/source/apis_reference.py'\n"
        "distribution = 'unused'\n"
        "package_root = 'scikitplot'\n"
        "require_distribution = false\n",
        encoding="utf-8",
    )
    path = tmp_path / "docs" / "source" / "apis_reference.py"
    assert not path.exists()

    out = io.StringIO()
    err = io.StringIO()
    status = api_ref.main(
        ["--repo-root", str(tmp_path), "rebuild", "--apply"],
        stdout=out,
        stderr=err,
    )
    assert status == 0
    assert path.is_file()
    assert path.read_text(encoding="utf-8") == api_ref.build_reference_source()

    out = io.StringIO()
    status = api_ref.main(
        ["--repo-root", str(tmp_path), "rebuild", "--check"],
        stdout=out,
        stderr=err,
    )
    assert status == 0
    assert "full rebuild is clean" in out.getvalue()


def test_full_rebuild_check_detects_generated_file_drift(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.scikitplot.maintenance.api_reference]\n"
        "reference_file = 'docs/source/apis_reference.py'\n",
        encoding="utf-8",
    )
    path = _reference(tmp_path, api_ref.build_reference_source() + "\n# local drift\n")
    out = io.StringIO()
    err = io.StringIO()
    status = api_ref.main(
        ["--repo-root", str(tmp_path), "rebuild", "--check"],
        stdout=out,
        stderr=err,
    )
    assert status == 1
    assert path.is_file()
    assert "requires a full rebuild" in out.getvalue()


def test_blueprint_copy_supports_customization_without_template_mutation():
    original = api_ref.reference_blueprint_copy()["api_reference"]["scikitplot"]["short_summary"]
    blueprint = api_ref.reference_blueprint_copy()
    blueprint["api_reference"]["scikitplot"]["short_summary"] = "Custom summary."
    source = api_ref.build_reference_source(blueprint)
    assert '"short_summary": "Custom summary."' in source
    assert (
        api_ref.reference_blueprint_copy()["api_reference"]["scikitplot"]["short_summary"]
        == original
    )


def test_blueprint_validation_rejects_inheritance_class_outside_autosummary():
    blueprint = api_ref.reference_blueprint_copy()
    section = blueprint["api_reference"]["scikitplot"]["sections"][0]
    section["classes"] = ["NotDocumented"]
    with pytest.raises(api_ref.APIReferenceError, match="outside autosummary"):
        api_ref.build_reference_source(blueprint)
