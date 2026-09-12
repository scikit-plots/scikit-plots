from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from architecture import (
    architecture_snapshot,
    effective_runtime_graph,
    validate_family_architecture,
    validate_manifest_architecture,
)


def _runtime(tmp_path: Path, packages: tuple[str, ...]) -> Path:
    root = tmp_path / "scikitplot/_externals/_sphinx_ext"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "_extension_setup.py").write_text("")
    for name in packages:
        pkg = root / name
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
    return root


def _manifest(
    source: str,
    *,
    edges=None,
    capabilities=None,
):
    return {
        "schema_version": 2,
        "subsystem": source,
        "runtime_dir": source,
        "state": "STATE.json",
        "tracker": "TRACKER.json",
        "handoff": "FRESH_CHAT_HANDOFF.md",
        "dependency_edges": [] if edges is None else edges,
        "capability_ownership": [] if capabilities is None else capabilities,
    }


def _edge(target: str, capability: str, *, kind="runtime_required", evidence="python_import"):
    return {
        "target": target,
        "kind": kind,
        "evidence": evidence,
        "capabilities": [capability],
    }


def _cap(capability_id: str, owner: str):
    return {
        "id": capability_id,
        "owner": owner,
        "description": f"Capability owned by {owner}",
    }


def test_source_reference_becomes_effective_runtime_edge(tmp_path):
    runtime = _runtime(tmp_path, ("_sphinx_a", "_sphinx_b"))
    (runtime / "_sphinx_a/__init__.py").write_text('TARGET = ".._sphinx_b"\n')
    manifest = _manifest(
        "_sphinx_a",
        edges=[_edge("_sphinx_b", "demo.dynamic", evidence="source_reference")],
        capabilities=[_cap("demo.dynamic", "_sphinx_b")],
    )
    graph = effective_runtime_graph(runtime, [manifest])
    assert "_sphinx_b" in graph["_sphinx_a"]
    assert validate_manifest_architecture(manifest, runtime) == []


def test_comment_cannot_fake_source_reference_evidence(tmp_path):
    runtime = _runtime(tmp_path, ("_sphinx_a", "_sphinx_b"))
    (runtime / "_sphinx_a/__init__.py").write_text("# dynamic target _sphinx_b\n")
    manifest = _manifest(
        "_sphinx_a",
        edges=[_edge("_sphinx_b", "demo.dynamic", evidence="source_reference")],
        capabilities=[_cap("demo.dynamic", "_sphinx_b")],
    )
    assert any(
        "lacks declared source reference evidence" in error
        for error in validate_manifest_architecture(manifest, runtime)
    )


def test_dynamic_edge_participates_in_cycle_detection(tmp_path):
    runtime = _runtime(tmp_path, ("_sphinx_a", "_sphinx_b"))
    (runtime / "_sphinx_a/__init__.py").write_text('TARGET = ".._sphinx_b"\n')
    (runtime / "_sphinx_b/__init__.py").write_text("from .._sphinx_a import x\nx = 1\n")
    manifest = _manifest(
        "_sphinx_a",
        edges=[_edge("_sphinx_b", "demo.dynamic", evidence="source_reference")],
        capabilities=[_cap("demo.dynamic", "_sphinx_b")],
    )
    errors = validate_manifest_architecture(manifest, runtime)
    assert any("runtime dependency cycle" in error for error in errors)


def test_non_runtime_classification_rejects_runtime_import(tmp_path):
    runtime = _runtime(tmp_path, ("_sphinx_a", "_sphinx_b"))
    (runtime / "_sphinx_a/__init__.py").write_text("from .._sphinx_b import x\n")
    manifest = _manifest(
        "_sphinx_a",
        edges=[
            _edge(
                "_sphinx_b",
                "demo.related",
                kind="family_related",
                evidence="architectural_relation",
            )
        ],
        capabilities=[_cap("demo.related", "_sphinx_b")],
    )
    assert any(
        "non-runtime dependency classification is imported by runtime" in error
        for error in validate_manifest_architecture(manifest, runtime)
    )


def test_edge_capability_owner_must_match_target(tmp_path):
    runtime = _runtime(tmp_path, ("_sphinx_a", "_sphinx_b", "_sphinx_c"))
    (runtime / "_sphinx_a/__init__.py").write_text("from .._sphinx_b import x\n")
    manifest = _manifest(
        "_sphinx_a",
        edges=[_edge("_sphinx_b", "demo.owner")],
        capabilities=[_cap("demo.owner", "_sphinx_c")],
    )
    errors = validate_manifest_architecture(manifest, runtime)
    assert any("not dependency target _sphinx_b" in error for error in errors)


def test_unknown_capability_fails_family_resolution(tmp_path):
    runtime = _runtime(tmp_path, ("_sphinx_a", "_sphinx_b"))
    (runtime / "_sphinx_a/__init__.py").write_text("from .._sphinx_b import x\n")
    manifest = _manifest("_sphinx_a", edges=[_edge("_sphinx_b", "demo.unknown")])
    path = tmp_path / "a.json"
    import json
    path.write_text(json.dumps(manifest))
    errors = validate_family_architecture([path], runtime)
    assert any("references unknown capability" in error for error in errors)


def test_duplicate_family_capability_ownership_fails(tmp_path):
    runtime = _runtime(tmp_path, ("_sphinx_a", "_sphinx_b"))
    import json
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    p1.write_text(json.dumps(_manifest("_sphinx_a", capabilities=[_cap("demo.shared", "_sphinx_a")])))
    p2.write_text(json.dumps(_manifest("_sphinx_b", capabilities=[_cap("demo.shared", "_sphinx_b")])))
    errors = validate_family_architecture([p1, p2], runtime)
    assert any("multiple ownership declarations" in error for error in errors)



def test_relative_alias_only_import_is_observed(tmp_path):
    runtime = _runtime(tmp_path, ("_sphinx_a", "_sphinx_b"))
    (runtime / "_sphinx_a/__init__.py").write_text("from .. import _sphinx_b\n")
    manifest = _manifest(
        "_sphinx_a",
        edges=[_edge("_sphinx_b", "demo.shared")],
        capabilities=[_cap("demo.shared", "_sphinx_b")],
    )
    assert validate_manifest_architecture(manifest, runtime) == []


def test_absolute_alias_only_import_is_observed(tmp_path):
    runtime = _runtime(tmp_path, ("_sphinx_a", "_sphinx_b"))
    (runtime / "_sphinx_a/__init__.py").write_text(
        "from scikitplot._externals._sphinx_ext import _sphinx_b\n"
    )
    manifest = _manifest(
        "_sphinx_a",
        edges=[_edge("_sphinx_b", "demo.shared")],
        capabilities=[_cap("demo.shared", "_sphinx_b")],
    )
    assert validate_manifest_architecture(manifest, runtime) == []


def test_v2_rejects_legacy_dependency_arrays(tmp_path):
    runtime = _runtime(tmp_path, ("_sphinx_a",))
    manifest = _manifest("_sphinx_a")
    manifest["runtime_requires"] = []
    manifest["family_related"] = []
    errors = validate_manifest_architecture(manifest, runtime)
    assert any("must not retain legacy dependency arrays" in error for error in errors)


def test_edge_kind_rejects_incompatible_evidence(tmp_path):
    runtime = _runtime(tmp_path, ("_sphinx_a", "_sphinx_b"))
    manifest = _manifest(
        "_sphinx_a",
        edges=[
            _edge(
                "_sphinx_b",
                "demo.related",
                kind="family_related",
                evidence="python_import",
            )
        ],
        capabilities=[_cap("demo.related", "_sphinx_b")],
    )
    errors = validate_manifest_architecture(manifest, runtime)
    assert any("incompatible with kind family_related" in error for error in errors)

def test_architecture_snapshot_is_deterministic_and_explainable(tmp_path):
    runtime = _runtime(tmp_path, ("_sphinx_a", "_sphinx_b"))
    (runtime / "_sphinx_a/__init__.py").write_text("from .._sphinx_b import x\n")
    import json
    manifest = _manifest(
        "_sphinx_a",
        edges=[_edge("_sphinx_b", "demo.shared")],
        capabilities=[_cap("demo.shared", "_sphinx_b")],
    )
    path = tmp_path / "a.json"
    path.write_text(json.dumps(manifest))
    snapshot = architecture_snapshot([path], runtime)
    assert snapshot["schema_version"] == 1
    assert snapshot["cycle"] is None
    assert snapshot["effective_runtime_graph"]["_sphinx_a"] == ["_sphinx_b"]
    assert snapshot["declared_edges"][0]["capabilities"] == ["demo.shared"]
    assert snapshot["capabilities"][0]["owner"] == "_sphinx_b"
