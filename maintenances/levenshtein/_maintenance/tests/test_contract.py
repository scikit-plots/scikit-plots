from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]
TOOLS = ROOT / "maintenances" / "levenshtein" / "_maintenance" / "tools"


def _load(name: str):
    path = TOOLS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"levmaint_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_structural_contract_is_current() -> None:
    result = _load("check_contract").check()
    assert result["runtime_status"] == "PASS", result["runtime_errors"]
    assert result["maintenance_status"] == "PASS", result["maintenance_errors"]


def test_recorded_open_findings_are_explicit() -> None:
    data = json.loads(
        (ROOT / "maintenances" / "levenshtein" / "REVIEW.json").read_text(encoding="utf-8")
    )
    assert {f["id"] for f in data["findings"] if f["status"] == "open"} == {
        "LV-001",
        "LV-002",
    }


def test_auto_backend_policy_excludes_gpl_backend_statically() -> None:
    source = (ROOT / "scikitplot" / "levenshtein" / "_core.py").read_text(encoding="utf-8")
    assert '("internal", "rapidfuzz", "python")' in source
    assert '("internal", "rapidfuzz", "levenshtein", "python")' not in source


def test_skill_and_docs_entry_points_exist() -> None:
    assert (ROOT / "skills" / "levenshtein" / "SKILL.md").is_file()
    assert (ROOT / "docs" / "source" / "user_guide" / "levenshtein" / "index.rst").is_file()
    assert (ROOT / "galleries" / "examples" / "levenshtein" / "README.txt").is_file()

def test_root_user_guide_discovers_levenshtein() -> None:
    text = (ROOT / "docs" / "source" / "user_guide" / "index.rst").read_text(encoding="utf-8")
    assert "Levenshtein <./levenshtein/index.rst>" in text
