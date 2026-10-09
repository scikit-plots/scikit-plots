"""
Static drift checks for the CleanPrompt user guide.

The checks avoid importing :mod:`scikitplot.cleanprompt` or optional packages.
They derive selected user-facing inventories directly from source syntax so the
documentation can detect drift in a source-only checkout.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _repository_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "scikitplot" / "cleanprompt" / "_cli.py"
        ).is_file():
            return candidate
    raise AssertionError("could not locate repository root")


ROOT = _repository_root()
SOURCE = ROOT / "scikitplot" / "cleanprompt"
GUIDE = ROOT / "docs" / "source" / "user_guide" / "cleanprompt" / "index.rst"
GALLERY_README = ROOT / "galleries" / "examples" / "cleanprompt" / "README.txt"


def _literal_string_keys(path: Path, assignment: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        target_name = None
        value = None
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target_name = node.targets[0].id
                value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value = node.value
        if target_name != assignment or not isinstance(value, ast.Dict):
            continue
        return {
            key.value
            for key in value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    raise AssertionError(f"could not statically find {assignment} in {path}")


def _command_names() -> set[str]:
    path = SOURCE / "_cli.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        value = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "COMMANDS":
                value = node.value
        elif isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "COMMANDS"
                for target in node.targets
            ):
                value = node.value
        if not isinstance(value, (ast.Tuple, ast.List)):
            continue
        names: set[str] = set()
        for item in value.elts:
            if not isinstance(item, ast.Call):
                continue
            for keyword in item.keywords:
                if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                    if isinstance(keyword.value.value, str):
                        names.add(keyword.value.value)
        return names
    raise AssertionError("could not statically find COMMANDS")


def _guide_text() -> str:
    return GUIDE.read_text(encoding="utf-8")


def test_guide_covers_every_canonical_cli_command() -> None:
    guide = _guide_text()
    missing = sorted(name for name in _command_names() if f"``{name}``" not in guide)
    assert not missing, f"CleanPrompt guide is missing CLI commands: {missing}"


def test_guide_covers_every_named_profile() -> None:
    guide = _guide_text()
    profiles = _literal_string_keys(SOURCE / "_policy.py", "PROFILES")
    missing = sorted(name for name in profiles if f"``{name}``" not in guide)
    assert not missing, f"CleanPrompt guide is missing profiles: {missing}"


def test_guide_covers_every_optional_tier() -> None:
    guide = _guide_text()
    tiers = _literal_string_keys(SOURCE / "_capabilities.py", "TIERS")
    missing = sorted(name for name in tiers if f"``{name}``" not in guide)
    assert not missing, f"CleanPrompt guide is missing optional tiers: {missing}"


def test_guide_keeps_the_sensitive_state_boundary_explicit() -> None:
    guide = _guide_text()
    required = (
        "must not be sent with the prompt",
        "Handle.clear",
        "doctor",
        "inspect",
        "detected values",
        "suggested terms derived from the input",
    )
    for phrase in required:
        assert phrase in guide


def test_gallery_entry_point_exists_and_is_linked() -> None:
    gallery = GALLERY_README.read_text(encoding="utf-8")
    guide = _guide_text()
    assert ".. _cleanprompt_examples:" in gallery
    assert ":ref:`cleanprompt_examples`" in guide
