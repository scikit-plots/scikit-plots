# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Canonical mirrored-test layout and stale-test regression guards.

The test tree is architecture, not release-history storage. Runtime ownership is
encoded by directory + filename; run chronology belongs under ``maintenances``.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import (
    MAINTENANCE_ROOT,
    RUNTIME_ROOT,
    TESTS_ROOT,
)

_EXECUTABLE_SUFFIXES = {".py", ".mjs", ".js"}
_HISTORICAL_RUN_FILE = re.compile(r"^test_run\d+", re.IGNORECASE)
_ALLOWED_RUN_SOURCE_TESTS = {"test_run_redis_chaos.py"}  # mirrors ci/run_redis_chaos.py
_BACKUP_MARKERS = (".bak", ".orig", ".rej", "~", ".pre-", ".backup")


def _test_files() -> list[Path]:
    return sorted(
        p
        for p in TESTS_ROOT.rglob("test_*")
        if p.is_file() and p.suffix in _EXECUTABLE_SUFFIXES
    )


def test_every_runtime_package_init_has_explicit_mirrored_test_init() -> None:
    runtime_inits = sorted(
        p
        for p in RUNTIME_ROOT.rglob("__init__.py")
        if TESTS_ROOT not in p.parents
    )
    missing: list[str] = []
    for source in runtime_inits:
        rel_dir = source.parent.relative_to(RUNTIME_ROOT)
        target = TESTS_ROOT / rel_dir / "test___init__.py"
        if not target.is_file():
            missing.append(str(rel_dir / "test___init__.py"))
    assert missing == [], "missing explicit __init__ test mirrors: " + ", ".join(missing)


def test_canonical_test_filenames_do_not_encode_historical_run_numbers() -> None:
    stale = [
        str(p.relative_to(TESTS_ROOT))
        for p in _test_files()
        if _HISTORICAL_RUN_FILE.match(p.name) and p.name not in _ALLOWED_RUN_SOURCE_TESTS
    ]
    assert stale == [], "release chronology belongs in maintenance, not filenames: " + ", ".join(stale)


def test_no_plain_javascript_or_stale_backup_tests_remain_canonical() -> None:
    plain_js = [str(p.relative_to(TESTS_ROOT)) for p in TESTS_ROOT.rglob("test_*.js")]
    # Python creates __pycache__ while this very test executes, so cache
    # presence is packaging hygiene rather than a runnable-tree invariant.
    # Backups/rejects, however, are static source debris and must never be
    # canonical test inputs.
    debris = [
        str(p.relative_to(TESTS_ROOT))
        for p in TESTS_ROOT.rglob("*")
        if p.is_file() and any(marker in p.name for marker in _BACKUP_MARKERS)
    ]
    assert plain_js == [], "Node tests must use the collected .mjs harness contract: " + ", ".join(plain_js)
    assert debris == [], "test tree contains generated/stale debris: " + ", ".join(debris)


def test_executable_test_files_are_byte_unique() -> None:
    by_digest: dict[str, list[str]] = defaultdict(list)
    for path in _test_files():
        by_digest[hashlib.sha256(path.read_bytes()).hexdigest()].append(
            str(path.relative_to(TESTS_ROOT))
        )
    duplicates = [paths for paths in by_digest.values() if len(paths) > 1]
    assert duplicates == [], f"duplicate canonical tests detected: {duplicates!r}"


def test_test_sources_do_not_derive_runtime_from_parent_depth() -> None:
    offenders: list[str] = []
    forbidden = re.compile(r"(?:Path\([^\n]*__file__[^\n]*\)|\b_THIS|\bROOT|\bHERE)\.parents\[\d+\]")
    for path in TESTS_ROOT.rglob("*.py"):
        if path.name == "test_test_layout.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "pathlib.RUNTIME_ROOT" in text or forbidden.search(text):
            offenders.append(str(path.relative_to(TESTS_ROOT)))
    assert offenders == [], "use tests._paths authorities instead of parent-depth inference: " + ", ".join(offenders)


def test_node_harnesses_live_under_runtime_owner_planes() -> None:
    bad: list[str] = []
    allowed = (
        TESTS_ROOT / "_static" / "ai_assistant",
        TESTS_ROOT / "_static" / "isolation",
        TESTS_ROOT / "_cf_worker",
    )
    for path in TESTS_ROOT.rglob("test_*.mjs"):
        if not any(parent == path.parent for parent in allowed):
            bad.append(str(path.relative_to(TESTS_ROOT)))
    assert bad == [], "Node harness outside an explicit runtime owner plane: " + ", ".join(bad)


def test_python_module_tests_have_exact_canonical_owner_names() -> None:
    bad: list[str] = []
    for path in TESTS_ROOT.rglob("test_*.py"):
        rel = path.relative_to(TESTS_ROOT)
        if rel.parts[0] in {"_architecture", "_integration"}:
            continue
        # Python adapters for ai-assistant.js/CSS intentionally belong to the
        # browser owner rather than to a Python runtime module.
        if rel.parts[:2] == ("_static", "ai_assistant"):
            continue
        owner_dir = RUNTIME_ROOT.joinpath(*rel.parts[:-1])
        candidates = []
        for source in owner_dir.glob("*.py"):
            expected = "test___init__.py" if source.name == "__init__.py" else f"test_{source.stem}.py"
            if path.name == expected:
                candidates.append(source)
        if len(candidates) != 1:
            bad.append(str(rel))
    assert bad == [], (
        "Python tests must use exact source ownership names "
        "(foo.py -> test_foo.py; __init__.py -> test___init__.py): "
        + ", ".join(bad)
    )


def test_large_contract_case_fragments_are_hidden_and_have_one_canonical_owner() -> None:
    bad: list[str] = []
    for case_root in TESTS_ROOT.rglob("_cases"):
        if not case_root.is_dir():
            continue
        for source_cases in sorted(p for p in case_root.iterdir() if p.is_dir()):
            fragments = sorted(
                p for p in source_cases.glob("*.py") if p.name != "__init__.py"
            )
            if not fragments:
                continue
            if any(p.name.startswith("test_") for p in fragments):
                bad.append(f"directly collectable case fragment under {source_cases.relative_to(TESTS_ROOT)}")
            source_stem = source_cases.name
            canonical = case_root.parent / (
                "test___init__.py" if source_stem == "root_init" else f"test_{source_stem}.py"
            )
            if source_stem == "root_init":
                canonical = TESTS_ROOT / "test___init__.py"
            if not canonical.is_file():
                bad.append(
                    f"{source_cases.relative_to(TESTS_ROOT)} -> missing {canonical.relative_to(TESTS_ROOT)}"
                )
    assert bad == [], "invalid hidden case-fragment ownership: " + "; ".join(bad)


def test_recorded_python_migration_targets_exist_and_old_flat_files_are_gone() -> None:
    manifest = MAINTENANCE_ROOT / "_maintenance" / "schemas" / "TEST_OWNERSHIP_MAP.json"
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    assert doc["schemaVersion"] >= 1
    missing: list[str] = []
    stale: list[str] = []
    for old, new in doc.get("pythonMoves", {}).items():
        if not (TESTS_ROOT / new).is_file():
            missing.append(new)
        old_path = TESTS_ROOT / old
        if old != new and old_path.exists():
            stale.append(old)
    assert missing == [], "recorded migration target missing: " + ", ".join(missing)
    assert stale == [], "stale pre-migration flat tests remain: " + ", ".join(stale)
