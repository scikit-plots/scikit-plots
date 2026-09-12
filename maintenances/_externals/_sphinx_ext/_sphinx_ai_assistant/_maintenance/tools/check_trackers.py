"""
Lightweight maintenance drift checker for ``_sphinx_ai_assistant``.

The maintenance control plane intentionally lives outside the runtime package::

    scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/
    maintenances/_externals/_sphinx_ext/_sphinx_ai_assistant/

This checker works both in a full repository checkout and in a standalone
maintenance archive. Runtime-dependent checks are enabled only when the sibling
``scikitplot`` tree can be located.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
MAINTENANCE_DIR = TOOLS_DIR.parent
MAINT_MODULE_ROOT = MAINTENANCE_DIR.parent
FAMILY_MAINTENANCE_ROOT = MAINT_MODULE_ROOT.parent
COMMON_TOOLS = FAMILY_MAINTENANCE_ROOT / "_maintenance_core" / "tools"
if str(COMMON_TOOLS) not in sys.path:
    sys.path.insert(0, str(COMMON_TOOLS))
from check_subsystem import check_subsystem as _check_common_subsystem

REQUIRED = [
    "MAINTENANCE_MODEL.md",
    "RULESET.md",
    "TRACKER_LOGICAL.md",
    "TRACKER_PHYSICAL.md",
    "TRACKER.json",
    "STATE.json",
    "SUBMODULE_STRUCTURE.md",
    "CONFIG_ARCHITECTURE.md",
    "INTEGRATION_CONTRACT.md",
    "RUNTIME_FLOW.md",
    "APP_STREAMING_RUNBOOK.md",
    "SECURITY_IMPLEMENTATION_RUNBOOK.md",
    "SECURITY_MODEL.md",
    "SECURITY_FINDINGS_INDEX.md",
    "REGISTRY.md",
    "VERIFICATION.md",
    "LEGACY_MAINTENANCE_MIGRATION.md",
    "CHECKPOINT_TEMPLATE.md",
    "HISTORY.md",
]
REQUIRED_SCHEMAS = [
    "state.schema.json",
    "tracker.schema.json",
    "checkpoint.schema.json",
    "discovery-contract.schema.json",
    "endpoint-profile.schema.json",
    "setting-definition.schema.json",
]
REQUIRED_TODO = [
    "lessons.md",
    "todo.md",
]


def _repository_root() -> Path | None:
    """Return the repository root when this tree is under ``maintenances/``."""
    for candidate in (MAINT_MODULE_ROOT, *MAINT_MODULE_ROOT.parents):
        if candidate.name == "maintenances":
            return candidate.parent
    return None


def _runtime_root() -> Path | None:
    repo = _repository_root()
    if repo is None:
        return None
    candidate = (
        repo
        / "scikitplot"
        / "_externals"
        / "_sphinx_ext"
        / "_sphinx_ai_assistant"
    )
    return candidate if candidate.is_dir() else None


def load_json(path: Path, errors: list[str]):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # ruff: ignore[blind-except]
        errors.append(f"invalid JSON {path}: {exc}")
        return None


def main() -> int:  # ruff: ignore[too-many-branches, undocumented-public-function]
    errors: list[str] = []
    errors.extend(f"common: {e}" for e in _check_common_subsystem(MAINT_MODULE_ROOT / "MAINTENANCE.json"))

    for name in REQUIRED:
        if not (MAINTENANCE_DIR / name).is_file():
            errors.append(f"missing maintenance file: {name}")
    for name in REQUIRED_SCHEMAS:
        path = MAINTENANCE_DIR / "schemas" / name
        if not path.is_file():
            errors.append(f"missing schema: schemas/{name}")
        else:
            load_json(path, errors)
    ownership_map = MAINTENANCE_DIR / "schemas" / "TEST_OWNERSHIP_MAP.json"
    if not ownership_map.is_file():
        errors.append("missing schema: schemas/TEST_OWNERSHIP_MAP.json")
    else:
        load_json(ownership_map, errors)

    for name in REQUIRED_TODO:
        if not (MAINT_MODULE_ROOT / "todo" / name).is_file():
            errors.append(f"missing maintenance todo file: todo/{name}")
    if not (MAINTENANCE_DIR / "history" / "design-stub-and-guards.md").is_file():
        errors.append("missing historical design: history/design-stub-and-guards.md")

    # The old source-local tasks/ directory is intentionally retired. Keeping
    # this invariant executable prevents maintenance prose from drifting back
    # into the runtime package.
    if (MAINT_MODULE_ROOT / "tasks").exists():
        errors.append("legacy maintenance directory present: tasks/ (use todo/)")

    state = load_json(MAINTENANCE_DIR / "STATE.json", errors) if (MAINTENANCE_DIR / "STATE.json").exists() else None
    tracker = load_json(MAINTENANCE_DIR / "TRACKER.json", errors) if (MAINTENANCE_DIR / "TRACKER.json").exists() else None

    if state:
        for key in [
            "schema_version",
            "subsystem",
            "source_anchor",
            "governing_rule",
            "phase",
            "active_checkpoint",
            "checkpoints",
            "verification_snapshot",
            "next_actions",
        ]:
            if key not in state:
                errors.append(f"STATE.json missing key: {key}")
        sha = state.get("source_anchor", {}).get("sha256", "")
        if sha and not re.fullmatch(r"[0-9a-f]{64}", sha):
            errors.append("STATE.json source sha256 is not 64 lowercase hex chars")
        for checkpoint_id in state.get("checkpoints", {}):
            matches = list((MAINTENANCE_DIR / "checkpoints").glob(f"{checkpoint_id}_*.md"))
            if len(matches) != 1:
                errors.append(
                    f"checkpoint {checkpoint_id} must map to exactly one file; found {len(matches)}"
                )

    if tracker:
        ids = [c.get("id") for c in tracker.get("logical_contracts", [])]
        if len(ids) != len(set(ids)):
            errors.append("TRACKER.json has duplicate logical contract IDs")

    runtime_root = _runtime_root()
    if runtime_root is not None:
        if (runtime_root / "tasks").exists():
            errors.append(
                "runtime maintenance leakage: scikitplot/.../_sphinx_ai_assistant/tasks/ exists"
            )
        if (runtime_root / "_maintenance").exists():
            errors.append(
                "runtime maintenance leakage: scikitplot/.../_sphinx_ai_assistant/_maintenance/ exists"
            )
        if (runtime_root / "MAINTAINING.md").exists():
            errors.append(
                "runtime maintenance leakage: scikitplot/.../_sphinx_ai_assistant/MAINTAINING.md exists"
            )
        if (runtime_root / "_backup").exists():
            errors.append(
                "runtime maintenance leakage: scikitplot/.../_sphinx_ai_assistant/_backup/ exists"
            )

        tests_root = runtime_root / "tests"
        if tests_root.is_dir():
            allowed_cross = {"_architecture", "_integration"}
            historical = re.compile(r"^test_run\d+", re.IGNORECASE)
            for path in tests_root.rglob("test_*.py"):
                rel = path.relative_to(tests_root)
                if rel.parts[0] in allowed_cross or rel.parts[:2] == ("_static", "ai_assistant"):
                    continue
                if historical.match(path.name) and path.name != "test_run_redis_chaos.py":
                    errors.append(f"historical test filename is canonical: {rel}")
                    continue
                owner_dir = runtime_root.joinpath(*rel.parts[:-1])
                matches = 0
                for source in owner_dir.glob("*.py"):
                    expected = "test___init__.py" if source.name == "__init__.py" else f"test_{source.stem}.py"
                    if path.name == expected:
                        matches += 1
                if matches != 1:
                    errors.append(f"non-canonical Python test owner: {rel}")
            for case_root in tests_root.rglob("_cases"):
                for fragment in case_root.rglob("test_*.py"):
                    errors.append(
                        f"case fragment is directly collectable: {fragment.relative_to(tests_root)}"
                    )

        # Enforce the frozen reverse-dependency boundary if the producer exists.
        producer = runtime_root.parent / "_sphinx_llm"
        if producer.exists():
            for path in producer.rglob("*.py"):
                if "tests" in path.parts:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                for line in text.splitlines():
                    stripped = line.strip()
                    if not stripped.startswith(("import ", "from ")):
                        continue
                    if "_sphinx_ai_assistant" in stripped:
                        errors.append(
                            f"reverse dependency: {path} imports _sphinx_ai_assistant"
                        )
                        break

        # Historical backups are maintenance-only. Runtime code must never
        # refer to them.
        for path in runtime_root.rglob("*.py"):
            if "tests" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "_static/_backup" in text or "_static\\_backup" in text:
                errors.append(
                    f"runtime backup dependency reference: {path.relative_to(runtime_root)}"
                )

    if errors:
        print("_sphinx_ai_assistant maintenance drift: FAIL")  # ruff: ignore[print]
        for error in errors:
            print(f" - {error}")  # ruff: ignore[print]
        return 1

    mode = "repository" if runtime_root is not None else "standalone-maintenance"
    print(  # ruff: ignore[print]
        f"_sphinx_ai_assistant maintenance drift: GREEN ({mode})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
