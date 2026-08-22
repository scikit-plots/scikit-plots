"""Lightweight maintenance drift checker for ``_sphinx_ai_assistant``."""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXT_ROOT = ROOT.parent

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


def load_json(  # ruff: ignore[undocumented-public-function]
    path: Path,
    errors: list[str],
):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # ruff: ignore[blind-except]
        errors.append(f"invalid JSON {path}: {exc}")
        return None


def main() -> int:  # ruff: ignore[too-many-branches, undocumented-public-function]
    errors: list[str] = []
    for name in REQUIRED:
        if not (HERE / name).is_file():
            errors.append(f"missing maintenance file: {name}")
    for name in REQUIRED_SCHEMAS:
        p = HERE / "schemas" / name
        if not p.is_file():
            errors.append(f"missing schema: schemas/{name}")
        else:
            load_json(p, errors)

    state = (
        load_json(HERE / "STATE.json", errors)
        if (HERE / "STATE.json").exists()
        else None
    )
    tracker = (
        load_json(HERE / "TRACKER.json", errors)
        if (HERE / "TRACKER.json").exists()
        else None
    )
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
    if tracker:
        ids = [c.get("id") for c in tracker.get("logical_contracts", [])]
        if len(ids) != len(set(ids)):
            errors.append("TRACKER.json has duplicate logical contract IDs")

    # If the actual sibling producer exists, enforce reverse-dependency rule by
    # scanning its production Python files for assistant references.
    producer = EXT_ROOT / "_sphinx_llm"
    if producer.exists():
        producer_maint = producer / "_maintenance"
        for p in producer.rglob("*.py"):
            if producer_maint in p.parents or "tests" in p.parts:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            if "_sphinx_ai_assistant" in text:
                errors.append(
                    f"reverse dependency: {p} references _sphinx_ai_assistant"
                )

    # Historical backup must remain non-runtime. Flag obvious Python/JS runtime
    # references outside backup itself when the actual source tree is present.
    for p in ROOT.rglob("*.py"):
        if HERE in p.parents or "tests" in p.parts:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if "_static/_backup" in text or "_static\\_backup" in text:
            errors.append(f"runtime backup dependency reference: {p.relative_to(ROOT)}")

    if errors:
        print("_sphinx_ai_assistant maintenance drift: FAIL")  # ruff: ignore[print]
        for e in errors:
            print(f" - {e}")  # ruff: ignore[print]
        return 1
    print("_sphinx_ai_assistant maintenance drift: GREEN")  # ruff: ignore[print]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
