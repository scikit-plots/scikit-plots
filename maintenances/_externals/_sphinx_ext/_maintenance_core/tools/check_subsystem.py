"""Reusable structural, dependency, evidence, and secret-hygiene checks."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from architecture import validate_manifest_architecture
from paths import discover_repository_root, discover_runtime_sphinx_ext

_HEX64 = re.compile(r"[0-9a-f]{64}")
_SECRET_KEY = re.compile(
    r"(?:password|secret|api[_-]?key|credential|private[_-]?key|"
    r"access[_-]?token|auth[_-]?token|bearer[_-]?token|refresh[_-]?token)",
    re.I,
)


def _load_json(path: Path, errors: list[str]):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # maintenance gate: report malformed state, never hide it
        errors.append(f"invalid JSON {path}: {exc}")
        return None


def _secret_paths(value, prefix: tuple[str, ...] = ()) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = (*prefix, str(key))
            if _SECRET_KEY.search(str(key)) and child not in (None, "", False, [], {}):
                yield ".".join(path)
            yield from _secret_paths(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _secret_paths(child, (*prefix, str(index)))




def _safe_child(root: Path, rel: object, label: str, errors: list[str]) -> Path | None:
    if not isinstance(rel, str) or not rel:
        return None
    candidate_rel = Path(rel)
    if candidate_rel.is_absolute():
        errors.append(f"{label} path must be relative to subsystem maintenance root: {rel}")
        return None
    candidate = (root / candidate_rel).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        errors.append(f"{label} path escapes subsystem maintenance root: {rel}")
        return None
    return candidate

def _leakage_imports(package_root: Path) -> list[str]:
    # Plane-separation check remains local because it is not a sibling-runtime edge.
    import ast

    leaks: list[str] = []
    for path in package_root.rglob("*.py"):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
        if any(
            n == "maintenances"
            or n.startswith("maintenances.")
            or n == "skills"
            or n.startswith("skills.")
            for n in names
        ):
            leaks.append(str(path.relative_to(package_root)))
    return leaks


def _check_handoff(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8", errors="ignore")
    for required in ("STATE.json", "TRACKER.json"):
        if required not in text:
            errors.append(f"fresh-chat handoff must reference {required}")
    lower = text.lower()
    if "chat history" not in lower or not any(
        phrase in lower for phrase in ("do not rely", "not authoritative", "not authority", "do not trust")
    ):
        errors.append("fresh-chat handoff must make previous chat history non-authoritative")


def _check_skill(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        errors.append("SKILL.md missing YAML frontmatter opening delimiter")
        return
    try:
        end = next(i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration:
        errors.append("SKILL.md missing YAML frontmatter closing delimiter")
        return
    frontmatter = "\n".join(lines[1:end])
    for key in ("name:", "description:"):
        if key not in frontmatter:
            errors.append(f"SKILL.md frontmatter missing {key[:-1]}")
    for required in ("MAINTAINING.md", "STATE.json"):
        if required not in text:
            errors.append(f"SKILL.md must route maintainers through {required}")


def check_subsystem(manifest_path: str | Path) -> list[str]:
    """Return drift errors for one subsystem; an empty list is GREEN."""
    manifest_path = Path(manifest_path).resolve()
    errors: list[str] = []
    manifest = _load_json(manifest_path, errors)
    if not isinstance(manifest, dict):
        return errors or [f"invalid maintenance manifest: {manifest_path}"]

    required = {
        "schema_version",
        "subsystem",
        "runtime_dir",
        "state",
        "tracker",
        "handoff",
    }
    schema_version = int(manifest.get("schema_version", 1))
    if schema_version < 2:
        required.update({"runtime_requires", "family_related"})
    else:
        required.update({"dependency_edges", "capability_ownership"})
    if schema_version >= 3:
        required.add("review_profile")
    for key in sorted(required - set(manifest)):
        errors.append(f"MAINTENANCE.json missing key: {key}")

    maint_root = manifest_path.parent
    file_keys = ["state", "tracker", "handoff"]
    if schema_version >= 3:
        file_keys.append("review_profile")
    resolved_files: dict[str, Path] = {}
    for key in file_keys:
        rel = manifest.get(key)
        path = _safe_child(maint_root, rel, key, errors)
        if path is not None:
            resolved_files[key] = path
            if not path.is_file():
                errors.append(f"missing {key}: {rel}")

    handoff_path = resolved_files.get("handoff")
    if handoff_path is not None:
        _check_handoff(handoff_path, errors)

    state_path = resolved_files.get("state")
    tracker_path = resolved_files.get("tracker")
    state = _load_json(state_path, errors) if state_path is not None and state_path.is_file() else None
    tracker = _load_json(tracker_path, errors) if tracker_path is not None and tracker_path.is_file() else None
    if isinstance(state, dict):
        for key in (
            "schema_version",
            "subsystem",
            "source_anchor",
            "phase",
            "active_checkpoint",
            "checkpoints",
            "verification_snapshot",
            "next_actions",
        ):
            if key not in state:
                errors.append(f"STATE.json missing key: {key}")
        sha = (
            state.get("source_anchor", {}).get("sha256", "")
            if isinstance(state.get("source_anchor"), dict)
            else ""
        )
        if sha and not _HEX64.fullmatch(sha):
            errors.append("STATE.json source sha256 is not 64 lowercase hex chars")
        checkpoint_dir = _safe_child(
            maint_root, manifest.get("checkpoint_dir", "_maintenance/checkpoints"),
            "checkpoint_dir", errors
        )
        for checkpoint_id in (
            state.get("checkpoints", {})
            if isinstance(state.get("checkpoints"), dict)
            else ()
        ):
            matches = (
                list(checkpoint_dir.glob(f"{checkpoint_id}_*.md"))
                if checkpoint_dir is not None and checkpoint_dir.is_dir()
                else []
            )
            if len(matches) != 1:
                errors.append(
                    f"checkpoint {checkpoint_id} must map to exactly one file; found {len(matches)}"
                )

    if isinstance(tracker, dict):
        ids = [
            c.get("id")
            for c in tracker.get("logical_contracts", [])
            if isinstance(c, dict)
        ]
        if len(ids) != len(set(ids)):
            errors.append("TRACKER.json has duplicate logical contract IDs")

    for label, data in (
        ("STATE.json", state),
        ("TRACKER.json", tracker),
        ("MAINTENANCE.json", manifest),
    ):
        if data is not None:
            for path in _secret_paths(data):
                errors.append(f"committed secret-looking state key in {label}: {path}")

    repo = discover_repository_root(manifest_path)
    runtime_family = None
    try:
        runtime_family = discover_runtime_sphinx_ext(manifest_path)
    except FileNotFoundError:
        pass

    if runtime_family is not None:
        runtime_dir = runtime_family / str(manifest.get("runtime_dir", ""))
        if not runtime_dir.is_dir():
            errors.append(f"runtime package missing: {runtime_dir.name}")
        else:
            errors.extend(validate_manifest_architecture(manifest, runtime_family))
            for leak in _leakage_imports(runtime_dir):
                errors.append(f"runtime imports maintenance/skill plane: {leak}")

    if repo is not None:
        skill_rel = manifest.get("skill_root")
        if skill_rel:
            skill = repo / skill_rel / "SKILL.md"
            if not skill.is_file():
                errors.append(f"skill entry missing: {skill_rel}/SKILL.md")
            else:
                _check_skill(skill, errors)

    return errors
