"""Run the common maintenance gate for every registered Sphinx subsystem."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from architecture import validate_family_architecture
from check_subsystem import check_subsystem
from review import validate_profile
from paths import discover_repository_root, discover_runtime_sphinx_ext

FAMILY = HERE.parent.parent
_CACHE_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def _json_error(path: Path) -> str | None:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"invalid family schema JSON {path.name}: {exc}"
    return None


def _residue(paths: list[Path]) -> list[str]:
    found: list[str] = []
    seen: set[Path] = set()
    for root in paths:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path in seen:
                continue
            seen.add(path)
            if path.name in _CACHE_NAMES or path.suffix == ".pyc":
                found.append(str(path))
    return sorted(found)


def _family_contract_collisions(manifests: list[Path]) -> list[str]:
    owners: dict[str, list[str]] = defaultdict(list)
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            tracker_path = manifest_path.parent / manifest["tracker"]
            tracker = json.loads(tracker_path.read_text(encoding="utf-8"))
        except Exception:
            # Per-subsystem checker reports malformed/missing files precisely.
            continue
        for contract in tracker.get("logical_contracts", []):
            if not isinstance(contract, dict):
                continue
            cid = contract.get("id")
            if isinstance(cid, str) and cid:
                owners[cid].append(manifest_path.parent.name)
    return [
        f"logical contract ID {cid} is owned by multiple subsystems: {', '.join(names)}"
        for cid, names in sorted(owners.items())
        if len(set(names)) > 1
    ]


def main() -> int:
    manifests = sorted(FAMILY.glob("_*/MAINTENANCE.json"))
    manifests = [p for p in manifests if p.parent.name != "_maintenance_core"]
    family_errors: list[str] = []

    if not manifests:
        print("_sphinx_ext maintenance family: FAIL")
        print(" - no subsystem MAINTENANCE.json files found")
        return 1

    for schema in sorted((FAMILY / "_maintenance_core" / "schemas").glob("*.json")):
        error = _json_error(schema)
        if error:
            family_errors.append(error)

    family_errors.extend(_family_contract_collisions(manifests))

    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if int(manifest.get("schema_version", 1)) >= 3:
                review_path = manifest_path.parent / str(manifest.get("review_profile", ""))
                if review_path.is_file():
                    profile = json.loads(review_path.read_text(encoding="utf-8"))
                    family_errors.extend(
                        f"{manifest_path.parent.name}: {error}"
                        for error in validate_profile(profile, manifest)
                    )
        except Exception as exc:
            family_errors.append(f"review profile validation failed for {manifest_path.parent.name}: {exc}")

    try:
        runtime_family = discover_runtime_sphinx_ext(FAMILY)
    except FileNotFoundError:
        runtime_family = None
    if runtime_family is not None:
        family_errors.extend(validate_family_architecture(manifests, runtime_family))

    residue_roots = [FAMILY]
    try:
        residue_roots.append(discover_runtime_sphinx_ext(FAMILY))
    except FileNotFoundError:
        pass
    repo = discover_repository_root(FAMILY)
    if repo is not None:
        skill_family = repo / "skills" / "_externals" / "_sphinx_ext"
        residue_roots.append(skill_family)
    for path in _residue(residue_roots):
        family_errors.append(f"cache/compiled residue: {path}")

    failed = bool(family_errors)
    for manifest in manifests:
        errors = check_subsystem(manifest)
        name = manifest.parent.name
        if errors:
            failed = True
            print(f"FAIL {name}")
            for error in errors:
                print(f" - {error}")
        else:
            print(f"GREEN {name}")

    if family_errors:
        print("FAIL family invariants")
        for error in family_errors:
            print(f" - {error}")

    print(
        f"_sphinx_ext maintenance family: {'FAIL' if failed else 'GREEN'} "
        f"({len(manifests)} subsystems)"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
