#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath


class ContractError(RuntimeError):
    pass


FORBIDDEN_META_KEYS = {"command", "commands", "cmd", "shell", "exec", "executable"}
TRACKED_SUFFIXES = {".py", ".json", ".md", ".sh"}
ALLOWED_REVIEW_CHECKS = {
    "runtime_presence",
    "contracts",
    "optionality",
    "parity",
    "io",
    "delegation",
    "planes",
    "handoff",
    "inventory",
    "hygiene",
    "evidence",
}


def load_json(path: Path):
    def hook(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ContractError(f"duplicate JSON key {key!r} in {path}")
            out[key] = value
        return out

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load {path}: {exc}") from exc


def reject_command_surface(obj, where="metadata"):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in FORBIDDEN_META_KEYS:
                raise ContractError(f"{where} contains unsupported executable field {key!r}")
            reject_command_surface(value, where)
    elif isinstance(obj, list):
        for value in obj:
            reject_command_surface(value, where)


def discover_repo(start: Path) -> Path:
    resolved = start.resolve()
    for candidate in [resolved, *resolved.parents]:
        if all((candidate / name).is_dir() for name in ("scikitplot", "maintenances", "skills")):
            return candidate
    raise ContractError(
        "could not locate wide repository root containing scikitplot/, maintenances/, and skills/"
    )


def safe_rel(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ContractError(f"unsafe repository path {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or "//" in value:
        raise ContractError(f"unsafe repository path {value!r}")
    return value


def safe_path(root: Path, value: str, exists=True) -> Path:
    value = safe_rel(value)
    path = root / value
    if exists and not path.exists():
        raise ContractError(f"required repository path does not exist: {value}")
    return path


def tracked_runtime_files(root: Path, runtime_root: str) -> list[Path]:
    rr = safe_path(root, runtime_root)
    files = []
    for path in sorted(rr.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.suffix in TRACKED_SUFFIXES:
            files.append(path)
    return files


def runtime_fingerprint(root: Path, runtime_root: str):
    rr = safe_path(root, runtime_root)
    h = hashlib.sha256()
    entries = []
    for path in tracked_runtime_files(root, runtime_root):
        rel = path.relative_to(rr).as_posix()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        h.update(rel.encode() + b"\0" + digest.encode() + b"\n")
        entries.append((rel, path, data))
    return h.hexdigest(), entries


def inventory(entries):
    out = {
        "tracked_files": len(entries),
        "python_files": 0,
        "runtime_python_files": 0,
        "test_python_files": 0,
        "command_python_files": 0,
        "frontend_python_files": 0,
        "markdown_files": 0,
    }
    for rel, _, _ in entries:
        if rel.endswith(".py"):
            out["python_files"] += 1
            if rel.startswith("tests/"):
                out["test_python_files"] += 1
            else:
                out["runtime_python_files"] += 1
            if rel.startswith("_commands/"):
                out["command_python_files"] += 1
            if rel.startswith("_frontends/"):
                out["frontend_python_files"] += 1
        if rel.endswith(".md"):
            out["markdown_files"] += 1
    return out


def parse_python(path: Path):
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise ContractError(f"cannot parse {path}: {exc}") from exc


def top_level_symbols(path: Path) -> set[str]:
    tree = parse_python(path)
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            found.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            def collect(target):
                if isinstance(target, ast.Name):
                    found.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for elt in target.elts:
                        collect(elt)
            for target in targets:
                collect(target)
    return found


def import_name(node) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        prefix = "." * node.level
        return [prefix + (node.module or "")]
    return []


def module_scope_imports(path: Path) -> list[tuple[int, str]]:
    tree = parse_python(path)
    out = []
    for node in tree.body:
        for name in import_name(node):
            out.append((getattr(node, "lineno", 0), name))
    return out


def production_python(root: Path, m: dict) -> list[Path]:
    rr = root / m["runtime_root"]
    return [
        p for p in sorted(rr.rglob("*.py"))
        if "tests" not in p.relative_to(rr).parts and "__pycache__" not in p.parts
    ]


def validate_manifest(root: Path, manifest: dict) -> list[str]:
    reject_command_surface(manifest, "MAINTENANCE.json")
    errors = []
    if manifest.get("schema_version") != 4:
        errors.append("MAINTENANCE.json schema_version must be integer 4")
    if manifest.get("subsystem") != "scikitplot._cli":
        errors.append("MAINTENANCE.json subsystem must be scikitplot._cli")
    for key in ("runtime_root", "maintenance_root", "skill"):
        try:
            safe_path(root, manifest[key])
        except (KeyError, ContractError) as exc:
            errors.append(str(exc))
    return errors


def runtime_presence_checks(root: Path, m: dict, inv: dict) -> list[str]:
    errors = []
    for rel in m["runtime_contract"]["required_files"]:
        if not (root / safe_rel(rel)).is_file():
            errors.append(f"missing required runtime file {rel}")
    tests_root = root / safe_rel(m["runtime_contract"]["tests_root"])
    if not tests_root.is_dir() or not any(tests_root.glob("test_*.py")):
        errors.append("_cli runtime tests are missing")
    if inv["command_python_files"] < 6:
        errors.append("command adapter surface is unexpectedly small/missing")
    if inv["frontend_python_files"] < 3:
        errors.append("frontend adapter surface is unexpectedly small/missing")
    return errors


def _literal_int_assignments(path: Path) -> dict[str, int]:
    out = {}
    for node in parse_python(path).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = node.value
            node = node  # placate type checkers
        else:
            continue
        target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
        if isinstance(value, ast.Constant) and isinstance(value.value, int):
            out[target.id] = value.value
    return out


def _declares_nonempty_capabilities(registry: Path) -> bool:
    tree = parse_python(registry)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "capabilities" and isinstance(kw.value, (ast.Tuple, ast.List)) and kw.value.elts:
                    return True
    return False


def _consumes_spec_capabilities(paths: list[Path]) -> bool:
    for path in paths:
        for node in ast.walk(parse_python(path)):
            if isinstance(node, ast.Attribute) and node.attr == "capabilities":
                return True
    return False


def contract_checks(root: Path, m: dict) -> list[str]:
    errors = []
    for rel, required in m["runtime_contract"]["required_symbols"].items():
        path = root / safe_rel(rel)
        if not path.is_file():
            errors.append(f"cannot check symbols: missing {rel}")
            continue
        symbols = top_level_symbols(path)
        for symbol in required:
            if symbol not in symbols:
                errors.append(f"{rel} is missing required top-level symbol {symbol}")

    codes = _literal_int_assignments(root / m["runtime_root"] / "exit_codes.py")
    expected = {"OK": 0, "ERROR": 1, "USAGE": 2, "UNAVAILABLE": 69, "SOFTWARE": 70, "INTERRUPTED": 130}
    for name, value in expected.items():
        if codes.get(name) != value:
            errors.append(f"exit code {name} must remain {value}; found {codes.get(name)!r}")

    app_text = (root / m["runtime_root"] / "app.py").read_text(encoding="utf-8")
    if m["boundary_contract"]["frontend_env"] not in app_text:
        errors.append("frontend selection environment contract disappeared from app.py")
    if 'return "argparse"' not in app_text:
        errors.append("argparse no longer appears to be the deterministic fallback frontend")

    registry = root / m["runtime_root"] / "registry.py"
    reg_text = registry.read_text(encoding="utf-8")
    target = m["boundary_contract"]["delegated_mcp_target"]
    if target not in reg_text:
        errors.append("registered MCP delegation target changed/disappeared")

    production = production_python(root, m)
    if _declares_nonempty_capabilities(registry) and not _consumes_spec_capabilities(
        [p for p in production if p.name not in {"_spec.py", "registry.py"}]
    ):
        errors.append(
            "CommandSpec.capabilities is declared as runtime contract metadata but no dispatch/frontend path consumes spec.capabilities"
        )

    refs = []
    for path in production:
        text = path.read_text(encoding="utf-8")
        if "EXTENDING.md" in text:
            refs.append(path.relative_to(root / m["runtime_root"]).as_posix())
    if refs and not (root / m["runtime_root"] / "EXTENDING.md").is_file():
        errors.append(f"runtime source references missing EXTENDING.md from {refs}")
    return errors


def optionality_checks(root: Path, m: dict) -> list[str]:
    errors = []
    rr = root / m["runtime_root"]
    click_allow = set(m["boundary_contract"]["module_scope_click_allow"])
    forbidden_roots = tuple(m["boundary_contract"]["forbid_module_scope_roots"])
    forbidden_optional = ("click", "yaml", "toml", "tomli_w", "rich")

    for path in production_python(root, m):
        rel = path.relative_to(rr).as_posix()
        for line, module in module_scope_imports(path):
            plain = module.lstrip(".")
            if plain == "click" or plain.startswith("click."):
                if rel not in click_allow:
                    errors.append(f"optional click import escaped its frontend owner at {rel}:{line}")
            for name in forbidden_optional[1:]:
                if plain == name or plain.startswith(name + "."):
                    errors.append(f"optional dependency {name!r} imported at module scope in {rel}:{line}")
            for prefix in forbidden_roots:
                if plain == prefix or plain.startswith(prefix + "."):
                    errors.append(f"forbidden subsystem/plane module-scope import {module!r} at {rel}:{line}")

    spec_path = rr / "_spec.py"
    allowed_spec_roots = {"__future__", "dataclasses", "typing"}
    for line, module in module_scope_imports(spec_path):
        root_name = module.lstrip(".").split(".")[0]
        if root_name not in allowed_spec_roots:
            errors.append(f"_spec.py lost stdlib-only bootstrap: {module!r} at line {line}")

    registry_path = rr / "registry.py"
    for line, module in module_scope_imports(registry_path):
        if "_commands" in module:
            errors.append(f"registry imports handler module at line {line}: {module}")
    return errors


def parity_checks(root: Path, m: dict) -> list[str]:
    errors = []
    rr = root / m["runtime_root"]
    argp = (rr / "_frontends/_argparse.py").read_text(encoding="utf-8")
    click = (rr / "_frontends/_click.py").read_text(encoding="utf-8")
    parity = (rr / "tests/test_cli_frontend_parity.py").read_text(encoding="utf-8")
    for marker in ("BUILTIN_COMMANDS", "spec.params", "spec.aliases"):
        if marker not in argp:
            errors.append(f"argparse frontend missing neutral-registry marker {marker!r}")
        if marker not in click:
            errors.append(f"click frontend missing neutral-registry marker {marker!r}")
    for marker in ("_argparse", "_click", "test_frontend_parity"):
        if marker not in parity:
            errors.append(f"frontend parity test missing marker {marker!r}")
    for marker in ("prm.multiple", "prm.count", "prm.negatable", "prm.choices"):
        if marker not in argp:
            errors.append(f"argparse parameter projection missing {marker}")
        if marker not in click:
            errors.append(f"click parameter projection missing {marker}")
    return errors


def io_checks(root: Path, m: dict) -> list[str]:
    errors = []
    rr = root / m["runtime_root"]
    ctx = (rr / "context.py").read_text(encoding="utf-8")
    output = (rr / "output.py").read_text(encoding="utf-8")
    logging = (rr / "logging.py").read_text(encoding="utf-8")
    app = (rr / "app.py").read_text(encoding="utf-8")
    for marker in ("stdout", "stderr"):
        if marker not in ctx:
            errors.append(f"Context lost {marker} stream")
    if "ctx.stdout" not in output:
        errors.append("output renderer no longer writes through ctx.stdout")
    if "logging.StreamHandler(sys.stderr)" not in logging:
        errors.append("CLI logging is no longer explicitly rooted on stderr")
    if "sys.stderr.write" not in app:
        errors.append("top-level CLI diagnostics no longer visibly target stderr")
    for path in production_python(root, m):
        tree = parse_python(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                errors.append(f"production CLI uses bare print() in {path.relative_to(rr).as_posix()}:{node.lineno}; route through context/explicit stream")
    return errors


def delegation_checks(root: Path, m: dict) -> list[str]:
    errors = []
    rr = root / m["runtime_root"]
    loader = (rr / "loader.py").read_text(encoding="utf-8")
    delegation_test = (rr / "tests/test_cli_delegation.py").read_text(encoding="utf-8")
    argp = (rr / "_frontends/_argparse.py").read_text(encoding="utf-8")
    click = (rr / "_frontends/_click.py").read_text(encoding="utf-8")
    for marker in ("run_delegate", "SystemExit", "CapabilityMissingError", "runpy.run_module"):
        if marker not in loader:
            errors.append(f"delegate loader missing behavior marker {marker!r}")
    for marker in ("forward", "verbatim", "--help"):
        if marker not in delegation_test:
            errors.append(f"delegation tests missing marker {marker!r}")
    if "_split_delegated" not in argp:
        errors.append("argparse no longer has an explicit pre-parse delegated argv split")
    if "click.UNPROCESSED" not in click or "allow_extra_args" not in click:
        errors.append("click no longer visibly preserves delegated argv as unprocessed pass-through")
    return errors


def plane_checks(root: Path, m: dict) -> list[str]:
    errors = []
    rr = root / m["runtime_root"]
    for path in production_python(root, m):
        for line, module in module_scope_imports(path):
            plain = module.lstrip(".")
            if plain == "maintenances" or plain.startswith("maintenances.") or plain == "skills" or plain.startswith("skills."):
                errors.append(f"runtime plane imports maintenance/skill at {path.relative_to(rr)}:{line}: {module}")
    maint = root / m["maintenance_root"]
    for path in maint.rglob("*.py"):
        if "history" in path.parts:
            continue
        for line, module in module_scope_imports(path):
            plain = module.lstrip(".")
            if plain == "scikitplot._cli" or plain.startswith("scikitplot._cli."):
                errors.append(f"maintenance plane imports runtime at {path.relative_to(root)}:{line}: {module}")
    return errors


def handoff_checks(root: Path, m: dict) -> list[str]:
    errors = []
    for rel in m.get("read_order", []):
        path = root / safe_rel(rel)
        if not path.is_file():
            errors.append(f"read-order file is missing: {rel}")
    handoff = root / m["maintenance_root"] / "_maintenance" / "FRESH_CHAT_HANDOFF.md"
    text = handoff.read_text(encoding="utf-8") if handoff.is_file() else ""
    for marker in ("`_cli` owns", "MCP owns", "stdout", "stderr", "release", "check_trackers.py"):
        if marker not in text:
            errors.append(f"fresh-chat handoff is missing marker {marker!r}")
    skill = root / safe_rel(m["skill"])
    if not skill.is_file() or len(skill.read_text(encoding="utf-8").splitlines()) < 40:
        errors.append("skill must be a substantive SKILL.md")
    elif not skill.read_text(encoding="utf-8").startswith("---\n"):
        errors.append("skill must contain YAML frontmatter")
    return errors


def hygiene_checks(root: Path, m: dict) -> list[str]:
    errors = []
    maint = root / m["maintenance_root"]
    live = maint / "_maintenance"
    required = {
        "README.md", "FRESH_CHAT_HANDOFF.md", "STATE.json", "FAMILY.md",
        "VERIFICATION.md", "TRACKER.json", "EVIDENCE.json", "HISTORY.md",
        "DEPENDENCY_MAP.md", "check_trackers.py", "review_subsystem.py",
    }
    missing = sorted(name for name in required if not (live / name).exists())
    if missing:
        errors.append(f"active maintenance surface missing {missing}")
    entry = (maint / "MAINTAINING.md").read_text(encoding="utf-8") if (maint / "MAINTAINING.md").is_file() else ""
    if "scikitplot/_cli/_maintenance" in entry:
        errors.append("MAINTAINING.md points at an obsolete runtime-local maintenance path")
    return errors


def evidence_checks(root: Path, m: dict, fingerprint: str) -> list[str]:
    errors = []
    path = root / m["maintenance_root"] / "_maintenance" / "EVIDENCE.json"
    evidence = load_json(path)
    reject_command_surface(evidence, "EVIDENCE.json")
    if evidence.get("runtime_fingerprint") != fingerprint:
        errors.append("EVIDENCE.json runtime_fingerprint is stale")
    gates = evidence.get("gates")
    if not isinstance(gates, dict):
        return errors + ["EVIDENCE.json gates must be an object"]
    allowed = {"GREEN", "RED", "UNAVAILABLE"}
    for name, gate in gates.items():
        if not isinstance(gate, dict) or gate.get("status") not in allowed:
            errors.append(f"evidence gate {name} has invalid status")
            continue
        log = gate.get("log")
        digest = gate.get("sha256")
        if log is None:
            if digest is not None:
                errors.append(f"evidence gate {name}: sha256 requires log")
            continue
        try:
            lp = safe_path(root, log)
        except ContractError as exc:
            errors.append(str(exc))
            continue
        actual = hashlib.sha256(lp.read_bytes()).hexdigest()
        if actual != digest:
            errors.append(f"evidence gate {name}: log hash mismatch")
    return errors


def review_profile(root: Path, m: dict):
    review = load_json(root / m["maintenance_root"] / "REVIEW.json")
    reject_command_surface(review, "REVIEW.json")
    errors = []
    if review.get("schema_version") != 1:
        errors.append("REVIEW.json schema_version must be integer 1")
    if review.get("subsystem") != "scikitplot._cli":
        errors.append("REVIEW.json subsystem must be scikitplot._cli")
    if set(review) - {"schema_version", "subsystem", "lanes", "release_gates"}:
        errors.append("REVIEW.json contains unsupported extra fields")
    seen = set()
    for lane in review.get("lanes", []):
        if not isinstance(lane, dict) or set(lane) - {"id", "checks"}:
            errors.append("review lane contains unsupported fields")
            continue
        for check in lane.get("checks", []):
            if check not in ALLOWED_REVIEW_CHECKS:
                errors.append(f"unknown review check {check}")
            seen.add(check)
    if seen != ALLOWED_REVIEW_CHECKS:
        errors.append(f"review checks must be exactly {sorted(ALLOWED_REVIEW_CHECKS)}")
    if not isinstance(review.get("release_gates"), list) or not review["release_gates"]:
        errors.append("review release_gates must be a non-empty list")
    if errors:
        raise ContractError("; ".join(errors))
    return review


def inspect(root: Path, refresh=False):
    maint = root / "maintenances" / "_cli"
    manifest = load_json(maint / "MAINTENANCE.json")
    manifest_errors = validate_manifest(root, manifest)
    review = review_profile(root, manifest)
    fingerprint, entries = runtime_fingerprint(root, manifest["runtime_root"])
    inv = inventory(entries)

    errors = {
        "manifest": manifest_errors,
        "runtime_presence": runtime_presence_checks(root, manifest, inv),
        "contracts": contract_checks(root, manifest),
        "optionality": optionality_checks(root, manifest),
        "parity": parity_checks(root, manifest),
        "io": io_checks(root, manifest),
        "delegation": delegation_checks(root, manifest),
        "planes": plane_checks(root, manifest),
        "handoff": handoff_checks(root, manifest),
        "hygiene": hygiene_checks(root, manifest),
        "evidence": evidence_checks(root, manifest, fingerprint),
    }

    tracker_path = root / manifest["maintenance_root"] / "_maintenance" / "TRACKER.json"
    tracker = load_json(tracker_path)
    reject_command_surface(tracker, "TRACKER.json")
    inv_errors = []
    if tracker.get("runtime_fingerprint") != fingerprint:
        inv_errors.append("TRACKER.json runtime_fingerprint is stale")
    if tracker.get("inventory") != inv:
        inv_errors.append(f"TRACKER.json inventory is stale: recorded={tracker.get('inventory')} actual={inv}")
    errors["inventory"] = inv_errors

    maintenance_keys = ["manifest", "handoff", "hygiene", "evidence", "inventory"]
    runtime_keys = ["runtime_presence", "contracts", "optionality", "parity", "io", "delegation", "planes"]
    maintenance_status = "PASS" if not any(errors[k] for k in maintenance_keys) else "FAIL"
    runtime_status = "PASS" if not any(errors[k] for k in runtime_keys) else "FAIL"

    if refresh:
        blockers = [
            item for key in maintenance_keys + runtime_keys
            for item in errors[key]
            if key not in {"evidence", "inventory"}
        ]
        if blockers:
            raise ContractError("refusing --update while contract checks fail: " + "; ".join(blockers))
        tracker = {
            "schema_version": 2,
            "subsystem": manifest["subsystem"],
            "runtime_root": manifest["runtime_root"],
            "runtime_fingerprint": fingerprint,
            "inventory": inv,
        }
        tracker_path.write_text(json.dumps(tracker, indent=2) + "\n", encoding="utf-8")
        errors["inventory"] = []

    evidence = load_json(root / manifest["maintenance_root"] / "_maintenance" / "EVIDENCE.json")
    release_status = (
        "PASS"
        if maintenance_status == "PASS"
        and runtime_status == "PASS"
        and all(evidence.get("gates", {}).get(gate, {}).get("status") == "GREEN" for gate in review["release_gates"])
        else "BLOCKED"
    )
    lanes = [
        {"id": lane["id"], "checks": {check: errors[check] for check in lane["checks"]}}
        for lane in review["lanes"]
    ]
    return {
        "subsystem": manifest["subsystem"],
        "maintenance_status": maintenance_status,
        "runtime_status": runtime_status,
        "release_status": release_status,
        "runtime_fingerprint": fingerprint,
        "inventory": inv,
        "lanes": lanes,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--inventory", action="store_true")
    args = parser.parse_args()
    try:
        root = Path(args.repo).resolve() if args.repo else discover_repo(Path(__file__))
        result = inspect(root, refresh=args.update)
        output = (
            {"subsystem": result["subsystem"], "runtime_inventory": result["inventory"], "runtime_fingerprint": result["runtime_fingerprint"]}
            if args.inventory else result
        )
        if args.json:
            print(json.dumps(output, indent=2))
        else:
            print(
                f"{result['subsystem']}: maintenance={result['maintenance_status']} "
                f"runtime={result['runtime_status']} release={result['release_status']}"
            )
            for name, values in result["errors"].items():
                for value in values:
                    print(f"{name.upper()}: {value}")
        if args.release:
            return 0 if result["release_status"] == "PASS" else 1
        return 0 if result["maintenance_status"] == "PASS" else 1
    except ContractError as exc:
        print(f"contract error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
