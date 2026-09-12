#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
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
    "protocol",
    "security",
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
                raise ContractError(
                    f"{where} contains unsupported executable field {key!r}"
                )
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
        if path.suffix in TRACKED_SUFFIXES or path.name.startswith(".") and path.suffix == ".json":
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
        h.update(rel.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
        entries.append((rel, path, data))
    return h.hexdigest(), entries


def inventory(entries):
    out = {
        "tracked_files": len(entries),
        "python_files": 0,
        "test_python_files": 0,
        "plugin_json_files": 0,
        "integration_python_files": 0,
        "markdown_files": 0,
    }
    for rel, _, _ in entries:
        if rel.endswith(".py"):
            out["python_files"] += 1
            if rel.startswith("tests/"):
                out["test_python_files"] += 1
            if rel.startswith("_integrations/"):
                out["integration_python_files"] += 1
        if rel.endswith(".json") and rel.startswith("_plugins/"):
            out["plugin_json_files"] += 1
        if rel.endswith(".md"):
            out["markdown_files"] += 1
    return out


def validate_manifest(root: Path, manifest: dict) -> list[str]:
    reject_command_surface(manifest, "MAINTENANCE.json")
    errors = []
    if manifest.get("schema_version") != 4:
        errors.append("MAINTENANCE.json schema_version must be integer 4")
    if manifest.get("subsystem") != "scikitplot.mcp":
        errors.append("MAINTENANCE.json subsystem must be scikitplot.mcp")
    for key in ("runtime_root", "maintenance_root", "skill"):
        try:
            safe_path(root, manifest[key])
        except (KeyError, ContractError) as exc:
            errors.append(str(exc))
    return errors


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
            for target in targets:
                if isinstance(target, ast.Name):
                    found.add(target.id)
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


def runtime_presence_checks(root: Path, m: dict, inv: dict) -> list[str]:
    errors = []
    contract = m["runtime_contract"]
    for rel in contract["required_files"]:
        if not (root / safe_rel(rel)).is_file():
            errors.append(f"missing required runtime file {rel}")
    tests_root = root / safe_rel(contract["tests_root"])
    if not tests_root.is_dir() or not any(tests_root.rglob("test_*.py")):
        errors.append("MCP runtime tests are missing")
    if inv["plugin_json_files"] < 6:
        errors.append(
            f"expected the multi-client plugin manifest surface; found only {inv['plugin_json_files']} JSON files"
        )
    if inv["integration_python_files"] < 3:
        errors.append("MCP integration adapter surface is unexpectedly small/missing")
    return errors


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

    init_path = root / m["runtime_root"] / "__init__.py"
    if init_path.is_file():
        init_text = init_path.read_text(encoding="utf-8")
        for name in ("CitationOutput", "SearchDocsOutput", "SearchService", "create_server"):
            if name not in init_text:
                errors.append(f"lazy server export {name} disappeared from __init__.py")

    caps_path = root / m["runtime_root"] / "_capabilities.py"
    if caps_path.is_file():
        caps = caps_path.read_text(encoding="utf-8")
        if '"stdio"' not in caps or '"streamable-http"' not in caps:
            errors.append("capability inventory no longer advertises both supported transports")
    return errors


def optionality_checks(root: Path, m: dict) -> list[str]:
    rr = root / m["runtime_root"]
    allow_pydantic = set(m["boundary_contract"]["module_scope_pydantic_allow"])
    allow_mcp = set(m["boundary_contract"]["module_scope_mcp_allow"])
    forbidden = tuple(m["boundary_contract"]["forbid_module_scope_roots"])
    errors = []

    for path in sorted(rr.rglob("*.py")):
        rel = path.relative_to(rr).as_posix()
        if rel.startswith("tests/") or "__pycache__" in path.parts:
            continue
        for line, module in module_scope_imports(path):
            plain = module.lstrip(".")
            head = plain.split(".")[0]
            if head == "pydantic" and rel not in allow_pydantic:
                errors.append(f"{rel}:{line} imports pydantic at module scope")
            if head == "mcp" and rel not in allow_mcp:
                errors.append(f"{rel}:{line} imports mcp at module scope; SDK must remain call-time")
            for blocked in forbidden:
                if plain == blocked or plain.startswith(blocked + "."):
                    errors.append(f"{rel}:{line} imports forbidden boundary {module!r} at module scope")

    server = rr / m["boundary_contract"]["official_sdk_import_owner"]
    server_text = server.read_text(encoding="utf-8")
    if "from mcp.server import" not in server_text or "from mcp.types import" not in server_text:
        errors.append("_server.py no longer contains the official SDK registration imports")
    return errors


def protocol_checks(root: Path, m: dict) -> list[str]:
    rr = root / m["runtime_root"]
    errors = []
    server = (rr / "_server.py").read_text(encoding="utf-8")
    if 'name="search_docs"' not in server or "@mcp.tool(" not in server:
        errors.append("official SDK search_docs tool registration is missing")
    if '"docs://chunk/{doc_id}"' not in server or "@mcp.resource(" not in server:
        errors.append("official SDK docs resource registration is missing")
    if "_forbid_unknown_tool_arguments" not in server:
        errors.append("unknown-tool-argument fail-closed seam is missing")

    offenders = []
    for path in sorted(rr.glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if all(marker in text for marker in ('"jsonrpc"', "protocolVersion", "tools/list")):
            offenders.append(path.name)
    if offenders:
        errors.append(f"hand-rolled JSON-RPC MCP dispatch detected in {offenders}")

    single = root / safe_rel(m["boundary_contract"]["single_protocol_test"])
    if not single.is_file():
        errors.append("single-protocol regression test is missing")
    return errors


def security_checks(root: Path, m: dict) -> list[str]:
    rr = root / m["runtime_root"]
    errors = []
    server = (rr / "_server.py").read_text(encoding="utf-8")
    core = (rr / "_core.py").read_text(encoding="utf-8")
    cli = (rr / "__main__.py").read_text(encoding="utf-8")
    if 'ConfigDict(extra="forbid")' not in server:
        errors.append("wire models are not visibly closed to unknown fields")
    if "_DOC_ID_RE" not in server or "_read_resource" not in server:
        errors.append("resource identifier validation seam is missing")
    if "untrusted_content" not in server or "UNTRUSTED" not in core.upper():
        errors.append("untrusted-content wire marker/safety notice is missing")
    if "allow_unauthenticated_remote" not in cli or "_LOCAL_HOSTS" not in cli:
        errors.append("explicit non-loopback bind acknowledgement guard is missing")
    if "MAX_QUERY_CHARS" not in core or "MAX_RESULTS" not in core or "MAX_CHUNK_CHARS" not in core:
        errors.append("bounded query/result/content constants are missing")
    return errors


def plane_checks(root: Path, m: dict) -> list[str]:
    errors = []
    rr = root / m["runtime_root"]
    for path in sorted(rr.rglob("*.py")):
        rel = path.relative_to(rr).as_posix()
        if rel.startswith("tests/"):
            continue
        for line, module in module_scope_imports(path):
            plain = module.lstrip(".")
            if plain == "maintenances" or plain.startswith("maintenances.") or plain == "skills" or plain.startswith("skills."):
                errors.append(f"runtime plane violation {rel}:{line}: {module}")

    maint_root = root / m["maintenance_root"] / "_maintenance"
    for path in sorted(maint_root.rglob("*.py")):
        if "history" in path.parts:
            continue
        for line, module in module_scope_imports(path):
            plain = module.lstrip(".")
            if plain == "scikitplot.mcp" or plain.startswith("scikitplot.mcp."):
                errors.append(
                    f"maintenance plane imports runtime {path.relative_to(root)}:{line}: {module}"
                )
    return errors


def handoff_checks(root: Path, m: dict) -> list[str]:
    errors = []
    for rel in m.get("read_order", []):
        path = root / safe_rel(rel)
        if not path.is_file():
            errors.append(f"read-order file is missing: {rel}")
    handoff = root / m["maintenance_root"] / "_maintenance" / "FRESH_CHAT_HANDOFF.md"
    text = handoff.read_text(encoding="utf-8") if handoff.is_file() else ""
    for marker in ("MCP owns", "Corpus", "Annoy", "release", "check_trackers.py"):
        if marker not in text:
            errors.append(f"fresh-chat handoff is missing marker {marker!r}")
    skill = root / safe_rel(m["skill"])
    if not skill.is_file() or len(skill.read_text(encoding="utf-8").splitlines()) < 30:
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
        "check_trackers.py", "review_subsystem.py",
    }
    missing = sorted(name for name in required if not (live / name).exists())
    if missing:
        errors.append(f"active maintenance surface missing {missing}")

    entry = (maint / "MAINTAINING.md").read_text(encoding="utf-8")
    if "scikitplot/mcp/_maintenance" in entry:
        errors.append("MAINTAINING.md still points at the obsolete runtime-local maintenance path")

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
    maint = root / "maintenances" / "mcp"
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
        "protocol": protocol_checks(root, manifest),
        "security": security_checks(root, manifest),
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
        inv_errors.append(
            f"TRACKER.json inventory is stale: recorded={tracker.get('inventory')} actual={inv}"
        )
    errors["inventory"] = inv_errors

    maintenance_keys = ["manifest", "handoff", "hygiene", "evidence", "inventory"]
    runtime_keys = ["runtime_presence", "contracts", "optionality", "protocol", "security", "planes"]
    maintenance_status = "PASS" if not any(errors[k] for k in maintenance_keys) else "FAIL"
    runtime_status = "PASS" if not any(errors[k] for k in runtime_keys) else "FAIL"

    if refresh:
        blockers = [item for key in maintenance_keys + runtime_keys for item in errors[key] if key not in {"evidence", "inventory"}]
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
            {
                "subsystem": result["subsystem"],
                "runtime_inventory": result["inventory"],
                "runtime_fingerprint": result["runtime_fingerprint"],
            }
            if args.inventory
            else result
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
