"""Dependency-free checks for the _annoy source owner in a wide repository.

No runtime package is imported. JSON is data, never a command source. These
static checks establish ownership and evidence freshness, not Cython ABI parity.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import tokenize

MODULE = "scikitplot.cexternals._annoy"
RUNTIME = "scikitplot/cexternals/_annoy"
MAINT = "maintenances/cexternals/_annoy"
CONTROL = MAINT + "/_maintenance"
SKILL = "skills/cexternals/_annoy/SKILL.md"
KINDS = {
    "annoy": "direct_source_and_native_api",
    "random": "direct_source",
    "memmap": "direct_source",
    "impute": "compiled_index",
    "corpus": "selectable_index",
    "mcp": "transitive_index",
}
CHECKS = {"dependencies", "planes", "hygiene", "inventory", "handoff", "evidence"}
GATES = {"maintenance_tests", "native_cpp", "native_extension",
         "consumer_build", "downstream_runtime", "windows_runtime"}
SOURCE_SUFFIXES = {".py", ".pyi", ".pyx", ".pxd", ".pxi", ".in",
                   ".h", ".hpp", ".c", ".cc", ".cpp", ".cu", ".i", ".sh"}
C_SUFFIXES = {".h", ".hpp", ".c", ".cc", ".cpp", ".cu"}
CY_SUFFIXES = {".pyx", ".pxd", ".pxi", ".in"}
GENERATED = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


class ContractError(ValueError):
    """Invalid or unsafe metadata; do not refresh a baseline on this input."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def repository_root(explicit=None):
    candidates = [Path(explicit).absolute()] if explicit else Path(__file__).resolve().parents
    for root in candidates:
        if all((root / p).is_dir() for p in (RUNTIME, MAINT, "skills")):
            return root.resolve()
    raise ContractError("wide repository not found; provide --repo-root containing scikitplot/, maintenances/ and skills/")


def safe_path(root, value, *, exists=True):
    require(isinstance(value, str) and value and "\\" not in value
            and ":" not in value and not any(ord(c) < 32 for c in value),
            f"invalid repository path: {value!r}")
    parts = value.split("/")
    require(not PurePosixPath(value).is_absolute()
            and all(p not in ("", ".", "..") for p in parts),
            f"noncanonical repository path: {value!r}")
    result = root
    for part in parts:
        result = result / part
        require(not result.is_symlink(), f"symlink is not allowed: {value}")
    require(result.resolve().is_relative_to(root.resolve()), f"path escapes repository: {value}")
    if exists:
        require(result.exists(), f"missing path: {value}")
    return result


def walk(root, relative):
    """Walk regular files, rejecting links even when the linked target is absent."""
    base = safe_path(root, relative)
    for current, dirs, names in os.walk(base, followlinks=False):
        dirs.sort()
        for name in dirs + names:
            p = Path(current) / name
            require(not p.is_symlink(), f"symlink is not allowed: {p.relative_to(root)}")
        dirs[:] = [d for d in dirs if d not in GENERATED and d != ".git"]
        for name in sorted(names):
            p = Path(current) / name
            require(p.is_file(), f"not a regular file: {p.relative_to(root)}")
            yield p


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    require(path.stat().st_size <= 2 * 1024 * 1024, f"metadata exceeds 2 MiB: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=lambda s: require(False, f"invalid JSON constant: {s}"))


def fields(value, keys, label):
    require(isinstance(value, dict) and set(value) == set(keys), f"invalid {label} fields; expected {sorted(keys)}")


def string_list(value, label):
    require(isinstance(value, list) and all(isinstance(s, str) and s for s in value)
            and len(value) == len(set(value)), f"invalid {label}: expected unique strings")


def load_contract(root):
    manifest = read_json(safe_path(root, MAINT + "/MAINTENANCE.json"))
    fields(manifest, {"schema_version", "subsystem", "runtime_root", "maintenance_root",
                      "skill", "consumers", "read_order", "public_references"}, "manifest")
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 3,
            "unsupported manifest schema")
    require((manifest["subsystem"], manifest["runtime_root"], manifest["maintenance_root"], manifest["skill"])
            == (MODULE, RUNTIME, MAINT, SKILL), "subsystem ownership mismatch")
    safe_path(root, RUNTIME); safe_path(root, MAINT); safe_path(root, SKILL)
    require(isinstance(manifest["consumers"], list), "consumers must be a list")
    names = []
    for edge in manifest["consumers"]:
        fields(edge, {"name", "root", "kind", "headers", "required_imports", "via"}, "consumer")
        name = edge["name"]
        require(isinstance(name, str) and name in KINDS, "unknown consumer")
        names.append(name)
        require(edge["kind"] == KINDS[name] and edge["root"] == "scikitplot/" + name,
                f"consumer ownership mismatch: {name}")
        safe_path(root, edge["root"])
        string_list(edge["headers"], name + " headers")
        string_list(edge["required_imports"], name + " imports")
        for header in edge["headers"]:
            require("/" not in header and header.endswith(".h"), f"invalid header: {header}")
            safe_path(root, RUNTIME + "/src/" + header)
        require(bool(edge["headers"]) == (name in {"annoy", "random", "memmap"}),
                f"wrong source-consumer contract: {name}")
        require(bool(edge["required_imports"]) == (name in {"annoy", "impute", "corpus", "mcp"}),
                f"missing import contract: {name}")
        require(edge["via"] == ("corpus" if name == "mcp" else None), f"wrong transitive owner: {name}")
    require(len(names) == len(KINDS) and set(names) == set(KINDS), "six distinct consumer edges are required")
    string_list(manifest["read_order"], "read order")
    required_read = {MAINT + "/MAINTAINING.md", CONTROL + "/FRESH_CHAT_HANDOFF.md",
                     CONTROL + "/STATE.json", CONTROL + "/FAMILY.md", CONTROL + "/VERIFICATION.md"}
    require(required_read <= set(manifest["read_order"]), "incomplete fresh-chat read order")
    for path in manifest["read_order"]:
        safe_path(root, path)
    string_list(manifest["public_references"], "public references")
    require(all(url.startswith("https://scikit-plots.github.io/") for url in manifest["public_references"]),
            "unexpected public-reference authority")
    profile = read_json(safe_path(root, MAINT + "/REVIEW.json"))
    fields(profile, {"schema_version", "subsystem", "lanes", "release_gates"}, "review")
    require(type(profile["schema_version"]) is int and profile["schema_version"] == 1
            and profile["subsystem"] == MODULE, "review identity mismatch")
    require(isinstance(profile["lanes"], list) and profile["lanes"], "review lanes required")
    selected, lane_ids = [], []
    for lane in profile["lanes"]:
        fields(lane, {"id", "checks"}, "review lane")
        require(isinstance(lane["id"], str) and lane["id"], "invalid lane id")
        string_list(lane["checks"], "registered checks")
        require(set(lane["checks"]) <= CHECKS, "unregistered check; metadata cannot execute commands")
        selected.extend(lane["checks"]); lane_ids.append(lane["id"])
    require(set(selected) == CHECKS and len(selected) == len(CHECKS)
            and len(lane_ids) == len(set(lane_ids)), "review must cover each registered check once")
    string_list(profile["release_gates"], "release gates")
    require(set(profile["release_gates"]) == GATES, "release gates cannot be disabled")
    return manifest, profile


def is_test(path):
    return "tests" in path.parts or path.name.startswith("test_") or path.stem.endswith("_test")


def inventory(root):
    files = []
    for path in walk(root, RUNTIME):
        data = path.read_bytes()
        kind = "test" if is_test(path) else "source" if path.suffix in SOURCE_SUFFIXES or path.name == "meson.build" else "asset"
        files.append({"path": path.relative_to(root).as_posix(), "kind": kind,
                      "size": len(data), "sha256": digest(data)})
    files.sort(key=lambda f: f["path"])
    return {"files": files, "digest": digest(canonical(files)),
            "totals": {k: sum(f["kind"] == k for f in files) for k in ("source", "test", "asset")}}


def python_imports(path, root):
    """Resolve absolute/relative imports at any indentation without importing code."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = ".".join(path.relative_to(root).parts[:-1])
    result, aliases = [], {}
    def absolute(module, level):
        parts = package.split(".")
        require(level <= len(parts), f"relative import escapes package: {path.relative_to(root)}")
        return ".".join(parts[:len(parts) - level + 1] + ([module] if module else [])) if level else module
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                result.append((item.name, node.lineno))
                aliases[item.asname or item.name.split(".")[0]] = item.name if item.asname else item.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            base = absolute(node.module or "", node.level)
            for item in node.names:
                name = base + "." + item.name if base else item.name
                result.append((name, node.lineno))
                aliases[item.asname or item.name] = name
    def qualified(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return qualified(node.value) + "." + node.attr
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and qualified(node.func) in {"importlib.import_module", "__import__", "builtins.__import__"}:
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                name = node.args[0].value
                if not name.startswith("."):
                    result.append((name, node.lineno))
    return sorted(set(result))


def cython_headers(path):
    """Inspect declaration literals, not Python syntax in Cython/Tempita input."""
    text = path.read_text(encoding="utf-8")
    literals = re.compile(r"#[^\n]*|(?:[rRuUbBfF]{0,2})(?:\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?'''|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')")
    previous = 0
    for match in literals.finditer(text):
        segment = re.sub(r"\\\r?\n", " ", text[previous:match.start()])
        if not match[0].startswith("#") and re.search(r"\bextern\s+from\s*$", segment):
            value = ast.literal_eval(match[0])
            require(isinstance(value, str), f"non-string extern path: {path}")
            yield value, text.count("\n", 0, match.start()) + 1
        previous = match.end()


def c_headers(path):
    # Preserve string literals while removing comments, including block comments.
    text = path.read_text(encoding="utf-8")
    text = re.sub(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/',
                  lambda m: "\n" * m[0].count("\n") if m[0].startswith(("//", "/*")) else m[0], text)
    for number, line in enumerate(text.splitlines(), 1):
        match = re.match(r'^\s*#\s*include\s*"([^"\n]+)"', line)
        if match:
            yield match[1], number


def architecture(root, manifest):
    """Inspect only _annoy edges outside the seven owned/observed runtime trees."""
    problems = {"dependencies": [], "planes": []}
    edges = {e["name"]: {"name": e["name"], "kind": e["kind"], "via": e["via"],
                          "headers": [], "imports": []} for e in manifest["consumers"]}
    shared = safe_path(root, RUNTIME + "/src")
    upstream = safe_path(root, RUNTIME)
    for path in walk(root, "scikitplot"):
        rel = path.relative_to(root).as_posix()
        if is_test(path) or "_backup" in path.parts:
            continue
        owner = path.relative_to(root / "scikitplot").parts[0]
        scoped = path.is_relative_to(upstream) or owner in edges
        if owner in {"annoy", "random", "memmap"} and path.suffix == ".h" and (shared / path.name).is_file():
            problems["dependencies"].append(f"{rel}: consumer duplicates a header owned by {RUNTIME}/src")
        if path.suffix in {".py", ".pyi"}:
            if not scoped and not any(s in path.read_text(encoding="utf-8")
                                      for s in ("annoy", "cexternals")):
                continue
            for target, line in python_imports(path, root):
                where = f"{rel}:{line}"
                if scoped and (target.split(".")[0] in {"maintenances", "skills"}
                               or "._maintenance" in target):
                    problems["planes"].append(f"{where}: runtime imports maintenance/skill code: {target}")
                if path.is_relative_to(upstream) and any(target == "scikitplot." + n or target.startswith("scikitplot." + n + ".") for n in KINDS):
                    problems["planes"].append(f"{where}: upstream imports consumer: {target}")
                to_index = target == "scikitplot.annoy" or target.startswith("scikitplot.annoy.")
                to_native = target == MODULE or target.startswith(MODULE + ".")
                to_corpus = target == "scikitplot.corpus" or target.startswith("scikitplot.corpus.")
                if owner == "mcp" and (to_index or to_native or target == "annoy" or target.startswith("annoy.")):
                    problems["dependencies"].append(f"{where}: MCP must reach Annoy through Corpus: {target}")
                if (to_index or to_native) and not path.is_relative_to(upstream):
                    if owner not in {"annoy", "impute", "corpus"}:
                        problems["dependencies"].append(f"{where}: undeclared index consumer: {owner} -> {target}")
                if (owner == "annoy" and to_native) or (owner in {"impute", "corpus"} and (to_index or to_native)) or (owner == "mcp" and to_corpus):
                    edges[owner]["imports"].append({"path": rel, "line": line, "target": target})
        if path.suffix in CY_SUFFIXES or path.suffix in C_SUFFIXES:
            refs = cython_headers(path) if path.suffix in CY_SUFFIXES else c_headers(path)
            for header, line in refs:
                if "\\" in header or ":" in header or "\x00" in header:
                    if scoped:
                        problems["dependencies"].append(f"{rel}:{line}: nonportable header path: {header!r}")
                    continue
                candidate = path.parent / header
                marked = "cexternals/_annoy" in header
                resolved = candidate.resolve()
                if path.is_relative_to(upstream) and any(resolved.is_relative_to(root / "scikitplot" / n) for n in KINDS):
                    problems["planes"].append(f"{rel}:{line}: upstream includes consumer source: {header}")
                if not marked and path.suffix in C_SUFFIXES and not resolved.is_file():
                    continue  # Quoted CPython/system headers use compiler include paths.
                is_shared = resolved.is_relative_to(shared)
                if not marked and not is_shared:
                    continue
                if not is_shared or not resolved.is_file():
                    problems["dependencies"].append(f"{rel}:{line}: shared header does not resolve inside {RUNTIME}/src: {header}")
                    continue
                cursor = path.parent
                for part in PurePosixPath(header).parts:
                    cursor /= part
                    require(not cursor.is_symlink(), f"symlink in header reference: {rel}:{line}")
                if not path.is_relative_to(upstream):
                    if owner not in edges or KINDS.get(owner) not in {"direct_source", "direct_source_and_native_api"}:
                        problems["dependencies"].append(f"{rel}:{line}: undeclared shared-source consumer: {owner}")
                    else:
                        edges[owner]["headers"].append({"path": rel, "line": line,
                                                       "target": resolved.relative_to(root).as_posix()})
    for expected in manifest["consumers"]:
        observed = edges[expected["name"]]
        headers = {Path(x["target"]).name for x in observed["headers"]}
        if headers != set(expected["headers"]):
            problems["dependencies"].append(f"{expected['name']}: header contract differs; expected {sorted(expected['headers'])}, observed {sorted(headers)}")
        imports = {x["target"] for x in observed["imports"]}
        for target in expected["required_imports"]:
            if target not in imports:
                problems["dependencies"].append(f"{expected['name']}: missing required import evidence: {target}")
    return {"schema_version": 1, "subsystem": MODULE,
            "consumers": [edges[n] for n in sorted(edges)]}, problems


def hygiene(root):
    errors = []
    for relative in (MAINT, str(PurePosixPath(SKILL).parent)):
        base = safe_path(root, relative)
        for current, dirs, files in os.walk(base, followlinks=False):
            for name in dirs + files:
                path = Path(current) / name
                require(not path.is_symlink(), f"symlink is not allowed: {path.relative_to(root)}")
                if name in GENERATED or path.suffix in {".pyc", ".pyo", ".o", ".obj", ".so", ".pyd"}:
                    errors.append(f"generated artifact in maintained source: {path.relative_to(root)}")
            dirs[:] = [d for d in dirs if d not in GENERATED and d != ".git"]
    return errors


def input_digest(root):
    """Bind verification to code, test sources, build inputs, manifest and skill.

    State, result logs and generated trackers are excluded to avoid hash cycles.
    Dependency observations include newly discovered edges elsewhere in the repo.
    """
    paths = set(walk(root, RUNTIME))
    for name in KINDS:
        paths.update(p for p in walk(root, "scikitplot/" + name)
                     if p.suffix in SOURCE_SUFFIXES or p.name == "meson.build")
    paths.update(p for p in walk(root, CONTROL) if p.suffix == ".py")
    for rel in (MAINT + "/MAINTENANCE.json", MAINT + "/REVIEW.json", SKILL,
                "meson.build", "meson.options", "pyproject.toml", "pytest.ini",
                "scikitplot/meson.build", "scikitplot/cexternals/meson.build",
                "scikitplot/_build_utils/tempita.py"):
        p = safe_path(root, rel, exists=False)
        if p.is_file():
            paths.add(p)
    rows = [{"path": p.relative_to(root).as_posix(), "sha256": digest(p.read_bytes())}
            for p in sorted(paths)]
    return digest(canonical(rows))


def physical_markdown(actual):
    lines = ["# Physical tracker", "", "Generated by `check_trackers.py --update`. Inventory authority:",
             f"`{RUNTIME}` only. Maintenance backups and consumer files are excluded.", "",
             "| Kind | Files |", "|---|---:|"]
    lines += [f"| {kind} | {count} |" for kind, count in actual["totals"].items()]
    lines += ["", f"Runtime SHA-256: `{actual['digest']}`", "",
              "`TRACKER.json` records every path, byte count and SHA-256. Any change",
              "requires reconciliation; there is no percentage tolerance.", ""]
    return "\n".join(lines)


def handoff_checks(root, manifest):
    errors = []
    state = read_json(safe_path(root, CONTROL + "/STATE.json"))
    fields(state, {"schema_version", "subsystem", "source", "sample", "scope", "status", "next_action", "findings"}, "state")
    require(type(state["schema_version"]) is int and state["schema_version"] == 2
            and state["subsystem"] == MODULE and state["scope"] == [MAINT, str(PurePosixPath(SKILL).parent)],
            "state ownership/schema mismatch")
    fields(state["source"], {"archive", "sha256", "size"}, "input provenance")
    require(isinstance(state["source"]["archive"], str)
            and re.fullmatch(r"[0-9a-f]{64}", state["source"]["sha256"] or "")
            and type(state["source"]["size"]) is int and state["source"]["size"] > 0,
            "invalid input provenance")
    fields(state["sample"], {"archive", "sha256", "size"}, "sample provenance")
    require(isinstance(state["sample"]["archive"], str) and isinstance(state["sample"]["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", state["sample"]["sha256"])
            and type(state["sample"]["size"]) is int and state["sample"]["size"] > 0, "invalid sample provenance")
    require(state["status"] in {"maintenance_onboarded_runtime_verification_pending", "maintenance_verified"}
            and isinstance(state["next_action"], str) and state["next_action"], "invalid continuation state")
    require(isinstance(state["findings"], list), "findings must be a list")
    ids = []
    for finding in state["findings"]:
        fields(finding, {"id", "status", "summary", "evidence", "next_action"}, "finding")
        require(all(isinstance(v, str) and v for v in finding.values()), "empty finding")
        require(finding["status"] in {"resolved", "disproved", "deferred"}, "invalid finding status")
        ids.append(finding["id"])
    require(len(ids) == len(set(ids)), "duplicate finding id")
    skill = safe_path(root, SKILL).read_text(encoding="utf-8")
    require(skill.startswith("---\n") and "\n---\n" in skill[4:], "skill frontmatter missing")
    front = skill.split("---", 2)[1]
    require(re.search(r"^name: annoy-source-maintainer$", front, re.M)
            and re.search(r"^description: .+", front, re.M), "invalid _annoy skill identity")
    for rel in manifest["read_order"]:
        if rel not in skill:
            errors.append(f"skill omits canonical read target: {rel}")
    for rel in (MAINT + "/MAINTAINING.md", CONTROL + "/FRESH_CHAT_HANDOFF.md"):
        text = safe_path(root, rel).read_text(encoding="utf-8")
        if "check_trackers.py" not in text or "review_subsystem.py" not in text:
            errors.append(f"{rel}: missing executable continuation route")
    # Check local Markdown links; historical documents and preserved backups are not active instructions.
    for path in [safe_path(root, SKILL), safe_path(root, MAINT + "/MAINTAINING.md"),
                 *safe_path(root, CONTROL).glob("*.md")]:
        if path.name in {"HISTORY.md", "PATTERN_COMPARISON.md"}:
            continue
        for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            target = target.split("#", 1)[0]
            if not target or re.match(r"[a-z]+://", target):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file():
                errors.append(f"{path.relative_to(root)}: broken local link: {target}")
    return errors


def evidence_checks(root, current):
    path = safe_path(root, CONTROL + "/EVIDENCE.json", exists=False)
    if not path.is_file():
        return ["EVIDENCE.json missing; verification has not been recorded"], {g: "UNAVAILABLE" for g in sorted(GATES)}
    value = read_json(path)
    fields(value, {"schema_version", "subsystem", "input_digest", "gates"}, "evidence")
    require(type(value["schema_version"]) is int and value["schema_version"] == 1 and value["subsystem"] == MODULE,
            "evidence identity mismatch")
    require(isinstance(value["gates"], dict) and set(value["gates"]) == GATES, "incomplete evidence gates")
    errors, statuses = [], {}
    if value["input_digest"] != current:
        errors.append("verification evidence is stale for the current code/build/skill inputs")
    for name, gate in value["gates"].items():
        fields(gate, {"status", "detail", "log", "sha256"}, "evidence gate")
        require(gate["status"] in {"PASS", "FAIL", "UNAVAILABLE"}
                and isinstance(gate["detail"], str) and gate["detail"], f"invalid evidence status: {name}")
        statuses[name] = gate["status"]
        if name == "maintenance_tests" and gate["status"] != "PASS":
            errors.append("maintenance behavior tests have not passed for this input")
        if gate["status"] == "FAIL":
            errors.append(f"verification gate failed: {name}")
        if gate["status"] in {"PASS", "FAIL"}:
            require(isinstance(gate["log"], str) and gate["log"].startswith(CONTROL + "/evidence/"),
                    f"evidence log must belong to this subsystem: {name}")
            log = safe_path(root, gate["log"])
            if digest(log.read_bytes()) != gate["sha256"]:
                errors.append(f"evidence log changed: {name}")
        else:
            require(gate["log"] is None and gate["sha256"] is None, "unavailable evidence cannot cite a passing log")
    return errors, statuses


def atomic_write(path, data):
    """Commit one complete file; interrupted updates leave detectable drift."""
    fd, tmp = tempfile.mkstemp(prefix=".annoy-maintenance-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def inspect(root, *, refresh=False):
    manifest, profile = load_contract(root)
    graph, results = architecture(root, manifest)
    results["hygiene"] = hygiene(root)
    results["handoff"] = handoff_checks(root, manifest)
    current = input_digest(root)
    evidence_errors, gates = evidence_checks(root, current)
    actual = inventory(root)
    expected = {"schema_version": 2, "subsystem": MODULE, "runtime_root": RUNTIME, "physical": actual}
    generated = {CONTROL + "/TRACKER.json": canonical(expected),
                 CONTROL + "/DEPENDENCY_GRAPH.json": canonical(graph),
                 CONTROL + "/TRACKER_PHYSICAL.md": physical_markdown(actual).encode()}
    results["inventory"] = []
    if refresh and not any(results.values()):
        # Validate every output path before the first write. Baselines cannot
        # bless architecture, profile, handoff or hygiene defects.
        paths = {rel: safe_path(root, rel, exists=False) for rel in generated}
        for rel, data in generated.items():
            atomic_write(paths[rel], data)
    for rel, data in generated.items():
        path = safe_path(root, rel, exists=False)
        if not path.is_file() or path.read_bytes() != data:
            results["inventory"].append(f"stale generated file: {rel}; review changes before --update")
    results["evidence"] = evidence_errors
    lanes = [{"id": lane["id"], "status": "FAIL" if any(results[c] for c in lane["checks"]) else "PASS",
              "checks": {c: results[c] for c in lane["checks"]}} for lane in profile["lanes"]]
    clean = not any(results.values())
    return {"subsystem": MODULE, "maintenance_status": "PASS" if clean else "FAIL",
            "release_status": "PASS" if clean and all(s == "PASS" for s in gates.values()) else "BLOCKED",
            "input_digest": current, "lanes": lanes, "gates": gates, "graph": graph,
            "runtime_inventory": {"digest": actual["digest"], "totals": actual["totals"]}}
