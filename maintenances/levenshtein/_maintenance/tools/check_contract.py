"""Structural checker for ``scikitplot.levenshtein``."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]
RUNTIME = ROOT / "scikitplot" / "levenshtein"
MAINT = ROOT / "maintenances" / "levenshtein"
DOCS = ROOT / "docs" / "source" / "user_guide" / "levenshtein"
GALLERY = ROOT / "galleries" / "examples" / "levenshtein"
SKILL = ROOT / "skills" / "levenshtein" / "SKILL.md"

REQUIRED_RUNTIME = {
    "__init__.py",
    "_core.py",
    "meson.build",
    "tests/test_levenshtein.py",
}
REQUIRED_DOCS = {
    "index.rst",
    "getting_started.rst",
    "how_it_works.rst",
    "backends_and_fallbacks.rst",
    "ranking_and_matching.rst",
    "corpus_integration.rst",
    "python_api.rst",
    "performance_and_limits.rst",
    "troubleshooting.rst",
    "test_user_guide_sync.py",
}
REQUIRED_GALLERY = {
    "README.txt",
    "plot_levenshtein_basics_script.py",
    "plot_levenshtein_backends_script.py",
    "plot_levenshtein_ranking_script.py",
    "plot_levenshtein_sequences_script.py",
    "plot_levenshtein_corpus_script.py",
}


def tree_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if not item.is_file() or "__pycache__" in item.parts or item.suffix in {".pyc", ".pyo"}:
            continue
        rel = item.relative_to(path).as_posix().encode()
        h.update(rel + b"\0" + item.read_bytes() + b"\0")
    return h.hexdigest()


def _public_names() -> list[str]:
    tree = ast.parse((RUNTIME / "_core.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return [ast.literal_eval(item) for item in node.value.elts]
    raise AssertionError("_core.__all__ not found")


def check() -> dict[str, object]:
    runtime_errors: list[str] = []
    maintenance_errors: list[str] = []

    runtime_files = {
        p.relative_to(RUNTIME).as_posix()
        for p in RUNTIME.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    for required in sorted(REQUIRED_RUNTIME):
        if required not in runtime_files:
            runtime_errors.append(f"missing runtime file: {required}")

    source = (RUNTIME / "_core.py").read_text(encoding="utf-8")
    if '("internal", "rapidfuzz", "python")' not in source:
        runtime_errors.append("auto backend order no longer visibly excludes GPL Levenshtein")
    if "make_corpus_scorer" not in _public_names():
        runtime_errors.append("make_corpus_scorer missing from public surface")

    docs_files = {p.name for p in DOCS.glob("*") if p.is_file()}
    gallery_files = {p.name for p in GALLERY.glob("*") if p.is_file()}
    for required in sorted(REQUIRED_DOCS - docs_files):
        maintenance_errors.append(f"missing user-guide file: {required}")
    for required in sorted(REQUIRED_GALLERY - gallery_files):
        maintenance_errors.append(f"missing gallery file: {required}")
    if not SKILL.is_file():
        maintenance_errors.append("missing skills/levenshtein/SKILL.md")

    guide_index = ROOT / "docs" / "source" / "user_guide" / "index.rst"
    if guide_index.is_file():
        guide_text = guide_index.read_text(encoding="utf-8")
        if "Levenshtein <./levenshtein/index.rst>" not in guide_text:
            maintenance_errors.append("root user-guide index does not link Levenshtein")
    else:
        maintenance_errors.append("missing root user-guide index")

    current_fp = tree_fingerprint(RUNTIME)
    evidence_path = MAINT / "_maintenance" / "EVIDENCE.json"
    state_path = MAINT / "_maintenance" / "STATE.json"
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("runtime_tree_fingerprint") != current_fp:
            maintenance_errors.append("EVIDENCE runtime_tree_fingerprint is stale")
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("runtime_tree_fingerprint") != current_fp:
            maintenance_errors.append("STATE runtime_tree_fingerprint is stale")

    return {
        "subsystem": "scikitplot.levenshtein",
        "runtime_status": "PASS" if not runtime_errors else "FAIL",
        "maintenance_status": "PASS" if not maintenance_errors else "FAIL",
        "release_status": "UNVERIFIED" if not runtime_errors and not maintenance_errors else "BLOCKED",
        "runtime_tree_fingerprint": current_fp,
        "public_names": _public_names(),
        "runtime_errors": runtime_errors,
        "maintenance_errors": maintenance_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"runtime={result['runtime_status']} "
            f"maintenance={result['maintenance_status']} "
            f"release={result['release_status']}"
        )
        for key in ("runtime_errors", "maintenance_errors"):
            for item in result[key]:
                print(f"- {item}")
    return 0 if result["runtime_status"] == result["maintenance_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
