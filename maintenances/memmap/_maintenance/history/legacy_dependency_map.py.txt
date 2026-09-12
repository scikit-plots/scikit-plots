#!/usr/bin/env python3
"""
Regenerate the cross-submodule dependency graph from the source tree.

Run from anywhere; resolves ``scikitplot/`` above this script.

Notes
-----
Imports are resolved with :mod:`ast`, not regex.  An earlier regex pass produced
three false edges -- stdlib ``import random`` read as ``scikitplot.random``, and
``:mod:`scikitplot.mcp``` in a docstring read as an import -- which would have
recorded a dependency cycle that does not exist.  Parsing is the difference
between a map and a guess.
"""

from __future__ import annotations

import ast
import collections
import json
import pathlib
import re
import sys

CY = re.compile(r'extern\s+from\s+"([^"]+)"')


def package_root() -> pathlib.Path:  # ruff: ignore[undocumented-public-function]
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "scikitplot":
            return parent
    raise SystemExit("could not locate scikitplot/ above this script")


def imports(path: pathlib.Path, mine: str, known: set[str]):
    """Yield ``(target, col_offset)`` for real intra-package imports."""
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            target = None
            if node.level >= 2 and node.module:  # ruff: ignore[magic-value-comparison]
                target = node.module.split(".")[0]
            elif (
                node.level == 0
                and node.module
                and node.module.startswith("scikitplot.")
            ):
                parts = node.module.split(".")
                target = parts[1] if len(parts) > 1 else None
            if target in known and target != mine:
                yield target, node.col_offset
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("scikitplot."):
                    parts = alias.name.split(".")
                    if len(parts) > 1 and parts[1] in known and parts[1] != mine:
                        yield parts[1], node.col_offset


def main() -> int:  # ruff: ignore[too-many-branches, undocumented-public-function]
    root = package_root()
    mods = sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith((".", "__"))
    )
    known = set(mods)
    edges = collections.defaultdict(lambda: collections.defaultdict(list))

    for mod in mods:
        for path in (root / mod).rglob("*"):
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or "_backup" in path.parts
            ):
                continue
            rel = str(path.relative_to(root))
            is_test = "/tests/" in "/" + rel
            if path.suffix == ".py":
                for target, col in imports(path, mod, known):
                    kind = "deferred" if col > 0 else "module-scope"
                    edges[mod][target].append((kind, rel, is_test))
            elif path.suffix in (".pyx", ".pxd", ".pxi", ".in"):
                for line in path.read_text(errors="replace").splitlines():
                    match = CY.search(line)
                    if match and "cexternals" in match.group(1):
                        edges[mod]["cexternals"].append(("cython-extern", rel, is_test))

    graph: dict = {}
    print(  # ruff: ignore[print]
        "%-12s %-12s %-14s %5s %5s"  # ruff: ignore[printf-string-formatting]
        % (
            "FROM",
            "TO",
            "KIND",
            "code",
            "test",
        )
    )
    for a in sorted(edges):
        for b in sorted(edges[a]):
            for kind in ("module-scope", "deferred", "cython-extern"):
                items = [x for x in edges[a][b] if x[0] == kind]
                if not items:
                    continue
                code = sum(1 for x in items if not x[2])
                print(  # ruff: ignore[print]
                    "%-12s %-12s %-14s %5d %5d"  # ruff: ignore[printf-string-formatting]
                    % (
                        a,
                        b,
                        kind,
                        code,
                        len(items) - code,
                    )
                )
                graph.setdefault(a, {}).setdefault(b, {})[kind] = {
                    "code": code,
                    "test": len(items) - code,
                    "files": sorted({x[1] for x in items if not x[2]})[:3],
                }

    out = root.parent / "dependency_graph.json"
    out.write_text(json.dumps({"graph": graph}, indent=2) + "\n")
    print(f"\nwrote {out}")  # ruff: ignore[print]

    # A cycle would invalidate the review order, so it is checked, not assumed.
    seen, stack = set(), set()

    def visit(node: str) -> bool:
        if node in stack:
            print(f"CYCLE detected at {node!r}", file=sys.stderr)  # ruff: ignore[print]
            return True
        if node in seen:
            return False
        seen.add(node)
        stack.add(node)
        found = any(visit(n) for n in graph.get(node, {}))
        stack.discard(node)
        return found

    if any(visit(n) for n in list(graph)):
        return 1
    print(  # ruff: ignore[print]
        "no cycles: the review order in DEPENDENCY_MAP.md is a topological sort"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
