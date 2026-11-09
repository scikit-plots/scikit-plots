#!/usr/bin/env python3
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Map each file a scikit-plots pull request changes to its owning area and kind.

Usage (from the root of a scikit-plots checkout)::

    python3 plugins/scikitplot-dev/skills/scikitplot-pr-walkthrough/scripts/pr_paths.py 123
    python3 plugins/scikitplot-dev/skills/scikitplot-pr-walkthrough/scripts/pr_paths.py --files a.py b.md

The area is the maintenance area under ``maintenances/`` that owns the path:
for ``scikitplot/<p>`` the longest ``<p>`` prefix that has a folder there;
``docs/`` is ``_docs``, ``plugins/`` is ``_plugins``, and the rest of the
repository (CI, tools, ``libs/``, build) is ``_repo``. The kind says how to
read the file in a walkthrough: ``generated`` files are consequences of a
source change (name the tool, skip their lines), ``record`` and
``release-note`` files document the change, and ``test``, ``docs`` and
``source`` are read as usual.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import pathlib
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

#: Files a tool writes (``maintenances/_plugins/MAINTAINING.md``, "Never edit a
#: generated file by hand"), with the tool that writes them.
GENERATED = (
    ("plugins/scikitplot/skills/*", "generate_usage_skills.py sync"),
    ("plugins/scikitplot/agents/*.md", "check_records.py --write-index"),
    ("plugins/scikitplot/README.md", "check_records.py --write-index"),
    ("plugins/scikitplot-dev/agents/*.md", "check_records.py --write-index"),
    ("plugins/INDEX.md", "check_records.py --write-index"),
    ("plugins/_shared/skills/scikitplot-engineering-rules/INDEX.md", "check_records.py --write-index"),
    (".github/labeler-by-module.yml", "generate_pr_labeler.py sync --apply"),
)


def shared_skills(root: pathlib.Path) -> List[str]:
    """Return the names of the shared skills (copied into every plugin)."""
    folder = root / "plugins" / "_shared" / "skills"
    return sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []


def generated_by(path: str, shared: List[str]) -> Optional[str]:
    """Return the tool that writes *path*, or ``None`` for a file written by hand."""
    for skill in shared:
        for plugin in ("scikitplot", "scikitplot-dev"):
            if path.startswith(f"plugins/{plugin}/skills/{skill}/"):
                return "check_records.py --write-index (copy of a shared skill)"
    for pattern, tool in GENERATED:
        if fnmatch.fnmatch(path, pattern):
            return tool
    return None


def area(path: str, root: pathlib.Path) -> str:
    """Return the maintenance area that owns *path*."""
    parts = pathlib.PurePosixPath(path).parts
    if parts and parts[0] == "maintenances":
        # An area is a top folder (``_repo``, ``_docs``, ``_plugins``) or the
        # mirror of a package; folders inside it (``_maintenance``, ``todo``)
        # are the area's own, not areas.
        sub = list(parts[1:-1])
        while len(sub) > 1 and not (root / "scikitplot" / pathlib.PurePosixPath(*sub)).is_dir():
            sub.pop()
        return "/".join(sub) or "_repo"
    if parts and parts[0] == "scikitplot":
        sub = list(parts[1:-1])
        while sub:
            if (root / "maintenances" / pathlib.PurePosixPath(*sub)).is_dir():
                return "/".join(sub)
            sub.pop()
        return "_repo"
    if parts and parts[0] == "docs":
        return "_docs"
    if parts and parts[0] == "plugins":
        return "_plugins"
    return "_repo"


def kind(path: str, shared: List[str]) -> Tuple[str, str]:
    """Return ``(kind, note)`` for *path*; the note names a generating tool."""
    tool = generated_by(path, shared)
    if tool:
        return "generated", tool
    name = pathlib.PurePosixPath(path).name
    if path.startswith("docs/source/whats_new/upcoming_changes/"):
        return "release-note", ""
    if path.startswith(("maintenances/", "upcoming_changes/")) and "/tests/" not in path:
        return "record", ""
    if "/tests/" in path or name.startswith("test_") or name == "conftest.py":
        return "test", ""
    if path.startswith("docs/") or path.endswith((".md", ".rst")):
        return "docs", ""
    return "source", ""


def table(paths: List[str], root: pathlib.Path) -> Dict[str, List[Tuple[str, str, str]]]:
    """Return area -> sorted ``(path, kind, note)`` for *paths*."""
    shared = shared_skills(root)
    out: Dict[str, List[Tuple[str, str, str]]] = {}
    for path in sorted(set(paths)):
        k, note = kind(path, shared)
        out.setdefault(area(path, root), []).append((path, k, note))
    return dict(sorted(out.items()))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("pr", nargs="?", type=int, help="pull request number")
    group.add_argument("--files", nargs="+", help="paths, instead of a pull request")
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.files:
        paths = args.files
    else:
        result = subprocess.run(["gh", "pr", "view", str(args.pr), "--json", "files",
                                 "--jq", ".files[].path"], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"gh failed: {result.stderr.strip()}", file=sys.stderr)
            return 1
        paths = result.stdout.split()
    grouped = table(paths, args.root)
    if args.json:
        print(json.dumps(grouped, indent=1))
        return 0
    for owner, rows in grouped.items():
        print(f"{owner}  (maintenances/{owner})")
        for path, k, note in rows:
            print(f"  {k:12s} {path}" + (f"  [{note}]" if note else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
