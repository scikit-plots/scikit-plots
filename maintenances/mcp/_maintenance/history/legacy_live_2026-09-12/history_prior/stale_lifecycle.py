# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Classify ``_maintenance/`` files by the stale-file lifecycle (see RULESET §7).

Lifecycle: **ACTIVE** (keep) → **STALE** (``git mv`` into ``history/``) →
**EXPIRED** (delete after two releases in ``history/``). Build artifacts
(``__pycache__``, ``*.pyc``) are removed immediately.

This tool is **dry-run by default** and never deletes or moves without ``--apply``;
it prints every proposed action first. It writes nothing except when ``--apply`` is
given, and even then only performs the shell moves/removals it printed.

Usage
-----
    python -m scikitplot.mcp._maintenance.stale_lifecycle            # report only
    python -m scikitplot.mcp._maintenance.stale_lifecycle --apply    # perform moves

Notes
-----
Classification is intentionally conservative: anything not matched by an explicit
STALE/EXPIRED rule is treated as ACTIVE. Update the rule sets as docs are added.
Stdlib-only; safe on Python 3.8+.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

# Filenames (basename) that are ACTIVE regardless of pattern. Extend as needed.
_ACTIVE = {
    "DESIGN.md",
    "CI_OUTPUT_ROUTING.md",
    "IDEMPOTENT_TESTING.md",
    "STRICT_WIRE_VALIDATION.md",
    "UNKNOWN_ARGUMENTS_AND_MANIFESTS.md",
    "CHANGELOG_IDEMPOTENT.md",
    "METHODOLOGY.md",
    "DOCKER.md",
    "ARTIFACT_SHA256SUMS.txt",
    "update_artifact_manifest.py",
    "stale_lifecycle.py",
    "STALE_FILES.md",
    "RULESET.md",
    "MCP_COMPATIBILITY_POLICY.md",
    "DROP_IN_README.md",
    "MCP_DEEP_REVIEW_REPORT.md",
    "MCP_REDESIGN_PLAN.md",
    "MCP_VERIFICATION_MATRIX.md",
}

# Patterns that mark a file STALE (move to history/ on the next review).
_STALE_PATTERNS = (
    re.compile(r"^VERIFICATION_\d+\.\d+\.\d+\.md$"),  # version-pinned snapshots
    re.compile(r"^VERIFICATION_.*\.md$"),  # other point-in-time verifications
    re.compile(r"^SESSION_LOG.*\.md$"),
    re.compile(r"^MCP_REVIEW_GUIDE\.md$"),  # superseded by the 3 artifacts
)


def classify(name: str) -> str:
    """Return one of ``ACTIVE`` / ``STALE`` / ``ARTIFACT``."""
    if name.endswith(".pyc") or name == "__pycache__":
        return "ARTIFACT"
    if name in _ACTIVE:
        return "ACTIVE"
    if any(p.match(name) for p in _STALE_PATTERNS):
        return "STALE"
    return "ACTIVE"  # conservative default


def _run(cmd: list[str], apply: bool) -> None:
    print(("APPLY: " if apply else "DRY  : ") + " ".join(cmd))  # ruff: ignore[print]
    if apply:
        subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            cmd,
            check=False,
        )


def main(argv: list[str] | None = None) -> int:
    """Run."""
    parser = argparse.ArgumentParser(
        prog="python -m scikitplot.mcp._maintenance.stale_lifecycle",
        description="Classify and (optionally) advance _maintenance/ files through "
        "the stale-file lifecycle. Dry-run unless --apply.",
    )
    parser.add_argument(
        "--apply", action="store_true", help="Perform the printed git mv / rm actions."
    )
    parser.add_argument(
        "--root",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="_maintenance directory (default: this file's dir).",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    root = args.root
    history = os.path.join(root, "history")
    active, stale, artifacts = [], [], []
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if entry == "history":
            continue
        if entry == "__pycache__" or entry.endswith(".pyc"):
            artifacts.append(entry)
            continue
        if not os.path.isfile(path):
            continue
        {"ACTIVE": active, "STALE": stale}[classify(entry)].append(entry)

    print(f"# stale-file lifecycle report for {root}")  # ruff: ignore[print]
    print(f"ACTIVE ({len(active)}): {', '.join(active) or '-'}")  # ruff: ignore[print]
    print(f"STALE  ({len(stale)}): {', '.join(stale) or '-'}")  # ruff: ignore[print]
    print(  # ruff: ignore[print]
        f"ARTIFACT ({len(artifacts)}): {', '.join(artifacts) or '-'}"
    )
    print()  # ruff: ignore[print]

    for entry in artifacts:  # immediate removal
        _run(["rm", "-rf", os.path.join(root, entry)], args.apply)
    if stale:
        _run(["mkdir", "-p", history], args.apply)
        for entry in stale:  # move to history/ (grace tier)
            _run(
                ["git", "mv", os.path.join(root, entry), os.path.join(history, entry)],
                args.apply,
            )

    if not args.apply:
        print(  # ruff: ignore[print]
            "\n(dry-run) re-run with --apply to perform the actions above."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
