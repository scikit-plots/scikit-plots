#!/usr/bin/env python3
# scikitplot/mcp/_maintenance/update_artifact_manifest.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Check or atomically regenerate the MCP source SHA-256 manifest.

The manifest covers source and maintenance inputs, not generated deliverables.
Archives, wheels, patches, caches, and build directories are deliberately
excluded so copying a release artifact beside the source cannot make checksum
verification non-idempotent.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = Path(__file__).resolve().with_name("ARTIFACT_SHA256SUMS.txt")
_EXCLUDED_DIR_NAMES = frozenset(
    {
        "__pycache__",
        ".hypothesis",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "dist",
    }
)
_EXCLUDED_FILE_NAMES = frozenset(
    {
        _MANIFEST.name,
        ".coverage",
        ".DS_Store",
        "Thumbs.db",
    }
)
_TRANSIENT_SUFFIXES = frozenset({".pyc", ".pyo"})
_ROOT_GENERATED_SUFFIXES = frozenset(
    {".bz2", ".gz", ".patch", ".tar", ".whl", ".xz", ".zip"}
)


def is_excluded_relative(relative: Path) -> bool:
    """Return whether a package-relative path is a generated/transient input."""
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("manifest paths must stay inside the package root")
    if any(part in _EXCLUDED_DIR_NAMES for part in relative.parts[:-1]):
        return True
    if relative.name in _EXCLUDED_FILE_NAMES:
        return True
    if relative.name.startswith(f".{_MANIFEST.name}."):
        return True
    suffix = relative.suffix.casefold()
    if suffix in _TRANSIENT_SUFFIXES:
        return True
    return len(relative.parts) == 1 and suffix in _ROOT_GENERATED_SUFFIXES


def tracked_files() -> list[Path]:
    """Return the deterministic source-file set represented by the manifest."""
    tracked: list[Path] = []
    for path in _PACKAGE_ROOT.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"source manifest refuses symbolic links: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(_PACKAGE_ROOT)
        if not is_excluded_relative(relative):
            tracked.append(path)
    return sorted(tracked)


def render_manifest() -> str:
    """Render stable package-relative SHA-256 entries."""
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
        f"./{path.relative_to(_PACKAGE_ROOT).as_posix()}"
        for path in tracked_files()
    ]
    return "\n".join(lines) + "\n"


def write_if_changed(content: str) -> bool:
    """Atomically update the manifest; return whether bytes changed."""
    try:
        current = _MANIFEST.read_text(encoding="utf-8")
    except FileNotFoundError:
        current = ""
    if current == content:
        return False

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{_MANIFEST.name}.",
        dir=_MANIFEST.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, _MANIFEST)
        try:
            directory_fd = os.open(_MANIFEST.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def build_parser() -> (  # ruff: ignore[undocumented-public-function]
    argparse.ArgumentParser
):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true", help="fail if the manifest is stale"
    )
    mode.add_argument(
        "--write", action="store_true", help="atomically update the manifest"
    )
    return parser


def main(  # ruff: ignore[undocumented-public-function]
    argv: list[str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    expected = render_manifest()

    if args.write:
        changed = write_if_changed(expected)
        print(  # ruff: ignore[print]
            f"manifest {'updated' if changed else 'unchanged'}: {_MANIFEST}"
        )
        return 0

    try:
        actual = _MANIFEST.read_text(encoding="utf-8")
    except FileNotFoundError:
        actual = ""
    if actual != expected:
        print(  # ruff: ignore[print]
            f"artifact manifest is stale; run {Path(__file__).name} --write",
        )
        return 1
    print(f"manifest ok: {_MANIFEST}")  # ruff: ignore[print]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
