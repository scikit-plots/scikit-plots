#!/usr/bin/env python
"""
Fail when two test files in one pytest session would import as the same module.

pytest's default import mode (``prepend``) imports a test file in a directory
without ``__init__.py`` under its bare file name, and a file inside a package
under its dotted package path. Two files that resolve to the same name cannot
both be collected: the second fails with "import file mismatch" and the whole
session stops at collection, before any test runs.

Usage
-----
``python tools/maint_tools/check_test_module_names.py [ROOT ...]``

Each ROOT is one test session (default: ``scikitplot``, the tree the test
shard and every partial distribution collect). Exit status 0 when every name
is unique within each root, 1 otherwise, with each clash listed.

Notes
-----
**Why a separate check.** The test shard reports the clash only after
installing and collecting for many minutes, and only the first clash. This
check needs nothing installed, runs in a second, and lists them all.

**Developer notes.** The module name is computed the way pytest's
``prepend`` mode does it: walk up from the file while the directory has an
``__init__.py``; the dotted path from the first directory without one is the
name. ``conftest.py`` files are not test modules and are not checked.
Directories named ``.git``, ``build``, ``node_modules`` and ``__pycache__``
are skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

_SKIP_DIRS = frozenset({".git", "build", "node_modules", "__pycache__"})


def is_test_file(name: str) -> bool:
    """Return whether *name* matches pytest's default ``python_files`` globs."""
    return name.endswith(".py") and (
        name.startswith("test_") or name.endswith("_test.py")
    )


def module_name(path: Path) -> str:
    """
    Return the module name pytest's ``prepend`` mode gives *path*.

    Parameters
    ----------
    path : pathlib.Path
        A test file.

    Returns
    -------
    str
        Bare stem when the directory has no ``__init__.py``; otherwise the
        dotted path from the first ancestor directory without one.
    """
    parts = [path.stem]
    directory = path.parent
    while (directory / "__init__.py").is_file():
        parts.insert(0, directory.name)
        directory = directory.parent
    return ".".join(parts)


def iter_test_files(root: Path) -> Iterable[Path]:
    """Yield test files under *root* in a stable order."""
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for name in sorted(files):
            if is_test_file(name):
                yield Path(current) / name


def clashes(root: Path) -> dict[str, list[str]]:
    """
    Return module names shared by more than one test file under *root*.

    Parameters
    ----------
    root : pathlib.Path
        One test session's root.

    Returns
    -------
    dict of str to list of str
        Module name to the POSIX paths (relative to *root*'s parent) that
        share it; empty when every name is unique.
    """
    seen: dict[str, list[str]] = defaultdict(list)
    for path in iter_test_files(root):
        seen[module_name(path)].append(path.as_posix())
    return {name: paths for name, paths in sorted(seen.items()) if len(paths) > 1}


def main(argv: Sequence[str] | None = None) -> int:
    """Check each root; print clashes; return 0 or 1."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("roots", nargs="*", default=["scikitplot"])
    args = parser.parse_args(argv)
    failed = False
    for raw in args.roots:
        root = Path(raw)
        if not root.is_dir():
            sys.stderr.write(f"error: not a directory: {root}\n")
            return 2
        found = clashes(root)
        for name, paths in found.items():
            failed = True
            sys.stdout.write(f"CLASH {name}\n")
            for path in paths:
                sys.stdout.write(f"  {path}\n")
        if not found:
            sys.stdout.write(f"ok {root}: every test module name is unique\n")
    if failed:
        sys.stdout.write(
            "Rename one file of each clash (prefix it with its owner), or add "
            "__init__.py so its package path makes it unique.\n"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
