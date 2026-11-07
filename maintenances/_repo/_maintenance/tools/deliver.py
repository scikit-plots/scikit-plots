# maintenances/_repo/_maintenance/tools/deliver.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Package one change as a git patch and a drop-in zip, and prove both apply.

Usage::

    python -B maintenances/_repo/_maintenance/tools/deliver.py OLD NEW NAME \
        [--out DIR]

``OLD`` is the tree the maintainer has (a directory), ``NEW`` the tree with
the change. Writes ``NAME.patch`` and ``NAME-dropin.zip`` to ``--out``
(default: the current directory) and exits 0 only when both are proved.

Notes
-----
**User notes.** The patch applies with ``git apply`` (renames detected); the
zip holds every added, modified or renamed-to file, plus ``REMOVED_FILES.txt``
naming each file to delete, because an archive cannot delete. Both are
verified before the tool reports success:

* the patch applies to a copy of ``OLD`` with
  ``git apply --whitespace=error-all`` (no trailing whitespace, no blank line
  at end of file) and the result equals ``NEW``;
* the zip unpacked over a copy of ``OLD``, with the listed files deleted,
  equals ``NEW``.

Caches (``__pycache__``, ``.pytest_cache``, ``*.pyc``) are never packaged or
compared.

**Developer notes.** Idempotent and deterministic: the same two trees give
byte-identical outputs on every run. Git runs with a fixed identity and
date in a scratch repository, the zip's entries are sorted with a fixed
timestamp and permission bits, and any earlier output of the same name is
replaced, not appended to. Nothing in ``OLD`` or ``NEW`` is changed. Needs
the ``git`` command; everything else is the standard library.
"""

from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

#: Names never packaged or compared.
IGNORED = frozenset({"__pycache__", ".pytest_cache", ".git"})
IGNORED_SUFFIXES = (".pyc",)

#: The fixed timestamp of every zip entry (the earliest a zip can hold).
ZIP_TIME = (1980, 1, 1, 0, 0, 0)

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "deliver",
    "GIT_AUTHOR_EMAIL": "deliver@localhost",
    "GIT_COMMITTER_NAME": "deliver",
    "GIT_COMMITTER_EMAIL": "deliver@localhost",
    "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
    "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
}


class DeliveryError(RuntimeError):
    """A patch or zip that does not reproduce ``NEW``; the message says why."""


def _ignored(name: str) -> bool:
    return name in IGNORED or name.endswith(IGNORED_SUFFIXES)


def copy_tree(source: Path, target: Path) -> None:
    """Copy ``source`` into ``target`` without caches or ``.git``."""
    shutil.copytree(
        source,
        target,
        symlinks=True,
        ignore=lambda _dir, names: [n for n in names if _ignored(n)],
        dirs_exist_ok=True,
    )


def tree_differences(left: Path, right: Path) -> list[str]:
    """
    Return every path that differs between two trees, caches ignored.

    Parameters
    ----------
    left, right : Path
        The trees.

    Returns
    -------
    list of str
        Sorted relative paths: present on one side only, or with different
        bytes. Empty when the trees are equal.
    """
    out: list[str] = []

    def walk(sub: Path) -> None:
        a, b = left / sub, right / sub
        names_a = {n for n in os.listdir(a) if not _ignored(n)} if a.is_dir() else set()
        names_b = {n for n in os.listdir(b) if not _ignored(n)} if b.is_dir() else set()
        out.extend((sub / name).as_posix() for name in sorted(names_a ^ names_b))
        for name in sorted(names_a & names_b):
            pa, pb = a / name, b / name
            if pa.is_dir() and pb.is_dir():
                walk(sub / name)
            elif pa.is_dir() != pb.is_dir() or not filecmp.cmp(pa, pb, shallow=False):
                out.append((sub / name).as_posix())

    walk(Path())
    return out


def _run_git(repo: Path, *args: str) -> bytes:
    """
    Run git in ``repo`` with a fixed identity and date; return its output.

    Notes
    -----
    **Developer notes.** The executable is resolved once to an absolute path
    and the arguments are a fixed list built here (paths are this tool's own
    scratch directories and the caller's patch file), never a shell string.
    """
    git = shutil.which("git")
    if git is None:
        msg = "the git command is required and was not found"
        raise DeliveryError(msg)
    result = subprocess.run(  # noqa: S603 - fixed argument list, no shell
        [git, *args],
        cwd=repo,
        env={**os.environ, **_GIT_ENV},
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        msg = f"git {' '.join(args)} failed: {result.stderr.decode(errors='replace')}"
        raise DeliveryError(msg)
    return result.stdout


def _git(repo: Path, *args: str, output: bool = False) -> str:
    out = _run_git(repo, *args)
    return out.decode("utf-8", errors="surrogateescape") if output else ""


def build_patch(
    old: Path, new: Path, work: Path
) -> tuple[bytes, list[tuple[str, ...]]]:
    """
    Return the binary git patch from ``old`` to ``new`` and its name-status rows.

    Parameters
    ----------
    old, new : Path
        The trees.
    work : Path
        An empty scratch directory.

    Returns
    -------
    patch : bytes
        ``git diff -M --binary`` output.
    status : list of tuple of str
        One row per changed path: ``(code, path)`` or ``(code, old, new)``
        for a rename.
    """
    repo = work / "repo"
    repo.mkdir()
    copy_tree(old, repo)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A", "-f")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "old")
    for child in repo.iterdir():
        if child.name != ".git":
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    copy_tree(new, repo)
    _git(repo, "add", "-A", "-f")
    patch = _run_git(repo, "diff", "--cached", "-M", "--binary", "--no-color")
    rows = [
        tuple(line.split("\t"))
        for line in _git(
            repo, "diff", "--cached", "-M", "--name-status", "--no-color", output=True
        ).splitlines()
        if line
    ]
    return patch, rows


def build_zip(new: Path, rows: list[tuple[str, ...]], target: Path) -> list[str]:
    """
    Write the drop-in zip deterministically and return the files to delete.

    Parameters
    ----------
    new : Path
        The tree with the change.
    rows : list of tuple of str
        Name-status rows from :func:`build_patch`.
    target : Path
        The zip to write; replaced if present.

    Returns
    -------
    list of str
        Sorted paths that were deleted or renamed away.
    """
    put, removed = set(), set()
    for row in rows:
        code = row[0][0]
        if code in "AMT":
            put.add(row[1])
        elif code == "R":
            removed.add(row[1])
            put.add(row[2])
        elif code == "C":
            put.add(row[2])
        elif code == "D":
            removed.add(row[1])
    if target.exists():
        target.unlink()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        if removed:
            text = "Delete these files (a drop-in archive cannot delete or rename):\n"
            text += "".join(f"{path}\n" for path in sorted(removed))
            info = zipfile.ZipInfo("REMOVED_FILES.txt", ZIP_TIME)
            info.external_attr = 0o644 << 16
            archive.writestr(info, text, zipfile.ZIP_DEFLATED)
        for path in sorted(put):
            source = new / path
            info = zipfile.ZipInfo(path, ZIP_TIME)
            mode = 0o755 if os.access(source, os.X_OK) else 0o644
            info.external_attr = mode << 16
            archive.writestr(info, source.read_bytes(), zipfile.ZIP_DEFLATED)
    return sorted(removed)


def verify(old: Path, new: Path, patch: Path, archive: Path, work: Path) -> None:
    """
    Prove that the patch and the zip each turn ``old`` into ``new``.

    Raises
    ------
    DeliveryError
        Naming the step that failed and the paths that differ.
    """
    applied = work / "applied"
    copy_tree(old, applied)
    _git(applied, "init", "-q")
    _git(applied, "apply", "--whitespace=error-all", str(patch))
    differences = tree_differences(applied, new)
    if differences:
        raise DeliveryError(f"patch does not reproduce NEW: {differences[:20]}")
    dropped = work / "dropped"
    copy_tree(old, dropped)
    removed: list[str] = []
    with zipfile.ZipFile(archive) as opened:
        for name in opened.namelist():
            if name == "REMOVED_FILES.txt":
                removed = opened.read(name).decode("utf-8").splitlines()[1:]
                continue
            target = dropped / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(opened.read(name))
    for path in removed:
        (dropped / path).unlink()
        parent = (dropped / path).parent
        while parent != dropped and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    differences = tree_differences(dropped, new)
    if differences:
        raise DeliveryError(f"drop-in zip does not reproduce NEW: {differences[:20]}")


def deliver(old: Path, new: Path, name: str, out: Path) -> dict:
    """
    Write and verify ``NAME.patch`` and ``NAME-dropin.zip``.

    Parameters
    ----------
    old, new : Path
        The maintainer's tree and the changed tree.
    name : str
        Output stem, for example ``scikit-plots-cleanprompt-de-pack``.
    out : Path
        Output directory, created if missing.

    Returns
    -------
    dict
        ``patch``, ``zip``, ``status`` (count per git status code) and
        ``removed`` (files to delete).

    Raises
    ------
    DeliveryError
        If the trees are equal, or a verification step fails.
    """
    if not tree_differences(old, new):
        raise DeliveryError("OLD and NEW are equal: nothing to deliver")
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        patch_bytes, rows = build_patch(old, new, work)
        patch = out / f"{name}.patch"
        patch.write_bytes(patch_bytes)
        archive = out / f"{name}-dropin.zip"
        removed = build_zip(new, rows, archive)
        verify(old, new, patch, archive, work)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row[0][0]] = counts.get(row[0][0], 0) + 1
    return {
        "patch": str(patch),
        "zip": str(archive),
        "status": dict(sorted(counts.items())),
        "removed": removed,
    }


def main(argv: list[str] | None = None) -> int:
    """Deliver and report; exit 1 on any failure."""
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("name")
    parser.add_argument("--out", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        report = deliver(args.old.resolve(), args.new.resolve(), args.name, args.out)
    except DeliveryError as error:
        sys.stderr.write(f"deliver: {error}\n")
        return 1
    sys.stdout.write(f"patch verified: {report['patch']}\n")
    sys.stdout.write(f"drop-in verified: {report['zip']}\n")
    sys.stdout.write(f"status: {report['status']}\n")
    for path in report["removed"]:
        sys.stdout.write(f"delete: {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
