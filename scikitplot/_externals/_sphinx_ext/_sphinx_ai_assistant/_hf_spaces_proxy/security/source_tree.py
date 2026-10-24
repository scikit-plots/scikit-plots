"""
Stable whole-extension source subject for release/CI evidence binding.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXTENSION_ROOT = HERE.parents[1]
_IGNORED_DIRS = {".git", ".pytest_cache", "__pycache__"}
_IGNORED_SUFFIXES = {".pyc", ".pyo"}
logger = logging.getLogger(__name__)


class SourceTreeError(RuntimeError):
    """Source tree cannot be represented by the release subject contract."""


# def _source_files(root: Path) -> list[Path]:
#     root = root.resolve()
#     files: list[Path] = []
#     for path in root.rglob("*"):
#         rel = path.relative_to(root)
#         if any(part in _IGNORED_DIRS for part in rel.parts):
#             continue
#         if path.suffix in _IGNORED_SUFFIXES:
#             continue
#         if path.is_symlink():
#             raise SourceTreeError("SOURCE_TREE_SYMLINK_FORBIDDEN")
#         if path.is_file():
#             files.append(path)
#     return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def _source_files(root: Path) -> list[Path]:
    root = root.resolve()
    files: list[Path] = []
    pending = [root]

    while pending:
        directory = pending.pop()

        with os.scandir(directory) as entries:
            for entry in entries:
                # Match the existing rule: ignore any path containing
                # an ignored directory name before checking for symlinks.
                if entry.name in _IGNORED_DIRS:
                    continue

                path = Path(entry.path)

                # Preserve the existing suffix exclusion. Directories
                # with these suffixes are still traversed, as with rglob.
                if path.suffix in _IGNORED_SUFFIXES:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                    continue

                if entry.is_symlink():
                    raise SourceTreeError("SOURCE_TREE_SYMLINK_FORBIDDEN")

                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    files.append(path)

    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def source_tree_sha256(root: Path = EXTENSION_ROOT) -> str:
    """Hash every release source file with path, Unix mode, length and content."""
    root = root.resolve()
    h = hashlib.sha256()
    for path in _source_files(root):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        mode = path.stat().st_mode & 0o777
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(mode.to_bytes(4, "big"))
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()


if __name__ == "__main__":
    logger.info(source_tree_sha256())
