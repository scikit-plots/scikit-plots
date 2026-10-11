"""
Rebuild selected project folders and empty files.

User notes:
- Run with `python tools/rebuild_tree.py --dry-run` to preview the actions.
- Run with `python tools/rebuild_tree.py` to apply them.
- The default target is the project root (the parent of this script's folder).
- Existing contents of the listed paths are permanently deleted. In particular,
  the existing Makefile and other listed files are replaced with empty files.
- Use `--root PATH` to target a different project root.

Developer notes:
- Uses only the Python standard library; supports Windows, Git Bash, and Linux.
- The paths to rebuild are defined in FOLDERS and FILES below.
- The script rejects absolute paths and parent-directory traversal in those lists.
- There is no rollback if an error occurs partway through the rebuild.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path, PureWindowsPath

logger = logging.getLogger("rebuild_tree")

FOLDERS = [
    ".circleci",
    ".github",
    "docker",
    "docs",
    "galleries",
    "libs",
    "LICENSES",
    "maintenances",
    "meson_cpu",
    "requirements",
    "scikitplot",
    "skills",
    "tasks",
    "upcoming_changes",
]

FILES = [
    "LICENSE.txt",
    "Makefile",
    "MANIFEST.in",
    "meson.build",
    "meson.options",
    "pyproject.toml",
    "pytest.ini",
]


def validate_relative_path(name: str) -> None:
    """Reject absolute paths and paths that could escape the project root."""
    path = Path(name)
    windows_path = PureWindowsPath(name)

    if (
        not name
        or path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in path.parts
        or ".." in windows_path.parts
    ):
        raise ValueError(f"Unsafe path in FOLDERS or FILES: {name!r}")


def remove_existing(path: Path) -> None:
    """Remove a file, directory, or symlink without following symlinks."""
    if path.is_symlink():
        logger.info("Removing symlink: %s", path)
        path.unlink()
    elif path.is_dir():
        logger.info("Removing directory and contents: %s", path)
        shutil.rmtree(path)
    elif path.exists():
        logger.info("Removing file: %s", path)
        path.unlink()
    else:
        logger.debug("Does not exist; nothing to remove: %s", path)


def rebuild(root: Path, dry_run: bool = False) -> None:
    """Remove the listed paths under root, then recreate their structure."""
    if not root.is_dir():
        raise NotADirectoryError(f"Project root is not a directory: {root}")

    if root == Path(root.anchor):
        raise ValueError(f"Refusing to operate on a filesystem root: {root}")

    for name in FOLDERS + FILES:
        validate_relative_path(name)

    if dry_run:
        logger.info("Dry run: no files or folders will be changed.")
        for name in FOLDERS + FILES:
            logger.info("Would remove, if present: %s", root / name)
        for name in FOLDERS:
            logger.info("Would create directory: %s", root / name)
        for name in FILES:
            logger.info("Would create empty file: %s", root / name)
        return

    # Remove all listed paths first so existing files or directories cannot
    # obstruct the requested layout.
    for name in FOLDERS + FILES:
        remove_existing(root / name)

    # Create the requested directories.
    for name in FOLDERS:
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        logger.info("Created directory: %s", path)

    # Create empty files. Parent directories are created if needed.
    for name in FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        logger.info("Created empty file: %s", path)


def main() -> int:
    default_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Remove and recreate the configured project folders and files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Project root directory (default: {default_root})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log planned actions without changing anything.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show additional diagnostic logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        root = args.root.resolve()
        logger.info("Project root: %s", root)
        rebuild(root, dry_run=args.dry_run)
    except (OSError, ValueError) as error:
        logger.error("%s", error)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
