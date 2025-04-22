#!/usr/bin/env python
"""
Advanced file and folder copying utility with support for:

- Dry-run mode
- File filtering (include/exclude)
- Archive mode (zip)
- Interactive prompts
- Progress bar
- Multiple sources
- JSON output

Cross-platform and customizable, usable as a script or importable module.

Consistency with Unix tools like `cp`. In Unix as you saw:
- cp -r my_folder dest/ → includes the folder itself
- cp -r my_folder/ dest/ → only contents
"""

import os
import sys
import re
import shutil
import argparse
import logging
import pprint
import fnmatch
import json
from pathlib import Path
from tqdm import tqdm

RAW_SRC_PATH = None


def setup_logging(verbose=False, debug=False):
    """
    Configure logging output level.

    Parameters
    ----------
    verbose : bool, optional
        If True, set logging level to DEBUG. Otherwise, INFO.
    """
    level = logging.WARNING if not verbose else logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)


def is_cp_like_copy_contents(raw_src: str) -> bool:
    """
    Determines if the source path implies copying folder contents,
    like Unix `cp -r folder/ dest/` vs `cp -r folder dest/`.

    - Trailing slash → copy contents
    - Paths like '.', '..', '../..', './..', etc. → copy contents
    - If any named folder appears → treat as copying folder itself
    """

    # Normalize slashes and strip trailing whitespace
    raw_src = raw_src.replace("\\", "/").rstrip()

    # 1. Trailing slash (means: copy contents)
    if raw_src.endswith("/"):
        return True

    # 2. Regex: only dots and slashes like "..", "../..", "./.." = copy contents
    # If path only consists of (./ or ../) repeated, treat as copy contents
    if re.fullmatch(r"(\.*/?)+", raw_src):
        return True

    # 3. Normalize to handle things like './' or 'a/../b'
    norm = os.path.normpath(raw_src)

    # 4. Normalized path: if it has no "real" folder name, like just ".." or "../.."
    # Reject if any part is not just "." or ".."
    parts = norm.split(os.sep)
    if all(part in (".", "..") for part in parts):
        return True

    # 5. Otherwise: assume folder name exists → copy folder itself
    return False


def resolve_path(path):
    """
    Normalize and expand user and relative paths.

    Parameters
    ----------
    path : str
        Path to resolve.

    Returns
    -------
    str
        Absolute, normalized path.
    """
    # path = os.path.expanduser(path)
    # path = os.path.abspath(os.path.join(os.getcwd(), path))
    return os.path.abspath(os.path.expanduser(path))


def confirm_action(prompt):
    """
    Prompt the user for confirmation.

    Parameters
    ----------
    prompt : str
        The message to display to the user.

    Returns
    -------
    bool
        True if the user confirms (enters 'y'), False otherwise.
    """
    response = input(f"{prompt} [y/N]: ").strip().lower()
    return response == "y"


def match_patterns(filename, include=None, exclude=None):
    """
    Check whether a file matches include/exclude patterns.

    Parameters
    ----------
    filename : str
        The filename to evaluate.
    include : list of str or None
        List of glob patterns to include.
    exclude : list of str or None
        List of glob patterns to exclude.

    Returns
    -------
    bool
        True if the filename matches filters, otherwise False.
    """
    if include and not any(fnmatch.fnmatch(filename, pat) for pat in include):
        return False
    if exclude and any(fnmatch.fnmatch(filename, pat) for pat in exclude):
        return False
    return True


def collect_files(src_dir, include=None, exclude=None):
    """
    Recursively collect files from a directory based on filters.

    Parameters
    ----------
    src_dir : str
        Source directory to search.
    include : list of str or None
        Inclusion glob patterns.
    exclude : list of str or None
        Exclusion glob patterns.

    Returns
    -------
    list of str
        List of matching file paths.
    """
    files = []
    # Lists only the immediate contents of a directory (files and subdirectories, no recursion). os.listdir(path)
    # Recursively walks the entire directory tree, yielding (root, dirs, files) tuples.
    for root, _, filenames in os.walk(src_dir):
        for f in filenames:
            if match_patterns(f, include, exclude):
                files.append(os.path.join(root, f))
    return files


def copy_file(src, dest, overwrite=True, dry_run=False, interactive=False):
    """
    Copy a single file.

    Parameters
    ----------
    src : str
        Source file path.
    dest : str
        Destination file path.
    overwrite : bool, optional
        Overwrite destination if it exists.
    dry_run : bool, optional
        Simulate the operation without making changes.
    interactive : bool, optional
        Prompt before overwriting.

    Returns
    -------
    bool
        True if the file was copied, False otherwise.
    """
    if os.path.exists(dest):
        if not overwrite:
            logging.warning(f"File exists, skipping (no overwrite): {dest}")
            return False
        if interactive and not confirm_action(f"Overwrite {dest}?"):
            logging.info(f"Skipped (user declined): {dest}")
            return False
        if not dry_run:
            os.remove(dest)
            logging.debug(f"Removed: {dest}")

    if not dry_run:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest)
    logging.info(f"{'[Dry Run] ' if dry_run else ''}Copied file: {src} -> {dest}")
    return True


def copy_directory(
    src,
    dest,
    include=None,
    exclude=None,
    clean=False,
    dry_run=False,
    interactive=False,
    progress=False,
    overwrite=True,
):
    """
    Copy a directory recursively with filtering and options.

    Parameters
    ----------
    src : str
        Source directory.
    dest : str
        Destination directory.
    clean : bool, optional
        Remove destination contents before copying.
    dry_run : bool, optional
        Simulate the operation.
    exclude : list of str or None
        Patterns to exclude.
    include : list of str or None
        Patterns to include.
    interactive : bool, optional
        Prompt before cleaning or overwriting.
    progress : bool, optional
        Show a progress bar.
    overwrite : bool, optional
        Overwrite files if they exist.

    Returns
    -------
    list of str
        List of copied file paths.
    """
    copied_files = []
    if clean and os.path.exists(dest):
        if interactive and not confirm_action(f"Clean destination {dest}?"):
            logging.info(f"Skipped clean (user declined): {dest}")
        elif not dry_run:
            shutil.rmtree(dest)
            logging.debug(f"Cleaned destination: {dest}")

    files = collect_files(src, include, exclude)
    iter_files = tqdm(files, desc=f"Copying from {src}") if progress else files

    for f in iter_files:
        rel_path = os.path.relpath(f, start=src)
        target = os.path.join(dest, rel_path)
        if copy_file(
            f, target, overwrite=overwrite, dry_run=dry_run, interactive=interactive
        ):
            copied_files.append(target)

    return copied_files


# def copy_directory_content(src, dest, clean=False, dry_run=False, interactive=False):
#     """
#     Copy a directory content.

#     Parameters
#     ----------
#     src : str
#         Source directory.
#     dest : str
#         Destination directory.
#     clean : bool, optional
#         Remove destination contents before copying.

#     Returns
#     -------
#     list of str
#         List of copied file paths.
#     """
#     copied_files = []
#     if clean and os.path.exists(dest):
#         if interactive and not confirm_action(f"Clean destination {dest}?"):
#             logging.info(f"Skipped clean (user declined): {dest}")
#         elif not dry_run:
#             shutil.rmtree(dest)
#             logging.debug(f"Cleaned destination: {dest}")

#     os.makedirs(dest, exist_ok=True)
#     shutil.copytree(src, dest, dirs_exist_ok=True)
#     copied_files.append(src)
#     return copied_files


def archive_directory(src, dest, dry_run=False):
    """
    Create a zip archive from a directory.

    Parameters
    ----------
    src : str
        Source directory.
    dest : str
        Destination file path (without .zip extension).
    dry_run : bool, optional
        Simulate archiving.

    Returns
    -------
    str
        Path to the resulting archive.
    """
    archive_path = f"{dest}.zip"
    if not dry_run:
        shutil.make_archive(dest, "zip", src)
    logging.info(f"{'[Dry Run] ' if dry_run else ''}Archived: {src} -> {archive_path}")
    return archive_path


def copy_items(sources, dest, **kwargs):
    """
    Copy multiple sources to a destination.

    Parameters
    ----------
    sources : list of str
        Source files or directories.
    dest : str
        Destination directory or path.
    **kwargs
        Keyword args for control: include, exclude, dry_run, interactive,
        archive, overwrite, etc.

    Returns
    -------
    dict
        Dictionary with lists of 'copied', 'archived', and 'skipped' items.
    """
    result = {
        "copied": [],
        "archived": [],
        "skipped": [],
    }

    for idx, src in enumerate(sources):
        if not os.path.exists(src):
            logging.error(f"Source not found: {src}")
            result["skipped"].append(src)
            continue

        if kwargs.get("archive"):
            archive = archive_directory(src, dest, dry_run=kwargs["dry_run"])
            result["archived"].append(archive)
        # For directory copy
        elif os.path.isdir(src):
            if not kwargs.get("recursive"):
                logging.warning(
                    f"Skipping directory (use -r to enable recursive copy): {src}"
                )
                result["skipped"].append(src)
                continue

            # if RAW_SRC_PATH and str(is_cp_like_copy_contents(RAW_SRC_PATH[idx])).lower() == 'true':
            #     base_name = os.path.basename(src.rstrip("/\\"))
            #     dest = dest if not os.path.isdir(dest) else os.path.join(dest, base_name)

            # if RAW_SRC_PATH and str(is_cp_like_copy_contents(RAW_SRC_PATH[idx])).lower() == 'true':
            #     logging.info(f"'UNIX_CP_LIKE_COPY' is -> {'true'}")
            #     copied = copy_directory_content(
            #         src,
            #         dest,
            #         clean=kwargs.get("clean"),
            #         dry_run=kwargs["dry_run"],
            #         interactive=kwargs["interactive"],
            #     )
            #     result["copied"].extend(copied)
            # else:
            copied = copy_directory(
                src,
                dest,
                clean=kwargs.get("clean"),
                dry_run=kwargs["dry_run"],
                exclude=kwargs.get("exclude"),
                include=kwargs.get("include"),
                interactive=kwargs["interactive"],
                progress=kwargs["progress"],
                overwrite=kwargs.get("overwrite", True),
            )
            result["copied"].extend(copied)
        # For file copy
        else:
            if copy_file(
                src,
                dest,
                overwrite=kwargs.get("overwrite", True),
                dry_run=kwargs["dry_run"],
                interactive=kwargs["interactive"],
            ):
                result["copied"].append(dest)

    return result


def parse_args():
    """
    Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Advanced file/directory copier with filtering and dry-run."
    )
    parser.add_argument("src", nargs="+", help="Source file(s) or directory(ies)")
    parser.add_argument("dest", help="Destination path")
    parser.add_argument(
        "-a",
        "--archive",
        action="store_true",
        help="Create zip archive instead of copy",
    )
    parser.add_argument(
        "-c", "--clean", action="store_true", help="Clean destination before copying"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making changes",
    )
    parser.add_argument(
        "-x", "--exclude", nargs="*", help="Exclude patterns (e.g., *.log)"
    )
    parser.add_argument(
        "-i", "--include", nargs="*", help="Include patterns (e.g., *.py)"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Prompt before overwriting/cleaning"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output results in JSON format"
    )
    parser.add_argument(
        "-p", "--progress", action="store_true", help="Show progress bar"
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Copy directories recursively (required for folders)",
    )
    # Overwrite toggle (default True)
    parser.set_defaults(overwrite=True)
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Do not overwrite existing files (default is overwrite)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def main():
    """
    Entry point for CLI use.
    """
    args = parse_args()
    setup_logging(args.verbose)

    global RAW_SRC_PATH
    RAW_SRC_PATH = args.src

    src_paths = [resolve_path(p) for p in args.src]
    dest_path = resolve_path(args.dest)

    result = copy_items(
        src_paths,
        dest_path,
        archive=args.archive,
        clean=args.clean,
        dry_run=args.dry_run,
        exclude=args.exclude,
        include=args.include,
        interactive=args.interactive,
        progress=args.progress,
        recursive=args.recursive,
        overwrite=args.overwrite,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        pprint.pprint(result, indent=2, width=100, sort_dicts=False)


if __name__ == "__main__":
    main()
