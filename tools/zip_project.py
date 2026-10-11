"""
Create a ZIP from selected project paths or a Git commit.

Notes
-----
Selected-path mode includes only the paths configured in ``FOLDERS`` and
``FILES``. Selected folders are included recursively. Symlinks are skipped
without being followed or changed.

Git mode archives a commit's tracked contents. It does not include uncommitted
changes or untracked files, and Git's ``export-ignore`` attributes apply.
Tracked symlinks are excluded from the resulting ZIP.

In both modes, the output ZIP must be outside the project or repository root.
The script prints a SHA-256 digest; ``--write-sha256`` also writes a sidecar
checksum file. Creating an archive does not modify the working tree.

Examples
--------
Create a ZIP from the configured paths::

    python tools/zip_project.py --write-sha256

Archive the current Git commit::

    python tools/zip_project.py --git-archive --write-sha256

Archive a specific Git reference::

    python tools/zip_project.py --git-archive --git-ref v1.2.3
"""

import argparse
import hashlib
import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath


logger = logging.getLogger("zip_project")


# Edit these lists to choose what goes into the selected-path ZIP.
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
    """
    Validate a configured selection path.

    Parameters
    ----------
    name : str
        Path relative to the project root. Use forward slashes.

    Raises
    ------
    ValueError
        If the path is empty, absolute, non-canonical, or could escape the
        project root.
    """
    path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)

    if (
        not name
        or name == "."
        or "\x00" in name
        or "\\" in name
        or path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in path.parts
        or ".." in windows_path.parts
        or path.as_posix() != name
    ):
        raise ValueError(
            f"Unsafe or non-canonical selection {name!r}; "
            "use a relative path with forward slashes."
        )


def validate_selections() -> None:
    """
    Validate configured paths and reject duplicate or conflicting entries.

    Folder/file selections that overlap are allowed; archive entries are
    deduplicated when the ZIP is built.

    Raises
    ------
    ValueError
        If a selection is unsafe, duplicated, or listed as both a folder and
        a file.
    """
    seen_folders = set()
    seen_files = set()

    for name in FOLDERS:
        validate_relative_path(name)
        if name in seen_folders:
            raise ValueError(f"Duplicate folder selection: {name!r}")
        seen_folders.add(name)

    for name in FILES:
        validate_relative_path(name)
        if name in seen_files:
            raise ValueError(f"Duplicate file selection: {name!r}")
        if name in seen_folders:
            raise ValueError(f"Path listed as both a folder and a file: {name!r}")
        seen_files.add(name)


def is_within(path: Path, parent: Path) -> bool:
    """
    Return whether a path is the parent or one of its descendants.

    Parameters
    ----------
    path : pathlib.Path
        Path to test.
    parent : pathlib.Path
        Candidate parent directory.

    Returns
    -------
    bool
        True if ``path`` is ``parent`` or is beneath it.
    """
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def has_symlink_component(root: Path, path: Path) -> bool:
    """
    Check whether a path or any component beneath the root is a symlink.

    Parameters
    ----------
    root : pathlib.Path
        Resolved project root.
    path : pathlib.Path
        Path to inspect.

    Returns
    -------
    bool
        True if a component from ``root`` through ``path`` is a symlink.

    Raises
    ------
    ValueError
        If ``path`` is not beneath ``root``.
    """
    try:
        relative_path = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Path is outside the project root: {path}") from error

    current = root
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            return True

    return False


def add_file(
    archive: zipfile.ZipFile,
    path: Path,
    root: Path,
    added: set,
) -> None:
    """
    Add a regular file to the ZIP once, skipping symlinks and non-files.

    Parameters
    ----------
    archive : zipfile.ZipFile
        ZIP archive being written.
    path : pathlib.Path
        File candidate.
    root : pathlib.Path
        Resolved project root.
    added : set
        Archive names already written.
    """
    if has_symlink_component(root, path):
        logger.info("Skipping symlink path: %s", path)
        return

    if not path.is_file():
        return

    archive_name = path.relative_to(root).as_posix()
    if archive_name not in added:
        archive.write(path, arcname=archive_name)
        added.add(archive_name)
        logger.debug("Added file: %s", archive_name)


def add_folder(
    archive: zipfile.ZipFile,
    folder: Path,
    root: Path,
    added: set,
) -> None:
    """
    Add a selected folder recursively without following symlinks.

    Empty directories are retained in the archive.

    Parameters
    ----------
    archive : zipfile.ZipFile
        ZIP archive being written.
    folder : pathlib.Path
        Selected folder.
    root : pathlib.Path
        Resolved project root.
    added : set
        Archive names already written.

    Raises
    ------
    OSError
        If walking or reading the folder fails.
    """
    if has_symlink_component(root, folder):
        logger.info("Skipping symlink path: %s", folder)
        return

    if not folder.is_dir():
        logger.warning("Selected folder is missing or not a directory: %s", folder)
        return

    folder_name = folder.relative_to(root).as_posix().rstrip("/") + "/"
    if folder_name not in added:
        archive.writestr(folder_name, "")
        added.add(folder_name)

    def raise_walk_error(error):
        raise error

    for current, dirnames, filenames in os.walk(
        folder, followlinks=False, onerror=raise_walk_error
    ):
        current_path = Path(current)

        # Prune symlinked directories before os.walk descends into them.
        kept_dirs = []
        for dirname in dirnames:
            directory = current_path / dirname
            if directory.is_symlink():
                logger.info("Skipping symlinked directory: %s", directory)
                continue

            kept_dirs.append(dirname)
            directory_name = directory.relative_to(root).as_posix().rstrip("/") + "/"
            if directory_name not in added:
                archive.writestr(directory_name, "")
                added.add(directory_name)

        dirnames[:] = kept_dirs

        for filename in filenames:
            add_file(archive, current_path / filename, root, added)


def sha256_file(path: Path) -> str:
    """
    Calculate a file's SHA-256 digest without loading it all into memory.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksum_file(output: Path, digest: str) -> Path:
    """
    Atomically write a conventional SHA-256 sidecar file.

    The sidecar is named ``<output-name>.sha256`` and contains the digest
    followed by the output filename.

    Parameters
    ----------
    output : pathlib.Path
        ZIP file whose digest is recorded.
    digest : str
        Lowercase hexadecimal digest.

    Returns
    -------
    pathlib.Path
        Path to the written sidecar file.

    Raises
    ------
    OSError
        If the sidecar cannot be written.
    ValueError
        If the sidecar destination is a symlink or is not a regular file.
    """
    checksum_path = output.with_name(output.name + ".sha256")

    if checksum_path.is_symlink():
        raise ValueError(f"Refusing to overwrite checksum symlink: {checksum_path}")
    if checksum_path.exists() and not checksum_path.is_file():
        raise ValueError(f"Checksum destination is not a file: {checksum_path}")

    temp_path = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{checksum_path.name}.",
            suffix=".tmp",
            dir=str(checksum_path.parent),
        )
        temp_path = Path(temp_name)

        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as file:
            file.write(f"{digest}  {output.name}\n")

        os.replace(temp_path, checksum_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    return checksum_path


def prepare_output(requested_output: Path, root: Path) -> Path:
    """
    Validate and prepare an output path outside a project root.

    The output parent is resolved so that symlinked parent directories are
    handled consistently. The output itself may not be a symlink.

    Parameters
    ----------
    requested_output : pathlib.Path
        Requested destination path.
    root : pathlib.Path
        Resolved project or repository root.

    Returns
    -------
    pathlib.Path
        Normalized output path.

    Raises
    ------
    OSError
        If the output parent cannot be created.
    ValueError
        If the output is a symlink, is not a regular file, or is inside the
        project root.
    """
    requested_output = requested_output.expanduser().absolute()

    if requested_output.is_symlink():
        raise ValueError(f"Refusing to overwrite output symlink: {requested_output}")

    output = requested_output.parent.resolve() / requested_output.name

    if is_within(output, root):
        raise ValueError(f"ZIP output must be outside the project root: {root}")
    if output.exists() and not output.is_file():
        raise ValueError(f"ZIP destination is not a regular file: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def report_archive(output: Path, write_sha256: bool, archive_type: str) -> None:
    """
    Print the ZIP digest and optionally write its checksum sidecar.

    Parameters
    ----------
    output : pathlib.Path
        Completed ZIP path.
    write_sha256 : bool
        Whether to write a ``.sha256`` sidecar.
    archive_type : str
        Description used in the completion log.
    """
    digest = sha256_file(output)
    print(f"SHA-256: {digest}  {output.name}")
    logger.info("Created %s: %s", archive_type, output)

    if write_sha256:
        checksum_path = write_checksum_file(output, digest)
        logger.info("Created checksum file: %s", checksum_path)


def create_zip(root: Path, requested_output: Path, write_sha256: bool) -> None:
    """
    Create a ZIP from the configured folders and files.

    Parameters
    ----------
    root : pathlib.Path
        Project root.
    requested_output : pathlib.Path
        Requested ZIP destination.
    write_sha256 : bool
        Whether to write a ``.sha256`` sidecar.

    Raises
    ------
    OSError
        If the project cannot be read or the ZIP cannot be written.
    ValueError
        If the root or output path is invalid, or selections are unsafe.
    """
    root = root.expanduser().resolve()

    if not root.is_dir():
        raise NotADirectoryError(f"Project root is not a directory: {root}")
    if root == Path(root.anchor):
        raise ValueError(f"Refusing to use a filesystem root: {root}")

    validate_selections()
    output = prepare_output(requested_output, root)

    temp_path = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=str(output.parent),
        )
        os.close(fd)
        temp_path = Path(temp_name)

        added = set()
        with zipfile.ZipFile(
            temp_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for name in FOLDERS:
                add_folder(archive, root / name, root, added)

            for name in FILES:
                path = root / name
                if has_symlink_component(root, path):
                    logger.info("Skipping symlink path: %s", path)
                elif not path.is_file():
                    logger.warning("Selected file is missing or not a file: %s", path)
                else:
                    add_file(archive, path, root, added)

        # Replace the destination only after the ZIP has been completed.
        os.replace(temp_path, output)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    report_archive(output, write_sha256, "ZIP")


def find_git_root(path: Path) -> Path:
    """
    Find the Git repository root containing a path.

    Parameters
    ----------
    path : pathlib.Path
        Path inside the repository.

    Returns
    -------
    pathlib.Path
        Resolved repository root.

    Raises
    ------
    subprocess.CalledProcessError
        If Git is unavailable or the path is not in a repository.
    """
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def git_symlink_paths(repo: Path, commit: str) -> set:
    """
    Get tracked symlink paths from a Git commit's tree.

    Parameters
    ----------
    repo : pathlib.Path
        Resolved repository root.
    commit : str
        Fully resolved commit ID.

    Returns
    -------
    set
        Repository-relative symlink paths decoded with ``surrogateescape``.

    Raises
    ------
    subprocess.CalledProcessError
        If Git cannot read the commit tree.
    """
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            commit,
        ],
        check=True,
        capture_output=True,
    )

    symlinks = set()
    for record in result.stdout.split(b"\0"):
        if not record:
            continue

        metadata, raw_path = record.split(b"\t", 1)
        mode = metadata.split(b" ", 1)[0]
        if mode == b"120000":  # Git tree mode for a symbolic link
            symlinks.add(raw_path.decode("utf-8", errors="surrogateescape"))

    return symlinks


def create_git_archive(
    repo: Path,
    requested_output: Path,
    git_ref: str,
    write_sha256: bool,
) -> None:
    """
    Archive a Git commit, excluding symlink entries.

    Git's ``export-ignore`` attributes are honored by ``git archive``. The
    working tree is not modified.

    Parameters
    ----------
    repo : pathlib.Path
        Repository root.
    requested_output : pathlib.Path
        Requested ZIP destination.
    git_ref : str
        Commit, branch, or tag to archive.
    write_sha256 : bool
        Whether to write a ``.sha256`` sidecar.

    Raises
    ------
    OSError
        If temporary files or the output cannot be written.
    ValueError
        If the repository or output path is invalid.
    zipfile.BadZipFile
        If Git produces an invalid ZIP archive.
    subprocess.CalledProcessError
        If a Git command fails.
    """
    repo = repo.resolve()

    if not repo.is_dir():
        raise NotADirectoryError(f"Repository root is not a directory: {repo}")
    if repo == Path(repo.anchor):
        raise ValueError(f"Refusing to use a filesystem root: {repo}")

    # Resolve the ref to a commit ID before passing it to other Git commands.
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{git_ref}^{{commit}}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if not commit:
        raise ValueError(f"Git returned an empty commit ID for ref {git_ref!r}")

    output = prepare_output(requested_output, repo)
    symlinks = git_symlink_paths(repo, commit)

    raw_path = None
    filtered_path = None
    try:
        fd, raw_name = tempfile.mkstemp(
            prefix=f".{output.name}.git-",
            suffix=".tmp",
            dir=str(output.parent),
        )
        os.close(fd)
        raw_path = Path(raw_name)

        fd, filtered_name = tempfile.mkstemp(
            prefix=f".{output.name}.filtered-",
            suffix=".tmp",
            dir=str(output.parent),
        )
        os.close(fd)
        filtered_path = Path(filtered_name)

        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "archive",
                "--format=zip",
                f"--output={raw_path}",
                commit,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # Git's tree identifies symlink names. ZIP Unix mode provides a second
        # check, including when a filename cannot be decoded identically.
        with zipfile.ZipFile(raw_path, "r") as source:
            with zipfile.ZipFile(filtered_path, "w") as destination:
                destination.comment = source.comment

                for info in source.infolist():
                    archive_name = info.filename.rstrip("/")
                    unix_mode = (info.external_attr >> 16) & 0xFFFF
                    is_symlink = (
                        archive_name in symlinks or stat.S_ISLNK(unix_mode)
                    )

                    if is_symlink:
                        logger.info("Skipping Git symlink: %s", archive_name)
                        continue

                    if info.is_dir():
                        destination.writestr(info, b"")
                    else:
                        with source.open(info, "r") as src:
                            with destination.open(info, "w") as dst:
                                shutil.copyfileobj(src, dst)

        # Replace the destination only after the archive was built successfully.
        os.replace(filtered_path, output)
        filtered_path = None

    finally:
        for temp_path in (raw_path, filtered_path):
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    report_archive(output, write_sha256, "Git archive")


def main() -> int:
    """
    Parse command-line arguments and create the requested ZIP.

    Returns
    -------
    int
        Process exit status: zero on success, nonzero on failure.
    """
    default_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="ZIP selected project paths or a Git commit."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Project root (default: {default_root})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="ZIP destination (default: ZIP beside the project folder)",
    )
    parser.add_argument(
        "--write-sha256",
        action="store_true",
        help="Also write OUTPUT.sha256 beside the ZIP.",
    )
    parser.add_argument(
        "--git-archive",
        action="store_true",
        help="Archive a Git commit's tracked contents instead of FOLDERS/FILES.",
    )
    parser.add_argument(
        "--git-ref",
        default="HEAD",
        help="Commit, branch, or tag to archive in Git mode (default: HEAD).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log each file added to the ZIP.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        root = args.root.expanduser().resolve()
        archive_root = find_git_root(root) if args.git_archive else root
        output = args.output or archive_root.parent / f"{archive_root.name}.zip"

        if args.git_archive:
            create_git_archive(
                archive_root,
                output,
                args.git_ref,
                args.write_sha256,
            )
        else:
            create_zip(root, output, args.write_sha256)

    except (
        OSError,
        ValueError,
        zipfile.BadZipFile,
        subprocess.CalledProcessError,
    ) as error:
        message = str(error)
        stderr = getattr(error, "stderr", None)

        if stderr:
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            message = f"{message}: {stderr.strip()}"

        logger.error("%s", message)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
