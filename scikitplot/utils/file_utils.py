# pylint: disable=broad-exception-caught
# pylint: disable=broad-exception-raised
# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-lines
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements

"""file_utils."""

# import errno
# import uuid
import atexit as _atexit
import codecs as _codecs
import concurrent as _concurrent
import contextlib as _contextlib
import fnmatch as _fnmatch
import gzip as _gzip
import json as _json
import math as _math
import os as _os
import pathlib as _pathlib
import posixpath as _posixpath
import shutil as _shutil
import stat as _stat

# import subprocess
import sys as _sys
import tarfile as _tarfile
import tempfile as _tempfile

# import textwrap
import time as _time
import urllib as _urllib
from collections.abc import Generator, Iterable, Iterator
from dataclasses import dataclass
from typing import Optional, Union

import requests as _requests

# import logging
from .. import logger as _logger
from ..environment_variables import (
    _SKPLT_MPD_NUM_RETRIES,
    _SKPLT_MPD_RETRY_INTERVAL_SECONDS,
    # SKPLT_DOWNLOAD_CHUNK_TIMEOUT,
    SKPLT_ENABLE_ARTIFACTS_PROGRESS_BAR,
)
from ..exceptions import ScikitplotException
from ..experimental._entities import FileInfo

# from ..utils import download_cloud_file_chunk
from ..utils.os import is_windows
from ..utils.process import cache_return_value_per_process
from ..utils.request_utils import (
    augmented_raise_for_status,
    cloud_storage_http_request,
    download_chunk,
)

# from ..protos.databricks_artifacts_pb2 import ArtifactCredentialType
# from ..protos.databricks_pb2 import INVALID_PARAMETER_VALUE
# from ..utils.databricks_utils import (
#     get_databricks_local_temp_dir,
#     get_databricks_nfs_temp_dir,
# )

INVALID_PARAMETER_VALUE = 0
ENCODING = "utf-8"
_PROGRESS_BAR_DISPLAY_THRESHOLD = 500_000_000  # 500 MB


# @dataclass
# class FileInfo:
#     """
#     Metadata about a file or directory.
#     """

#     path: str                 # Relative path
#     is_dir: bool              # Is it a directory?
#     file_size: Optional[int]  # Size in bytes (None for directories)
#     modified_time: float      # Last modified time (Unix timestamp)
#     permissions: str          # String like 'rwxr-xr--'
#     absolute_path: str        # Full resolved path


def get_file_info(
    path: str,
    rel_path: str,
) -> FileInfo:
    """
    Retrieve metadata about a file or directory.

    Parameters
    ----------
    path : str
        Absolute path to the file or directory.
    rel_path : str
        Path relative to some root directory (e.g., for display or storage reference).

    Returns
    -------
    FileInfo
        Metadata including size, type, permissions, and modification time.
        A FileInfo object containing the relative path, whether the target is a directory,
        and file size (if applicable).

    Raises
    ------
    FileNotFoundError
        If the given path does not exist.

    Examples
    --------
    >>> get_file_info("/home/user/data.csv", "data.csv")
    FileInfo(path='data.csv', is_dir=False, size=1284)
    """
    if not _os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    def _permissions(mode: int) -> str:
        """Convert permission bits to symbolic notation."""
        return _stat.filemode(mode)

    is_dir_ = _os.path.isdir(path)
    stat_result = _os.stat(path)  # _os.path.getsize(path)

    return FileInfo(
        path=rel_path,
        is_dir=is_dir_,
        file_size=None if is_dir_ else stat_result.st_size,
        # Not implemented
        # modified_time=stat_result.st_mtime,
        # permissions=_permissions(stat_result.st_mode),
        # absolute_path=str(_pathlib.Path(path).resolve()),
    )


######################################################################
## basic path
######################################################################


def contains_path_separator(
    path: Union[str, _os.PathLike],
) -> bool:
    r"""
    Check if the given path contains any path separator.

    Parameters
    ----------
    path : str or _os.PathLike
        The path to check.

    Returns
    -------
    bool
        True if the path contains a path separator (`_os.path.sep` or `_os.path.altsep`),
        False otherwise.

    Notes
    -----
    - `_os.path.sep` is the primary path separator for the current OS
      (e.g., '/' on Unix, '\\' on Windows).
    - `_os.path.altsep` is an alternative separator if one exists
      (e.g., '/' is the altsep on Windows).

    Examples
    --------
    >>> contains_path_separator("folder/file.txt")
    True

    >>> contains_path_separator("file.txt")
    False
    """
    return any(
        (sep in path) for sep in (_os.path.sep, _os.path.altsep) if sep is not None
    )


def contains_percent(
    path: Union[str, bytes],
) -> bool:
    """
    Check if the given path contains a percent character (`%`).

    Parameters
    ----------
    path : str or bytes
        The path or URI to check.

    Returns
    -------
    bool
        True if the path contains a percent (`%`), False otherwise.

    Notes
    -----
    - This is often used to detect percent-encoded URIs (e.g., `%20` for space).
    - Useful for deciding whether to decode a URI using `urllib.parse.unquote`.

    Examples
    --------
    >>> contains_percent("my%20file.txt")
    True

    >>> contains_percent("data/file.txt")
    False
    """
    return b"%" in path if isinstance(path, bytes) else "%" in path


def exists(
    name: Union[str, _pathlib.Path],
) -> bool:
    """
    Check whether a given file or directory exists.

    Parameters
    ----------
    name : str or _pathlib.Path
        Path to the file or directory.

    Returns
    -------
    bool
        True if the path exists, False otherwise.

    Examples
    --------
    >>> exists("data/file.txt")
    True
    >>> exists("/some/missing/folder")
    False
    """
    return _os.path.exists(str(name))


def is_file(
    name: Union[str, _pathlib.Path],
) -> bool:
    """
    Check if the given path is a file.

    Parameters
    ----------
    name : str or _pathlib.Path
        Path to check.

    Returns
    -------
    bool
        True if the path exists and is a file, False otherwise.

    Examples
    --------
    >>> is_file("example.txt")
    True
    >>> is_file("some_directory/")
    False
    """
    return _os.path.isfile(str(name))


def is_dir(
    path: Union[str, _pathlib.Path],
) -> bool:
    """
    Check if the given path is an existing directory.

    Parameters
    ----------
    path : str or _pathlib.Path
        Path to check.

    Returns
    -------
    bool
        True if path exists and is a directory, False otherwise.

    Examples
    --------
    >>> is_dir("/tmp")
    True
    >>> is_dir("/tmp/somefile.txt")
    False
    """
    return _os.path.isdir(str(path))


def abspath(
    path: str = ".",
) -> _pathlib.Path:
    """
    Resolve a given relative or user path to an absolute `Path` object.

    This function expands environment variables, user shortcuts like `~`,
    and resolves symbolic links or relative references.

    Parameters
    ----------
    path : str, optional
        A relative or absolute file path. If empty, resolves to the current working directory.

    Returns
    -------
    _pathlib.Path
        The fully resolved absolute path as a `Path` object.

    Examples
    --------
    >>> abspath("~/my_project/.env")
    PosixPath('/home/user/my_project/.env')

    >>> abspath()
    PosixPath('/current/working/directory')
    """
    # Stay in the current directory
    # current_path = _os.path.join(_os.getcwd(), _os.curdir)
    # path = _os.path.abspath(_os.path.join(_os.getcwd(), path))
    # path = _os.path.abspath(_os.path.expanduser(path))
    # path = Path.cwd() / path or f"{_os.getcwd()}/.env"
    # If you don't want symlinks to be resolved (preserve exact structure):
    # Replace .resolve() with .absolute().
    path = _pathlib.Path(path).expanduser().resolve()


def relpath(
    root_path: Union[str, _pathlib.Path],
    target_path: Union[str, _pathlib.Path],
) -> str:
    """
    Return the part of `target_path` that is relative to `root_path`.

    Parameters
    ----------
    root_path : str or _pathlib.Path
        The base or root directory path.

    target_path : str or _pathlib.Path
        The target path from which the relative path is calculated.

    Returns
    -------
    str
        The relative path from `root_path` to `target_path`.

    Raises
    ------
    ValueError
        If the target path is not under the root path.

    Notes
    -----
    - Uses `_pathlib.Path.relative_to()` which avoids issues with simple prefix matching.
    - Symbolic links are resolved by default to avoid false mismatches.
    - `~` is expanded using `expanduser()`.

    Examples
    --------
    >>> relpath("/home/user/projects", "/home/user/projects/app/main.py")
    'app/main.py'
    """
    # if len(root_path) > len(target_path):
    #     raise Exception(f"Root path '{root_path}' longer than target path '{target_path}'")
    # common_prefix = _os.path.commonprefix([root_path, target_path])
    # return _os.path.relpath(target_path, common_prefix)
    root = _pathlib.Path(root_path).expanduser().resolve()
    target = _pathlib.Path(target_path).expanduser().resolve()

    try:
        return str(target.relative_to(root))
    except ValueError as e:
        raise ValueError(
            f"Target path '{target}' is not located under root path '{root}'{e}"
        ) from e


def parent(
    path: Union[str, _pathlib.Path],
) -> str:
    """
    Return the absolute path to the parent directory of the given path,
    with support for `~`, symlink resolution, and normalized case.

    Parameters
    ----------
    path : str or _pathlib.Path or _os.PathLike
        A file or directory path. May include `~` to indicate the home directory.

    Returns
    -------
    str
        The resolved, absolute path to the parent directory.

    Examples
    --------
    >>> parent("~/projects/myrepo")
    '/home/username/projects'

    >>> parent("my_folder/sub")
    '/absolute/path/to/my_folder'

    Notes
    -----
    - Uses `_pathlib.Path` for modern path handling.
    - `expanduser()` supports `~` expansion to home directory.
    - `resolve()` resolves symlinks and returns an absolute path.
    - On Windows, the returned path is case-normalized using `.casefold()`
      for comparison consistency.
    - `_os.pardir` is a constant string representing the parent directory (typically `'..'`)
      and is used here to navigate one level up in the directory tree.
    """  # noqa: D205
    # expanded_path = _os.path.expanduser(path)
    # Move to the parent of the current directory
    # dirname = _os.path.dirname(path)
    # return _os.path.abspath(_os.path.join(expanded_path, _os.pardir))
    path_obj = _pathlib.Path(path).expanduser().resolve(strict=False)
    dirname = path_obj.parent

    # Optional: normalize case only on Windows (mostly relevant for internal comparison)
    if dirname.drive and dirname.anchor:
        return str(dirname)
    return str(dirname)


get_parent_dir = parent

######################################################################
## convert path to uri or vice versa
######################################################################


def relative_path_to_artifact_path(
    path: Union[str, _os.PathLike],
) -> str:
    r"""
    Convert a relative file path to a POSIX-style artifact path.

    This is useful for converting local relative file paths (especially on Windows)
    into a consistent POSIX-style format for storing or logging artifact paths in
    machine learning pipelines, MLflow, or cloud storage systems.

    Parameters
    ----------
    path : str or _os.PathLike
        A relative file path string or Path-like object.

    Returns
    -------
    str
        A POSIX-style artifact path with forward slashes (e.g., 'some/path/file.txt').

    Raises
    ------
    ValueError
        If an absolute path is passed instead of a relative path.

    Notes
    -----
    - On POSIX systems (Linux/macOS), the path is returned unchanged.
    - On Windows systems, it transforms backslashes to forward slashes and
      escapes special characters via `urllib.request.pathname2url`.
    - This function assumes that the input is a valid relative file path.
    - Use this when you want consistent storage or logging formats across platforms.

    Examples
    --------
    >>> relative_path_to_artifact_path("models/output.pkl")
    'models/output.pkl'

    >>> relative_path_to_artifact_path("subdir\\model.joblib")
    'subdir/model.joblib'

    >>> relative_path_to_artifact_path("C:/abs/path/file.txt")
    Traceback (most recent call last):
        ...
    ValueError: This method only works with relative paths.
    """
    # Normalize input to string
    path = str(path)
    # Check if path is absolute
    if _os.path.isabs(path):
        raise ValueError("This method only works with relative paths.")
    # On POSIX, return unchanged
    if _os.path == _posixpath:
        return path  # Already POSIX-style
    # On Windows, convert path to URL format then unquote to get posix style
    # with escaped chars handled
    return _urllib.parse.unquote(_urllib.request.pathname2url(path))


def path_to_local_file_uri(
    path: Union[str, _pathlib.Path],
) -> str:
    r"""
    Convert a local filesystem path to a file URI.

    This is useful for representing local paths in a URI format
    (e.g., for logging, references in web UIs, or distributed systems).

    Parameters
    ----------
    path : str or _pathlib.Path
        The local file path to convert. Can be relative or absolute.

    Returns
    -------
    str
        A `file://` URI corresponding to the absolute local path.

    Notes
    -----
    - This function uses `Path.as_uri()` which works reliably on all platforms.
    - On Windows, it handles drive letters correctly (e.g., `file:///C:/...`).
    - Useful in ML metadata tracking systems, documentation tools, etc.

    Examples
    --------
    >>> path_to_local_file_uri("data/output.txt")
    'file:///home/user/project/data/output.txt'

    >>> path_to_local_file_uri("/tmp/logs.txt")
    'file:///tmp/logs.txt'

    >>> path_to_local_file_uri("C:\\logs\\run.log")
    'file:///C:/logs/run.log'
    """
    return _pathlib.Path(_os.path.abspath(path)).as_uri()


def local_file_uri_to_path(
    uri: str,
) -> str:
    r"""
    Convert a local file URI to a local filesystem path.

    If the URI does not start with the "file:" scheme, it returns the input unchanged.

    Parameters
    ----------
    uri : str
        The file URI to convert.

    Returns
    -------
    str
        The corresponding local filesystem path.

    Notes
    -----
    - Handles UNC paths on Windows by preserving the server name.
    - Uses urllib to correctly decode URL-encoded characters.

    Example
    -------
    >>> local_file_uri_to_path("file:///C:/Users/user/file.txt")
    'C:\\Users\\user\\file.txt'  # on Windows

    >>> local_file_uri_to_path("file:///home/user/file.txt")
    '/home/user/file.txt'  # on Unix-like OS

    >>> local_file_uri_to_path("/home/user/file.txt")
    '/home/user/file.txt'  # no scheme, returned as-is
    """
    path = uri
    if uri.startswith("file:"):
        parsed = _urllib.parse.urlparse(uri)
        path = parsed.path
        # Fix for retaining server name in UNC path.
        # For Windows UNC paths, include the server/network share name
        if is_windows() and parsed.netloc:
            # UNC path, e.g. file://server/share/file.txt
            return _urllib.request.url2pathname(rf"\\{parsed.netloc}{path}")
    return _urllib.request.url2pathname(path)


def local_file_uri_to_path2(
    uri: str,
) -> _pathlib.Path:
    """
    Convert a file URI to a local filesystem path.

    Parameters
    ----------
    uri : str
        The file URI to convert (e.g., 'file:///home/user/file.txt').

    Returns
    -------
    _pathlib.Path
        The corresponding local file system path.

    Notes
    -----
    - Works cross-platform.
    - Handles percent-encoded characters in the URI.
    - Input URI must start with 'file://'.
    - Raises ValueError if the URI scheme is not 'file'.

    Examples
    --------
    >>> local_file_uri_to_path2("file:///home/user/data.txt")
    PosixPath("/home/user/data.txt")

    >>> local_file_uri_to_path2("file:///C:/Users/Name/Documents/file.txt")
    WindowsPath("C:/Users/Name/Documents/file.txt")
    """
    parsed = _urllib.parse.urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"URI scheme must be 'file', got '{parsed.scheme}'")
    path = _urllib.parse.unquote(parsed.path)
    # On Windows, urlparse includes an initial '/' before drive letter, remove if needed
    if path.startswith("/") and _pathlib.Path(path[1:]).drive:
        path = path[1:]
    return _pathlib.Path(path)


def get_local_path_or_none(
    path_or_uri: str,
) -> Optional[str]:
    r"""
    Return the local filesystem path if the input is a local path or file URI,
    otherwise return None.

    Parameters
    ----------
    path_or_uri : str
        The path or URI string to check.

    Returns
    -------
    Optional[str]
        The corresponding local path if input is local or a local file URI,
        None if input is a non-local URI.

    Notes
    -----
    - Local path means no URI scheme (e.g., "/home/user/file.txt") or
      a 'file' URI without a network location (e.g., "file:///home/user/file.txt").

    Example
    -------
    >>> get_local_path_or_none("/home/user/file.txt")
    '/home/user/file.txt'

    >>> get_local_path_or_none("file:///C:/Users/user/file.txt")
    'C:\\Users\\user\\file.txt'  # on Windows

    >>> get_local_path_or_none("https://example.com/file.txt")
    None
    """  # noqa: D205
    parsed_uri = _urllib.parse.urlparse(path_or_uri)
    if (
        # No scheme means it's a local path
        len(parsed_uri.scheme) == 0
        or
        # Local file URI without network location
        (parsed_uri.scheme == "file" and len(parsed_uri.netloc) == 0)
    ):
        return local_file_uri_to_path(path_or_uri)
    return None


def path_to_local_sqlite_uri(
    path: Union[str, _os.PathLike],
) -> str:
    r"""
    Convert a local filesystem path to a SQLite URI suitable for SQLAlchemy or other tools.

    Parameters
    ----------
    path : Union[str, _os.PathLike]
        The local filesystem path to convert.

    Returns
    -------
    str
        SQLite URI string.

    Notes
    -----
    - On Unix-like systems, the prefix is 'sqlite:///' (three slashes).
    - On Windows, the URI prefix is 'sqlite:///' followed by the absolute path with forward slashes.
    - On Windows32, the URI prefix is 'sqlite://' (two slashes).
    - The path is converted to a posix-style URL path and absolute.

    Example
    -------
    >>> path_to_local_sqlite_uri("mydb.sqlite")
    'sqlite:///home/user/mydb.sqlite'  # on Unix

    >>> path_to_local_sqlite_uri("C:\\Users\\user\\mydb.sqlite")
    'sqlite:///C:/Users/user/mydb.sqlite'  # on Windows

    >>> path_to_local_sqlite_uri("C:\\Users\\user\\mydb.sqlite")
    'sqlite://C:/Users/user/mydb.sqlite'  # on Windows32
    """
    abs_path = _pathlib.Path(path).expanduser().resolve()
    # Convert path to a URL-encoded path (with forward slashes)
    posix_path = _posixpath.abspath(_urllib.request.pathname2url(abs_path))
    # SQLite URI requires three slashes after sqlite:
    # This is consistent on both Windows and Unix.
    prefix = "sqlite://" if _sys.platform == "win32" else "sqlite:///"
    return prefix + posix_path


######################################################################
## list path
######################################################################


def list_all(
    root: Union[str, _pathlib.Path],
    full_path: bool = False,
    filter_func: "callable[[_pathlib.Path], bool]" = lambda x: True,
    only_files: bool = False,
    only_dirs: bool = False,
    exclude_symlinks: bool = False,
    sort: "Union[bool, callable[[_pathlib.Path], any]]" = False,
    return_path_objects: bool = False,
) -> list[Union[str, _pathlib.Path]]:
    """
    List all entries in a directory that match a filter function.

    Parameters
    ----------
    root : str or _pathlib.Path
        Path to the directory whose immediate contents are to be listed.
    filter_func : Callable[[_pathlib.Path], bool], optional
        Function to filter entries. Defaults to include all.
    full_path : bool, optional
        If True, return absolute paths (as strings) to each item;
        otherwise, return just the filenames.
    only_files : bool, optional
        If True, only return regular files (symlinks to files included).
    only_dirs : bool, optional
        If True, only return directories (symlinks to directories included).
    exclude_symlinks : bool, optional
        If True, exclude symbolic links entirely.
    sort : bool or Callable[[_pathlib.Path], Any], optional
        If True, sort alphabetically by name.
        If a callable is provided, use it as the sort key function.
    return_path_objects : bool, optional
        If True, return _pathlib.Path objects instead of strings.

    Returns
    -------
    List[Union[str, _pathlib.Path]]
        List of files or directories under `root` matching the criteria.

    Raises
    ------
    ValueError
        If the root path is not a valid directory.

    Notes
    -----
    - Non-recursive; lists items directly under `root`.
    - Symlinks are followed in `is_file` and `is_dir` checks unless excluded.
    - If both `only_files` and `only_dirs` are True, `only_files` takes precedence.

    See Also
    --------
    _pathlib.Path.glob, _os.walk

    Examples
    --------
    >>> list_all("/tmp", lambda p: p.endswith(".log"))
    ['example.log']

    >>> list_all("/tmp", _os.path.isfile, full_path=True)
    ['/tmp/example.log', '/tmp/test.txt']

    >>> list_all("my_folder", only_files=True)
    ['file1.txt', 'file2.csv']

    >>> list_all("my_folder", full_path=True, only_dirs=True)
    ['/abs/path/my_folder/data', '/abs/path/my_folder/images']
    """
    # if not is_dir(root):
    #     raise Exception(f"Invalid parent directory '{root}'")

    # entries = _os.listdir(root)
    # matches = [x for x in entries if filter_func(_os.path.join(root, x))]
    # return [_os.path.join(root, m) for m in matches] if full_path else matches
    root_path = _pathlib.Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ValueError(f"Invalid directory: {root}")

    entries = list(root_path.iterdir())

    # Filter out symlinks if requested
    if exclude_symlinks:
        entries = [p for p in entries if not p.is_symlink()]

    # Filter for only files or only dirs (files prioritized if both True)
    if only_files:
        entries = [p for p in entries if p.is_file()]
    elif only_dirs:
        entries = [p for p in entries if p.is_dir()]

    # Apply user filter function
    entries = [p for p in entries if filter_func(p)]

    # Sort if requested
    if sort:
        if callable(sort):
            entries.sort(key=sort)
        else:
            entries.sort(key=lambda p: p.name)

    # Format return values
    if return_path_objects:
        return entries
    return [str(p) if full_path else p.name for p in entries]


def list_subdirs(
    dir_name: Union[str, _pathlib.Path],
    full_path: bool = False,
    exclude_symlinks: bool = False,
    sort: bool = False,
) -> list[Union[str, _pathlib.Path]]:
    """
    List all immediate subdirectories (non-recursive) under the given directory.

    Equivalent to the Unix shell command:
        `find $dir_name -maxdepth 1 -type d`

    Parameters
    ----------
    dir_name : str or _pathlib.Path
        Path to the directory to search for subdirectories.
    full_path : bool, optional
        If True, returns full `Path` objects. If False, returns just names.
    exclude_symlinks : bool, optional
        If True, excludes symbolic links to directories.
    sort : bool, optional
        If True, returns results sorted alphabetically by name.

    Returns
    -------
    List[Union[str, _pathlib.Path]]
        List of immediate subdirectories as `Path` or `str`.

    Raises
    ------
    ValueError
        If the given path is not a valid directory.

    Notes
    -----
    - This function performs a shallow search (non-recursive).
    - Uses `Path.iterdir()` for efficient traversal.

    Examples
    --------
    >>> list_subdirs("/tmp")
    ['data', 'logs']

    >>> list_subdirs("/tmp", full_path=True)
    [PosixPath('/tmp/data'), PosixPath('/tmp/logs')]

    >>> list_subdirs("/tmp", exclude_symlinks=True, sort=True)
    ['data', 'logs']
    """
    # return list_all(dir_name, _os.path.isdir, full_path)
    # Normalize and resolve the root path
    root = _pathlib.Path(dir_name).expanduser().resolve()

    # Validate that the root path is a directory
    if not root.is_dir():
        raise ValueError(f"Invalid directory: {root}")

    # List all entries that are directories (including symlinks to directories)
    subdirs = [
        p
        for p in root.iterdir()
        if p.is_dir() and (not exclude_symlinks or not p.is_symlink())
    ]
    if sort:
        subdirs.sort(key=lambda p: p.name)
    # Return full paths or just names based on the `full_path` flag
    return [p if full_path else p.name for p in subdirs]


def list_files(
    dir_name: Union[str, _pathlib.Path],
    full_path: bool = False,
    exclude_symlinks: bool = False,
    sort: bool = False,
) -> list[Union[str, _pathlib.Path]]:
    """
    List all immediate files (non-recursive) under the specified directory.

    Equivalent to the Unix shell command:
        `find $dir_name -maxdepth 1 -type f`

    Parameters
    ----------
    dir_name : str or _pathlib.Path
        Path to the directory in which to look for files.
    full_path : bool, optional
        If True, returns absolute Path objects. If False, returns just file names.
    exclude_symlinks : bool, optional
        If True, symbolic links to files are excluded.
    sort : bool, optional
        If True, returns results sorted alphabetically by name.

    Returns
    -------
    List[Union[str, _pathlib.Path]]
        A list of file names or full paths, depending on `full_path`.

    Raises
    ------
    ValueError
        If the provided path is not a valid directory.

    Notes
    -----
    - This is a shallow search; subdirectories are not traversed.
    - Uses `_pathlib.Path.iterdir()` for performance and simplicity.

    Examples
    --------
    >>> list_files("/tmp")
    ['example.txt', 'log.csv']

    >>> list_files("/tmp", full_path=True)
    [PosixPath('/tmp/example.txt'), PosixPath('/tmp/log.csv')]

    >>> list_files("/tmp", exclude_symlinks=True, sort=True)
    ['example.txt', 'log.csv']
    """
    # return list_all(dir_name, _os.path.isfile, full_path)
    # Normalize and resolve the input directory path
    root = _pathlib.Path(dir_name).expanduser().resolve()

    # Ensure the input is a valid directory
    if not root.is_dir():
        raise ValueError(f"Invalid directory: {root}")

    # Collect all files (including symlinks that point to files)
    files = [
        p
        for p in root.iterdir()
        if p.is_file() and (not exclude_symlinks or not p.is_symlink())
    ]
    if sort:
        files.sort(key=lambda p: p.name)
    # Return as full paths or names depending on `full_path` flag
    return [p if full_path else p.name for p in files]


######################################################################
## create or remove path
######################################################################


@_contextlib.contextmanager
def chdir(path: str) -> Generator[None, None, None]:
    """
    Temporarily change the current working directory to the specified path.

    Parameters
    ----------
    path : str
        The path to use as the temporary working directory.

    Yields
    ------
    None
        Control returns to the caller with the working directory set.

    Notes
    -----
    Restores the original working directory after the context exits,
    even if an exception is raised.

    Examples
    --------
    >>> with chdir("/tmp"):
    ...     print(_os.getcwd())
    """
    # original_dir = _os.getcwd()
    original_dir = _pathlib.Path.cwd()
    try:
        _os.chdir(_pathlib.Path(path).expanduser().resolve())
        yield
    finally:
        _os.chdir(original_dir)


def find(
    root: Union[str, _pathlib.Path],
    name: str,
    full_path: bool = False,
) -> list[Union[str, _pathlib.Path]]:
    """
    Search for an entry with the given name directly under the specified root directory.

    This is equivalent to:
        `find $root -name "$name" -maxdepth 1` in Unix.

    Parameters
    ----------
    root : str or _pathlib.Path
        The directory path where the search is performed.
    name : str
        The exact name (not pattern) of the file or directory to find.
    full_path : bool, optional
        If True, returns full absolute paths as `Path` objects.
        If False, returns only the matching names.

    Returns
    -------
    list of str or _pathlib.Path
        A list of matching names or full paths under the specified root.

    Raises
    ------
    ValueError
        If `root` is not a valid existing directory.

    Examples
    --------
    >>> find("/tmp", "log.txt")
    ['log.txt']

    >>> find("/tmp", "log.txt", full_path=True)
    [PosixPath('/tmp/log.txt')]
    """
    # path_name = _os.path.join(root, name)
    # return list_all(root, lambda x: x == path_name, full_path)
    # Normalize root path
    root_path = _pathlib.Path(root).expanduser().resolve()

    # Validate that root exists and is a directory
    if not root_path.is_dir():
        raise ValueError(f"Invalid directory: {root_path}")

    # Iterate over entries directly under root
    matches = [entry for entry in root_path.iterdir() if entry.name == name]

    # Return full path or just names based on flag
    return [entry if full_path else entry.name for entry in matches]


def mkdir(
    root: Union[str, _pathlib.Path],
    name: Optional[str] = None,
) -> _pathlib.Path:
    """
    Create a directory at `root/name` if `name` is provided, or just `root` otherwise.

    Parameters
    ----------
    root : str or _pathlib.Path
        The base or parent directory where the new directory should be created.
    name : str, optional
        Name of the subdirectory to create. If None, `root` itself is created.

    Returns
    -------
    _pathlib.Path
        A `Path` object pointing to the created (or pre-existing) directory.

    Raises
    ------
    OSError
        If the path cannot be created and does not already exist as a directory.

    Examples
    --------
    >>> mkdir("/tmp", "my_logs")
    PosixPath('/tmp/my_logs')

    >>> mkdir("~/logs")  # Expands user home
    PosixPath('/home/user/logs')

    Notes
    -----
    - Uses `mkdir(parents=True, exist_ok=True)` to support nested creation.
    - Resolves symbolic links and expands `~` via `expanduser().resolve()`.
    - Does nothing if the directory already exists.
    """
    # target = _os.path.join(root, name) if name is not None else root
    # try:
    #     _os.makedirs(target, exist_ok=True)
    # except OSError as e:
    #     if e.errno != errno.EEXIST or not _os.path.isdir(target):
    #         raise e
    # return target
    # Normalize and combine paths using _pathlib
    target = _pathlib.Path(root).expanduser().resolve()
    if name is not None:
        target = target / name

    try:
        # Try to create the directory (including parents)
        target.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # If mkdir fails and the path is not a directory, re-raise
        if not target.is_dir():
            _logger.error(f"Failed to create directory: {target} — {e}")
            raise

    _logger.debug(f"Directory ensured: {target}")
    return target


def mkdir_parent(
    path: Union[str, _pathlib.Path],
) -> None:
    """
    Ensure that all parent directories for a given file path exist.

    Parameters
    ----------
    path : str or _pathlib.Path
        The file path whose parent directory tree will be created if missing.

    Returns
    -------
    None

    Notes
    -----
    - This function only creates the parent directories, not the file itself.
    - Existing directories are silently accepted (no-op).
    - Uses `expanduser()` to support tilde (`~`) and `resolve()` for absolute path resolution.
    - Safe for concurrent usage: uses `mkdir(..., exist_ok=True)`.

    Examples
    --------
    >>> mkdir_parent("/tmp/myfolder/data.json")
    # Ensures that /tmp/myfolder exists.

    >>> mkdir_parent("~/logs/app/output.log")
    # Ensures that ~/logs/app exists.
    """
    # dirname = _os.path.dirname(path)
    # if not _os.path.exists(dirname):
    #     _os.makedirs(dirname, exist_ok=True)
    try:
        # Resolve absolute path and get its parent directory
        dir_path = _pathlib.Path(path).expanduser().resolve().parent

        # Recursively create parent directories if they don't exist
        dir_path.mkdir(parents=True, exist_ok=True)

        _logger.debug(f"Ensured parent directory: {dir_path}")
    except Exception as e:
        _logger.error(f"Failed to ensure parent directory for {path}: {e}")
        raise


def mv(
    target: Union[str, _os.PathLike],
    new_parent: Union[str, _os.PathLike],
) -> None:
    """
    Move a file or directory into a new parent directory.

    Parameters
    ----------
    target : str or _os.PathLike
        Path to the file or directory to move.
    new_parent : str or _os.PathLike
        Path to the destination directory.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the `target` or `new_parent` path does not exist.
    NotADirectoryError
        If `new_parent` is not a directory.
    FileExistsError
        If a file or directory with the same name already exists at the destination.

    Examples
    --------
    >>> mv("data/file.txt", "archive/")
    >>> mv("~/downloads/myfolder", "/mnt/storage")

    Notes
    -----
    - Equivalent to: `mv target new_parent` in the Unix shell.
    - Symbolic links are moved as-is (not dereferenced).
    - Overwriting is not allowed by default. The function will raise if
      the destination path already exists.
    """
    target_path = _pathlib.Path(target).expanduser().resolve()
    new_parent_path = _pathlib.Path(new_parent).expanduser().resolve()

    # Validate source and destination
    if not target_path.exists():
        raise FileNotFoundError(f"Target not found: {target_path}")
    if not new_parent_path.exists():
        raise FileNotFoundError(f"Destination directory not found: {new_parent_path}")
    if not new_parent_path.is_dir():
        raise NotADirectoryError(f"Destination is not a directory: {new_parent_path}")

    destination_path = new_parent_path / target_path.name

    if destination_path.exists():
        raise FileExistsError(f"Destination already exists: {destination_path}")

    try:
        _shutil.move(str(target_path), str(new_parent_path))
        _logger.info(f"Moved: {target_path} -> {new_parent_path}")
    except Exception as e:
        _logger.error(f"Failed to move {target_path} to {new_parent_path}: {e}")
        raise


@_contextlib.contextmanager
def rmtree_on_error(
    path: Union[str, _os.PathLike],
    onerror: "Optional[callable[[Exception], None]]" = None,
) -> Generator[None, None, None]:
    """
    Context manager that attempts to clean up a file or directory at `path`
    if an exception is raised within the managed block.

    Parameters
    ----------
    path : str or _os.PathLike
        The file or directory path to remove if an error occurs.
    onerror : callable, optional
        A callback accepting the raised exception as input. Useful for
        custom logging, alerting, or diagnostics.

    Yields
    ------
    None
        Control is yielded to the caller's context block.

    Raises
    ------
    Exception
        Re-raises the original exception after attempting cleanup.

    Examples
    --------
    >>> with rmtree_on_error("/tmp/scratch"):
    ...     raise ValueError("Boom!")
    WARNING - Removed path after error: /tmp/scratch

    Notes
    -----
    - Converts `path` to `Path` internally using `_pathlib`.
    - Only attempts cleanup if an exception is raised.
    - Uses `unlink()` for files and `rmtree()` for directories.
    - Cleanup failure is logged but does not mask the original exception.
    """  # noqa: D205
    _path = _pathlib.Path(path)

    try:
        # Yield control to the context block
        yield
    except Exception as exc:
        if onerror:
            try:
                onerror(exc)
            except Exception as cb_err:
                _logger.warning(f"onerror callback raised an exception: {cb_err}")

        # Attempt to remove the file or directory
        try:
            if _path.exists():
                if _path.is_file() or _path.is_symlink():
                    _path.unlink()
                    _logger.warning(f"Removed file after error: {_path}")
                elif _path.is_dir():
                    _shutil.rmtree(_path)
                    _logger.warning(f"Removed directory after error: {_path}")
            else:
                _logger.debug(f"No cleanup needed; path does not exist: {_path}")
        except Exception as cleanup_err:
            _logger.warning(f"Cleanup failed for path '{_path}': {cleanup_err}")

        # Re-raise the original exception after cleanup attempt
        raise


# def copytree_without_file_permissions(
#     src_dir: Union[str, _os.PathLike],
#     dst_dir: Union[str, _os.PathLike],
# ) -> None:
#     """
#     Copy the directory src_dir into dst_dir, without preserving filesystem permissions.
#     """
#     for dirpath, dirnames, filenames in _os.walk(src_dir):
#         for dirname in dirnames:
#             relative_dir_path = _os.path.relpath(_os.path.join(dirpath, dirname), src_dir)
#             # For each directory <dirname> immediately under <dirpath>,
#             # create an equivalently-named
#             # directory under the destination directory
#             abs_dir_path = _os.path.join(dst_dir, relative_dir_path)
#             _os.mkdir(abs_dir_path)
#         for filename in filenames:
#             # For each file with name <filename> immediately under <dirpath>, copy that file to
#             # the appropriate location in the destination directory
#             file_path = _os.path.join(dirpath, filename)
#             relative_file_path = _os.path.relpath(file_path, src_dir)
#             abs_file_path = _os.path.join(dst_dir, relative_file_path)
#             _shutil.copy2(file_path, abs_file_path)


def copytree_without_file_permissions(  # noqa: PLR0912
    src_dir: Union[str, _pathlib.Path],
    dst_dir: Union[str, _pathlib.Path],
    skip_extensions: Optional[Iterable[str]] = None,
    *,
    follow_symlinks: bool = False,
    dry_run: bool = False,
    max_workers: int = 4,
) -> None:
    """
    Recursively copies the contents of `src_dir` into `dst_dir`, skipping file permissions,
    and optionally skipping certain extensions. Supports dry-run, threading, and summary stats.

    Parameters
    ----------
    src_dir : str or _pathlib.Path
        Source directory to copy from.
    dst_dir : str or _pathlib.Path
        Destination directory to copy to.
    skip_extensions : Iterable[str], optional
        A list of file extensions to skip, e.g. {".tmp", ".log"}.
    follow_symlinks : bool, default False
        Whether to follow and copy the contents of symbolic links.
        False, symlinks are recreated.
    dry_run : bool, default False
        If True, does not copy anything—only logs actions.
    max_workers : int, default 4
        Number of threads to use for parallel file copying.

    Notes
    -----
    - Uses `_shutil.copyfile` to avoid copying metadata (unlike `copy2`).
    - Does not preserve original permissions, timestamps, or other metadata.
    - Directories are created with `mkdir(parents=True, exist_ok=True)`.
    - If `follow_symlinks=False`, symbolic links are preserved.
    - Fails if the source doesn't exist or destination is invalid.

    Examples
    --------
    >>> copytree_without_file_permissions(
    ...     "/path/to/src",
    ...     "/path/to/dst",
    ...     skip_extensions={".tmp", ".log"},
    ... )
    """  # noqa: D205
    src_path = _pathlib.Path(src_dir).expanduser().resolve()
    dst_path = _pathlib.Path(dst_dir).expanduser().resolve()

    if not src_path.exists() or not src_path.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_path}")

    if src_path == dst_path:
        raise ValueError("Source and destination directories must be different.")

    skip_exts: set[str] = set(skip_extensions or [])

    file_tasks = []  # List of (source_path, dest_path) pairs
    total_bytes = 0
    files_copied = 0
    files_skipped = 0

    def _copy_file(source: _pathlib.Path, dest: _pathlib.Path) -> int:
        """Copy file from `source` to `dest` and return the size in bytes."""
        if dry_run:
            _logger.info(f"[Dry-run] Would copy: {source} -> {dest}")
            return 0
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            _shutil.copyfile(source, dest)
            size = source.stat().st_size
            _logger.info(f"Copied: {source} -> {dest} ({size} bytes)")
            return size
        except Exception as e:
            _logger.error(f"Error copying {source} -> {dest}: {e}")
            raise

    # Traverse source directory
    for path in src_path.rglob("*"):
        # Handle symbolic links
        if path.is_symlink() and not follow_symlinks:
            _logger.info(f"Skipped symlink: {path}")
            files_skipped += 1
            continue

        rel_path = path.relative_to(src_path)
        target_path = dst_path / rel_path

        # Create directories (dry_run safe)
        if path.is_dir():
            if dry_run:
                _logger.info(f"[Dry-run] Would create directory: {target_path}")
            else:
                try:
                    target_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    _logger.error(f"Failed to create directory {target_path}: {e}")
            continue

        # Skip files with ignored extensions
        if path.is_file():
            if path.suffix in skip_exts:
                _logger.info(f"Skipped file due to extension: {path}")
                files_skipped += 1
                continue
            file_tasks.append((path, target_path))

    # Parallel file copying
    with _concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_copy_file, src, dst): (src, dst) for src, dst in file_tasks
        }
        for future in _concurrent.futures.as_completed(futures):
            src, dst = futures[future]
            try:
                total_bytes += future.result()
                files_copied += 1
            except Exception:
                _logger.error(f"Failed to copy file: {src} -> {dst}")

    # Log summary
    _logger.info("---- Copy Summary ----")
    _logger.info(f"Files copied:  {files_copied}")
    _logger.info(f"Files skipped: {files_skipped}")
    _logger.info(f"Total bytes:   {total_bytes:,}")
    if dry_run:
        _logger.info("[Dry-run] No files were copied.")


def _copy_file_or_tree(
    src: "str | _pathlib.Path",
    dst: "str | _pathlib.Path",
    dst_dir: "Optional[str | _pathlib.Path]" = None,
) -> str:
    """
    Copy a file or directory tree from `src` into the destination `dst`, optionally
    nesting under `dst_dir`.

    Parameters
    ----------
    src : str or _pathlib.Path
        Source file or directory to copy.
    dst : str or _pathlib.Path
        Destination root directory.
    dst_dir : str or _pathlib.Path, optional
        Optional subdirectory inside `dst` where the `src` will be copied.

    Returns
    -------
    str
        The relative path (from `dst`) to the copied artifact.

    Raises
    ------
    FileNotFoundError
        If the source path does not exist.
    OSError
        If copying fails due to OS errors such as permission issues.

    Notes
    -----
    - If `src` is a file, it is copied preserving content but not metadata.
    - If `src` is a directory, the entire directory tree is copied,
      ignoring `__pycache__` directories.
    - The destination directory structure is created as needed.
    """  # noqa: D205
    # Convert inputs to _pathlib.Path for consistent handling
    src_path = _pathlib.Path(src).expanduser().resolve()
    dst_path = _pathlib.Path(dst).expanduser().resolve()

    if dst_dir is not None:  # noqa: SIM108
        dst_dir = _pathlib.Path(dst_dir)
    else:
        dst_dir = _pathlib.Path()

    if not src_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {src_path}")

    # Construct relative destination path under dst (and dst_dir)
    dst_subpath = dst_dir / src_path.name
    full_dst_path = dst_path / dst_subpath

    try:
        if src_path.is_file():
            # Ensure destination directory exists
            full_dst_path.parent.mkdir(parents=True, exist_ok=True)
            # Copy file contents only
            # _shutil.copy(src=src, dst=dst_path)
            _shutil.copyfile(src=src_path, dst=full_dst_path)
        elif src_path.is_dir():
            # Copy entire directory tree, ignore __pycache__
            _shutil.copytree(
                src=src_path,
                dst=full_dst_path,
                ignore=_shutil.ignore_patterns("__pycache__"),
                dirs_exist_ok=True,  # Python 3.8+
            )
        else:
            raise ValueError(f"Source path is neither a file nor directory: {src_path}")
    except Exception as e:
        raise OSError(f"Failed to copy {src_path} to {full_dst_path}: {e}") from e
    return str(dst_subpath)


def _handle_readonly_on_windows(
    func: "callable[[str], any]", path: str, exc_info: tuple[type, BaseException, any]
) -> None:
    """
    Retry deleting a read-only file on Windows by setting it writable.

    This is intended to be used as the `onerror` handler for `_shutil.rmtree`
    when cleaning up files or directories that might be read-only (common in
    temporary caches, build artifacts, or cloned repos).

    Parameters
    ----------
    func : callable
        The function that raised the error (e.g., _os.unlink or _os.rmdir).
    path : str
        The path that caused the error.
    exc_info : tuple
        The exception info as returned by `_sys.exc_info()`.

    Raises
    ------
    Exception
        If the error is not a Windows-specific permission issue or if retry fails.

    References
    ----------
    - https://bugs.python.org/issue19643
    - https://bugs.python.org/issue43657

    Notes
    -----
    - Only applies to Windows with `winerror == 5` (Access Denied).
    - Makes the file writable via `_os.chmod(path, _stat.S_IWRITE)` before retrying.
    """
    _exc_type, exc_value = exc_info[:2]

    should_reattempt = (
        # _os.name == "nt"
        is_windows()
        and func in (_os.unlink, _os.rmdir)
        and isinstance(exc_value, PermissionError)
        and getattr(exc_value, "winerror", None) == 5  # Access denied  # noqa: PLR2004
    )

    if not should_reattempt:
        raise exc_value

    _os.chmod(path, _stat.S_IWRITE)
    try:
        func(path)
    except Exception as retry_error:  # noqa: TRY203
        raise retry_error


def _copy_project(
    src_path: str,
    dst_path: str = "",
) -> str:
    """
    Internal utility to copy an MLflow project directory during development.

    This function copies the entire project directory tree while excluding files and
    directories that match patterns specified in a `.dockerignore` file, if present.
    It is assumed that the MLflow project is available as a local directory and that
    `pyproject.toml` is present at the root.

    Parameters
    ----------
    src_path : str
        Source path of the MLflow project to copy.
    dst_path : str, optional
        Destination path where the MLflow project should be copied.
        Defaults to the current working directory.

    Returns
    -------
    str
        The name of the copied MLflow project directory (default: "mlflow-project").

    Raises
    ------
    AssertionError
        If `pyproject.toml` is not found in the source directory.

    Notes
    -----
    - Mimics behavior of `docker build` by supporting `.dockerignore` rules.
    - Useful in dev environments for staging MLflow projects prior to packaging or containerizing.

    Examples
    --------
    >>> _copy_project("/path/to/project")
    'mlflow-project'
    """  # noqa: D401

    def _docker_ignore(root: str):
        # Construct path to potential `.dockerignore` file
        docker_ignore_path = _os.path.join(root, ".dockerignore")
        patterns = []

        # Read ignore patterns if `.dockerignore` exists
        if _os.path.exists(docker_ignore_path):
            with open(docker_ignore_path, encoding=ENCODING) as f:
                # Strip whitespace and ignore empty lines
                patterns = [line.strip() for line in f if line.strip()]

        def ignore(_, names):
            # Apply all ignore patterns using fnmatch (like glob)
            ignored = set()
            for pattern in patterns:
                ignored.update(_fnmatch.filter(names, pattern))
            return list(ignored)

        # Return the ignore callable only if patterns were found
        return ignore if patterns else None

    project_dir = "mlflow-project"

    # Ensure this is a valid MLflow project by checking for `pyproject.toml`
    pyproject_path = _os.path.abspath(_os.path.join(src_path, "pyproject.toml"))
    assert _os.path.isfile(  # noqa: S101
        pyproject_path,
    ), f"file not found: {pyproject_path}"  # noqa: S101

    # Perform recursive copy, applying the ignore rules if applicable
    _shutil.copytree(
        src=src_path,
        dst=_os.path.join(dst_path, project_dir),
        ignore=_docker_ignore(src_path),
    )

    # Return the name of the copied project directory
    return project_dir


######################################################################
## size of file
######################################################################


def _get_local_file_size(file: Union[str, _pathlib.Path]) -> float:
    """
    Get the size of a local file in kilobytes (KB).

    Parameters
    ----------
    file : str or _pathlib.Path
        Path to the file.

    Returns
    -------
    float
        File size in kilobytes, rounded to 1 decimal place.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    OSError
        If the file size cannot be determined.

    Examples
    --------
    >>> _get_local_file_size("example.txt")
    12.3
    """
    file_path = _pathlib.Path(file).expanduser().resolve()

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        size_bytes = file_path.stat().st_size
        return round(size_bytes / 1024.0, 1)
    except OSError as e:
        raise OSError(f"Unable to get file size for {file_path}: {e}") from e


def get_total_file_size(
    path: Union[str, _pathlib.Path],
) -> Optional[int]:
    """
    Return the total size (in bytes) of all files under the given directory path,
    including all nested files in subdirectories.

    Parameters
    ----------
    path : str or _pathlib.Path
        Absolute or relative path to a local directory.

    Returns
    -------
    Optional[int]
        The total size in bytes of all files under the directory.
        Returns None if the path is invalid or an error occurs.

    Raises
    ------
    ScikitplotException
        If the path does not exist or is not a directory.

    Notes
    -----
    - Uses _os.walk to recursively traverse directory tree.
    - Logs and swallows unexpected errors, returning None on failure.
    """  # noqa: D205
    try:
        # Convert _pathlib.Path to string if necessary
        if isinstance(path, _pathlib.Path):
            path = str(path)

        # Validate path existence
        if not _os.path.exists(path):
            raise ScikitplotException(
                message=f"The given path does not exist: {path}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # Ensure the path is a directory
        if not _os.path.isdir(path):
            raise ScikitplotException(
                message=f"The given path is not a directory: {path}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        total_size = 0

        # Traverse directory tree, summing file sizes
        for current_dir, _dirs, files in _os.walk(path):
            # Build full file paths for all files in current directory
            full_paths = [_os.path.join(current_dir, file) for file in files]

            # Accumulate file sizes
            total_size += sum(_os.path.getsize(file) for file in full_paths)

        return total_size

    except Exception as e:
        # Log error and return None if anything unexpected happens
        _logger.info(f"Failed to get total size for {path}: {e}")
        return None


def _get_local_project_dir_size(
    project_path: Union[str, _pathlib.Path],
) -> float:
    """
    Compute the total size of all files in a local project directory.

    This internal utility is used for CLI logging or pre-copy diagnostics to estimate
    how large the project directory is before copying it (e.g., for MLflow packaging).

    Parameters
    ----------
    project_path : str or _pathlib.Path
        The root path of the local project directory.

    Returns
    -------
    float
        Total size of all files in the directory (and subdirectories),
        in kilobytes (KB), rounded to 1 decimal place.

    Notes
    -----
    - Ignores symbolic links to avoid circular references.
    - Includes hidden files.
    - Uses `_os.walk` to traverse the directory tree.
    """
    # Ensure path is a string
    if isinstance(project_path, _pathlib.Path):
        project_path = str(project_path)

    total_size = 0

    # Walk through all subdirectories and files
    for root, _, files in _os.walk(project_path):
        for f in files:
            path = _os.path.join(root, f)
            # Accumulate file size (in bytes)
            total_size += _os.path.getsize(path)

    # Convert bytes to kilobytes and round for readability
    return round(total_size / 1024.0, 1)


######################################################################
## size of file
######################################################################


def read_file(
    parent_path: Union[str, _pathlib.Path],
    file_name: Union[str, _pathlib.Path],
) -> str:
    r"""
    Read and return the full contents of a text file.

    Parameters
    ----------
    parent_path : str or _pathlib.Path
        Path to the directory containing the file. Can include `~` for the home directory.
    file_name : str or _pathlib.Path
        Name of the file to read (can be a simple file name or subpath).

    Returns
    -------
    str
        The entire contents of the file as a string.

    Examples
    --------
    >>> read_file("~/logs", "output.txt")
    'Log started...\n'

    Notes
    -----
    - Supports both string and _pathlib inputs.
    - Uses `expanduser()` to resolve user home (~).
    - Uses `resolve()` to get an absolute canonical path.
    - Assumes the file is encoded in UTF-8 unless otherwise configured.
    """
    # Convert and normalize path
    path = _pathlib.Path(parent_path).expanduser().resolve() / file_name
    # Open file with specified encoding and read entire content
    with _codecs.open(path, mode="r", encoding=ENCODING) as f:
        return f.read()


def read_file_lines(
    parent_path: Union[str, _pathlib.Path],
    file_name: Union[str, _pathlib.Path],
) -> list[str]:
    r"""
    Read the contents of a text file and return lines as a list.

    This function constructs the full file path using `parent_path` and `file_name`,
    opens the file with the specified encoding, and returns all lines as a list.

    Parameters
    ----------
    parent_path : str or _pathlib.Path
        Path to the directory containing the file. Can include `~` for the home directory.
    file_name : str or _pathlib.Path
        Name of the file (may include subdirectories relative to `parent_path`).

    Returns
    -------
    List[str]
        A list of strings, each representing a line from the file.

    Examples
    --------
    >>> read_file_lines("~/logs", "output.log")
    ['line1\n', 'line2\n', ...]

    Notes
    -----
    - Uses `expanduser()` to resolve `~` to the user home directory.
    - Uses `resolve()` to get a canonical absolute path.
    - Uses `codecs.open` for consistent encoding handling across platforms.
    """
    # Convert and normalize path
    file_path = _pathlib.Path(parent_path).expanduser().resolve() / file_name
    # Open the file with the specified encoding and return all lines
    with _codecs.open(file_path, mode="r", encoding=ENCODING) as f:
        return f.readlines()


def read_chunk(
    path: Union[str, _os.PathLike],
    size: int,
    start_byte: int = 0,
) -> bytes:
    """
    Read a chunk of bytes from a file.

    Parameters
    ----------
    path : str or _os.PathLike
        Path to the file. Supports `~` for home directory expansion.
    size : int
        The number of bytes to read.
    start_byte : int, optional
        The byte offset from which to start reading. Default is 0.

    Returns
    -------
    bytes
        A chunk of bytes read from the file.

    Raises
    ------
    ValueError
        If the path does not exist or is not a file.
    """
    file_path = _pathlib.Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise ValueError(f"Invalid file path: {file_path}")

    with open(file_path, "rb") as f:
        if start_byte > 0:
            f.seek(start_byte)
        return f.read(size)


def yield_file_in_chunks(
    file: Union[str, _pathlib.Path],
    chunk_size: int = 100_000_000,
) -> Generator[bytes, None, None]:
    """
    Generate to read a file in binary mode and yield it in chunks.

    This is useful for processing or uploading large files without loading the entire
    file into memory at once.

    Parameters
    ----------
    file : str or _pathlib.Path
        Path to the input file. Supports `~` for home directory.
    chunk_size : int, optional
        Number of bytes to read per chunk. Default is 100,000,000 (≈100 MB).

    Yields
    ------
    bytes
        A chunk of the file content as a byte string.

    Examples
    --------
    >>> for chunk in yield_file_in_chunks("large_file.bin", chunk_size=1024 * 1024):
    ...     process(chunk)  # Handle 1MB chunks
    """
    file_path = _pathlib.Path(file).expanduser().resolve()

    # Open the file in binary mode and yield chunks
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)  # Read up to `chunk_size` bytes
            if not chunk:
                break
            yield chunk


def write_to(
    filename: Union[str, _pathlib.Path],
    data: str,
) -> None:
    """
    Write string data to a file, overwriting any existing content.

    Parameters
    ----------
    filename : str or _pathlib.Path
        The target file path. Supports `~` for home directory.
    data : str
        The content to write.

    Notes
    -----
    - Uses the global ENCODING constant.
    - Automatically creates the parent directory if it does not exist.
    """
    file_path = _pathlib.Path(filename).expanduser().resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with _codecs.open(file_path, mode="w", encoding=ENCODING) as handle:
        handle.write(data)


def append_to(
    filename: Union[str, _pathlib.Path],
    data: str,
) -> None:
    """
    Append string data to a file. Creates the file if it doesn't exist.

    Parameters
    ----------
    filename : str or _pathlib.Path
        The target file path. Supports `~` for home directory.
    data : str
        The content to append.

    Notes
    -----
    - Appends using system default encoding unless explicitly modified.
    - Automatically creates the parent directory if it does not exist.
    """
    file_path = _pathlib.Path(filename).expanduser().resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a", encoding=ENCODING) as handle:
        handle.write(data)


def create_tar_gz_archive(
    output_filename: str,
    source_dir: str,
    archive_name: str,
    custom_filter: "Optional[callable[[_tarfile.TarInfo], Optional[_tarfile.TarInfo]]]" = None,
) -> None:
    """
    Create a reproducible gzip-compressed tar archive (`.tar.gz`) from a directory.

    This function archives the contents of `source_dir` into a `.tar.gz` file named
    `output_filename`. The archive will contain a top-level directory named `archive_name`.
    File modification timestamps are zeroed to ensure reproducible builds, and an optional
    `custom_filter` allows further customization of archive entries (e.g., exclude files).

    Parameters
    ----------
    output_filename : str
        The target filename for the `.tar.gz` archive to create. If the file exists,
        it will be overwritten.
    source_dir : str
        Path to the directory whose contents will be archived.
    archive_name : str
        The name of the root directory inside the archive. This controls the folder
        name when extracting the archive.
    custom_filter : callable, optional
        A callable that takes a `_tarfile.TarInfo` object and returns a modified
        `TarInfo` object or `None` to exclude the entry. Useful to filter or modify
        metadata such as permissions, ownership, or file inclusion.
        If `None`, no additional filtering is applied.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If `source_dir` does not exist.
    OSError
        If there are issues reading files or writing the output archive.
    Exception
        For other unforeseen errors during archiving or compression.

    Notes
    -----
    - The function zeroes file modification times (`mtime=0`) inside the archive to
      support reproducible builds where file timestamps should not affect the archive hash.
    - The archive is created in two steps: first a `.tar` file is created as a temporary
      file, then compressed into `.tar.gz`.
    - The temporary `.tar` file is securely removed after compression.
    - Uses `gzip.GzipFile` with `mtime=0` to omit gzip file timestamp metadata.
    - On Windows, ensure proper permissions to write temporary files.

    Examples
    --------
    Create a `.tar.gz` archive from a directory, with a top-level folder name 'my_project':

    >>> create_tar_gz_archive("archive.tar.gz", "/path/to/my_project", "my_project")

    Use a custom filter to exclude `.pyc` files:

    >>> def filter_pyc(tarinfo):
    ...     if tarinfo.name.endswith(".pyc"):
    ...         return None
    ...     return tarinfo

    >>> create_tar_gz_archive(
    ...     "filtered_archive.tar.gz",
    ...     "./src",
    ...     "src_folder",
    ...     custom_filter=filter_pyc,
    ... )

    References
    ----------
    - Python `tarfile` module: https://docs.python.org/3/library/tarfile.html
    - Python `gzip` module: https://docs.python.org/3/library/gzip.html
    - Reproducible builds: https://reproducible-builds.org/
    """

    def _filter_timestamps(tar_info: _tarfile.TarInfo) -> Optional[_tarfile.TarInfo]:
        # Zero modification time for reproducible archives
        tar_info.mtime = 0
        # Apply user-provided filter if any
        return custom_filter(tar_info) if custom_filter else tar_info

    # Create a temporary file for the uncompressed tar archive
    unzipped_fd, unzipped_filename = _tempfile.mkstemp()
    try:
        _os.close(unzipped_fd)  # Close the file descriptor, tarfile will open it
        # Create tar archive with filtered timestamps and optional filtering
        with _tarfile.open(unzipped_filename, "w") as tar:
            tar.add(source_dir, arcname=archive_name, filter=_filter_timestamps)

        # Compress the tar archive with gzip, omitting timestamp metadata
        # When gzipping the tar, don't include the tar's filename or modification time in the
        # zipped archive (see https://docs.python.org/3/library/gzip.html#gzip.GzipFile)
        # ⚠️ Cannot use parentheses within a `with` statement on Python 3.8 (syntax was added in Python 3.9)
        with open(unzipped_filename, "rb") as raw_tar, _gzip.GzipFile(
            filename="",
            fileobj=open(output_filename, "wb"),
            mode="wb",
            mtime=0,
        ) as gzipped_tar:
            gzipped_tar.write(raw_tar.read())

    finally:
        # Clean up temporary tar file
        _os.remove(unzipped_filename)


######################################################################
## tmp dir path
######################################################################


def _get_tmp_dir() -> Optional[_pathlib.Path]:
    """
    Get a secure temporary directory path, with special handling for Databricks environments.

    This function provides a writable temporary directory location suitable for
    cross-platform use and Databricks notebooks/jobs.

    Behavior
    --------
    - If running inside a Databricks environment:
        1. Attempts to return a new temporary directory inside the path from
           `get_databricks_local_temp_dir()`.
        2. If that fails, attempts to create a REPL-specific subdirectory under the
           system temp directory to avoid conflicts between users or sessions.

    - If not in a Databricks environment:
        - Returns a new temporary directory under the default system temp location.

    Returns
    -------
    Optional[_pathlib.Path]
        A _pathlib.Path to a newly created temporary directory.
        Returns None only if an unexpected failure occurs.

    Notes
    -----
    - The returned directory is guaranteed to exist.
    - Complies with Ruff rule S108 by avoiding hardcoded "/tmp" paths.
    - Caller is responsible for cleanup if desired.

    Examples
    --------
    >>> tmpdir = _get_tmp_dir()
    >>> if tmpdir:
    ...     print(tmpdir)
    ...     # Use the directory here
    """
    try:
        from mlflow.utils.databricks_utils import is_in_databricks_runtime

        if is_in_databricks_runtime():
            # Attempt to use the Databricks-provided temp directory
            from mlflow.utils.databricks_utils import get_databricks_local_temp_dir

            base_dir = _pathlib.Path(get_databricks_local_temp_dir())
            base_dir = base_dir.expanduser().resolve()
            base_dir.mkdir(parents=True, exist_ok=True)
            return _pathlib.Path(_tempfile.mkdtemp(dir=base_dir))
    except Exception:
        pass  # If Databricks-specific utilities not available or fail, fallback

    try:
        from mlflow.utils.databricks_utils import get_repl_id

        repl_id = get_repl_id()
        if repl_id:
            base_dir = _pathlib.Path(_tempfile.gettempdir()) / "repl_tmp_data" / repl_id
            base_dir = base_dir.expanduser().resolve()
            base_dir.mkdir(parents=True, exist_ok=True)
            return _pathlib.Path(_tempfile.mkdtemp(dir=base_dir))
    except Exception:
        pass  # Silently ignore if repl_id not retrievable or other errors

    # Default fallback to system temp directory
    base_dir = _pathlib.Path(_tempfile.gettempdir())
    base_dir.mkdir(parents=True, exist_ok=True)
    return _pathlib.Path(_tempfile.mkdtemp(dir=base_dir))


def create_tmp_dir() -> str:
    """
    Create a secure temporary directory, optionally nested inside a Databricks-specific location.

    This function wraps `_tempfile.mkdtemp()` with special handling for Databricks environments
    by attempting to place the temporary directory inside a platform-appropriate parent directory.

    Behavior
    --------
    - If `_get_tmp_dir()` returns a valid directory (e.g., in Databricks REPL or runtime),
      the temporary directory will be created inside it.
    - Otherwise, falls back to using the system-wide default temp directory.

    Returns
    -------
    str
        The absolute path to the created temporary directory.

    Examples
    --------
    >>> tmp_dir = create_tmp_dir()
    >>> with open(_os.path.join(tmp_dir, "temp.txt"), "w") as f:
    ...     f.write("example")

    Notes
    -----
    - Always returns a valid, unique, and writable directory path.
    - `_get_tmp_dir()` should return a Path or str. It must resolve to an existing directory.
    """
    base_dir = _get_tmp_dir()

    # Fallback to system temp if _get_tmp_dir() returns None
    if base_dir is None:
        base_dir = _pathlib.Path(_tempfile.gettempdir())
    else:
        base_dir = _pathlib.Path(base_dir).expanduser().resolve()
        base_dir.mkdir(parents=True, exist_ok=True)

    # Create a new unique subdirectory within base_dir
    tmp_path = _pathlib.Path(_tempfile.mkdtemp(dir=base_dir))
    return str(tmp_path)


class TempDir:
    """
    Context manager for creating and optionally entering a temporary directory.

    Supports:
    - Changing the working directory upon entry (`chdr=True`)
    - Deleting the temp directory on exit (`remove_on_exit=True`)
    - Using environment-aware temp directories like those in Databricks (`use_env_tmpdir=True`)

    Parameters
    ----------
    chdr : bool, default=False
        If True, changes the current working directory to the temp directory upon entry.
    remove_on_exit : bool, default=True
        If True, removes the temp directory and all contents upon exit.
    use_env_tmpdir : bool, default=True
        If True, attempts to use a platform/environment-specific temporary base path.
    """

    def __init__(
        self,
        chdr: bool = False,
        remove_on_exit: bool = True,
        use_env_tmpdir: bool = True,
    ):
        self._chdr = chdr
        self._remove = remove_on_exit
        self._use_env_tmpdir = use_env_tmpdir

        self._original_dir: Optional[_pathlib.Path] = None
        self._temp_dir: Optional[_pathlib.Path] = None

    def __enter__(self) -> "TempDir":  # noqa: D105
        base_dir = _get_tmp_dir() if self._use_env_tmpdir else None
        self._temp_dir = _pathlib.Path(_tempfile.mkdtemp(dir=base_dir)).resolve()

        if self._chdr:
            self._original_dir = _pathlib.Path.cwd()
            _os.chdir(self._temp_dir)

        return self

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: D105
        if self._chdr and self._original_dir:
            _os.chdir(self._original_dir)
            self._original_dir = None

        if self._remove and self._temp_dir and self._temp_dir.exists():
            _shutil.rmtree(self._temp_dir, ignore_errors=True)

        if self._remove and self._temp_dir and self._temp_dir.exists():
            raise RuntimeError(
                f"Failed to remove temporary directory: {self._temp_dir}"
            )

    def path(self, *parts: Union[str, _pathlib.Path]) -> _pathlib.Path:
        """
        Construct a path inside the temporary directory.

        Parameters
        ----------
        *parts : str or _pathlib.Path
            Path components to append to the temp directory path.

        Returns
        -------
        _pathlib.Path
            A resolved Path inside the temp directory.

        Raises
        ------
        RuntimeError
            If called before the context has been entered.
        """
        if not self._temp_dir:
            raise RuntimeError("TempDir must be entered before using `.path()`.")
        subpath = _pathlib.Path(*parts).expanduser()
        return (
            _pathlib.Path(".") / subpath if self._chdr else self._temp_dir / subpath
        ).resolve()

    @property
    def root(self) -> _pathlib.Path:
        """
        Get the root temporary directory path.

        Returns
        -------
        _pathlib.Path
            The root temp directory as a Path object.

        Raises
        ------
        RuntimeError
            If the context has not been entered yet.
        """
        if not self._temp_dir:
            raise RuntimeError("TempDir is not active. Use within a `with` block.")
        return self._temp_dir


@cache_return_value_per_process
def get_or_create_tmp_dir() -> _pathlib.Path:
    """
    Get or create a persistent temporary directory scoped to the current Python process.

    Databricks behavior:
    - If running in a Databricks REPL:
        - Try `get_databricks_local_temp_dir()` as the base directory.
        - If it fails, fallback to: `/tmp/repl_tmp_data/{repl_id}/mlflow`.
        - This directory is *not* cleaned up with `atexit`,
          Databricks handles cleanup on session end.

    Non-Databricks behavior:
    - Creates a secure temporary directory via `_tempfile.mkdtemp()`.
    - Registers `_shutil.rmtree()` via `atexit` for cleanup on process exit.
    - Changes permissions to `0o777` so subprocesses like Spark UDFs can access it.

    Returns
    -------
    _pathlib.Path
        A path object pointing to the persistent temporary directory.

    Notes
    -----
    - The returned directory is created immediately and guaranteed to exist.
    - This function is cached per process and always returns the same directory.
    """
    try:
        from mlflow.utils.databricks_utils import (
            get_databricks_local_temp_dir,
            get_repl_id,
            is_in_databricks_runtime,
        )

        if is_in_databricks_runtime() and (repl_id := get_repl_id()):
            # Note: For python process attached to databricks notebook, atexit does not work.
            # The directory returned by `get_databricks_local_tmp_dir`
            # will be removed once databricks notebook detaches.
            # The temp directory is designed to be used by all kinds of applications,
            # so create a child directory "mlflow" for storing mlflow temp data.
            try:
                base_dir = _pathlib.Path(get_databricks_local_temp_dir())
            except Exception:
                # "/tmp"
                base_dir = (
                    _pathlib.Path(_tempfile.gettempdir()) / "repl_tmp_data" / repl_id
                )

            tmp_dir = base_dir.expanduser().resolve() / "mlflow"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            return tmp_dir

    except Exception:
        pass  # Silently fall back to generic tmp dir

    # Fallback: standard tmpdir with atexit cleanup
    tmp_dir = _pathlib.Path(_tempfile.mkdtemp()).resolve()
    # mkdtemp creates a directory with permission 0o700
    # change it to be 0o777 to ensure it can be seen in spark UDF
    _os.chmod(
        tmp_dir,
        0o777,  # noqa: S103
    )  # Needed for access in subprocesses like Spark UDFs  # noqa: S103
    _atexit.register(_shutil.rmtree, tmp_dir, ignore_errors=True)
    return tmp_dir


@cache_return_value_per_process
def get_or_create_nfs_tmp_dir() -> _pathlib.Path:
    """
    Get or create a temporary NFS-backed directory scoped to the current Python process.

    - If running on Databricks and `mlflow` is available:
        Attempts to place the temp directory inside a REPL-scoped or Databricks NFS location.
    - Otherwise:
        Falls back to using a local temporary directory inside the system temp location.

    In all cases:
    - Directory is created with 0o777 permissions.
    - Directory is cleaned up on process exit (if created locally).

    Returns
    -------
    _pathlib.Path
        Path to the created or reused NFS temporary directory.
    """
    try:
        from mlflow.utils.databricks_utils import (
            get_repl_id,
            is_in_databricks_runtime,
        )
        from mlflow.utils.nfs_on_spark import get_nfs_cache_root_dir
    except Exception:
        # mlflow not available: use default local temp directory
        tmp_nfs_dir = _pathlib.Path(_tempfile.mkdtemp()).resolve()
        _os.chmod(tmp_nfs_dir, 0o777)  # noqa: S103
        _atexit.register(_shutil.rmtree, tmp_nfs_dir, ignore_errors=True)
        return tmp_nfs_dir

    try:
        nfs_root = _pathlib.Path(get_nfs_cache_root_dir()).expanduser().resolve()

        if is_in_databricks_runtime() and (repl_id := get_repl_id()):
            # Note: In databricks, atexit hook does not work.
            # The directory returned by `get_databricks_nfs_tmp_dir`
            # will be removed once databricks notebook detaches.
            # The temp directory is designed to be used by all kinds of applications,
            # so create a child directory "mlflow" for storing mlflow temp data.
            try:
                from mlflow.utils.databricks_utils import get_databricks_nfs_temp_dir

                repl_nfs_base = _pathlib.Path(get_databricks_nfs_temp_dir())
            except Exception:
                repl_nfs_base = nfs_root / "repl_tmp_data" / repl_id

            tmp_nfs_dir = (repl_nfs_base / "mlflow").resolve()
            tmp_nfs_dir.mkdir(parents=True, exist_ok=True)
            return tmp_nfs_dir

    except Exception:
        pass  # Ignore errors and fallback below

    # Fallback if not in Databricks or if REPL/NFS logic fails
    tmp_nfs_dir = _pathlib.Path(_tempfile.mkdtemp(dir=nfs_root)).resolve()
    # mkdtemp creates a directory with permission 0o700
    # change it to be 0o777 to ensure it can be seen in spark UDF
    _os.chmod(tmp_nfs_dir, 0o777)  # noqa: S103
    _atexit.register(_shutil.rmtree, tmp_nfs_dir, ignore_errors=True)
    return tmp_nfs_dir


######################################################################
## Artifact ProgressBar
######################################################################


class ProgressBar:
    """
    A context-managed progress bar wrapper for file and chunk operations, using `tqdm` if available.

    This utility is commonly used for tracking upload/download or file processing progress,
    while gracefully degrading when `tqdm` is not installed.

    Parameters
    ----------
    desc : str
        Description to show next to the progress bar.
    total : int
        Total number of units to track (e.g., bytes or files).
    step : int
        Amount to increment on each update.
    disable : bool, optional
        If True, disables progress bar entirely regardless of global flag.
        Default is False.
    **kwargs : dict
        Additional keyword arguments passed to `tqdm`.

    Notes
    -----
    - This wrapper checks the `SKPLT_ENABLE_ARTIFACTS_PROGRESS_BAR` flag before displaying
      and optional `disable=True` override.
    - Will silently disable the progress bar if `tqdm` is not available.
    - Use `.chunks()` for tracking bytes and `.files()` for counting files.

    Examples
    --------
    >>> with ProgressBar.chunks(
    >>>     file_size=1024 * 1024, desc="Uploading", chunk_size=4096
    >>> ) as bar:
    ...     while not done:
    ...         upload_chunk(...)
    ...         bar.update()

    >>> with ProgressBar.files(desc="Processing", total=10) as bar:
    ...     for file in files:
    ...         process(file)
    ...         bar.update()
    """

    def __init__(
        self,
        desc: str,
        total: int,
        step: int,
        disable: bool = False,
        **kwargs,
    ) -> None:
        self.desc = desc  # Label for progress bar
        self.total = max(total, 0)  # Ensure total is non-negative
        self.step = max(step, 1)  # Ensure positive progress
        self.disable = disable  # Override to disable bar
        self.kwargs = kwargs  # Additional tqdm options
        self.pbar = None  # tqdm instance (or None if disabled)
        self.progress = 0  # Internal tracker

    def set_pbar(self) -> None:
        """
        Initialize the tqdm progress bar if allowed.
        """
        if self.disable or not SKPLT_ENABLE_ARTIFACTS_PROGRESS_BAR.get():
            return  # Globally or explicitly disabled

        try:
            from tqdm.auto import tqdm

            self.pbar = tqdm(
                total=self.total,
                desc=self.desc,
                **self.kwargs,
            )
        except ImportError:
            self.pbar = None  # Silently degrade

    @classmethod
    def chunks(
        cls,
        file_size: int,
        desc: str,
        chunk_size: int,
        **kwargs,
    ) -> "ProgressBar":
        """
        Create a chunk-wise progress bar (e.g., for byte streams).

        Parameters
        ----------
        file_size : int
            Total number of bytes.
        desc : str
            Progress label.
        chunk_size : int
            Step size in bytes.
        **kwargs : dict
            Additional keyword arguments passed to `tqdm`.

        Returns
        -------
        ProgressBar
            A configured progress bar instance.
        """
        bar_ = cls(
            desc,
            total=file_size,
            step=chunk_size,
            unit="iB",  # Binary unit: kibibytes, etc.
            unit_scale=True,  # Human-readable scaling
            unit_divisor=1024,  # Use 1024 for binary units
            miniters=1,  # Show updates after each iteration
            **kwargs,
        )
        if file_size >= _PROGRESS_BAR_DISPLAY_THRESHOLD:
            bar_.set_pbar()
        return bar_

    @classmethod
    def files(
        cls,
        desc: str,
        total: int,
        **kwargs,
    ) -> "ProgressBar":
        """
        Create a file-counting progress bar.

        Parameters
        ----------
        desc : str
            Progress label.
        total : int
            Total number of files.
        **kwargs : dict
            Additional keyword arguments passed to `tqdm`.

        Returns
        -------
        ProgressBar
            A configured progress bar instance.
        """
        bar_ = cls(desc, total=total, step=1, **kwargs)  # Step is 1 file at a time
        bar_.set_pbar()
        return bar_

    def update(self) -> None:
        """
        Advance the progress bar by one step or to completion, whichever is smaller.
        """
        if not self.pbar:
            return

        try:
            remaining = self.total - self.progress
            update_step = min(self.step, remaining)  # Avoid over-stepping
            if update_step > 0:
                self.pbar.update(update_step)  # Update tqdm bar
                self.pbar.refresh()  # Force visual update
                self.progress += update_step  # Track internally
        except Exception:
            # Never raise from progress bar
            self.pbar = None

    def __enter__(self) -> "ProgressBar":
        """
        Enter the context manager.

        Returns
        -------
        ProgressBar
            Self, for usage in a `with` block.
        """
        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ) -> None:
        """
        Exit the context manager and clean up progress bar.
        """
        if self.pbar:
            try:
                self.pbar.refresh()  # Ensure final state is printed
                self.pbar.close()  # Clean up, close tqdm bar to release stdout
            except Exception:
                pass  # Tolerate tqdm exceptions


def download_file_using_http_uri(
    http_uri: str,
    download_path: str,
    chunk_size: int = 100000000,  # 100_000_000,
    headers: "dict | None" = None,
) -> None:
    """
    Download a file from an HTTP URI to a local path in chunks to avoid high memory usage.

    This function is especially designed to download files from presigned URLs
    provided by cloud storage services.

    Parameters
    ----------
    http_uri : str
        The HTTP(s) URL of the file to download.
    download_path : str
        The local filesystem path where the file should be saved.
    chunk_size : int, optional
        Size (in bytes) of each chunk to read during streaming download. Default is 100_000_000.
    headers : dict, optional
        Optional HTTP headers to include in the request (e.g., authentication).

    Returns
    -------
    None

    Raises
    ------
    HTTPError
        If the HTTP request returns an unsuccessful status code.
    IOError
        If there is an error writing to the local file.

    Notes
    -----
    - The function uses streaming download (`stream=True`) to avoid loading
      the entire file into memory.
    - It is recommended to handle retries and exponential backoff externally if needed.

    Examples
    --------
    >>> download_file_using_http_uri(
    ...     "https://example.com/presigned-url",
    ...     "/tmp/myfile.dat",
    ...     chunk_size=10_000_000,
    ... )
    """
    if headers is None:
        headers = {}

    # Validate chunk_size
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    # Open an HTTP GET request with streaming enabled to avoid loading whole file in memory
    with cloud_storage_http_request(
        "get", http_uri, stream=True, headers=headers
    ) as response:
        # Raise exception for HTTP errors (e.g., 404, 403)
        augmented_raise_for_status(response)

        # Open the target file in binary write mode
        with open(download_path, "wb") as output_file:
            # Iterate over the response in chunks
            for chunk in response.iter_content(chunk_size=chunk_size):
                # Break if chunk is empty (end of content)
                if not chunk:
                    break
                # Write the chunk to the output file
                output_file.write(chunk)


######################################################################
## _Chunk
######################################################################


@dataclass(frozen=True)
class _Chunk:
    """
    Immutable data structure representing a chunk of a file for partial download.

    Attributes
    ----------
    index : int
        The zero-based index of the chunk in the sequence of chunks.
    start : int
        The starting byte (inclusive) of this chunk in the file.
    end : int
        The ending byte (inclusive) of this chunk in the file.
    path : str
        The local file path where this chunk's data is stored.

    Examples
    --------
    >>> chunk = _Chunk(index=0, start=0, end=999999, path="/tmp/chunk0.part")
    >>> chunk.index
    0
    """

    index: int
    start: int
    end: int
    path: str


def _yield_chunks(
    path: str,
    file_size: int,
    chunk_size: int,
) -> Iterator[_Chunk]:
    """
    Yield sequential/consecutive file chunks for downloading or processing.

    Parameters
    ----------
    path : str
        The file path where the chunks will be saved or referenced.
    file_size : int
        Total size of the file in bytes.
    chunk_size : int
        Size of each chunk in bytes.

    Yields
    ------
    _Chunk
        Metadata for each chunk.
        A dataclass instance describing each chunk's byte range and path.

    Raises
    ------
    ValueError
        If `file_size` or `chunk_size` is not positive.

    Examples
    --------
    >>> list(_yield_chunks("/tmp/file", 2500, 1000))
    [_Chunk(index=0, start=0, end=999, path='/tmp/file'),
     _Chunk(index=1, start=1000, end=1999, path='/tmp/file'),
     _Chunk(index=2, start=2000, end=2499, path='/tmp/file')]
    """
    if file_size <= 0:
        raise ValueError("file_size must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    # Calculate the total number of chunks needed, rounding up for any remainder
    num_chunks = _math.ceil(file_size / chunk_size)

    for i in range(num_chunks):
        range_start = i * chunk_size
        # End is inclusive, so subtract 1, but never exceed file_size - 1
        range_end = min(range_start + chunk_size - 1, file_size - 1)

        # Yield a _Chunk instance for this byte range
        yield _Chunk(index=i, start=range_start, end=range_end, path=path)


def download_chunk_direct(chunk: _Chunk, headers: dict, http_uri: str) -> None:
    """
    Download a chunk using HTTP Range requests with `requests` library.

    Parameters
    ----------
    chunk : _Chunk
        The chunk metadata.
    headers : dict
        HTTP headers.
    http_uri : str
        Source URI.

    Raises
    ------
    HTTPError
        If request fails.
    """
    headers = headers.copy()
    headers["Range"] = f"bytes={chunk.start}-{chunk.end}"

    response = _requests.get(http_uri, headers=headers, stream=True, timeout=99)
    response.raise_for_status()

    with open(chunk.path, "r+b") as f:
        f.seek(chunk.start)
        for block in response.iter_content(1024 * 1024):  # 1MB
            f.write(block)


def parallelized_download_file(
    executor: _concurrent.futures.Executor,
    http_uri: str,
    download_path: str,
    file_size: int,
    chunk_size: int,
    headers: Optional[dict] = None,
) -> dict[int, Exception]:
    """
    Download a file in parallel using HTTP range requests.

    Parameters
    ----------
    executor : Executor
        Thread or process pool executor.
    http_uri : str
        The source HTTP URI.
    download_path : str
        Where to save the downloaded file.
    file_size : int
        Total size of the file.
    chunk_size : int
        Size of each chunk in bytes.
    headers : dict, optional
        Headers for HTTP requests.

    Returns
    -------
    Dict[int, Exception]
        Failed chunk indices mapped to exceptions.

    Notes
    -----
    Pre-creates/truncates output file. Uses direct requests for downloading.
    """
    headers = headers or {}

    # Ensure file exists and is sized correctly
    with open(download_path, "wb") as f:
        f.truncate(file_size)

    # Generate chunks
    chunks = list(_yield_chunks(download_path, file_size, chunk_size))

    def _download(chunk):
        try:
            download_chunk_direct(chunk, headers, http_uri)
        except Exception as e:
            _logger.warning(f"Chunk {chunk.index} failed: {e}")
            raise

    futures = {executor.submit(_download, c): c for c in chunks}
    failures: dict[int, Exception] = {}

    with ProgressBar.chunks(
        file_size, f"Downloading {download_path}", chunk_size
    ) as pbar:
        for future in _concurrent.futures.as_completed(futures):
            chunk = futures[future]
            try:
                future.result()
                pbar.update()
            except Exception as e:
                failures[chunk.index] = e

    return failures


# def parallelized_download_file_using_http_uri(
#     thread_pool_executor: concurrent.futures.Executor,
#     http_uri: str,
#     download_path: str,
#     remote_file_path: str,
#     file_size: int,
#     uri_type: Optional[Union[str, 'ArtifactCredentialType']],
#     chunk_size: int,
#     env: Optional[dict] = None,
#     headers: Optional[dict] = None,
# ) -> dict[int, Exception]:
#     """
#     Download a large file in parallel using HTTP range requests.

#     Splits the file into byte-range chunks and downloads each chunk concurrently
#     using subprocess calls to an external chunk downloader script. Handles
#     retries and aggregates failed chunks.

#     Parameters
#     ----------
#     thread_pool_executor : concurrent.futures.Executor
#         Executor to manage parallel thread or process pool.
#     http_uri : str
#         The HTTP URI of the file to download.
#     download_path : str
#         Local filesystem path to save the downloaded file.
#     remote_file_path : str
#         Path identifying the remote file for chunk generation.
#     file_size : int
#         Total size of the remote file in bytes.
#     uri_type : Optional[Union[str, ArtifactCredentialType]]
#         The credential or URI type that may affect download behavior (e.g., GCP signed URLs).
#     chunk_size : int
#         Size of each chunk in bytes.
#     env : Optional[dict], default=None
#         Environment variables for subprocesses.
#     headers : Optional[dict], default=None
#         HTTP headers to send with each chunk request.

#     Returns
#     -------
#     Dict[int, Exception]
#         Mapping of chunk indices to exceptions for failed downloads.

#     Notes
#     -----
#     - For certain URI types (like GCP signed URLs), this function downloads
#       one chunk serially first to check for transcoding behavior.
#     - The file at `download_path` is pre-created/truncated before downloads.
#     """
#     headers = headers or {}

#     def run_download(chunk: _Chunk):
#         """Run chunk download as subprocess and handle errors."""
#         try:
#             # Call external downloader script with required arguments
#             subprocess.run(  # noqa: S603
#                 [
#                     _sys.executable,
#                     download_cloud_file_chunk.__file__,
#                     "--range-start",
#                     str(chunk.start),
#                     "--range-end",
#                     str(chunk.end),
#                     "--headers",
#                     json.dumps(headers),
#                     "--download-path",
#                     download_path,
#                     "--http-uri",
#                     http_uri,
#                 ],
#                 text=True,
#                 check=True,
#                 capture_output=True,
#                 timeout=SKPLT_DOWNLOAD_CHUNK_TIMEOUT.get(),
#                 env=env,
#             )
#         except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
#             # Raise custom exception with captured stdout and stderr
#             raise ScikitplotException(
#                 textwrap.dedent(
#                     f"""\
#                     ----- stdout -----
#                     {e.stdout.strip() if e.stdout else 'No output'}

#                     ----- stderr -----
#                     {e.stderr.strip() if e.stderr else 'No error output'}
#                     """
#                 )
#             ) from e

#     # Generate chunks to download
#     chunks = list(_yield_chunks(remote_file_path, file_size, chunk_size))

#     # Create file if it doesn't exist or erase the contents if it does. We should do this here
#     # before sending to the workers so they can each individually seek to their respective
#     # positions and write chunks without overwriting.
#     # Pre-create or truncate the file before parallel writes to allow seeking
#     with open(download_path, "wb") as f:
#         pass  # File is created empty or truncated

#     # Handle special cases for certain URI types
#     if uri_type == ArtifactCredentialType.GCP_SIGNED_URL or uri_type is None:
#         first_chunk = next(chunks)  # chunks[0]
#         # GCP files could be transcoded, in which case the range header is ignored.
#         # Test if this is the case by downloading one chunk and seeing if it's larger than the
#         # requested size. If yes, let that be the file; if not, continue downloading more chunks.
#         # Download the first chunk serially to detect transcoding behavior
#         download_chunk(
#             range_start=first_chunk.start,
#             range_end=first_chunk.end,
#             headers=headers,
#             download_path=download_path,
#             http_uri=http_uri,
#         )
#         downloaded_size = _os.path.getsize(download_path)
#         # If downloaded size was equal to the chunk size it would have been downloaded serially,
#         # so we don't need to consider this here
#         # If size indicates transcoding, skip parallel chunked downloads
#         if downloaded_size > chunk_size:
#             return {}

#     # Submit chunk downloads to thread pool executor
#     futures = {thread_pool_executor.submit(run_download, chunk): chunk for chunk in chunks}
#     failed_downloads: dict[int, Exception] = {}
#     # Use progress bar for feedback
#     with ProgressBar.chunks(
#         file_size,
#         f"Downloading {download_path}",
#         chunk_size,
#     ) as pbar:
#         for future in concurrent.futures.as_completed(futures):
#             chunk = futures[future]
#             try:
#                 # Raises exception if occurred in the thread
#                 future.result()
#             except Exception as e:
#                 _logger.debug(
#                     f"Failed to download chunk {chunk.index} for {chunk.path}: {e}. "
#                     f"The download of this chunk will be retried later."
#                 )
#                 # failed_downloads[chunk] = future.exception()
#                 failed_downloads[chunk.index] = e
#             else:
#                 pbar.update()

#     return failed_downloads


def retry_failed_chunks(
    failed_chunks: list[_Chunk],
    http_uri: str,
    headers: dict,
    max_retries: int = 3,
    retry_interval: float = 5.0,
) -> None:
    """
    Retries downloading failed chunks sequentially.

    Parameters
    ----------
    failed_chunks : list of _Chunk
        Failed download chunks.
    http_uri : str
        URI to fetch the chunks from.
    headers : dict
        HTTP headers.
    max_retries : int
        Maximum retries.
    retry_interval : float
        Seconds between retries.

    Raises
    ------
    Exception
        If all retries fail.
    """
    max_retries = _SKPLT_MPD_NUM_RETRIES.get() or max_retries
    retry_interval = _SKPLT_MPD_RETRY_INTERVAL_SECONDS.get() or retry_interval

    for chunk in failed_chunks:
        _logger.info(f"Retrying chunk {chunk.index}")
        for attempt in range(max_retries):
            try:
                download_chunk_direct(chunk, headers, http_uri)
                _logger.info(f"Chunk {chunk.index} succeeded on attempt {attempt + 1}")
                break
            except Exception as e:
                _logger.warning(
                    f"Attempt {attempt + 1} failed for chunk {chunk.index}: {e}"
                )
                if attempt == max_retries - 1:
                    raise


def download_chunk_retries(
    *,
    chunks: list[_Chunk],
    http_uri: str,
    headers: dict,
    download_path: str,
    max_retries: int = 3,
    retry_interval: float = 5.0,
) -> None:
    """
    Retry downloading failed chunks sequentially with retries and intervals.

    Parameters
    ----------
    chunks : list[_Chunk]
        List of chunks that failed to download.
    http_uri : str
        HTTP URI of the file to download.
    headers : dict
        HTTP headers to include in requests.
    download_path : str
        Local path to save the file.
    max_retries : int, default=3
        Number of retry attempts per chunk.
    retry_interval : float, default=5.0
        Seconds to wait between retries.

    Raises
    ------
    Exception
        If all retries fail for a chunk, the exception is propagated.
    """
    max_retries = _SKPLT_MPD_NUM_RETRIES.get() or max_retries
    retry_interval = _SKPLT_MPD_RETRY_INTERVAL_SECONDS.get() or retry_interval
    for chunk in chunks:
        _logger.info(f"Retrying download of chunk {chunk.index} for {chunk.path}")
        for attempt in range(max_retries):
            try:
                download_chunk(
                    range_start=chunk.start,
                    range_end=chunk.end,
                    headers=headers,
                    download_path=download_path,
                    http_uri=http_uri,
                )
                _logger.info(
                    f"Successfully downloaded chunk {chunk.index} for {chunk.path}"
                )
                break  # Success: break retry loop
            except Exception as e:
                _logger.warning(
                    f"Attempt {attempt + 1} failed for chunk {chunk.index}: {e}"
                )
                if attempt == max_retries - 1:
                    # Raise exception if max retries exceeded
                    raise
                _time.sleep(retry_interval)


# --- JSON ---


def read_json(
    root: str,
    file_name: str,
) -> dict[str, any]:
    """
    Read JSON content from a file and return it as a Python dictionary.

    Parameters
    ----------
    root : str
        Directory path where the JSON file is located.
    file_name : str
        Name of the JSON file to read.

    Returns
    -------
    dict[str, Any]
        The data parsed from the JSON file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    json.JSONDecodeError
        If the file content is not valid JSON.

    Examples
    --------
    >>> data = read_json("/configs", "settings.json")
    """
    file_path = _os.path.join(root, file_name)

    with open(file_path, "r", encoding="utf-8") as f:
        return _json.load(f)


def write_json(
    root: str,
    file_name: str,
    data: dict[str, any],
    overwrite: bool = False,
    ensure_json_extension: bool = True,
    indent: int = 4,
) -> None:
    """
    Write a Python dictionary to a JSON file.

    Parameters
    ----------
    root : str
        Directory path where the JSON file will be saved.
    file_name : str
        Name of the JSON file to write.
    data : dict[str, Any]
        Python dictionary data to serialize to JSON.
    overwrite : bool, optional (default=False)
        If False, raises FileExistsError if the file already exists.
        If True, overwrites the existing file.
    ensure_json_extension : bool, optional (default=True)
        If True and file_name does not end with '.json', appends '.json'.
    indent : int, optional (default=4)
        Number of spaces for JSON indentation (pretty printing).

    Returns
    -------
    None
        Writes the JSON content to the specified file path.

    Raises
    ------
    FileExistsError
        If `overwrite` is False and the target file already exists.

    Examples
    --------
    >>> config = {"model": "ModelSight", "version": 1.0}
    >>> write_json("/etc/configs", "settings.json", config, overwrite=True)
    """
    if ensure_json_extension and not file_name.lower().endswith(".json"):
        file_name += ".json"

    file_path = _os.path.join(root, file_name)

    if not overwrite and _os.path.exists(file_path):
        raise FileExistsError(
            f"File '{file_path}' already exists and overwrite is False."
        )

    with open(file_path, "w", encoding="utf-8") as f:
        _json.dump(data, f, indent=indent)


# --- YAML ---


def read_yaml(
    root: str,
    file_name: str,
) -> dict[str, any]:
    """
    Load a YAML file and parse it into a Python dictionary.

    This function reads a YAML file from the specified directory and returns
    its contents as a Python dictionary. It uses `yaml.safe_load` to safely
    parse the YAML content.

    Parameters
    ----------
    root : str
        Directory path where the YAML file is located.
    file_name : str
        Name of the YAML file to load.

    Returns
    -------
    dict[str, Any]
        Parsed YAML content as a dictionary.

    Notes
    -----
    - Requires PyYAML package (`pip install pyyaml`).
    - Uses `safe_load` to avoid executing arbitrary code.
    - File is opened in text mode ('r') with system default encoding.
    - NEVER TOUCH THIS FUNCTION. KEPT FOR BACKWARD COMPATIBILITY with
      databricks-feature-engineering<=0.10.2

    Examples
    --------
    >>> config = read_yaml("/etc/configs", "settings.yaml")
    >>> print(config["database"]["host"])
    localhost

    """
    import yaml

    file_path = _os.path.join(root, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(
    root: str,
    file_name: str,
    data: dict[str, any],
    overwrite: bool = False,
    sort_keys: bool = True,
    ensure_yaml_extension: bool = True,
) -> None:
    """
    Write a Python dictionary to a YAML file.

    This function serializes a Python dictionary to a YAML file, supporting
    options to prevent overwriting existing files and to enforce `.yaml` file extension.

    Parameters
    ----------
    root : str
        Directory path where the YAML file will be saved.
    file_name : str
        Name of the YAML file to write.
    data : dict[str, Any]
        Python dictionary data to serialize to YAML.
    overwrite : bool, optional (default=False)
        If False, raises FileExistsError if the file already exists.
        If True, overwrites the existing file.
    sort_keys : bool, optional (default=True)
        Whether to sort dictionary keys in the output YAML.
    ensure_yaml_extension : bool, optional (default=True)
        If True and file_name does not end with '.yaml' or '.yml', appends '.yaml'.

    Returns
    -------
    None
        Writes the YAML content to the specified file path.

    Raises
    ------
    FileExistsError
        If `overwrite` is False and the target file already exists.

    Notes
    -----
    - Requires PyYAML package (`pip install pyyaml`).
    - Uses `safe_dump` for safe serialization.
    - The file is opened in write mode ('w') with system default encoding.
    - The output YAML is human-readable (block style).
    - NEVER TOUCH THIS FUNCTION. KEPT FOR BACKWARD COMPATIBILITY with
      databricks-feature-engineering<=0.10.2

    Examples
    --------
    >>> config = {"model": "ModelSight", "version": 1.0}
    >>> write_yaml("/etc/configs", "settings.yaml", config, overwrite=True)
    """
    import yaml

    if ensure_yaml_extension and not file_name.lower().endswith((".yaml", ".yml")):
        file_name += ".yaml"

    file_path = _os.path.join(root, file_name)

    if not overwrite and _os.path.exists(file_path):
        raise FileExistsError(
            f"File '{file_path}' already exists and overwrite is False."
        )

    with open(file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=sort_keys,
        )


# --- parquet_as_pandas ---

# def read_parquet_as_pandas_df(data_parquet_path: str):
#     """Deserialize and load the specified parquet file as a Pandas DataFrame.

#     Args:
#         data_parquet_path: String, path object (implementing _os.PathLike[str]),
#             or file-like object implementing a binary read() function. The string
#             could be a URL. Valid URL schemes include http, ftp, s3, gs, and file.
#             For file URLs, a host is expected. A local file could
#             be: file://localhost/path/to/table.parquet. A file URL can also be a path to a
#             directory that contains multiple partitioned parquet files. Pyarrow
#             support paths to directories as well as file URLs. A directory
#             path could be: file://localhost/path/to/tables or s3://bucket/partition_dir.

#     Returns:
#         pandas dataframe
#     """
#     import pandas as pd

#     return pd.read_parquet(data_parquet_path, engine="pyarrow")


# def write_pandas_df_as_parquet(df, data_parquet_path: str):
#     """Write a DataFrame to the binary parquet format.

#     Args:
#         df: pandas data frame.
#         data_parquet_path: String, path object (implementing _os.PathLike[str]),
#             or file-like object implementing a binary write() function.

#     """
#     df.to_parquet(data_parquet_path, engine="pyarrow")


# def write_spark_dataframe_to_parquet_on_local_disk(spark_df, output_path):
#     """Write spark dataframe in parquet format to local disk.

#     Args:
#         spark_df: Spark dataframe.
#         output_path: Path to write the data to.

#     """
#     from mlflow.utils.databricks_utils import is_in_databricks_runtime

#     if is_in_databricks_runtime():
#         dbfs_path = _os.path.join(".mlflow", "cache", str(uuid.uuid4()))
#         spark_df.coalesce(1).write.format("parquet").save(dbfs_path)
#         _shutil.copytree("/dbfs/" + dbfs_path, output_path)
#         _shutil.rmtree("/dbfs/" + dbfs_path)
#     else:
#         spark_df.coalesce(1).write.format("parquet").save(output_path)
