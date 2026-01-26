# scikitplot/utils/_path.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Path and filename utilities.

This module provides small, robust helpers for generating portable and
collision-resistant file and folder names. The default format is designed to be:

- lexicographically sortable by timestamp (UTC)
- safe across Windows/macOS/Linux filesystems
- collision-resistant across threads/processes/machines

The core building block is `PathNamer`, plus a zero-argument convenience wrapper
`make_path`.

Filename format:

- ``[ {prefix}- ]{YYYYMMDDTHHMMSSmmmZ}-{counter:06d}-{uuid4hex}[ -{secret} ][ -{suffix} ][ .{ext} ]``
"""

from __future__ import annotations

import itertools
import os
import re
import secrets
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported during type checking
    from typing import Optional  # noqa: F401

__all__ = [
    "PathNamer",
    "make_path",
    "make_temp_path",
    "normalize_directory_path",
    "normalize_extension",
    "sanitize_path_component",
]

# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------

# Counter breaks ties when multiple names are generated within the same
# millisecond. UUID4 already makes collisions extremely unlikely; the counter
# also improves local ordering and protects tight loops.
_COUNTER = itertools.count()
_COUNTER_LOCK = threading.Lock()

# Comment contains ambiguous `–` (EN DASH). Did you mean `-` (HYPHEN-MINUS)?  # noqa: RUF003
# Windows invalid path characters, plus ASCII control characters (0x00–0x1F).  # noqa: RUF003
_INVALID_WIN_CHARS = r'<>:"/\\|?*\x00-\x1F'
_INVALID_RE = re.compile(f"[{_INVALID_WIN_CHARS}]+")

# Windows reserved device names (case-insensitive).
_RESERVED_WIN_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *{f"COM{i}" for i in range(1, 10)},
    *{f"LPT{i}" for i in range(1, 10)},
}


def _ensure_aware_utc(now: datetime | None) -> datetime:
    """
    Return a timezone-aware UTC datetime.

    If `now` is None, current UTC time is used.
    If `now` is naive, it is treated as UTC.

    Parameters
    ----------
    now : datetime or None
        Datetime to normalize.

    Returns
    -------
    now_utc : datetime
        Timezone-aware datetime in UTC.
    """
    if now is None:
        return datetime.now(timezone.utc)  # .timestamp(), .isoformat()
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _utc_timestamp_ms(now_utc: datetime) -> str:
    """Format a UTC datetime as YYYYMMDDTHHMMSSmmmZ."""
    # Example: 20260101T031522123Z
    return now_utc.strftime("%Y%m%dT%H%M%S") + f"{now_utc.microsecond // 1000:03d}Z"


def _uuid4_hex():
    return uuid.uuid4().hex


def _secret_token_hex():
    # 8 bytes => 16 hex chars (stronger opacity than the earlier 4 bytes).
    return secrets.token_hex(8)


def _next_counter(modulus: int = 1_000_000) -> int:
    """Return the next counter value (modulo)."""
    with _COUNTER_LOCK:
        return next(_COUNTER) % modulus


# -----------------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------------


def normalize_directory_path(
    path: str | Path,
    *,
    expand_user: bool = True,
    expand_vars: bool = True,
    resolve: bool = True,
) -> Path:
    """
    Normalize a directory path for use as an output root.

    Expand "~" and env vars, then normalize to an absolute path without
    requiring the file to already exist.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory path. Supports "~" user expansion and optional environment
        variable expansion.
    expand_user : bool, default=True
        Expand "~" to the current user's home directory.
    expand_vars : bool, default=True
        Expand environment variables like "$HOME" on POSIX or "%USERPROFILE%"
        on Windows.
    resolve : bool, default=True
        If True, normalize "." / ".." and return an absolute path using
        Path.resolve(strict=False).

    Returns
    -------
    normalized : pathlib.Path
        Normalized directory path.

    Examples
    --------
    >>> from scikitplot.utils._path import normalize_directory_path
    >>> p = normalize_directory_path("~/artifacts")
    >>> isinstance(p, Path)
    True

    See Also
    --------
    pathlib.Path.expanduser : Expand "~".
    pathlib.Path.resolve : Normalize dot segments and make absolute.
    """
    p = Path(os.fspath(path))

    if expand_user:
        p = p.expanduser()

    if expand_vars:
        p = Path(os.path.expandvars(str(p)))

    if resolve:
        p = p.resolve(strict=False)

    return p


def sanitize_path_component(
    name: str | None,
    *,
    default: str = "",
    max_len: int = 80,
) -> str:
    """
    Sanitize a single path component (file or folder name).

    The result is portable across common filesystems and avoids Windows-specific
    pitfalls (invalid characters, reserved device names, trailing dot/space).

    Parameters
    ----------
    name : str or None
        Input name to sanitize. If None or empty after sanitization, `default`
        is used.
    default : str, default=""
        Fallback name when the input is empty or becomes empty after cleanup.
    max_len : int, default=80
        Maximum length of the resulting component.

    Returns
    -------
    component : str
        Sanitized component string.

    See Also
    --------
    normalize_extension : Normalize a filename extension.
    normalize_directory_path : Expand "~" and environment variables for roots.
    pathlib.Path : Path manipulation utilities.

    Notes
    -----
    - Spaces are converted to underscores.
    - Invalid characters (including path separators) are replaced with "_".
    - Multiple underscores are collapsed.
    - "." and ".." are not allowed as components (replaced by `default`).
    - Trailing dots/spaces are removed (Windows restriction).

    Examples
    --------
    Basic sanitization:

    >>> from scikitplot.utils._path import sanitize_path_component
    >>> sanitize_path_component("my report: v1")
    'my_report_v1'
    >>> sanitize_path_component("..", default="data")
    'data'

    Windows-reserved names are avoided:

    >>> sanitize_path_component("CON")
    'CON_'

    Empty values fall back to the default:

    >>> sanitize_path_component("file")
    'file'
    """
    s = (name or "").strip().replace(" ", "_")

    # Replace invalid characters with underscores (portable & predictable).
    s = _INVALID_RE.sub("_", s)

    # Collapse underscores.
    s = re.sub(r"_+", "_", s).strip()

    # Explicitly disallow dot-directory components.
    if s in {".", ".."}:
        s = ""

    # Windows forbids trailing dot/space.
    s = s.rstrip(" .")

    if not s:
        s = (default or "").strip()

    # Avoid Windows reserved device names.
    if s.upper() in _RESERVED_WIN_NAMES:
        s = f"{s}_"

    return s[:max_len]


def normalize_extension(ext: str | None) -> str:
    """
    Normalize a filename extension.

    Parameters
    ----------
    ext : str or None
        Extension like ``"csv"`` or ``".csv"``. If None/empty, returns ``""``.

    Returns
    -------
    normalized : str
        Normalized extension including a leading dot, or ``""``.

    See Also
    --------
    sanitize_path_component : Sanitize a file/folder component safely.

    Notes
    -----
    This function does not attempt to validate content types; it only normalizes
    the string form.

    Examples
    --------
    >>> from scikitplot.utils._path import normalize_extension
    >>> normalize_extension("csv")
    '.csv'
    >>> normalize_extension(".parquet")
    '.parquet'
    >>> normalize_extension("")
    ''
    """
    e = (ext or "").strip()
    if not e:
        return ""
    return e if e.startswith(".") else f".{e}"


@dataclass(frozen=True)
class PathNamer:
    """
    Generate portable, collision-resistant filenames and paths.

    Parameters
    ----------
    prefix : str, default=""
        Default filename prefix (sanitized before use).
    suffix : str, default=""
        Default filename suffix (sanitized before use).
    ext : str, default=""
        Default extension, with or without a leading dot (e.g., ``"csv"``).
    root : pathlib.Path, default=Path("scikitplot-artifacts")
        Base directory where paths are created. "~" and env vars are expanded.
    by_day : bool, default=False
        If True, nest outputs under ``YYYY/MM/DD`` using UTC dates.
    add_secret : bool, default=False
        If True, append a cryptographically strong random token, making names
        harder to guess. UUID4 already provides uniqueness; this option is for
        opacity when filenames may be exposed publicly.
    private : bool, default=False
        If True, force adding an extra random token (opacity). This is a more
        user-friendly alias for "make it hard to guess" and implies secret token
        behavior regardless of `add_secret`.
    mkdir : bool, default=True
        If True, create the target directory (parents included).

    See Also
    --------
    make_path : Convenience wrapper callable with zero arguments.
    uuid.uuid4 : UUID generator used for collision resistance.
    secrets.token_hex : Optional token source when `add_secret=True`.

    Notes
    -----
    Generated filename format (default):

    ``[ {prefix}- ]{YYYYMMDDTHHMMSSmmmZ}-{counter:06d}-{uuid4hex}[ -{secret} ][ -{suffix} ][ .{ext} ]``

    Examples
    --------
    >>> import scikitplot.utils as sp

    >>> with sp.Timer(verbose=True, logging_level="debug"):
    ...     sp.PathNamer()

    >>> with sp.Timer(logging_level="debug"):
    ...     sp.PathNamer().make_filename()

    >>> with sp.Timer(logging_level="info"):
    ...     sp.PathNamer().make_path()

    >>> with sp.Timer(logging_level="warn"):
    ...     sp.make_path()

    Create a path with defaults (UTC date folders):

    >>> from scikitplot.utils._path import PathNamer
    >>> namer = PathNamer()
    >>> path = namer.make_path(prefix="report")
    >>> path.parts[-1].startswith("report-")
    True

    Write outputs under a project directory:

    >>> from pathlib import Path
    >>> namer = PathNamer(
    ...     root=Path("artifacts"), prefix="report", suffix="report", ext="csv"
    ... )
    >>> p = namer.make_path()
    >>> p.as_posix().startswith("artifacts/")
    True
    >>> namer = PathNamer(root="~/artifacts", prefix="run", suffix="report", ext="json")
    >>> p = namer.make_path()
    >>> p.name.startswith("run-")
    True

    Disable date folders and group under a custom subdirectory:

    >>> namer = PathNamer(
    ...     root=Path("scikitplot-artifacts"),
    ...     by_day=False,
    ...     prefix="snapshot",
    ...     ext="parquet",
    ... )
    >>> p = namer.make_path(subdir="models")
    >>> "models" in p.as_posix()
    True

    Generate only the filename (no directory):

    >>> fname = namer.make_filename(prefix="metrics", ext="json")
    >>> fname.startswith("metrics-")
    True
    >>> fname.endswith(".json")
    True

    Private (unguessable) names:

    >>> namer = PathNamer(
    ...     root="scikitplot-artifacts",
    ...     prefix="report",
    ...     ext="csv",
    ...     private=True,
    ... )
    >>> p = namer.make_path()
    >>> p.name.endswith(".csv")
    True
    """

    root: Path = Path("scikitplot-artifacts")
    prefix: str = ""
    suffix: str = ""
    ext: str = ""
    by_day: bool = False
    add_secret: bool = False
    private: bool = False
    mkdir: bool = True

    def make_filename(
        self,
        prefix: str | None = None,
        suffix: str | None = None,
        ext: str | None = None,
        *,
        now: datetime | None = None,
    ) -> str:
        """
        Create a unique, portable filename.

        Parameters
        ----------
        prefix : str or None, default=None
            Filename prefix. If None, uses `prefix` attr.
        suffix : str or None, default=None
            Filename suffix. If None, uses `suffix` attr.
        ext : str or None, default=None
            File extension. If None, uses `ext` attr.
        now : datetime or None, default=None
            Timestamp to use. If None, uses current time in UTC.
            Naive datetimes are treated as UTC.

        Returns
        -------
        filename : str
            A filename (no directory) suitable for common filesystems.
        """
        prefix_s = sanitize_path_component(
            prefix or self.prefix,
            default=self.prefix,
        )
        suffix_s = sanitize_path_component(
            suffix or self.suffix,
            default=self.suffix,
        )
        ext_s = normalize_extension(ext if ext is not None else self.ext)

        now_utc = _ensure_aware_utc(now)
        ts = _utc_timestamp_ms(now_utc)
        ctr = _next_counter()
        uid = _uuid4_hex()
        token = _secret_token_hex()  # 8 hex chars

        # ``[ {prefix}- ]{YYYYMMDDTHHMMSSmmmZ}-{counter:06d}-{uuid4hex}[ -{secret} ][ -{suffix} ][ .{ext} ]``
        # Filtering with None (Truthiness Check)
        parts = [prefix_s, ts, f"{ctr:06d}", uid]
        if self.private or self.add_secret:
            parts.append(token)
        parts += [suffix_s]

        return "-".join(filter(None, parts)) + ext_s

    def make_path(
        self,
        prefix: str | None = None,
        suffix: str | None = None,
        ext: str | None = None,
        *,
        subdir: str | None = None,
        now: datetime | None = None,
    ) -> Path:
        """
        Create a full path (folder + unique filename).

        Parameters
        ----------
        prefix : str or None, default=None
            Filename prefix. If None, uses `prefix` attr.
        suffix : str or None, default=None
            Filename suffix. If None, uses `suffix` attr.
        ext : str or None, default=None
            File extension. If None, uses `ext` attr.
        subdir : str or None, default=None
            Optional subdirectory (sanitized) when `by_day=False`.
        now : datetime or None, default=None
            Timestamp to use. If None, uses current time in UTC.
            Naive datetimes are treated as UTC.

        Returns
        -------
        path : pathlib.Path
            Full path to a unique file location.

        See Also
        --------
        make_filename : Build only the filename portion.
        pathlib.Path.mkdir : Directory creation.

        Notes
        -----
        A single timestamp (`now`) is used for both folder selection and filename,
        preventing mismatches at day boundaries.
        """
        base = normalize_directory_path(self.root)
        now_utc = _ensure_aware_utc(now)

        if self.by_day:
            base = base / now_utc.strftime("%Y/%m/%d")
        elif subdir:
            base = base / sanitize_path_component(subdir, default="data", max_len=40)

        if self.mkdir:
            base.mkdir(parents=True, exist_ok=True)

        return base / self.make_filename(
            prefix=prefix, suffix=suffix, ext=ext, now=now_utc
        )


_DEFAULT_NAMER = PathNamer()


def make_path(
    prefix: str = _DEFAULT_NAMER.prefix,
    suffix: str = _DEFAULT_NAMER.suffix,
    ext: str = _DEFAULT_NAMER.ext,
    root: str | Path = _DEFAULT_NAMER.root,
    *,
    by_day: bool = _DEFAULT_NAMER.by_day,
    add_secret: bool = _DEFAULT_NAMER.add_secret,
    private: bool = _DEFAULT_NAMER.private,
    mkdir: bool = _DEFAULT_NAMER.mkdir,
    subdir: str | None = None,
    now: datetime | None = None,
) -> Path:
    """
    Make Convenience wrapper to build a unique path (callable with zero args).

    Parameters
    ----------
    prefix : str, default=""
        Filename prefix.
    suffix : str, default=""
        Filename suffix.
    ext : str, default=""
        File extension (e.g., ``"csv"``).
    root : str or pathlib.Path, default=Path("scikitplot-artifacts")
        Base output directory.
    by_day : bool, default=False
        If True, nest outputs under ``YYYY/MM/DD`` in UTC.
    add_secret : bool, default=False
        If True, append an extra random token for opacity.
    private : bool, default=False
        If True, append an extra random token for opacity.
    mkdir : bool, default=True
        If True, create the output directory if needed.
    subdir : str or None, default=None
        Optional subdirectory when `by_day=False`.
    now : datetime or None, default=None
        Timestamp to use. If None, uses current UTC time. Naive datetimes are
        treated as UTC.

    Returns
    -------
    path : pathlib.Path
        Full path to a unique file location.

    See Also
    --------
    PathNamer : Configurable generator for repeated use.

    Notes
    -----
    For repeated use with the same configuration, prefer :class:`PathNamer`
    to avoid re-specifying parameters.

    If `private=True`, a random token is appended to make names hard to guess.
    This affects naming only (not OS-level file permissions).

    Examples
    --------
    Zero-argument usage:

    >>> from scikitplot.utils._path import make_path
    >>> p = make_path()
    >>> p.name.startswith("file-")
    True

    Choose a prefix and extension:

    >>> p = make_path(prefix="report", ext="csv")
    >>> p.name.endswith(".csv")
    True

    Disable daily folders and write under a stable subdirectory:

    >>> p = make_path(
    ...     root="scikitplot-artifacts",
    ...     by_day=False,
    ...     subdir="runs",
    ...     prefix="run",
    ...     ext="json",
    ... )
    >>> "runs" in p.as_posix()
    True

    Expand "~" and create a private filename:

    >>> p = make_path(root="~/artifacts", prefix="run", ext="json", private=True)
    >>> p.name.endswith(".json")
    True
    """
    namer = PathNamer(
        prefix=prefix,
        suffix=suffix,
        ext=ext,
        root=normalize_directory_path(root),
        by_day=by_day,
        add_secret=add_secret,
        private=private,
        mkdir=mkdir,
    )
    return namer.make_path(subdir=subdir, now=now)


def make_temp_path(
    prefix: str = _DEFAULT_NAMER.prefix,
    suffix: str = _DEFAULT_NAMER.suffix,
    ext: str = _DEFAULT_NAMER.ext,
    root: str | Path = _DEFAULT_NAMER.root,
):
    fd, temp_build_path = tempfile.mkstemp(
        prefix=prefix + suffix,
        suffix=ext,
        dir=root or Path.cwd(),
    )
    os.close(fd)
    return temp_build_path


######################################################################
## get_result_image_path
######################################################################


def _filename_extension_normalizer(
    filename: str,
    ext: str | None = None,
    allowed_exts: list[str] | None = None,
) -> tuple[str, str]:
    """
    Normalize a filename and ensure it has a valid extension.

    Ensures that the given filename has a valid extension from a predefined list.
    If not provided or not found, it falls back to a default extension.

    Parameters
    ----------
    filename : str
        The input filename, with or without an extension.
    ext : str, optional
        An explicit extension to use. If provided, it overrides the one in `filename`.
    allowed_exts : list of str, optional
        A tuple of allowed extensions. Defaults to (".png", ".jpg", ".jpeg", ".pdf").

    Returns
    -------
    tuple of (str, str)
        - `filename`: The filename without extension.
        - `ext`: The normalized extension (with leading dot).

    Examples
    --------
    >>> _filename_extension_normalizer("chart.png")
    ('chart', '.png')

    >>> _filename_extension_normalizer("photo", ext=".jpg")
    ('photo', '.jpg')

    >>> _filename_extension_normalizer("document.PDF", allowed_exts=(".pdf",))
    ('document', '.pdf')

    >>> _filename_extension_normalizer("archive", ext="zip")
    ('archive', '.png')  # fallback to ext

    >>> _filename_extension_normalizer("output")
    ('output', '.png')  # Uses ext

    Notes
    -----
    - Case-insensitive matching of allowed extensions is applied.
    - The returned extension always includes the leading dot.
    - If the provided extension is not allowed, `ext` is used.
    - Errors in parsing are safely caught, and the default extension is used.
    """
    allowed_exts = allowed_exts or []
    filename_lower = filename.lower()
    try:
        if ext is None:
            for allowed in allowed_exts:
                if filename_lower.endswith(allowed.lower()):
                    filename, ext = os.path.splitext(filename)  # noqa: PTH122
                    break
            if ext in [None, ""]:
                filename, ext = os.path.splitext(filename)  # noqa: PTH122
        elif ext and not ext.startswith(".") and not ext.endswith("."):
            ext = f".{ext}"
        return filename, ext
    except Exception:
        # Gracefully fallback if anything goes wrong
        return filename, ext


def _filename_sanitizer(filename: str) -> str:
    """
    Sanitize filename by removing or replacing invalid characters for most filesystems.

    Parameters
    ----------
    filename : str
        The original filename.

    Returns
    -------
    str
        A sanitized filename safe for saving.
    """
    # Replace invalid characters with '_'
    return re.sub(r'[<>:"/\\|?*]', "_", filename)


def _filename_uniquer(full_path, file_path, filename):
    """
    Check if the file already exists, and if so, modify the filename to avoid overwriting.

    Parameters
    ----------
    full_path : str
        The complete path of the file to check.
    file_path : str
        The directory where the file should be saved.
    filename : str
        The base filename to check.

    Returns
    -------
    tuple
        A tuple containing the full path, file path, and filename,
        ensuring uniqueness.
    """
    base, extension = os.path.splitext(filename)  # noqa: PTH122
    counter = 1
    while os.path.exists(full_path):  # noqa: PTH110
        new_filename = f"{base}_{counter}{extension}"
        full_path = os.path.join(file_path, new_filename)  # noqa: PTH118
        counter += 1
    filename = os.path.basename(full_path)  # noqa: PTH119
    return full_path, file_path, filename


def get_path(  # noqa: D417
    *,
    filename=None,
    ext=None,
    file_path=None,
    subfolder=None,
    add_timestamp=False,
    return_parts=False,
    overwrite=True,
    verbose=False,
    **kwargs,
):
    """
    Generate a full file path for saving result images, ensuring the target directory exists.

    Parameters
    ----------
    filename : str, optional
        Base name of the image file. Defaults to 'plot_image'.
    ext : str, optional, default=None
        File extension (e.g., '.png', '.jpg').
        Defaults to try to find `filename` if not fallback to '.png'.
    file_path : str, optional
        Directory path to save the image.
        Defaults to the current working directory.
    subfolder : str, optional
        Optional subdirectory inside the main path.
    add_timestamp : bool, optional, default=False
        Whether to append a timestamp to the filename.
        Default is False.
    overwrite : bool, optional, default=True
        If False and a file exists, auto-increments the filename to avoid overwriting.
    return_parts : bool, optional
        If True, returns (full_path, file_path, filename) instead of just the full path.
    verbose : bool, optional
        If True, prints the final save path.

    Returns
    -------
    str or tuple
        The full file path, or a tuple (full_path, file_path, filename) if return_parts=True.

    Raises
    ------
    ValueError
        If the provided file extension is not supported.
    """
    # Validate file extension
    allowed_exts = (".png", ".jpg", ".jpeg", ".pdf")
    ext = ext or ".png"

    # Default to 'plot_image' if no filename provided
    if filename is None:
        filename = "plot_image"
    filename = _filename_sanitizer(filename)

    filename, ext = _filename_extension_normalizer(filename, ext, allowed_exts)

    if ext.lower() not in allowed_exts:
        raise ValueError(f"Extension '{ext}' not supported. Use one of: {allowed_exts}")

    # Add timestamp to filename if specified
    if add_timestamp:
        if filename.endswith(ext):
            filename = filename.rstrip(ext)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%SZ")
        filename = f"{filename}_{timestamp}"

    # Ensure the extension is included
    if not filename.endswith(ext):
        filename += ext

    # Set the default file path if not provided
    if file_path is None:
        file_path = os.path.join(os.getcwd(), "result_images")  # noqa: PTH109,PTH118

    # Add subfolder to path if provided
    if subfolder:
        file_path = os.path.join(  # noqa: PTH118
            file_path, _filename_sanitizer(subfolder)
        )  # noqa: PTH118

    # Ensure the directory exists
    os.makedirs(file_path, exist_ok=True)  # noqa: PTH103

    # Full path of the file
    full_path = os.path.join(file_path, filename)  # noqa: PTH118

    # Handle file overwriting if the flag is set to False
    if not overwrite:
        full_path, file_path, filename = _filename_uniquer(
            full_path, file_path, filename
        )

    # Verbose output for debugging
    if verbose:
        print(f"[INFO] Saving path to: {full_path}")  # noqa: T201

    # Return full path or path components based on the return_parts flag
    if return_parts:
        return full_path, file_path, filename

    return full_path


######################################################################
## remove
######################################################################


def remove_path(
    paths: list[str] | None = None,
    base_path: str | None = None,
) -> None:
    """
    Remove unwanted files or directories from a specified base path.

    Parameters
    ----------
    paths : List[str], optional
        A list of directory or file names to be removed.
        If these exist in the `base_path`, they will be deleted.
        (default ['__MACOSX', 'bank-additional']).

    base_path : str, optional
        The base directory where the unwanted paths will be removed from.
        If None, it defaults to the current working directory (cwd).

    Notes
    -----
    - It checks if the path exists before trying to remove it.
    - Uses `shutil.rmtree()` for directories and `os.remove()` for files.
    - Any exceptions raised during removal will be silently ignored.
    """
    if base_path is None:
        # Default to current working directory
        base_path = os.getcwd()  # noqa: PTH109

    if paths is None:
        paths = [
            "__MACOSX",
            "bank-additional",
        ]  # Default for modelplotpy bank data

    for p in paths:
        try:
            path_to_remove = os.path.join(base_path, p)  # noqa: PTH118

            # Check if it's a file and remove it
            if os.path.isfile(  # noqa: PTH113
                path_to_remove
            ) and os.path.exists(  # noqa: PTH110, PTH113
                path_to_remove
            ):  # noqa: PTH110
                os.remove(path_to_remove)  # noqa: PTH107

            # Check if it's a directory and remove it
            elif os.path.isdir(  # noqa: PTH112
                path_to_remove
            ) and os.path.exists(  # noqa: PTH110, PTH112
                path_to_remove
            ):  # noqa: PTH110
                shutil.rmtree(path_to_remove)
        except Exception:
            # Log the error silently or add specific logging if needed
            pass
