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

- ``{prefix}-{YYYYMMDDTHHMMSSmmmZ}-{counter:06d}-{uuid4hex}[ -{secret} ].{ext}``
"""

from __future__ import annotations

import itertools
import os
import re
import secrets
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

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
    default: str = "file",
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
    default : str, default="file"
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

    >>> sanitize_path_component("")
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
        s = (default or "file").strip() or "file"

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
    default_prefix : str, default="file"
        Default filename prefix (sanitized before use).
    default_ext : str, default=""
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

    ``{prefix}-{YYYYMMDDTHHMMSSmmmZ}-{counter:06d}-{uuid4hex}[ -{secret} ].{ext}``

    Examples
    --------
    Create a path with defaults (UTC date folders):

    >>> from scikitplot.utils._path import PathNamer
    >>> namer = PathNamer()
    >>> path = namer.make_path()
    >>> path.parts[-1].startswith("file-")
    True

    Write outputs under a project directory:

    >>> from pathlib import Path
    >>> namer = PathNamer(
    ...     root=Path("artifacts"), default_prefix="report", default_ext="csv"
    ... )
    >>> p = namer.make_path()
    >>> p.as_posix().startswith("artifacts/")
    True
    >>> namer = PathNamer(root="~/artifacts", default_prefix="run", default_ext="json")
    >>> p = namer.make_path()
    >>> p.name.startswith("run-")
    True

    Disable date folders and group under a custom subdirectory:

    >>> namer = PathNamer(
    ...     root=Path("scikitplot-artifacts"),
    ...     by_day=False,
    ...     default_prefix="snapshot",
    ...     default_ext="parquet",
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
    ...     default_prefix="report",
    ...     default_ext="csv",
    ...     private=True,
    ... )
    >>> p = namer.make_path()
    >>> p.name.endswith(".csv")
    True
    """

    default_prefix: str = "file"
    default_ext: str = ""
    root: Path = Path("scikitplot-artifacts")
    by_day: bool = False
    add_secret: bool = False
    private: bool = False
    mkdir: bool = True

    def make_filename(
        self,
        prefix: str | None = None,
        ext: str | None = None,
        *,
        now: datetime | None = None,
    ) -> str:
        """
        Create a unique, portable filename.

        Parameters
        ----------
        prefix : str or None, default=None
            Filename prefix. If None, uses `default_prefix`.
        ext : str or None, default=None
            File extension. If None, uses `default_ext`.
        now : datetime or None, default=None
            Timestamp to use. If None, uses current time in UTC.
            Naive datetimes are treated as UTC.

        Returns
        -------
        filename : str
            A filename (no directory) suitable for common filesystems.
        """
        prefix_s = sanitize_path_component(
            prefix or self.default_prefix,
            default=self.default_prefix,
        )
        ext_s = normalize_extension(ext if ext is not None else self.default_ext)

        now_utc = _ensure_aware_utc(now)
        ts = _utc_timestamp_ms(now_utc)
        ctr = _next_counter()
        uid = _uuid4_hex()
        token = _secret_token_hex()  # 8 hex chars

        # ``{prefix}-{YYYYMMDDTHHMMSSmmmZ}-{counter:06d}-{uuid4hex}[ -{secret} ].{ext}``
        parts = [prefix_s, ts, f"{ctr:06d}", uid]  # type: list[str]
        if self.private or self.add_secret:
            parts.append(token)

        return "-".join(parts) + ext_s

    def make_path(
        self,
        prefix: str | None = None,
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
            Filename prefix. If None, uses `default_prefix`.
        ext : str or None, default=None
            File extension. If None, uses `default_ext`.
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

        return base / self.make_filename(prefix=prefix, ext=ext, now=now_utc)


_DEFAULT_NAMER = PathNamer()


def make_path(
    prefix: str = _DEFAULT_NAMER.default_prefix,
    ext: str = _DEFAULT_NAMER.default_ext,
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
    prefix : str, default="file"
        Filename prefix.
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
        default_prefix=prefix,
        default_ext=ext,
        root=normalize_directory_path(root),
        by_day=by_day,
        add_secret=add_secret,
        private=private,
        mkdir=mkdir,
    )
    return namer.make_path(subdir=subdir, now=now)


def make_temp_path(
    prefix: str = _DEFAULT_NAMER.default_prefix,
    ext: str = _DEFAULT_NAMER.default_ext,
    root: str | Path = _DEFAULT_NAMER.root,
):
    fd, temp_build_path = tempfile.mkstemp(
        prefix=prefix,
        suffix=ext,
        dir=root or Path.cwd(),
    )
    os.close(fd)
    return temp_build_path
