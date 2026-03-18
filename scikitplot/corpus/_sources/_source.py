"""
scikitplot.corpus._sources._source
====================================
Concrete implementation of :class:`CorpusSource`.

Design invariants
-----------------
* ``CorpusSource`` is immutable (frozen dataclass). Mutating state
  returns a new instance via helper class methods.
* ``iter_entries()`` is always a generator — never loads all paths into
  memory at once. Large directory trees with millions of files are safe.
* All file-system access (``glob``, ``stat``, existence checks) is
  deferred to ``iter_entries()`` so construction is instant and testable
  without touching the filesystem.
* URL manifest files must be UTF-8, one URL per line. Blank lines and
  ``#``-prefixed comment lines are skipped.
* ``source_provenance`` is merged into every yielded :class:`SourceEntry`
  so downstream readers inherit corpus-level metadata (author, title, etc.)
  without repeating it per file.

Python compatibility
--------------------
Python 3.8-3.15. No use of ``match``, walrus operator only in 3.8-safe
positions. ``from __future__ import annotations`` for all annotations.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
import pathlib
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import (  # noqa: F401
    Any,
    Dict,
    Final,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
)

if sys.version_info >= (3, 11):
    from enum import StrEnum as _StrEnumBase
else:

    class _StrEnumBase(str, Enum):  # type: ignore[no-redef]
        """Backport of StrEnum for Python < 3.11."""


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL pattern — basic http/https detection
# ---------------------------------------------------------------------------

_URL_RE: re.Pattern[str] = re.compile(r"^https?://", re.IGNORECASE)

# Glob patterns considered binary / non-text (skipped in auto-discovery)
_SKIP_EXTENSIONS: frozenset[str] = frozenset(
    {".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe", ".bin"}
)

_PREFIX: Final[str] = "**/"


def remove_glob_prefix(pattern: str) -> str:
    if not isinstance(pattern, str):
        raise TypeError("pattern must be str")

    if sys.version_info >= (3, 9):
        return pattern.removeprefix(_PREFIX)

    if pattern.startswith(_PREFIX):
        return pattern[len(_PREFIX) :]

    return pattern


# ===========================================================================
# SourceKind — what kind of source is this entry?
# ===========================================================================


class SourceKind(_StrEnumBase):
    """
    Discriminant for the kind of source an entry represents.

    Values
    ------
    FILE
        A local filesystem path to a single file.
    URL
        An ``http://`` or ``https://`` URL.
    DIRECTORY
        A local directory (expanded into file entries by
        :meth:`CorpusSource.iter_entries`).
    MANIFEST
        A text file containing one URL or file path per line.
    """

    FILE = "file"
    URL = "url"
    DIRECTORY = "directory"
    MANIFEST = "manifest"


# ===========================================================================
# SourceEntry — one resolved (path/url, provenance) pair
# ===========================================================================


@dataclass(frozen=True)
class SourceEntry:
    """
    A single resolved source entry yielded by :meth:`CorpusSource.iter_entries`.

    Parameters
    ----------
    path_or_url : str
        Filesystem path (absolute or relative) or full URL.
    kind : SourceKind
        Whether this is a local file or a URL.
    provenance : dict
        Merged provenance metadata to propagate into
        :meth:`~scikitplot.corpus._base.DocumentReader.create` calls.
    """

    path_or_url: str
    kind: SourceKind
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def is_url(self) -> bool:
        """Return ``True`` if this entry is a URL."""
        return self.kind == SourceKind.URL

    @property
    def as_path(self) -> pathlib.Path:
        """
        Return ``path_or_url`` as a :class:`pathlib.Path`.

        Raises
        ------
        ValueError
            If this entry is a URL (not a local path).
        """
        if self.is_url:
            raise ValueError(
                f"SourceEntry.as_path: entry is a URL, not a path: "
                f"{self.path_or_url!r}."
            )
        return pathlib.Path(self.path_or_url)


# ===========================================================================
# CorpusSource — declarative source descriptor
# ===========================================================================


@dataclass(frozen=True)
class CorpusSource:
    """
    Declarative descriptor for one or more document sources.

    :class:`CorpusSource` is a value object — it describes *where* to
    find documents and what provenance metadata to attach. The actual
    file-system access is deferred to :meth:`iter_entries`.

    Parameters
    ----------
    kind : SourceKind
        What kind of source this is.
    root : pathlib.Path or None
        Base path (directory root, single file, or manifest file).
        ``None`` when ``kind=URL`` without a manifest.
    urls : list[str]
        Explicit list of URLs. Only relevant when ``kind=URL``.
    pattern : str
        Glob pattern used when ``kind=DIRECTORY``. Default: ``"**/*"``.
    recursive : bool
        When ``True``, globs descend into sub-directories. Default: ``True``.
    extensions : list[str] or None
        Whitelist of file extensions (lowercase, with leading dot) to
        include when globbing. ``None`` means accept all. Default: ``None``.
    source_provenance : dict
        Metadata propagated into every yielded :class:`SourceEntry`
        (e.g. ``{"source_title": "Hamlet", "source_author": "Shakespeare"}``).
    follow_symlinks : bool
        Whether to follow symbolic links during directory traversal.
        Default: ``True``.

    See Also
    --------
    scikitplot.corpus._pipeline.CorpusPipeline : Consumes ``CorpusSource``.

    Examples
    --------
    Single file:

    >>> from pathlib import Path
    >>> src = CorpusSource.from_file(Path("article.txt"))
    >>> list(src.iter_entries())
    [SourceEntry(path_or_url='article.txt', kind=<SourceKind.FILE: 'file'>, ...)]

    Directory glob:

    >>> src = CorpusSource.from_directory(Path("corpus/"), pattern="*.txt")
    >>> entries = list(src.iter_entries())

    URL list:

    >>> src = CorpusSource.from_urls(["https://a.com/p1", "https://b.com/p2"])

    URL manifest file:

    >>> src = CorpusSource.from_manifest(Path("urls.txt"))
    """

    kind: SourceKind
    root: pathlib.Path | None = field(default=None)
    urls: list[str] = field(default_factory=list)
    pattern: str = field(default="**/*")
    recursive: bool = field(default=True)
    extensions: list[str] | None = field(default=None)
    source_provenance: dict[str, Any] = field(default_factory=dict)
    follow_symlinks: bool = field(default=True)

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        path: pathlib.Path | str,
        source_provenance: dict[str, Any] | None = None,
    ) -> CorpusSource:
        """
        Create a source for a single local file.

        Parameters
        ----------
        path : pathlib.Path or str
            Path to the file.
        source_provenance : dict, optional
            Provenance metadata merged into every yielded entry.

        Returns
        -------
        CorpusSource
        """
        return cls(
            kind=SourceKind.FILE,
            root=pathlib.Path(path),
            source_provenance=source_provenance or {},
        )

    @classmethod
    def from_directory(
        cls,
        directory: pathlib.Path | str,
        pattern: str = "**/*",
        recursive: bool = True,
        extensions: list[str] | None = None,
        source_provenance: dict[str, Any] | None = None,
        follow_symlinks: bool = True,
    ) -> CorpusSource:
        """
        Create a source that globs a directory.

        Parameters
        ----------
        directory : pathlib.Path or str
            Root directory to glob.
        pattern : str, optional
            Glob pattern relative to *directory*. Default: ``"**/*"``
            (all files recursively).
        recursive : bool, optional
            Whether ``**`` in *pattern* should recurse into
            sub-directories. Default: ``True``.
        extensions : list[str] or None, optional
            Whitelist of lowercase file extensions with leading dot.
            ``None`` accepts all. Default: ``None``.
        source_provenance : dict, optional
            Provenance metadata for all entries. Default: ``{}``.
        follow_symlinks : bool, optional
            Follow symlinks during traversal. Default: ``True``.

        Returns
        -------
        CorpusSource
        """
        return cls(
            kind=SourceKind.DIRECTORY,
            root=pathlib.Path(directory),
            pattern=pattern,
            recursive=recursive,
            extensions=extensions,
            source_provenance=source_provenance or {},
            follow_symlinks=follow_symlinks,
        )

    @classmethod
    def from_urls(
        cls,
        urls: list[str],
        source_provenance: dict[str, Any] | None = None,
    ) -> CorpusSource:
        """
        Create a source from an explicit list of URLs.

        Parameters
        ----------
        urls : list[str]
            List of ``http://`` or ``https://`` URLs.
        source_provenance : dict, optional
            Provenance metadata for all entries.

        Returns
        -------
        CorpusSource

        Raises
        ------
        ValueError
            If *urls* is empty or any entry is not a valid URL.
        """
        if not urls:
            raise ValueError("CorpusSource.from_urls: urls must not be empty.")
        invalid = [u for u in urls if not _URL_RE.match(u)]
        if invalid:
            raise ValueError(
                f"CorpusSource.from_urls: non-URL entries found: {invalid!r}. "
                "All entries must start with 'http://' or 'https://'."
            )
        return cls(
            kind=SourceKind.URL,
            urls=list(urls),
            source_provenance=source_provenance or {},
        )

    @classmethod
    def from_manifest(
        cls,
        manifest_path: pathlib.Path | str,
        source_provenance: dict[str, Any] | None = None,
    ) -> CorpusSource:
        """
        Create a source from a UTF-8 manifest file (one entry per line).

        Lines starting with ``#`` and blank lines are ignored. Each
        non-comment line is treated as either a URL or a filesystem path.

        Parameters
        ----------
        manifest_path : pathlib.Path or str
            Path to the manifest text file.
        source_provenance : dict, optional
            Provenance metadata for all entries.

        Returns
        -------
        CorpusSource

        Raises
        ------
        ValueError
            If the manifest file does not exist.
        """
        path = pathlib.Path(manifest_path)
        if not path.exists():
            raise ValueError(
                f"CorpusSource.from_manifest: manifest file not found: {path}."
            )
        return cls(
            kind=SourceKind.MANIFEST,
            root=path,
            source_provenance=source_provenance or {},
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Assert that this source is internally consistent.

        Raises
        ------
        ValueError
            On any configuration inconsistency.
        """
        if self.kind == SourceKind.FILE:
            if self.root is None:
                raise ValueError("CorpusSource (FILE): root path must be set.")
        elif self.kind == SourceKind.DIRECTORY:
            if self.root is None:
                raise ValueError(
                    "CorpusSource (DIRECTORY): root directory must be set."
                )
            if self.extensions is not None:
                bad = [e for e in self.extensions if not e.startswith(".")]
                if bad:
                    raise ValueError(
                        f"CorpusSource (DIRECTORY): extensions must start with "
                        f"'.'; bad values: {bad!r}."
                    )
        elif self.kind == SourceKind.URL:
            if not self.urls:
                raise ValueError(
                    "CorpusSource (URL): urls must contain at least one entry."
                )
        elif self.kind == SourceKind.MANIFEST:  # noqa: SIM102
            if self.root is None:
                raise ValueError(
                    "CorpusSource (MANIFEST): root (manifest path) must be set."
                )

    # ------------------------------------------------------------------
    # Core iterator
    # ------------------------------------------------------------------

    def iter_entries(self) -> Generator[SourceEntry, None, None]:
        """
        Yield resolved :class:`SourceEntry` objects for this source.

        The generator is lazy — filesystem access happens per-entry, not
        upfront. This keeps memory proportional to working-set size, not
        corpus size.

        Yields
        ------
        SourceEntry
            One entry per file or URL.

        Raises
        ------
        ValueError
            If configuration is invalid (delegated to :meth:`validate`).
        FileNotFoundError
            If a FILE source path does not exist at iteration time.
        """
        self.validate()

        if self.kind == SourceKind.FILE:
            yield from self._iter_file()
        elif self.kind == SourceKind.DIRECTORY:
            yield from self._iter_directory()
        elif self.kind == SourceKind.URL:
            yield from self._iter_urls()
        elif self.kind == SourceKind.MANIFEST:
            yield from self._iter_manifest()

    def _iter_file(self) -> Generator[SourceEntry, None, None]:
        """Yield a single file entry."""
        assert self.root is not None  # noqa: S101
        path = self.root
        if not path.exists():
            raise FileNotFoundError(f"CorpusSource (FILE): file not found: {path}.")
        yield SourceEntry(
            path_or_url=str(path),
            kind=SourceKind.FILE,
            provenance=dict(self.source_provenance),
        )

    def _iter_directory(self) -> Generator[SourceEntry, None, None]:
        """Glob the root directory and yield one entry per matching file."""
        assert self.root is not None  # noqa: S101
        root = self.root
        if not root.is_dir():
            raise FileNotFoundError(
                f"CorpusSource (DIRECTORY): directory not found: {root}."
            )
        pattern = self.pattern
        # Use rglob when the pattern starts with ** for recursive traversal
        if self.recursive and "**" in pattern:
            candidates: Iterator[pathlib.Path] = root.rglob(remove_glob_prefix(pattern))
        else:
            candidates = root.glob(pattern)

        n_yielded = 0
        n_skipped = 0
        for path in candidates:
            if not path.is_file():
                continue
            if not self.follow_symlinks and path.is_symlink():
                n_skipped += 1
                continue
            ext = path.suffix.lower()
            if ext in _SKIP_EXTENSIONS:
                n_skipped += 1
                continue
            if self.extensions is not None and ext not in self.extensions:
                n_skipped += 1
                continue
            n_yielded += 1
            yield SourceEntry(
                path_or_url=str(path),
                kind=SourceKind.FILE,
                provenance=dict(self.source_provenance),
            )

        logger.info(
            "CorpusSource (DIRECTORY): yielded %d entries, skipped %d from %s.",
            n_yielded,
            n_skipped,
            root,
        )

    def _iter_urls(self) -> Generator[SourceEntry, None, None]:
        """Yield one URL entry per item in self.urls."""
        for url in self.urls:
            yield SourceEntry(
                path_or_url=url,
                kind=SourceKind.URL,
                provenance=dict(self.source_provenance),
            )

    def _iter_manifest(self) -> Generator[SourceEntry, None, None]:
        """Read a manifest file and yield one entry per non-comment line."""
        assert self.root is not None  # noqa: S101
        manifest = self.root
        if not manifest.exists():
            raise FileNotFoundError(
                f"CorpusSource (MANIFEST): manifest not found: {manifest}."
            )
        n_yielded = 0
        n_skipped = 0
        for lineno, raw_line in enumerate(
            manifest.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                n_skipped += 1
                continue
            is_url = bool(_URL_RE.match(line))
            kind = SourceKind.URL if is_url else SourceKind.FILE
            if not is_url:
                # Resolve relative paths against manifest directory
                resolved = manifest.parent / pathlib.Path(line)
                line = str(resolved)
            n_yielded += 1
            yield SourceEntry(
                path_or_url=line,
                kind=kind,
                provenance=dict(self.source_provenance),
            )
            logger.debug(
                "CorpusSource (MANIFEST): line %d → %s (%s).",
                lineno,
                line,
                kind.value,
            )

        logger.info(
            "CorpusSource (MANIFEST): yielded %d entries, skipped %d "
            "comment/blank lines from %s.",
            n_yielded,
            n_skipped,
            manifest,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def count(self) -> int:
        """
        Return the total number of entries this source will yield.

        .. warning::
           For ``DIRECTORY`` sources this iterates all matching files.
           For large directory trees (100k+ files) this may be slow.

        Returns
        -------
        int
            Number of entries.
        """
        return sum(1 for _ in self.iter_entries())

    def __repr__(self) -> str:  # noqa: D105
        root_str = str(self.root) if self.root else "—"
        return (
            f"CorpusSource("
            f"kind={self.kind.value!r}, "
            f"root={root_str!r}, "
            f"pattern={self.pattern!r})"
        )


__all__ = [
    "CorpusSource",
    "SourceEntry",
    "SourceKind",
]
