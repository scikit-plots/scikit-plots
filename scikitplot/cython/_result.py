# scikitplot/cython/_result.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Result types for :mod:`scikitplot.cython`.

This module defines small, frozen dataclasses returned by the public API when the
user requests structured metadata (instead of only a loaded module object).

All types in this module are safe for Sphinx ``autodoc`` and ``autosummary``
because every public attribute is explicitly declared.

Notes
-----
The default public API returns ``types.ModuleType`` for ergonomics. Power users
and tooling can request structured results via ``*_result`` functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from types import ModuleType as _ModuleType
from typing import Any, Mapping, Sequence

__all__ = [
    "BatchBuildError",
    "BatchBuildResult",
    "BatchFailure",
    "BuildResult",
    "CacheGCResult",
    "CacheStats",
    "PackageBuildResult",
]


def _uninitialized_module() -> ModuleType:
    """
    Return a sentinel module used only as a dataclass default.

    Notes
    -----
    The public API always returns fully-populated result objects. The sentinel
    exists purely to satisfy documentation tooling that expects dataclass
    attributes to have defaults at the class level.
    """

    return _ModuleType("__scikitplot_cython_uninitialized__")


def _annotation_reports(build_dir: Path, meta: Mapping[str, Any]) -> dict[str, Path]:
    """
    Return the Cython annotation reports of a cache entry that exist on disk.

    Parameters
    ----------
    build_dir : pathlib.Path
        The cache entry's directory.
    meta : Mapping[str, Any]
        The entry's ``meta.json`` content.

    Returns
    -------
    dict of str to pathlib.Path
        Module name to the absolute path of its ``.html`` report, in module
        name order. A module is listed only when its report is a file that
        exists now; the dictionary is empty when the build did not annotate.

    Notes
    -----
    **Developer.** ``meta["annotation_html"]`` maps module name to a path
    *relative to the entry*, for module builds and package builds alike. The
    entry is written in a staging directory and then renamed, and a cache can
    be moved, so only the pair (entry directory, relative path) names the
    report reliably; this is the one place that joins them. Anything else
    under that key (an entry written before the paths were relative held a
    single absolute string) is not a mapping and yields no report.
    """
    recorded = meta.get("annotation_html")
    if not isinstance(recorded, Mapping):
        return {}
    reports: dict[str, Path] = {}
    for name in sorted(key for key in recorded if isinstance(key, str)):
        relative = recorded[name]
        if isinstance(relative, str) and (Path(build_dir) / relative).is_file():
            reports[name] = Path(build_dir) / relative
    return reports


@dataclass(frozen=True, slots=True)
class BuildResult:
    """
    Result of compiling/importing a single Cython extension module.

    Parameters
    ----------
    module : types.ModuleType
        The imported extension module.
    key : str
        Deterministic cache key for this build configuration.
    module_name : str
        Module name the artifact was compiled for (init symbol name).
    build_dir : pathlib.Path
        Cache/build directory containing source, metadata, and artifacts.
    artifact_path : pathlib.Path
        Path to the compiled extension artifact (``.so`` / ``.pyd``).
    used_cache : bool
        Whether an existing artifact was reused without recompilation.
    created_utc : str or None
        ISO 8601 UTC timestamp (``...Z``) if available.
    fingerprint : Mapping[str, Any] or None
        Runtime fingerprint used for caching (Python/Cython/NumPy/platform).
    source_sha256 : str or None
        SHA-256 digest of the main source content, if available.
    meta : Mapping[str, Any]
        The full metadata dictionary persisted in ``meta.json`` for this entry.

    Notes
    -----
    Importing an extension artifact requires the *same* ``module_name`` it was
    compiled for.
    """

    # NOTE: Defaults exist to satisfy documentation tooling that expects class
    # attributes to have defaults. The public API always returns fully-filled
    # results; callers should treat empty/default values as "uninitialized".
    module: ModuleType = field(default_factory=_uninitialized_module)
    key: str = ""
    module_name: str = ""
    build_dir: Path = field(default_factory=lambda: Path("."))
    artifact_path: Path = field(default_factory=lambda: Path("."))
    used_cache: bool = False
    created_utc: str | None = None
    fingerprint: Mapping[str, Any] | None = None
    source_sha256: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def annotation_html(self) -> Path | None:
        """
        Absolute path of this module's Cython annotation report, if there is one.

        Returns
        -------
        pathlib.Path or None
            The ``.html`` file Cython wrote for this module when the build was
            made with ``annotate=True`` (or ``profile="annotate"``); ``None``
            when the build did not annotate or the file is not on disk.

        Notes
        -----
        **User.** Open the file in a browser to read the generated C/C++ next
        to each line of the ``.pyx`` source; lines are shaded by how much they
        call into the Python interpreter. ``meta["annotate_html"]`` holds the
        same file as a path relative to ``build_dir``.

        Examples
        --------
        >>> result = compile_and_load_result(code, annotate=True)  # doctest: +SKIP
        >>> result.annotation_html.name  # doctest: +SKIP
        'scikitplot_cython_0123456789abcdef.html'
        """
        return _annotation_reports(self.build_dir, self.meta).get(self.module_name)

    def __repr__(self) -> str:  # pragma: no cover
        # NOTE: Include *all* declared dataclass fields to keep the repr
        # stable, introspectable, and Sphinx-friendly.
        fingerprint = dict(self.fingerprint) if self.fingerprint is not None else None
        meta = dict(self.meta)
        return (
            "BuildResult("
            f"module={self.module!r}, "
            f"key={self.key!r}, "
            f"module_name={self.module_name!r}, "
            f"build_dir={self.build_dir!r}, "
            f"artifact_path={self.artifact_path!r}, "
            f"used_cache={self.used_cache!r}, "
            f"created_utc={self.created_utc!r}, "
            f"fingerprint={fingerprint!r}, "
            f"source_sha256={self.source_sha256!r}, "
            f"meta={meta!r}"
            ")"
        )


@dataclass(frozen=True, slots=True)
class PackageBuildResult:
    """
    Result of compiling/importing a *package* of extension modules.

    A "package build" compiles multiple Cython extension modules in a single
    build directory under one Python package name (e.g., ``mypkg.mod1``,
    ``mypkg.mod2``). This supports workflows where a logical unit is split across
    several ``.pyx`` modules, and/or a package has multiple extension entrypoints.

    Parameters
    ----------
    package_name : str
        Python package name (e.g., ``"mypkg"``).
    key : str
        Deterministic cache key for this package build configuration.
    build_dir : pathlib.Path
        Cache/build directory containing the package directory and artifacts.
    results : Sequence[BuildResult]
        Per-module build results, in deterministic module-name order.
    used_cache : bool
        Whether an existing artifact set was reused without recompilation.
    created_utc : str or None
        ISO 8601 UTC timestamp (``...Z``) if available.
    fingerprint : Mapping[str, Any] or None
        Runtime fingerprint used for caching (Python/Cython/NumPy/platform).
    meta : Mapping[str, Any]
        The full metadata dictionary persisted in ``meta.json`` for this entry.
    """

    # NOTE: Defaults exist to satisfy documentation tooling that expects class
    # attributes to have defaults. The public API always returns fully-filled
    # results; callers should treat empty/default values as "uninitialized".
    package_name: str = ""
    key: str = ""
    build_dir: Path = field(default_factory=lambda: Path("."))
    results: Sequence[BuildResult] = field(default_factory=tuple)
    used_cache: bool = False
    created_utc: str | None = None
    fingerprint: Mapping[str, Any] | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def modules(self) -> Sequence[ModuleType]:
        """
        Loaded modules in the same order as ``results``.

        Returns
        -------
        Sequence[types.ModuleType]
            Loaded extension modules.
        """
        return [r.module for r in self.results]

    @property
    def annotation_html(self) -> Mapping[str, Path]:
        """
        Absolute paths of the Cython annotation reports, one per module.

        Returns
        -------
        Mapping[str, pathlib.Path]
            Full module name (``"mypkg.mod1"``) to the ``.html`` file Cython
            wrote for it, in module name order; empty when the package was
            built without ``annotate=True``. ``results[i].annotation_html``
            gives the same path for one module.

        Notes
        -----
        **User.** ``meta["annotation_html"]`` holds the same mapping with paths
        relative to ``build_dir``, and ``meta["annotate_html"]`` the first of
        them.
        """
        return _annotation_reports(self.build_dir, self.meta)

    def __repr__(self) -> str:  # pragma: no cover
        # NOTE: Include *all* declared dataclass fields to keep the repr
        # stable, introspectable, and Sphinx-friendly.
        results = tuple(self.results)
        fingerprint = dict(self.fingerprint) if self.fingerprint is not None else None
        meta = dict(self.meta)
        return (
            "PackageBuildResult("
            f"package_name={self.package_name!r}, "
            f"key={self.key!r}, "
            f"build_dir={self.build_dir!r}, "
            f"results={results!r}, "
            f"used_cache={self.used_cache!r}, "
            f"created_utc={self.created_utc!r}, "
            f"fingerprint={fingerprint!r}, "
            f"meta={meta!r}"
            ")"
        )


@dataclass(frozen=True, slots=True)
class CacheStats:
    """
    Cache statistics for the compiled-artifact cache root.

    Parameters
    ----------
    cache_root : pathlib.Path
        Cache root directory.
    n_modules : int
        Number of module entries (``kind == 'module'``).
    n_packages : int
        Number of package entries (``kind == 'package'``).
    total_bytes : int
        Total disk usage in bytes for all entries.
    pinned_aliases : int
        Number of pin aliases in the pin registry.
    pinned_keys : int
        Number of unique pinned keys in the pin registry.
    newest_mtime_utc : str or None
        UTC timestamp (``...Z``) of the newest entry's last-modified time.
    oldest_mtime_utc : str or None
        UTC timestamp (``...Z``) of the oldest entry's last-modified time.
    """

    cache_root: Path = field(default_factory=lambda: Path("."))
    n_modules: int = 0
    n_packages: int = 0
    total_bytes: int = 0
    pinned_aliases: int = 0
    pinned_keys: int = 0
    newest_mtime_utc: str | None = None
    oldest_mtime_utc: str | None = None

    def __repr__(self) -> str:  # pragma: no cover
        # NOTE: Include *all* declared dataclass fields.
        return (
            "CacheStats("
            f"cache_root={self.cache_root!r}, "
            f"n_modules={self.n_modules!r}, "
            f"n_packages={self.n_packages!r}, "
            f"total_bytes={self.total_bytes!r}, "
            f"pinned_aliases={self.pinned_aliases!r}, "
            f"pinned_keys={self.pinned_keys!r}, "
            f"newest_mtime_utc={self.newest_mtime_utc!r}, "
            f"oldest_mtime_utc={self.oldest_mtime_utc!r}"
            ")"
        )


@dataclass(frozen=True, slots=True)
class CacheGCResult:
    """
    Result of a cache garbage-collection operation.

    Parameters
    ----------
    cache_root : pathlib.Path
        Cache root directory.
    deleted_keys : Sequence[str]
        Cache keys deleted (hex digests).
    skipped_pinned_keys : Sequence[str]
        Cache keys preserved because they are pinned.
    skipped_missing_keys : Sequence[str]
        Cache keys requested for deletion but missing on disk.
    skipped_active_keys : Sequence[str]
        Cache keys preserved because a build lock was held (an active build) or
        the entry became pinned during the GC transaction.
    freed_bytes : int
        Estimated bytes freed (best effort, computed pre-delete).
    """

    cache_root: Path = field(default_factory=lambda: Path("."))
    deleted_keys: Sequence[str] = field(default_factory=tuple)
    skipped_pinned_keys: Sequence[str] = field(default_factory=tuple)
    skipped_missing_keys: Sequence[str] = field(default_factory=tuple)
    skipped_active_keys: Sequence[str] = field(default_factory=tuple)
    freed_bytes: int = 0

    def __repr__(self) -> str:  # pragma: no cover
        # NOTE: Include *all* declared dataclass fields.
        deleted_keys = tuple(self.deleted_keys)
        skipped_pinned_keys = tuple(self.skipped_pinned_keys)
        skipped_missing_keys = tuple(self.skipped_missing_keys)
        skipped_active_keys = tuple(self.skipped_active_keys)
        return (
            "CacheGCResult("
            f"cache_root={self.cache_root!r}, "
            f"deleted_keys={deleted_keys!r}, "
            f"skipped_pinned_keys={skipped_pinned_keys!r}, "
            f"skipped_missing_keys={skipped_missing_keys!r}, "
            f"skipped_active_keys={skipped_active_keys!r}, "
            f"freed_bytes={self.freed_bytes!r}"
            ")"
        )


@dataclass(frozen=True)
class BatchFailure:
    """A single failure within a batch build (CYTHON-BATCH-001).

    Parameters
    ----------
    name : str
        The batch key (e.g. file stem) that failed.
    source : pathlib.Path
        The source path that failed to build/import.
    error_type : str
        The exception class name.
    error : str
        The stringified exception message.
    """

    name: str
    source: Path
    error_type: str
    error: str


@dataclass(frozen=True)
class BatchBuildResult:
    """Structured result of a batch compile/import (CYTHON-BATCH-001).

    ``cython_import_all`` was fail-fast: a mid-batch failure left earlier
    compiled artifacts and imported modules committed but returned no report of
    what had already succeeded.  This type makes the partial outcome explicit:
    ordered successes, ordered failures, and the list of side effects that were
    committed before the batch stopped (or completed).

    Parameters
    ----------
    successes : Mapping[str, BuildResult]
        Ordered mapping of batch key → :class:`BuildResult` for each item that
        built and imported successfully.
    failures : Sequence[BatchFailure]
        Ordered failures (empty when every item succeeded).
    committed : Sequence[str]
        Ordered batch keys whose native side effects (compiled artifact +
        imported module) were committed.  Equal to ``list(successes)``; provided
        explicitly so callers can reason about what to roll back or resume.
    policy : str
        The batch policy that produced this result: ``"fail_fast"`` (stop at the
        first failure) or ``"collect"`` (attempt every item).

    Notes
    -----
    - ``ok`` is ``True`` only when there were no failures.
    - ``resume_token`` is the sorted tuple of keys still to attempt after a
      fail-fast stop; feed it back via the ``only`` argument to resume.
    """

    successes: Mapping[str, BuildResult]
    failures: Sequence[BatchFailure]
    committed: Sequence[str]
    policy: str

    @property
    def ok(self) -> bool:
        """Whether the batch completed with no failures."""
        return len(self.failures) == 0

    @property
    def n_succeeded(self) -> int:
        """Number of items that built and imported successfully."""
        return len(self.successes)

    @property
    def n_failed(self) -> int:
        """Number of items that failed."""
        return len(self.failures)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            "BatchBuildResult("
            f"policy={self.policy!r}, "
            f"succeeded={self.n_succeeded}, "
            f"failed={self.n_failed}, "
            f"ok={self.ok}"
            ")"
        )


class BatchBuildError(RuntimeError):
    """Raised by fail-fast batch builds, carrying the partial result.

    Attributes
    ----------
    result : BatchBuildResult
        The partial result up to and including the failure, so callers can see
        what was already committed and resume (CYTHON-BATCH-001).
    """

    def __init__(self, result: BatchBuildResult) -> None:
        self.result = result
        first = result.failures[0] if result.failures else None
        detail = (
            f"{first.name} ({first.error_type}: {first.error})"
            if first is not None
            else "unknown"
        )
        super().__init__(
            f"batch build failed at {detail}; "
            f"{result.n_succeeded} item(s) already committed: "
            f"{list(result.committed)!r}"
        )
