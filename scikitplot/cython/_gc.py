# scikitplot/cython/_gc.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Cache management utilities for :mod:`scikitplot.cython`.

This module provides deterministic, safe cache inspection and garbage collection.

Notes
-----
Deletion is intentionally strict:
- Only directories whose name is a valid cache key (64 hex chars) are removed.
- Pinned keys are never removed.
- Deletion is protected by a cache-root lock to avoid concurrent mutations.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from ._cache import is_valid_key, iter_all_entry_dirs, peek_cache_dir, resolve_cache_dir
from ._lock import build_lock
from ._pins import list_pins
from ._result import CacheGCResult, CacheStats


def _utc_iso_from_epoch(ts: float) -> str:
    # ISO 8601 with Z suffix, second resolution.
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _dir_size_bytes(root: Path) -> int:
    total = 0
    for p in root.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except FileNotFoundError:
            # Concurrent delete; ignore in best-effort stats.
            pass
    return total


def _dir_mtime_epoch(root: Path) -> float:
    try:
        return root.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def cache_stats(cache_dir: str | Path | None = None) -> CacheStats:
    """
    Compute cache statistics.

    Parameters
    ----------
    cache_dir : str or pathlib.Path or None, default=None
        Cache root. If None, uses the default cache location.

    Returns
    -------
    scikitplot.cython.CacheStats
        Cache statistics snapshot.
    """
    root = peek_cache_dir(cache_dir)
    if not root.exists():
        return CacheStats(
            cache_root=root,
            n_modules=0,
            n_packages=0,
            total_bytes=0,
            pinned_aliases=0,
            pinned_keys=0,
            newest_mtime_utc=None,
            oldest_mtime_utc=None,
        )

    pins = list_pins(root)
    pinned_aliases = len(pins)
    pinned_keys = len(set(pins.values()))

    total_bytes = 0
    mtimes: list[float] = []
    n_modules = 0
    n_packages = 0

    for d in iter_all_entry_dirs(root):
        total_bytes += _dir_size_bytes(d)
        mtimes.append(_dir_mtime_epoch(d))
        # classify kind
        meta = d / "meta.json"
        kind = None
        if meta.exists():
            try:
                import json  # noqa: PLC0415

                data = json.loads(meta.read_text(encoding="utf-8"))
                kind = data.get("kind")
            except Exception:
                kind = None
        if kind == "package":
            n_packages += 1
        else:
            n_modules += 1

    newest = _utc_iso_from_epoch(max(mtimes)) if mtimes else None
    oldest = _utc_iso_from_epoch(min(mtimes)) if mtimes else None

    return CacheStats(
        cache_root=root,
        n_modules=n_modules,
        n_packages=n_packages,
        total_bytes=total_bytes,
        pinned_aliases=pinned_aliases,
        pinned_keys=pinned_keys,
        newest_mtime_utc=newest,
        oldest_mtime_utc=oldest,
    )


def gc_cache(  # noqa: PLR0912
    *,
    cache_dir: str | Path | None = None,
    keep_n_newest: int | None = None,
    max_age_days: int | None = None,
    max_bytes: int | None = None,
    dry_run: bool = False,
    lock_timeout_s: float = 60.0,
) -> CacheGCResult:
    """
    Deterministically garbage-collect cached build entries.

    Parameters
    ----------
    cache_dir : str or pathlib.Path or None, default=None
        Cache root. If None, uses the default cache location.
    keep_n_newest : int or None, default=None
        If provided, keep at least the N newest entries (across modules and packages).
    max_age_days : int or None, default=None
        If provided, delete entries older than this many days.
    max_bytes : int or None, default=None
        If provided, delete oldest entries until total cache size is <= max_bytes.
    dry_run : bool, default=False
        If True, do not delete anything; only report what would be deleted.
    lock_timeout_s : float, default=60.0
        Max seconds to wait for a cache-root GC lock.

    Returns
    -------
    scikitplot.cython.CacheGCResult
        GC report.

    Raises
    ------
    ValueError
        If any numeric parameter is invalid.
    """
    if keep_n_newest is not None and keep_n_newest < 0:
        raise ValueError("keep_n_newest must be >= 0")
    if max_age_days is not None and max_age_days < 0:
        raise ValueError("max_age_days must be >= 0")
    if max_bytes is not None and max_bytes < 0:
        raise ValueError("max_bytes must be >= 0")

    root = peek_cache_dir(cache_dir)
    if not root.exists():
        return CacheGCResult(
            cache_root=root,
            deleted_keys=(),
            skipped_pinned_keys=(),
            skipped_missing_keys=(),
            freed_bytes=0,
        )

    root = resolve_cache_dir(root)
    pins = list_pins(root)
    pinned_keys = set(pins.values())

    gc_lock = root / ".gc.lock"
    deleted: list[str] = []
    skipped_pinned: list[str] = []
    skipped_missing: list[str] = []
    freed = 0

    # Collect entries with mtime and size.
    entries: list[tuple[str, Path, float, int]] = []
    for d in iter_all_entry_dirs(root):
        key = d.name
        if not is_valid_key(key):
            continue
        entries.append((key, d, _dir_mtime_epoch(d), _dir_size_bytes(d)))

    # Newest-first ordering for keep rule.
    entries_sorted_newest = sorted(entries, key=lambda t: (-t[2], t[0]))
    keep_set: set[str] = set()
    if keep_n_newest is not None:
        keep_set = {k for (k, _, _, _) in entries_sorted_newest[:keep_n_newest]}

    # Age threshold
    age_cutoff_epoch: float | None = None
    if max_age_days is not None:
        age_cutoff_epoch = time.time() - (max_age_days * 86400.0)

    # Start candidate set: entries not pinned and not forced-keep
    candidates: list[tuple[str, Path, float, int]] = []
    for k, d, mt, sz in entries:
        if k in pinned_keys:
            skipped_pinned.append(k)
            continue
        if k in keep_set:
            continue
        if age_cutoff_epoch is not None and mt >= age_cutoff_epoch:
            # not old enough to delete by age
            continue
        candidates.append((k, d, mt, sz))

    # If max_bytes is set, we may need to delete more even if not old-by-age.
    if max_bytes is not None:
        total = sum(sz for (_, _, _, sz) in entries)
        if total > max_bytes:
            # Expand candidate pool with non-pinned, non-keep entries ordered oldest-first
            pool = [
                e for e in entries if e[0] not in pinned_keys and e[0] not in keep_set
            ]
            pool_oldest = sorted(pool, key=lambda t: (t[2], t[0]))  # oldest first
            # delete until under threshold
            need = total - max_bytes
            acc = 0
            extra: list[tuple[str, Path, float, int]] = []
            for e in pool_oldest:
                if e in candidates:
                    continue
                extra.append(e)
                acc += e[3]
                if acc >= need:
                    break
            candidates.extend(extra)

    # Deterministic deletion order: oldest-first
    candidates_sorted = sorted(
        {c[0]: c for c in candidates}.values(), key=lambda t: (t[2], t[0])
    )

    with build_lock(gc_lock, timeout_s=lock_timeout_s):
        for key, d, _, sz in candidates_sorted:
            if key in pinned_keys:
                continue
            if not d.exists():
                skipped_missing.append(key)
                continue
            if dry_run:
                deleted.append(key)
                freed += sz
                continue
            # safety: only delete direct children of cache root with valid key names
            if d.parent != root or not is_valid_key(key):
                continue
            freed += sz
            shutil.rmtree(d, ignore_errors=False)
            deleted.append(key)

    return CacheGCResult(
        cache_root=root,
        deleted_keys=tuple(deleted),
        skipped_pinned_keys=tuple(sorted(set(skipped_pinned))),
        skipped_missing_keys=tuple(sorted(set(skipped_missing))),
        freed_bytes=freed,
    )
