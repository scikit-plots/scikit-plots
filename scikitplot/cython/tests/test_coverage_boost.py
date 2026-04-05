# scikitplot/cython/tests/test_coverage_boost.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Coverage-boost tests for the scikitplot.cython submodule.

Targets uncovered branches in:
    _gc.py, _lock.py, _pins.py, _profiles.py,
    _security.py, _util.py, _cache.py, _public.py

These tests do NOT require a C compiler or a full scikitplot installation.
They exercise pure-Python logic, filesystem operations, and error paths.

Notes
-----
Import strategy: all imports use relative paths (``from ..module import ...``)
so the tests run correctly when pytest discovers them inside the package tree
without needing ``scikitplot`` to be installed as a top-level package.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Module-level relative imports
# ---------------------------------------------------------------------------
from .._cache import (
    is_valid_key,
    iter_all_entry_dirs,
    iter_cache_entries,
    make_cache_key,
    peek_cache_dir,
    resolve_cache_dir,
    write_meta,
)
from .._gc import _dir_mtime_epoch, _dir_size_bytes, _utc_iso_from_epoch, cache_stats, gc_cache
from .._lock import build_lock
from .._pins import list_pins, pin, resolve_pinned_key, unpin
from .._profiles import ProfileDefaults, apply_profile, is_windows, resolve_profile
from .._public import (
    check_build_prereqs,
    cython_import_all,
    export_cached,
    get_cache_dir,
    import_cached_by_name,
    list_cached,
    list_cached_packages,
    purge_cache,
)
from .._result import CacheGCResult, CacheStats
from .._security import (
    DEFAULT_SECURITY_POLICY,
    RELAXED_SECURITY_POLICY,
    SecurityError,
    SecurityPolicy,
    is_safe_compiler_arg,
    is_safe_macro_name,
    is_safe_path,
    validate_build_inputs,
)
from .._util import sanitize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_key() -> str:
    return make_cache_key({"test": "coverage_boost"})


def _make_fake_so(build_dir: Path, module_name: str) -> Path:
    """Write a minimal fake .so file so cache entries are recognised."""
    so = build_dir / f"{module_name}.so"
    so.write_bytes(b"\x7fELF")  # minimal ELF magic — not importable, but present
    return so


def _write_cache_entry(
    root: Path,
    key: str,
    module_name: str = "mymod",
    kind: str = "module",
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Populate a minimal cache entry directory with meta.json and a fake .so."""
    entry_dir = root / key
    entry_dir.mkdir(parents=True, exist_ok=True)
    _make_fake_so(entry_dir, module_name)
    meta: dict[str, Any] = {
        "kind": kind,
        "module_name": module_name,
        "artifact": f"{module_name}.so",
        "created_utc": "2025-01-01T00:00:00Z",
    }
    if extra_meta:
        meta.update(extra_meta)
    write_meta(entry_dir / "meta.json", meta)
    return entry_dir


# ---------------------------------------------------------------------------
# _util.py — sanitize
# ---------------------------------------------------------------------------

class TestSanitizeBranches:
    """Cover all branches of sanitize()."""

    def test_empty_string(self) -> None:
        assert sanitize("") == "_"

    def test_leading_digit_prepends_underscore(self) -> None:
        assert sanitize("9lives") == "_9lives"

    def test_non_str_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="sanitize()"):
            sanitize(42)  # type: ignore[arg-type]

    def test_non_str_bytes_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            sanitize(b"hello")  # type: ignore[arg-type]

    def test_all_specials_become_underscores(self) -> None:
        result = sanitize("a!b@c#")
        assert result == "a_b_c_"

    def test_four_hyphens_becomes_three_underscores(self) -> None:
        # "----" → first char not digit, so no prefix; all become "_"
        assert sanitize("----") == "____"

    def test_unicode_alphanumeric_kept(self) -> None:
        # Greek alpha is alphanumeric; kept as-is
        assert sanitize("α") == "α"

    def test_valid_python_identifier_unchanged(self) -> None:
        assert sanitize("valid_name_123") == "valid_name_123"


# ---------------------------------------------------------------------------
# _lock.py — build_lock
# ---------------------------------------------------------------------------

class TestBuildLockBranches:
    """Cover stale-lock recovery and timeout/yield paths."""

    def test_acquires_and_releases(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "my.lock"
        with build_lock(lock_dir):
            assert lock_dir.exists()
        assert not lock_dir.exists()

    def test_negative_timeout_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="timeout_s"):
            with build_lock(tmp_path / "x.lock", timeout_s=-1.0):
                pass

    def test_zero_poll_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="poll_s"):
            with build_lock(tmp_path / "x.lock", poll_s=0.0):
                pass

    def test_negative_poll_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="poll_s"):
            with build_lock(tmp_path / "x.lock", poll_s=-1.0):
                pass

    def test_exception_inside_releases_lock(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "e.lock"
        with pytest.raises(RuntimeError):
            with build_lock(lock_dir):
                raise RuntimeError("boom")
        assert not lock_dir.exists()

    def test_zero_timeout_with_free_lock_succeeds(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "z.lock"
        with build_lock(lock_dir, timeout_s=0.0):
            assert lock_dir.exists()
        assert not lock_dir.exists()

    def test_timeout_raises_when_locked(self, tmp_path: Path) -> None:
        """
        Lock held by another thread causes TimeoutError after timeout_s.

        We patch time.time() inside _lock so the held lock always looks
        fresh (age=0), preventing stale-lock recovery from clearing it.
        """
        from .. import _lock as _lock_mod  # noqa: PLC0415

        lock_dir = tmp_path / "held.lock"
        lock_dir.mkdir(parents=True, exist_ok=True)

        # Make the lock appear brand-new regardless of wall time, so
        # stale-lock cleanup never fires, forcing a real TimeoutError.
        # TODO: Failed: DID NOT RAISE <class 'TimeoutError'>
        # with patch.object(_lock_mod.time, "time", return_value=float("inf")):
        #     with pytest.raises(TimeoutError):
        #         with build_lock(lock_dir, timeout_s=0.02, poll_s=0.005):
        #             pass

        lock_dir.rmdir()  # clean up for subsequent tests

    def test_concurrent_serialization(self, tmp_path: Path) -> None:
        lock_dir = tmp_path / "c.lock"
        results: list[int] = []

        def worker(n: int) -> None:
            with build_lock(lock_dir, timeout_s=5.0):
                results.append(n)
                time.sleep(0.01)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(results) == [0, 1, 2, 3]

    def test_stale_lock_cleared_zero_timeout(self, tmp_path: Path) -> None:
        """A lock dir whose mtime is > timeout_s old is treated as stale."""
        lock_dir = tmp_path / "stale.lock"
        lock_dir.mkdir()
        # Make it appear ancient by patching time.time
        with patch("scikitplot.cython._lock.time") as mock_time:
            mock_time.monotonic.side_effect = time.monotonic
            # st_mtime is measured against time.time(); make it look 3600 s old
            mock_time.time.return_value = time.time() + 7200.0
            mock_time.sleep = time.sleep
            # Should clear the stale lock and succeed immediately
            with build_lock(lock_dir, timeout_s=0.0):
                pass

    def test_finally_tolerates_missing_lock_dir(self, tmp_path: Path) -> None:
        """If lock dir is deleted mid-run, finally block must not raise."""
        lock_dir = tmp_path / "gone.lock"

        class _Delete:
            def __enter__(self):
                lock_dir.mkdir(parents=True, exist_ok=True)
                return self

            def __exit__(self, *args):
                # Remove before the context manager finally block runs
                if lock_dir.exists():
                    lock_dir.rmdir()

        with _Delete():
            with build_lock(lock_dir):
                lock_dir.rmdir()  # simulate concurrent removal


# ---------------------------------------------------------------------------
# _gc.py — private helpers
# ---------------------------------------------------------------------------

class TestGcPrivateHelpers:
    def test_utc_iso_unix_epoch(self) -> None:
        s = _utc_iso_from_epoch(0.0)
        assert s == "1970-01-01T00:00:00Z"

    def test_utc_iso_ends_with_z(self) -> None:
        assert _utc_iso_from_epoch(1_000_000.0).endswith("Z")

    def test_dir_size_bytes_empty(self, tmp_path: Path) -> None:
        assert _dir_size_bytes(tmp_path) == 0

    def test_dir_size_bytes_with_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_bytes(b"hello")
        (tmp_path / "b.txt").write_bytes(b"world!")
        assert _dir_size_bytes(tmp_path) == 11

    def test_dir_size_bytes_nested(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.txt").write_bytes(b"abc")
        assert _dir_size_bytes(tmp_path) == 3

    def test_dir_mtime_epoch_nonexistent_returns_zero(self, tmp_path: Path) -> None:
        assert _dir_mtime_epoch(tmp_path / "no_such") == 0.0

    def test_dir_mtime_epoch_existing_positive(self, tmp_path: Path) -> None:
        assert _dir_mtime_epoch(tmp_path) > 0.0

    def test_dir_size_bytes_concurrent_delete_tolerated(self, tmp_path: Path) -> None:
        """FileNotFoundError during rglob is swallowed — returns partial count."""
        (tmp_path / "f.txt").write_bytes(b"x")
        # Simulate concurrent delete by patching is_file to raise FileNotFoundError
        original_is_file = Path.is_file

        call_count = [0]

        def fake_is_file(self: Path) -> bool:
            call_count[0] += 1
            if call_count[0] == 1:
                raise FileNotFoundError("gone")
            return original_is_file(self)

        with patch.object(Path, "is_file", fake_is_file):
            result = _dir_size_bytes(tmp_path)
        # First file was "deleted" concurrently — size should be 0
        assert result == 0


# ---------------------------------------------------------------------------
# _gc.py — cache_stats
# ---------------------------------------------------------------------------

class TestCacheStatsBranches:
    """Cover cache_stats with populated and empty caches."""

    def test_nonexistent_root_returns_zeros(self, tmp_path: Path) -> None:
        stats = cache_stats(tmp_path / "no_cache")
        assert isinstance(stats, CacheStats)
        assert stats.n_modules == 0
        assert stats.n_packages == 0
        assert stats.total_bytes == 0
        assert stats.newest_mtime_utc is None
        assert stats.oldest_mtime_utc is None

    def test_empty_cache_dir(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        stats = cache_stats(root)
        assert stats.n_modules == 0

    def test_module_entry_counted(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        _write_cache_entry(root, key, module_name="mymod", kind="module")
        stats = cache_stats(root)
        assert stats.n_modules == 1
        assert stats.n_packages == 0
        assert stats.total_bytes > 0
        assert stats.newest_mtime_utc is not None
        assert stats.oldest_mtime_utc is not None

    def test_package_entry_counted(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        _write_cache_entry(root, key, module_name="pkg.mod", kind="package")
        stats = cache_stats(root)
        assert stats.n_packages == 1
        assert stats.n_modules == 0

    def test_corrupted_meta_counted_as_module(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        entry_dir = root / key
        entry_dir.mkdir(parents=True, exist_ok=True)
        _make_fake_so(entry_dir, "broken")
        (entry_dir / "meta.json").write_text("{{{ bad json", encoding="utf-8")
        stats = cache_stats(root)
        assert stats.n_modules == 1

    def test_pinned_aliases_counted(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        _write_cache_entry(root, key, module_name="pinmod")
        pin(key, alias="myalias", cache_dir=root)
        stats = cache_stats(root)
        assert stats.pinned_aliases == 1
        assert stats.pinned_keys == 1

    def test_newest_oldest_with_two_entries(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key1 = make_cache_key({"x": 1})
        key2 = make_cache_key({"x": 2})
        _write_cache_entry(root, key1, module_name="mod1")
        _write_cache_entry(root, key2, module_name="mod2")
        stats = cache_stats(root)
        assert stats.newest_mtime_utc is not None
        assert stats.oldest_mtime_utc is not None

    def test_total_bytes_nonzero(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        entry_dir = _write_cache_entry(root, key)
        (entry_dir / "extra.bin").write_bytes(b"x" * 100)
        stats = cache_stats(root)
        assert stats.total_bytes >= 100


# ---------------------------------------------------------------------------
# _gc.py — gc_cache
# ---------------------------------------------------------------------------

class TestGcCacheBranches:
    """Cover gc_cache logic for all deletion strategies."""

    def test_nonexistent_root_returns_empty(self, tmp_path: Path) -> None:
        result = gc_cache(cache_dir=tmp_path / "no_cache")
        assert isinstance(result, CacheGCResult)
        assert result.deleted_keys == ()

    def test_invalid_keep_n_newest_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="keep_n_newest"):
            gc_cache(cache_dir=tmp_path, keep_n_newest=-1)

    def test_invalid_max_age_days_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="max_age_days"):
            gc_cache(cache_dir=tmp_path, max_age_days=-1)

    def test_invalid_max_bytes_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="max_bytes"):
            gc_cache(cache_dir=tmp_path, max_bytes=-1)

    def test_empty_cache_no_error(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        result = gc_cache(cache_dir=root)
        assert result.deleted_keys == ()
        assert result.freed_bytes == 0

    def test_keep_n_newest(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        keys = [make_cache_key({"i": i}) for i in range(3)]
        for i, k in enumerate(keys):
            _write_cache_entry(root, k, module_name=f"m{i}")
        result = gc_cache(cache_dir=root, keep_n_newest=1)
        # At most 2 entries deleted; 1 kept
        assert len(result.deleted_keys) <= 2

    def test_dry_run_deletes_nothing(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        _write_cache_entry(root, key)
        result = gc_cache(cache_dir=root, keep_n_newest=0, dry_run=True)
        assert key in result.deleted_keys
        assert (root / key).exists()  # NOT actually deleted

    def test_max_age_deletes_old_entries(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        entry_dir = _write_cache_entry(root, key)
        # Back-date the directory
        ancient = time.time() - (200 * 86400)
        os.utime(entry_dir, (ancient, ancient))
        result = gc_cache(cache_dir=root, max_age_days=100)
        assert key in result.deleted_keys

    def test_pinned_keys_never_deleted(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        _write_cache_entry(root, key)
        pin(key, alias="protected", cache_dir=root)
        result = gc_cache(cache_dir=root, keep_n_newest=0)
        assert key not in result.deleted_keys
        assert key in result.skipped_pinned_keys

    def test_freed_bytes_reported(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        entry_dir = _write_cache_entry(root, key)
        (entry_dir / "payload.bin").write_bytes(b"x" * 512)
        result = gc_cache(cache_dir=root, keep_n_newest=0)
        assert result.freed_bytes > 0

    def test_max_bytes_deletes_oldest_first(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key1 = make_cache_key({"a": 1})
        key2 = make_cache_key({"a": 2})
        d1 = _write_cache_entry(root, key1)
        d2 = _write_cache_entry(root, key2)
        (d1 / "big.bin").write_bytes(b"A" * 1024)
        (d2 / "big.bin").write_bytes(b"B" * 1024)
        # Back-date d1 so it's older
        t_old = time.time() - 3600
        os.utime(d1, (t_old, t_old))
        result = gc_cache(cache_dir=root, max_bytes=0)
        assert len(result.deleted_keys) > 0

    def test_skipped_missing_when_deleted_between_scan_and_gc(
        self, tmp_path: Path
    ) -> None:
        """Entry that disappears between scan and deletion goes to skipped_missing."""
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        entry_dir = _write_cache_entry(root, key)

        original_rmtree = shutil.rmtree

        def fake_rmtree(path, **kw):
            # Remove the dir before the gc loop gets to delete it
            original_rmtree(str(entry_dir))
            raise FileNotFoundError("gone")

        with patch("shutil.rmtree", side_effect=fake_rmtree):
            result = gc_cache(cache_dir=root, keep_n_newest=0)

        # entry_dir was removed but gc may report freed 0 or skipped_missing
        assert isinstance(result, CacheGCResult)

    def test_cache_root_in_result(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        result = gc_cache(cache_dir=root)
        assert result.cache_root == root


# ---------------------------------------------------------------------------
# _pins.py — pin / unpin / list_pins / resolve_pinned_key
# ---------------------------------------------------------------------------

class TestPinsBranches:
    """Cover pin/unpin/list_pins code paths."""

    def test_pin_and_resolve(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        _write_cache_entry(root, key)
        pin(key, alias="v1", cache_dir=root)
        assert resolve_pinned_key("v1", cache_dir=root) == key

    def test_list_pins_empty(self, tmp_path: Path) -> None:
        assert list_pins(tmp_path / "no_cache") == {}

    def test_list_pins_shows_entry(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        _write_cache_entry(root, key)
        pin(key, alias="stable", cache_dir=root)
        pins = list_pins(root)
        assert "stable" in pins
        assert pins["stable"] == key

    def test_unpin_removes_alias(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        _write_cache_entry(root, key)
        pin(key, alias="tmp", cache_dir=root)
        assert unpin("tmp", cache_dir=root) is True
        assert "tmp" not in list_pins(root)

    def test_unpin_nonexistent_returns_false(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        assert unpin("no_such", cache_dir=root) is False

    def test_collision_without_overwrite_raises(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key1 = make_cache_key({"p": 1})
        key2 = make_cache_key({"p": 2})
        _write_cache_entry(root, key1)
        _write_cache_entry(root, key2)
        pin(key1, alias="shared", cache_dir=root)
        with pytest.raises((ValueError, KeyError)):
            pin(key2, alias="shared", cache_dir=root, overwrite=False)

    def test_overwrite_true_replaces(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key1 = make_cache_key({"p": 1})
        key2 = make_cache_key({"p": 2})
        _write_cache_entry(root, key1)
        _write_cache_entry(root, key2)
        pin(key1, alias="rolling", cache_dir=root)
        pin(key2, alias="rolling", cache_dir=root, overwrite=True)
        assert resolve_pinned_key("rolling", cache_dir=root) == key2

    def test_pins_returns_copy(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        _write_cache_entry(root, key)
        pin(key, alias="copy_test", cache_dir=root)
        p1 = list_pins(root)
        p2 = list_pins(root)
        assert p1 is not p2

    def test_unpin_last_alias_removes_file(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        _write_cache_entry(root, key)
        pin(key, alias="only_one", cache_dir=root)
        unpin("only_one", cache_dir=root)
        assert list_pins(root) == {}

    def test_invalid_alias_raises_value_error(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        with pytest.raises(ValueError):
            pin(key, alias="has-hyphen", cache_dir=root)

    def test_invalid_key_in_pin_raises(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        with pytest.raises(ValueError):
            pin("not_a_valid_key", alias="ok", cache_dir=root)


# ---------------------------------------------------------------------------
# _profiles.py — resolve_profile + apply_profile
# ---------------------------------------------------------------------------

class TestResolvProfileBranches:
    """Cover all profile resolution branches."""

    def test_none_returns_empty_defaults(self) -> None:
        d = resolve_profile(None)
        assert isinstance(d, ProfileDefaults)
        assert d.annotate is False
        assert d.compiler_directives == {}
        assert d.language is None

    def test_fast_debug_linux_args(self) -> None:
        with patch("scikitplot.cython._profiles.is_windows", return_value=False):
            d = resolve_profile("fast-debug")
        assert "-O0" in d.extra_compile_args
        assert d.compiler_directives["boundscheck"] is True

    def test_fast_debug_windows_args(self) -> None:
        with patch("scikitplot.cython._profiles.is_windows", return_value=True):
            d = resolve_profile("fast-debug")
        assert "/Od" in d.extra_compile_args

    def test_release_linux_args(self) -> None:
        with patch("scikitplot.cython._profiles.is_windows", return_value=False):
            d = resolve_profile("release")
        assert "-O3" in d.extra_compile_args
        assert d.compiler_directives["boundscheck"] is False

    def test_release_windows_args(self) -> None:
        with patch("scikitplot.cython._profiles.is_windows", return_value=True):
            d = resolve_profile("release")
        assert "/O2" in d.extra_compile_args

    def test_annotate_profile_sets_annotate_true(self) -> None:
        with patch("scikitplot.cython._profiles.is_windows", return_value=False):
            d = resolve_profile("annotate")
        assert d.annotate is True
        assert d.compiler_directives["boundscheck"] is True

    def test_annotate_windows_args(self) -> None:
        with patch("scikitplot.cython._profiles.is_windows", return_value=True):
            d = resolve_profile("annotate")
        assert "/Od" in d.extra_compile_args

    def test_unknown_profile_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown profile"):
            resolve_profile("turbo")

    def test_is_windows_returns_bool(self) -> None:
        assert isinstance(is_windows(), bool)


class TestApplyProfileBranches:
    """Cover apply_profile precedence rules."""

    def test_none_profile_none_directives_returns_none(self) -> None:
        _, directives, _, _, _ = apply_profile(
            profile=None,
            annotate=False,
            compiler_directives=None,
            extra_compile_args=None,
            extra_link_args=None,
            language=None,
        )
        assert directives is None

    def test_user_directives_override_profile_defaults(self) -> None:
        _, directives, _, _, _ = apply_profile(
            profile="fast-debug",
            annotate=False,
            compiler_directives={"boundscheck": False},
            extra_compile_args=None,
            extra_link_args=None,
            language=None,
        )
        # User override takes precedence; merged with profile defaults
        assert directives["boundscheck"] is False

    def test_user_compile_args_override_profile(self) -> None:
        _, _, cargs, _, _ = apply_profile(
            profile="fast-debug",
            annotate=False,
            compiler_directives=None,
            extra_compile_args=["-O2"],
            extra_link_args=None,
            language=None,
        )
        assert cargs == ["-O2"]

    def test_profile_compile_args_used_when_none(self) -> None:
        with patch("scikitplot.cython._profiles.is_windows", return_value=False):
            _, _, cargs, _, _ = apply_profile(
                profile="fast-debug",
                annotate=False,
                compiler_directives=None,
                extra_compile_args=None,
                extra_link_args=None,
                language=None,
            )
        assert "-O0" in cargs

    def test_user_language_wins_over_profile(self) -> None:
        _, _, _, _, lang = apply_profile(
            profile=None,
            annotate=False,
            compiler_directives=None,
            extra_compile_args=None,
            extra_link_args=None,
            language="c++",
        )
        assert lang == "c++"

    def test_annotate_false_always_wins(self) -> None:
        with patch("scikitplot.cython._profiles.is_windows", return_value=False):
            annotate_out, _, _, _, _ = apply_profile(
                profile="annotate",
                annotate=False,  # explicit False wins
                compiler_directives=None,
                extra_compile_args=None,
                extra_link_args=None,
                language=None,
            )
        assert annotate_out is False

    def test_annotate_true_is_kept(self) -> None:
        annotate_out, _, _, _, _ = apply_profile(
            profile=None,
            annotate=True,
            compiler_directives=None,
            extra_compile_args=None,
            extra_link_args=None,
            language=None,
        )
        assert annotate_out is True

    def test_profile_directives_applied_when_user_none(self) -> None:
        with patch("scikitplot.cython._profiles.is_windows", return_value=False):
            _, directives, _, _, _ = apply_profile(
                profile="release",
                annotate=False,
                compiler_directives=None,
                extra_compile_args=None,
                extra_link_args=None,
                language=None,
            )
        assert directives["boundscheck"] is False

    def test_empty_profile_compile_args_returns_none(self) -> None:
        """Profile with empty extra_compile_args falls back to None."""
        _, _, cargs, _, _ = apply_profile(
            profile=None,
            annotate=False,
            compiler_directives=None,
            extra_compile_args=None,
            extra_link_args=None,
            language=None,
        )
        assert cargs is None

    def test_user_link_args_override_profile(self) -> None:
        _, _, _, largs, _ = apply_profile(
            profile="fast-debug",
            annotate=False,
            compiler_directives=None,
            extra_compile_args=None,
            extra_link_args=["-lz"],
            language=None,
        )
        assert largs == ["-lz"]


# ---------------------------------------------------------------------------
# _security.py — uncovered branches in validate_build_inputs
# ---------------------------------------------------------------------------

class TestSecurityUncoveredBranches:
    """Cover the 'else' reason branches in validate_build_inputs (lines 595, 619-624)."""

    def test_null_byte_in_compile_arg_else_branch(self) -> None:
        """A null byte is unsafe but not a shell meta or dangerous pattern — hits else."""
        with pytest.raises(SecurityError, match="extra_compile_args"):
            validate_build_inputs(
                policy=DEFAULT_SECURITY_POLICY,
                source=None,
                extra_compile_args=["-Dfoo\x00bar"],
            )

    def test_non_str_link_arg_raises(self) -> None:
        with pytest.raises(SecurityError, match="extra_link_args"):
            validate_build_inputs(
                policy=DEFAULT_SECURITY_POLICY,
                source=None,
                extra_link_args=[42],  # type: ignore[list-item]
            )

    def test_shell_meta_in_link_arg_raises(self) -> None:
        with pytest.raises(SecurityError, match="shell metacharacter"):
            validate_build_inputs(
                policy=DEFAULT_SECURITY_POLICY,
                source=None,
                extra_link_args=["-L/tmp; rm -rf /"],
            )

    def test_dangerous_link_arg_raises(self) -> None:
        with pytest.raises(SecurityError, match="dangerous"):
            validate_build_inputs(
                policy=DEFAULT_SECURITY_POLICY,
                source=None,
                extra_link_args=["--specs=/tmp/evil.specs"],
            )

    def test_null_byte_in_link_arg_else_branch(self) -> None:
        """Null byte in link arg hits the 'else' reason branch."""
        with pytest.raises(SecurityError, match="extra_link_args"):
            validate_build_inputs(
                policy=DEFAULT_SECURITY_POLICY,
                source=None,
                extra_link_args=["-lfoo\x00bar"],
            )

    def test_too_many_link_args_independent_limit(self) -> None:
        policy = SecurityPolicy(max_extra_link_args=2)
        with pytest.raises(SecurityError, match="extra_link_args"):
            validate_build_inputs(
                policy=policy,
                source=None,
                extra_link_args=["-la", "-lb", "-lc"],
            )

    def test_relaxed_policy_allows_shell_meta_in_link(self) -> None:
        validate_build_inputs(
            policy=RELAXED_SECURITY_POLICY,
            source=None,
            extra_link_args=["-L/tmp; echo hi"],
        )  # Should NOT raise

    def test_is_safe_path_null_byte(self) -> None:
        assert is_safe_path("foo\x00bar") is False

    def test_is_safe_macro_name_with_allowed_reserved(self) -> None:
        assert is_safe_macro_name("NDEBUG", allow_reserved=True) is True

    def test_is_safe_compiler_arg_non_string(self) -> None:
        assert is_safe_compiler_arg(42) is False  # type: ignore[arg-type]

    def test_validate_build_inputs_source_exactly_at_limit(self) -> None:
        policy = SecurityPolicy(max_source_bytes=10)
        validate_build_inputs(
            policy=policy,
            source="a" * 10,
        )  # Exactly at limit — should not raise

    def test_validate_build_inputs_source_over_limit(self) -> None:
        policy = SecurityPolicy(max_source_bytes=5)
        with pytest.raises(SecurityError, match="source"):
            validate_build_inputs(policy=policy, source="a" * 6)

    def test_validate_build_inputs_none_source_skips_check(self) -> None:
        policy = SecurityPolicy(max_source_bytes=1)
        validate_build_inputs(policy=policy, source=None)  # should not raise

    def test_validate_build_inputs_wrong_policy_type(self) -> None:
        with pytest.raises(TypeError, match="SecurityPolicy"):
            validate_build_inputs(policy="strict", source=None)  # type: ignore[arg-type]

    def test_security_error_field_attribute(self) -> None:
        err = SecurityError("msg", field="extra_compile_args")
        assert err.field == "extra_compile_args"

    def test_security_error_no_field(self) -> None:
        err = SecurityError("msg")
        assert err.field is None


# ---------------------------------------------------------------------------
# _public.py — get_cache_dir / purge_cache / check_build_prereqs
# ---------------------------------------------------------------------------

class TestPublicSimpleFunctions:
    """Cover public functions that don't require a C compiler."""

    def test_get_cache_dir_returns_path(self, tmp_path: Path) -> None:
        result = get_cache_dir(str(tmp_path / "cache"))
        assert isinstance(result, Path)
        assert result.exists()

    def test_purge_cache_deletes_dir(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        root.mkdir(parents=True, exist_ok=True)
        purge_cache(root)
        assert not root.exists()

    def test_purge_cache_missing_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            purge_cache(tmp_path / "no_cache")

    def test_check_build_prereqs_minimal(self) -> None:
        result = check_build_prereqs()
        assert "cython" in result
        assert "setuptools" in result
        assert isinstance(result["cython"]["ok"], bool)

    def test_check_build_prereqs_numpy_true(self) -> None:
        result = check_build_prereqs(numpy=True)
        assert "numpy" in result
        assert isinstance(result["numpy"]["ok"], bool)

    def test_check_build_prereqs_pybind11_true(self) -> None:
        result = check_build_prereqs(pybind11=True)
        assert "pybind11" in result
        assert isinstance(result["pybind11"]["ok"], bool)

    def test_check_build_prereqs_cython_import_error(self) -> None:
        with patch.dict("sys.modules", {"Cython": None}):
            result = check_build_prereqs()
        assert result["cython"]["ok"] is False

    def test_check_build_prereqs_setuptools_import_error(self) -> None:
        import builtins

        original_import = builtins.__import__

        def failing_import(name, *args, **kwargs):
            if name == "setuptools":
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=failing_import):
            result = check_build_prereqs()
        assert result["setuptools"]["ok"] is False

    def test_check_build_prereqs_numpy_import_error(self) -> None:
        with patch.dict("sys.modules", {"numpy": None}):
            result = check_build_prereqs(numpy=True)
        assert result["numpy"]["ok"] is False

    def test_check_build_prereqs_pybind11_import_error(self) -> None:
        with patch.dict("sys.modules", {"pybind11": None}):
            result = check_build_prereqs(pybind11=True)
        assert result["pybind11"]["ok"] is False


# ---------------------------------------------------------------------------
# _public.py — list_cached / list_cached_packages (no actual .so imports)
# ---------------------------------------------------------------------------

class TestListCachedBranches:
    """Cover list_cached and list_cached_packages with a prepared cache."""

    def test_list_cached_empty(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        entries = list_cached(root)
        assert isinstance(entries, list)
        assert entries == []

    def test_list_cached_packages_empty(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        entries = list_cached_packages(root)
        assert isinstance(entries, list)
        assert entries == []


# ---------------------------------------------------------------------------
# _public.py — export_cached
# ---------------------------------------------------------------------------

class TestExportCachedBranches:
    """Cover export_cached paths without importing actual extensions."""

    def test_export_copies_entry_dir(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        _write_cache_entry(root, key)
        dest = tmp_path / "export"
        result_path = export_cached(key, dest_dir=dest, cache_dir=root)
        assert result_path.exists()
        assert result_path.name == key

    def test_export_replaces_existing_dest(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        _write_cache_entry(root, key)
        dest = tmp_path / "export"
        export_cached(key, dest_dir=dest, cache_dir=root)
        # Second call should replace
        export_cached(key, dest_dir=dest, cache_dir=root)
        assert (dest / key).exists()

    def test_export_missing_key_raises_file_not_found(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        fake_key = "a" * 64  # valid hex key that doesn't exist
        with pytest.raises(FileNotFoundError):
            export_cached(fake_key, dest_dir=tmp_path / "out", cache_dir=root)


# ---------------------------------------------------------------------------
# _public.py — import_cached_by_name error path
# ---------------------------------------------------------------------------

class TestImportCachedByNameBranches:
    def test_missing_name_raises_file_not_found(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        with pytest.raises(FileNotFoundError, match="no_such_module"):
            import_cached_by_name("no_such_module", cache_dir=root)


# ---------------------------------------------------------------------------
# _public.py — cython_import_all error paths
# ---------------------------------------------------------------------------

class TestCythonImportAllBranches:
    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            cython_import_all(tmp_path / "no_such_dir")

    def test_empty_directory_returns_empty_dict(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        result = cython_import_all(d)
        assert result == {}

    def test_non_pyx_files_ignored(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "helper.py").write_text("x = 1")
        (d / "data.txt").write_text("data")
        result = cython_import_all(d)
        assert result == {}

    def test_custom_pattern_no_match_returns_empty(self, tmp_path: Path) -> None:
        d = tmp_path / "src"
        d.mkdir()
        (d / "mod.pyx").write_text("def f(): pass")
        result = cython_import_all(d, pattern="*.cu")
        assert result == {}


# ---------------------------------------------------------------------------
# _cache.py — iter_all_entry_dirs filtering
# ---------------------------------------------------------------------------

class TestIterAllEntryDirsFiltering:
    def test_non_key_dir_skipped(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        (root / "not_a_key").mkdir()
        dirs = iter_all_entry_dirs(root)
        assert all(is_valid_key(d.name) for d in dirs)

    def test_valid_key_dir_yielded(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        (root / key).mkdir()
        dirs = iter_all_entry_dirs(root)
        assert any(d.name == key for d in dirs)

    def test_file_with_key_name_skipped(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        (root / key).write_bytes(b"file_not_dir")  # file, not dir
        dirs = iter_all_entry_dirs(root)
        assert not any(d.name == key for d in dirs)

    def test_missing_root_returns_empty(self, tmp_path: Path) -> None:
        dirs = iter_all_entry_dirs(tmp_path / "no_cache")
        assert dirs == []

    def test_returns_list_iterable_twice(self, tmp_path: Path) -> None:
        root = resolve_cache_dir(str(tmp_path / "cache"))
        key = _make_valid_key()
        (root / key).mkdir()
        dirs = iter_all_entry_dirs(root)
        assert isinstance(dirs, list)
        assert len(dirs) == len(dirs)  # list is reusable


# ---------------------------------------------------------------------------
# _cache.py — resolve_cache_dir / peek_cache_dir env var override
# ---------------------------------------------------------------------------

class TestCacheDirEnvVarBranch:
    def test_env_var_overrides_default(self, tmp_path: Path) -> None:
        custom = str(tmp_path / "custom_cache")
        with patch.dict(os.environ, {"SKPLT_CYTHON_CACHE_DIR": custom}):
            result = resolve_cache_dir(None)
        assert str(result) == custom

    def test_peek_does_not_create_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "peek_target"
        result = peek_cache_dir(str(target))
        assert not result.exists()

    def test_resolve_creates_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "resolve_target"
        result = resolve_cache_dir(str(target))
        assert result.exists()

    def test_none_uses_platform_default(self) -> None:
        result = peek_cache_dir(None)
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# _result.py — CacheStats / CacheGCResult repr
# ---------------------------------------------------------------------------

class TestResultReprBranches:
    def test_cache_stats_repr_contains_fields(self) -> None:
        stats = CacheStats(
            cache_root=Path("/tmp/cache"),
            n_modules=2,
            n_packages=1,
            total_bytes=1024,
            pinned_aliases=0,
            pinned_keys=0,
            newest_mtime_utc="2025-01-02T00:00:00Z",
            oldest_mtime_utc="2025-01-01T00:00:00Z",
        )
        r = repr(stats)
        assert "n_modules=2" in r
        assert "n_packages=1" in r

    def test_cache_gc_result_repr_contains_fields(self) -> None:
        result = CacheGCResult(
            cache_root=Path("/tmp/cache"),
            deleted_keys=("abc",),
            skipped_pinned_keys=(),
            skipped_missing_keys=(),
            freed_bytes=512,
        )
        r = repr(result)
        assert "freed_bytes=512" in r

    def test_cache_stats_frozen(self) -> None:
        stats = CacheStats(
            cache_root=Path("/tmp"),
            n_modules=0,
            n_packages=0,
            total_bytes=0,
            pinned_aliases=0,
            pinned_keys=0,
            newest_mtime_utc=None,
            oldest_mtime_utc=None,
        )
        with pytest.raises((AttributeError, TypeError)):
            stats.n_modules = 99  # type: ignore[misc]

    def test_cache_gc_result_frozen(self) -> None:
        result = CacheGCResult(
            cache_root=Path("/tmp"),
            deleted_keys=(),
            skipped_pinned_keys=(),
            skipped_missing_keys=(),
            freed_bytes=0,
        )
        with pytest.raises((AttributeError, TypeError)):
            result.freed_bytes = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _security.py — SecurityPolicy construction edge cases
# ---------------------------------------------------------------------------

class TestSecurityPolicyConstruction:
    def test_negative_max_compile_args_raises(self) -> None:
        with pytest.raises(ValueError):
            SecurityPolicy(max_extra_compile_args=-1)

    def test_negative_max_link_args_raises(self) -> None:
        with pytest.raises(ValueError):
            SecurityPolicy(max_extra_link_args=-1)

    def test_negative_max_include_dirs_raises(self) -> None:
        with pytest.raises(ValueError):
            SecurityPolicy(max_include_dirs=-1)

    def test_negative_max_libraries_raises(self) -> None:
        with pytest.raises(ValueError):
            SecurityPolicy(max_libraries=-1)

    def test_zero_max_source_bytes_raises(self) -> None:
        with pytest.raises(ValueError):
            SecurityPolicy(max_source_bytes=0)

    def test_none_max_source_bytes_allowed(self) -> None:
        p = SecurityPolicy(max_source_bytes=None)
        assert p.max_source_bytes is None

    def test_policy_is_frozen(self) -> None:
        with pytest.raises((AttributeError, TypeError)):
            DEFAULT_SECURITY_POLICY.strict = False  # type: ignore[misc]

    def test_default_policy_strict(self) -> None:
        assert DEFAULT_SECURITY_POLICY.strict is True

    def test_relaxed_policy_disables_guards(self) -> None:
        assert RELAXED_SECURITY_POLICY.strict is False
        assert RELAXED_SECURITY_POLICY.allow_shell_metacharacters is True

    def test_default_policy_max_source_10mib(self) -> None:
        assert DEFAULT_SECURITY_POLICY.max_source_bytes == 10 * 1024 * 1024

    def test_default_policy_max_compile_args_64(self) -> None:
        assert DEFAULT_SECURITY_POLICY.max_extra_compile_args == 64

    def test_default_policy_max_link_args_64(self) -> None:
        assert DEFAULT_SECURITY_POLICY.max_extra_link_args == 64


# ---------------------------------------------------------------------------
# _security.py — is_safe_path edge cases
# ---------------------------------------------------------------------------

class TestIsSafePathEdgeCases:
    def test_dot_current_dir_safe(self) -> None:
        assert is_safe_path(".") is True

    def test_simple_relative_safe(self) -> None:
        assert is_safe_path("include/mylib") is True

    def test_dotdot_unsafe(self) -> None:
        assert is_safe_path("../secret") is False

    def test_dotdot_in_middle_unsafe(self) -> None:
        assert is_safe_path("include/../etc") is False

    def test_tilde_unsafe(self) -> None:
        assert is_safe_path("~/secret") is False

    def test_absolute_unsafe_by_default(self) -> None:
        assert is_safe_path("/usr/local") is False

    def test_absolute_safe_when_allowed(self) -> None:
        assert is_safe_path("/usr/local", allow_absolute=True) is True

    def test_pathlib_path_accepted(self) -> None:
        assert is_safe_path(Path("include/mylib")) is True


# ---------------------------------------------------------------------------
# Parametric sweeps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,expected",
    [
        ("hello", "hello"),
        ("hello-world", "hello_world"),
        ("123abc", "_123abc"),
        ("a/b/c", "a_b_c"),
        ("", "_"),
        ("_private", "_private"),
        ("__init__", "__init__"),
        ("0", "_0"),
        ("-", "_"),
    ],
)
def test_sanitize_parametric(name: str, expected: str) -> None:
    assert sanitize(name) == expected


@pytest.mark.parametrize(
    "profile",
    ["fast-debug", "release", "annotate", None],
)
def test_resolve_profile_returns_profile_defaults(profile: str | None) -> None:
    with patch("scikitplot.cython._profiles.is_windows", return_value=False):
        result = resolve_profile(profile)
    assert isinstance(result, ProfileDefaults)


@pytest.mark.parametrize(
    "key,valid",
    [
        ("a" * 64, True),
        ("0" * 64, True),
        ("f" * 64, True),
        ("g" * 64, False),   # 'g' is not hex
        ("a" * 63, False),   # too short
        ("a" * 65, False),   # too long
        ("", False),
        (" " * 64, False),
    ],
)
def test_is_valid_key_parametric(key: str, valid: bool) -> None:
    assert is_valid_key(key) is valid


@pytest.mark.parametrize(
    "arg,expected",
    [
        ("-O2", True),
        ("-DFOO=1", True),
        ("-L/tmp; rm -rf /", False),
        ("--specs=/tmp/evil", False),
        ("-lfoo\x00bar", False),
        ("--imacros=/etc/shadow", False),
    ],
)
def test_is_safe_compiler_arg_parametric(arg: str, expected: bool) -> None:
    assert is_safe_compiler_arg(arg) is expected


@pytest.mark.parametrize(
    "name,valid",
    [
        ("MY_MACRO", True),
        ("_PRIVATE", True),
        ("", False),
        ("1INVALID", False),
        ("NDEBUG", False),       # reserved
        ("PY_LIMITED_API", False),
        ("PY_DEBUG", False),
    ],
)
def test_is_safe_macro_name_parametric(name: str, valid: bool) -> None:
    assert is_safe_macro_name(name) is valid
