# scikitplot/cython/tests/test__pins_transactional.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-PIN-001.

``list_pins`` swallowed all read/parse errors and returned ``{}``, so a corrupt
``pins.json`` silently looked empty — pins could be overwritten (lost) and
pinned entries could be garbage-collected.  ``pin``/``unpin`` wrote the registry
non-atomically, so a crash mid-write could produce that corruption.

The fix makes corruption an explicit ``PinRegistryError`` and writes the
registry atomically (temp file + os.replace).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from .._cache import make_cache_key
from .._pins import PinRegistryError, list_pins, pin, unpin


KEY = "a" * 64
KEY2 = "b" * 64


class TestExplicitCorruption:
    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        root = tmp_path / "cache"
        root.mkdir()
        (root / "pins.json").write_text("not json <<<", encoding="utf-8")
        with pytest.raises(PinRegistryError):
            list_pins(root)

    def test_non_object_json_raises(self, tmp_path: Path) -> None:
        root = tmp_path / "cache"
        root.mkdir()
        (root / "pins.json").write_text("[1, 2, 3]", encoding="utf-8")  # not an object
        with pytest.raises(PinRegistryError):
            list_pins(root)

    def test_pin_refuses_to_clobber_corrupt(self, tmp_path: Path) -> None:
        root = tmp_path / "cache"
        root.mkdir()
        pin(KEY, alias="first", cache_dir=root)
        (root / "pins.json").write_text("broken<<<", encoding="utf-8")
        with pytest.raises(PinRegistryError):
            pin(KEY2, alias="second", cache_dir=root)

    def test_pin_registry_error_is_value_error(self) -> None:
        # Backward compatibility: existing `except ValueError` still catches it.
        assert issubclass(PinRegistryError, ValueError)


class TestValidRegistryStillWorks:
    def test_pin_list_unpin_roundtrip(self, tmp_path: Path) -> None:
        root = tmp_path / "cache"
        root.mkdir()
        pin(KEY, alias="alpha", cache_dir=root)
        assert list_pins(root) == {"alpha": KEY}
        assert unpin("alpha", cache_dir=root) is True
        assert list_pins(root) == {}

    def test_invalid_entries_filtered_in_valid_json(self, tmp_path: Path) -> None:
        """A well-formed object with some bad entries is still read (filtered)."""
        root = tmp_path / "cache"
        root.mkdir()
        (root / "pins.json").write_text(
            json.dumps({"good": KEY, "bad!alias": KEY, "shortkey": "abc"}),
            encoding="utf-8",
        )
        assert list_pins(root) == {"good": KEY}


class TestAtomicWrite:
    def test_no_temp_files_left_after_pin(self, tmp_path: Path) -> None:
        root = tmp_path / "cache"
        root.mkdir()
        pin(KEY, alias="alpha", cache_dir=root)
        assert list(root.glob(".pins-*")) == [], "atomic-write temp file leaked"

    def test_pins_file_is_valid_json_after_write(self, tmp_path: Path) -> None:
        root = tmp_path / "cache"
        root.mkdir()
        pin(KEY, alias="alpha", cache_dir=root)
        # The published file must be complete, parseable JSON.
        data = json.loads((root / "pins.json").read_text(encoding="utf-8"))
        assert data == {"alpha": KEY}
