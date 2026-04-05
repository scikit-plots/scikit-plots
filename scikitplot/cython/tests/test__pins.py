# scikitplot/cython/tests/test__pins.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for :mod:`scikitplot.cython._pins`.

Covers
------
- ``_validate_alias()``    : valid/invalid identifier rules
- ``pin()``                : alias assignment, overwrite, collision, invalid args
- ``unpin()``              : removal, non-existent, last-alias file cleanup,
                             FileNotFoundError swallowed on unlink
- ``list_pins()``          : empty/populated/corrupted registry, returns copy
- ``resolve_pinned_key()`` : unknown-alias KeyError, success path
"""
from __future__ import annotations

from pathlib import Path

import pytest

from .._cache import make_cache_key, resolve_cache_dir
from .._pins import _validate_alias, list_pins, pin, resolve_pinned_key, unpin

from .conftest import make_valid_key, write_simple_cache_entry

# ---------------------------------------------------------------------------
# Module-level helpers (aliases to conftest utilities for legacy test classes)
# ---------------------------------------------------------------------------
from .conftest import make_valid_key as _make_valid_key, write_simple_cache_entry as _write_cache_entry
import json
import os
from .conftest import FAKE_KEY as _FAKE_KEY, FAKE_KEY2 as _FAKE_KEY2
from unittest.mock import patch



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


class TestValidateAlias:
    """Tests for :func:`scikitplot.cython._pins._validate_alias`."""

    @pytest.mark.parametrize(
        "alias",
        ["fast_fft", "myalias", "_internal", "Alias1", "a"],
    )
    def test_valid_aliases(self, alias: str) -> None:
        _validate_alias(alias)  # must not raise

    @pytest.mark.parametrize(
        "alias",
        ["", "1invalid", "has-hyphen", "has.dot", "has space", None, 123],
    )
    def test_invalid_aliases_raise(self, alias) -> None:
        with pytest.raises((ValueError, TypeError)):
            _validate_alias(alias)  # type: ignore[arg-type]


class TestPinRegistry:
    """Tests for pin/unpin/list_pins/resolve_pinned_key."""

    def test_pin_and_resolve(self, tmp_cache: Path) -> None:
        key = make_cache_key({"test": "pin"})
        pin(key, alias="my_alias", cache_dir=tmp_cache)
        resolved = resolve_pinned_key("my_alias", cache_dir=tmp_cache)
        assert resolved == key

    def test_list_pins_empty(self, tmp_cache: Path) -> None:
        assert list_pins(tmp_cache) == {}

    def test_list_pins_shows_pinned(self, tmp_cache: Path) -> None:
        key = make_cache_key({"test": "list_pins"})
        pin(key, alias="alias1", cache_dir=tmp_cache)
        pins = list_pins(tmp_cache)
        assert "alias1" in pins
        assert pins["alias1"] == key

    def test_unpin_removes_alias(self, tmp_cache: Path) -> None:
        key = make_cache_key({"test": "unpin"})
        pin(key, alias="to_remove", cache_dir=tmp_cache)
        result = unpin("to_remove", cache_dir=tmp_cache)
        assert result is True
        assert "to_remove" not in list_pins(tmp_cache)

    def test_unpin_nonexistent_returns_false(self, tmp_cache: Path) -> None:
        assert unpin("ghost_alias", cache_dir=tmp_cache) is False

    def test_collision_without_overwrite_raises(self, tmp_cache: Path) -> None:
        key1 = make_cache_key({"k": "1"})
        key2 = make_cache_key({"k": "2"})
        pin(key1, alias="shared", cache_dir=tmp_cache)
        with pytest.raises(ValueError, match="collision"):
            pin(key2, alias="shared", cache_dir=tmp_cache, overwrite=False)

    def test_overwrite_replaces_alias(self, tmp_cache: Path) -> None:
        key1 = make_cache_key({"k": "1"})
        key2 = make_cache_key({"k": "2"})
        pin(key1, alias="shared", cache_dir=tmp_cache)
        pin(key2, alias="shared", cache_dir=tmp_cache, overwrite=True)
        assert resolve_pinned_key("shared", cache_dir=tmp_cache) == key2

    def test_one_to_one_key_constraint(self, tmp_cache: Path) -> None:
        key = make_cache_key({"k": "unique"})
        pin(key, alias="alias_a", cache_dir=tmp_cache)
        with pytest.raises(ValueError, match="already pinned"):
            pin(key, alias="alias_b", cache_dir=tmp_cache, overwrite=False)

    def test_resolve_unknown_alias_raises_key_error(self, tmp_cache: Path) -> None:
        with pytest.raises(KeyError, match="ghost"):
            resolve_pinned_key("ghost", cache_dir=tmp_cache)

    def test_invalid_key_raises_value_error(self, tmp_cache: Path) -> None:
        with pytest.raises(ValueError, match="Invalid"):
            pin("not_a_valid_key", alias="alias", cache_dir=tmp_cache)

    def test_invalid_alias_raises_value_error(self, tmp_cache: Path) -> None:
        key = make_cache_key({"x": "y"})
        with pytest.raises(ValueError):
            pin(key, alias="invalid-alias!", cache_dir=tmp_cache)

    def test_missing_root_unpin_returns_false(self, tmp_path: Path) -> None:
        result = unpin("ghost", cache_dir=tmp_path / "nonexistent")
        assert result is False

    def test_list_pins_nonexistent_root(self, tmp_path: Path) -> None:
        result = list_pins(tmp_path / "nonexistent")
        assert result == {}

    def test_pins_returns_copy(self, tmp_cache: Path) -> None:
        key = make_cache_key({"test": "copy"})
        pin(key, alias="zcopy", cache_dir=tmp_cache)
        pins = list_pins(tmp_cache)
        pins["new_key"] = "mutated"
        # Original file should be unchanged
        pins2 = list_pins(tmp_cache)
        assert "new_key" not in pins2

    def test_unpin_removes_file_when_empty(self, tmp_cache: Path) -> None:
        key = make_cache_key({"k": "sole"})
        pin(key, alias="sole_pin", cache_dir=tmp_cache)
        unpin("sole_pin", cache_dir=tmp_cache)
        pins_file = tmp_cache / "pins.json"
        assert not pins_file.exists()


class TestPinsCorruptedRegistry:
    """``list_pins`` when pins.json is corrupted."""

    def test_corrupted_pins_returns_empty(self, tmp_cache: Path) -> None:
        # Create a pins.json with invalid JSON
        key = make_cache_key({"p": "1"})
        pin(key, alias="zz_test", cache_dir=tmp_cache)
        # Corrupt it
        (tmp_cache / "pins.json").write_text("NOT JSON <<<", encoding="utf-8")
        result = list_pins(tmp_cache)
        assert result == {}

    def test_pins_with_invalid_entries_filtered(self, tmp_cache: Path) -> None:
        # Write a pins.json with mixed valid/invalid entries
        pins_data = {
            "good_alias": "a" * 64,
            "bad_alias!": "a" * 64,    # invalid alias
            "another_good": "b" * 64,  # valid format
        }
        (tmp_cache / "pins.json").write_text(
            json.dumps(pins_data, indent=2) + "\n", encoding="utf-8"
        )
        result = list_pins(tmp_cache)
        assert "good_alias" in result
        assert "bad_alias!" not in result  # filtered (invalid alias)


class TestPinOverwriteTrue:
    """``pin()`` with ``overwrite=True`` covers all three overwrite branches."""

    def test_same_alias_same_key_overwrite_is_idempotent(self, tmp_cache: Path) -> None:
        key = make_cache_key({"idem": "same"})
        pin(key, alias="idem_alias", cache_dir=tmp_cache)
        # Re-pin same key + same alias with overwrite=True must not raise
        pin(key, alias="idem_alias", cache_dir=tmp_cache, overwrite=True)
        assert resolve_pinned_key("idem_alias", cache_dir=tmp_cache) == key

    def test_overwrite_true_replaces_different_key(self, tmp_cache: Path) -> None:
        key1 = make_cache_key({"v": "old"})
        key2 = make_cache_key({"v": "new"})
        pin(key1, alias="swap_me", cache_dir=tmp_cache)
        pin(key2, alias="swap_me", cache_dir=tmp_cache, overwrite=True)
        assert resolve_pinned_key("swap_me", cache_dir=tmp_cache) == key2

    def test_overwrite_true_allows_key_repin_to_new_alias(self, tmp_cache: Path) -> None:
        key = make_cache_key({"v": "repin"})
        pin(key, alias="old_alias", cache_dir=tmp_cache)
        # overwrite=True: bypass one-to-one constraint
        pin(key, alias="new_alias", cache_dir=tmp_cache, overwrite=True)
        assert resolve_pinned_key("new_alias", cache_dir=tmp_cache) == key


class TestUnpinInvalidAlias:
    """``unpin()`` with an invalid alias raises ``ValueError``."""

    def test_hyphen_in_alias_raises(self, tmp_cache: Path) -> None:
        with pytest.raises(ValueError):
            unpin("invalid-alias", cache_dir=tmp_cache)

    def test_empty_alias_raises(self, tmp_cache: Path) -> None:
        with pytest.raises(ValueError):
            unpin("", cache_dir=tmp_cache)

    def test_digit_leading_alias_raises(self, tmp_cache: Path) -> None:
        with pytest.raises(ValueError):
            unpin("1bad_alias", cache_dir=tmp_cache)

    def test_space_in_alias_raises(self, tmp_cache: Path) -> None:
        with pytest.raises(ValueError):
            unpin("has space", cache_dir=tmp_cache)


class TestUnpinLastAliasRemovesFile:
    """Removing the last alias must delete pins.json, not write an empty file."""

    def test_pins_file_removed_when_empty(self, tmp_path: Path) -> None:
        from .._pins import pin, unpin

        pin(_FAKE_KEY, alias="sole_alias", cache_dir=tmp_path)
        pins_file = tmp_path / "pins.json"
        assert pins_file.exists()

        unpin("sole_alias", cache_dir=tmp_path)
        assert not pins_file.exists(), (
            "pins.json must be deleted when the last alias is removed"
        )

    def test_pins_file_unlink_filenotfound_swallowed(
        self, tmp_path: Path
    ) -> None:
        """unlink() raising FileNotFoundError must be silently swallowed."""
        from .._pins import pin, unpin

        pin(_FAKE_KEY, alias="alias_x", cache_dir=tmp_path)
        pins_file = tmp_path / "pins.json"

        original_unlink = Path.unlink

        def failing_unlink(self: Path, *args: any, **kw: any) -> None:
            if self == pins_file:
                raise FileNotFoundError("already gone")
            original_unlink(self, *args, **kw)

        with patch.object(Path, "unlink", failing_unlink):
            # Must not raise
            result = unpin("alias_x", cache_dir=tmp_path)

        assert result is True


@pytest.mark.parametrize(
    ("alias", "valid"),
    [
        ("good_alias", True),
        ("GoodAlias1", True),
        ("_ok", True),
        ("", False),
        ("1bad", False),
        ("bad-alias", False),
        ("bad alias", False),
    ],
)
def test_validate_alias_parametric(alias, valid: bool) -> None:
    if valid:
        _validate_alias(alias)
    else:
        with pytest.raises((ValueError, TypeError)):
            _validate_alias(alias)
