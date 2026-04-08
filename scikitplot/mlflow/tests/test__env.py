# scikitplot/mlflow/tests/test__env.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Canonical tests for scikitplot.mlflow._env.

Naming convention: test__<module_name>.py

Covers
------
- EnvSnapshot.capture        : captures current environment
- EnvSnapshot.data           : returns a copy (not the internal reference)
- EnvSnapshot.restore        : removes session-added keys, restores original values
- parse_dotenv               : basic KEY=VALUE, quoted values (matched pair only),
                               export prefix, comment/blank-line skipping,
                               missing file raises FileNotFoundError,
                               values containing '=' are preserved whole
- apply_env                  : set_defaults_only=True does not overwrite existing keys,
                               set_defaults_only=False overwrites,
                               extra_env always overrides,
                               None env_file is a no-op on file loading

Notes
-----
All environment mutations are isolated via pytest's monkeypatch fixture or
explicit snapshot/restore to avoid polluting the process environment between tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scikitplot.mlflow._env import EnvSnapshot, apply_env, parse_dotenv


# ===========================================================================
# EnvSnapshot
# ===========================================================================


class TestEnvSnapshot:
    """Tests for EnvSnapshot correctness."""

    def test_data_returns_copy_not_reference(self) -> None:
        """
        EnvSnapshot.data must return a fresh dict each call.

        Notes
        -----
        If data returned self._data directly, callers mutating the result
        would silently corrupt the snapshot and break restore().
        """
        snap = EnvSnapshot.capture()
        d1 = snap.data
        d2 = snap.data
        assert d1 is not d2, "data must return a new dict each time"
        assert d1 == d2, "both copies must have identical contents"

    def test_data_mutation_does_not_corrupt_snapshot(self) -> None:
        """Mutating the returned dict must not affect future restore()."""
        snap = EnvSnapshot.capture()
        original_keys = set(snap.data.keys())
        d = snap.data
        d["_SHOULD_NOT_APPEAR_IN_SNAP"] = "poison"
        assert "_SHOULD_NOT_APPEAR_IN_SNAP" not in snap.data
        assert set(snap.data.keys()) == original_keys

    def test_capture_restore_roundtrip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        snap = EnvSnapshot.capture()
        monkeypatch.setenv("_SP_SNAP_TEST", "yes")
        assert os.environ.get("_SP_SNAP_TEST") == "yes"
        snap.restore()
        assert os.environ.get("_SP_SNAP_TEST") is None

    def test_restore_removes_session_added_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """restore() must clear keys added after capture, not just restore values."""
        monkeypatch.delenv("_SP_ADDED_AFTER", raising=False)
        snap = EnvSnapshot.capture()
        os.environ["_SP_ADDED_AFTER"] = "1"
        snap.restore()
        assert "_SP_ADDED_AFTER" not in os.environ

    def test_capture_reflects_current_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("_SP_CAPTURE_CHECK", "value123")
        snap = EnvSnapshot.capture()
        assert snap.data.get("_SP_CAPTURE_CHECK") == "value123"

    def test_restore_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Calling restore() twice must not raise."""
        snap = EnvSnapshot.capture()
        snap.restore()
        snap.restore()


# ===========================================================================
# parse_dotenv
# ===========================================================================


class TestParseDotenv:
    """Tests for parse_dotenv correctness."""

    def test_basic_key_value(self, tmp_path: Path) -> None:
        f = tmp_path / ".env"
        f.write_text("FOO=bar\nBAZ=qux\n", encoding="utf-8")
        result = parse_dotenv(str(f))
        assert result["FOO"] == "bar"
        assert result["BAZ"] == "qux"

    def test_matched_double_quotes_stripped(self, tmp_path: Path) -> None:
        f = tmp_path / ".env"
        f.write_text('A="hello world"\n', encoding="utf-8")
        assert parse_dotenv(str(f))["A"] == "hello world"

    def test_matched_single_quotes_stripped(self, tmp_path: Path) -> None:
        f = tmp_path / ".env"
        f.write_text("A='hello'\n", encoding="utf-8")
        assert parse_dotenv(str(f))["A"] == "hello"

    def test_mismatched_quotes_not_stripped(self, tmp_path: Path) -> None:
        """
        Mismatched outer quotes must be left unchanged.

        Notes
        -----
        A naive implementation using ``.strip('"').strip("'")`` would silently
        strip ``"value'`` because it strips each character independently.
        The correct implementation checks that both outer characters are identical.
        """
        f = tmp_path / ".env"
        f.write_text("A=\"mismatched'\n", encoding="utf-8")
        result = parse_dotenv(str(f))
        assert "mismatched" in result["A"]
        # At least one outer quote must survive.
        assert result["A"].startswith('"') or result["A"].endswith("'")

    def test_export_prefix_stripped(self, tmp_path: Path) -> None:
        f = tmp_path / ".env"
        f.write_text("export MY_VAR=myval\n", encoding="utf-8")
        assert parse_dotenv(str(f))["MY_VAR"] == "myval"

    def test_comment_lines_ignored(self, tmp_path: Path) -> None:
        f = tmp_path / ".env"
        f.write_text("# comment\nFOO=bar\n", encoding="utf-8")
        result = parse_dotenv(str(f))
        assert "comment" not in str(result)
        assert result["FOO"] == "bar"

    def test_empty_lines_ignored(self, tmp_path: Path) -> None:
        f = tmp_path / ".env"
        f.write_text("\n\nFOO=bar\n\n", encoding="utf-8")
        assert parse_dotenv(str(f)) == {"FOO": "bar"}

    def test_line_without_equals_ignored(self, tmp_path: Path) -> None:
        f = tmp_path / ".env"
        f.write_text("NOEQUALS\nFOO=bar\n", encoding="utf-8")
        result = parse_dotenv(str(f))
        assert "NOEQUALS" not in result
        assert result["FOO"] == "bar"

    def test_empty_value(self, tmp_path: Path) -> None:
        f = tmp_path / ".env"
        f.write_text("EMPTY=\n", encoding="utf-8")
        assert parse_dotenv(str(f))["EMPTY"] == ""

    def test_missing_file_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_dotenv("/nonexistent/.env")

    def test_value_with_equals_inside_preserved(self, tmp_path: Path) -> None:
        """Values containing '=' must not be truncated at the second '='."""
        f = tmp_path / ".env"
        f.write_text("TOKEN=abc=def=ghi\n", encoding="utf-8")
        assert parse_dotenv(str(f))["TOKEN"] == "abc=def=ghi"

    def test_returns_dict(self, tmp_path: Path) -> None:
        f = tmp_path / ".env"
        f.write_text("X=1\n", encoding="utf-8")
        result = parse_dotenv(str(f))
        assert isinstance(result, dict)

    def test_keys_are_stripped(self, tmp_path: Path) -> None:
        f = tmp_path / ".env"
        f.write_text(" SPACED = value\n", encoding="utf-8")
        result = parse_dotenv(str(f))
        assert "SPACED" in result


# ===========================================================================
# apply_env
# ===========================================================================


class TestApplyEnv:
    """Tests for apply_env defaults-only vs overwrite semantics."""

    def test_defaults_only_does_not_overwrite_existing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = tmp_path / ".env"
        f.write_text("EXISTING=fromfile\nNEW=fromfile\n", encoding="utf-8")
        monkeypatch.setenv("EXISTING", "original")
        monkeypatch.delenv("NEW", raising=False)
        apply_env(env_file=str(f), extra_env=None, set_defaults_only=True)
        assert os.environ["EXISTING"] == "original"
        assert os.environ["NEW"] == "fromfile"

    def test_overwrite_mode_replaces_existing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = tmp_path / ".env"
        f.write_text("EXISTING=fromfile\n", encoding="utf-8")
        monkeypatch.setenv("EXISTING", "original")
        apply_env(env_file=str(f), extra_env=None, set_defaults_only=False)
        assert os.environ["EXISTING"] == "fromfile"

    def test_extra_env_always_overrides(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = tmp_path / ".env"
        f.write_text("K=fromfile\n", encoding="utf-8")
        monkeypatch.setenv("K", "original")
        apply_env(env_file=str(f), extra_env={"K": "override"}, set_defaults_only=True)
        assert os.environ["K"] == "override"

    def test_no_env_file_is_noop_for_file_loading(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("_SP_X", raising=False)
        apply_env(env_file=None, extra_env={"_SP_X": "1"})
        assert os.environ["_SP_X"] == "1"

    def test_missing_env_file_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(FileNotFoundError):
            apply_env(env_file="/no/such/.env", extra_env=None)

    def test_extra_env_none_does_not_raise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = tmp_path / ".env"
        f.write_text("Z=1\n", encoding="utf-8")
        monkeypatch.delenv("Z", raising=False)
        apply_env(env_file=str(f), extra_env=None)
        assert os.environ["Z"] == "1"


# ===========================================================================
# Module structure
# ===========================================================================


def test_env_snapshot_restore_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backward-compatible module-level test retained from original test__env.py."""
    snap = EnvSnapshot.capture()
    monkeypatch.setenv("SP_TEST_ENV", "1")
    snap.restore()
    assert os.environ.get("SP_TEST_ENV") is None


def test_apply_env_defaults_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envf = tmp_path / ".env"
    envf.write_text("FOO=fromfile\nBAR=fromfile\n", encoding="utf-8")
    monkeypatch.setenv("FOO", "existing")
    monkeypatch.delenv("BAR", raising=False)
    apply_env(env_file=str(envf), extra_env=None, set_defaults_only=True)
    assert os.environ["FOO"] == "existing"
    assert os.environ["BAR"] == "fromfile"


def test_apply_env_extra_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envf = tmp_path / ".env"
    envf.write_text("FOO=fromfile\n", encoding="utf-8")
    monkeypatch.setenv("FOO", "existing")
    apply_env(
        env_file=str(envf),
        extra_env={"FOO": "override", "BAZ": "1"},
        set_defaults_only=True,
    )
    assert os.environ["FOO"] == "override"
    assert os.environ["BAZ"] == "1"
