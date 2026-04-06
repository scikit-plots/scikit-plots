# scikitplot/mlflow/tests/test__cli_caps.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Canonical tests for scikitplot.mlflow._cli_caps.

Naming convention: test__<module_name>.py  (double underscore to match _cli_caps).

Covers
------
- _run_mlflow_server_help : subprocess happy path, empty output raises RuntimeError (line 70)
- _extract_long_flags     : all flag patterns, hyphenated names, empty text
- get_mlflow_server_cli_caps : lru_cache semantics, fresh cache after cache_clear
- ensure_flags_supported  : passes for valid flags, equals-form, skips short flags,
                            raises MlflowCliIncompatibleError for unknown flags,
                            empty args list, single short flag only

Notes
-----
All subprocess calls are fully mocked so the suite runs without a working MLflow install.
The lru_cache on get_mlflow_server_cli_caps is cleared before and after every test that
exercises it to prevent cross-test pollution.
"""

from __future__ import annotations

import subprocess
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import scikitplot.mlflow._cli_caps as _m
from scikitplot.mlflow._cli_caps import (
    MlflowServerCliCaps,
    _extract_long_flags,
    _run_mlflow_server_help,
    ensure_flags_supported,
    get_mlflow_server_cli_caps,
)
from scikitplot.mlflow._errors import MlflowCliIncompatibleError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_HELP = """\
Usage: mlflow server [OPTIONS]

Options:
  -h, --host TEXT                 Hostname to listen on.
  -p, --port INTEGER              Port to listen on.
  --backend-store-uri TEXT        Backend store URI.
  --default-artifact-root TEXT    Default artifact root.
  --serve-artifacts               Serve artifacts.
  --no-serve-artifacts            Disable artifact serving.
  --gunicorn-opts TEXT            Extra Gunicorn args.
  --workers INTEGER               Worker count.
  --help                          Show this message and exit.
"""


@pytest.fixture(autouse=True)
def _clear_lru_cache():
    """Clear the lru_cache before and after every test to prevent pollution."""
    get_mlflow_server_cli_caps.cache_clear()
    yield
    get_mlflow_server_cli_caps.cache_clear()


# ===========================================================================
# _run_mlflow_server_help
# ===========================================================================


class TestRunMlflowServerHelp:
    """Tests for _run_mlflow_server_help."""

    def test_returns_stdout_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Happy path: subprocess returns non-empty stdout."""

        def _fake_run(cmd, **kwargs):
            return SimpleNamespace(stdout=_SAMPLE_HELP)

        monkeypatch.setattr(subprocess, "run", _fake_run)
        result = _run_mlflow_server_help()
        assert "--host" in result
        assert "--port" in result

    def test_empty_output_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Line 70: empty stdout must raise RuntimeError immediately.

        Developer note: this directly covers the uncovered branch in
        _run_mlflow_server_help where stdout strips to an empty string.
        The lru_cache on get_mlflow_server_cli_caps must NOT be used here
        so we test the raw function.
        """

        def _fake_run(cmd, **kwargs):
            return SimpleNamespace(stdout="   \n  \t  ")  # whitespace only → strips to ""

        monkeypatch.setattr(subprocess, "run", _fake_run)
        with pytest.raises(RuntimeError, match="empty output"):
            _run_mlflow_server_help()

    def test_none_stdout_treated_as_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Subprocess stdout=None (e.g., if PIPE not captured) → empty string → raises."""

        def _fake_run(cmd, **kwargs):
            return SimpleNamespace(stdout=None)

        monkeypatch.setattr(subprocess, "run", _fake_run)
        with pytest.raises(RuntimeError, match="empty output"):
            _run_mlflow_server_help()

    def test_uses_sys_executable_in_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Command must include sys.executable to stay in the same venv."""
        captured: list[list[str]] = []

        def _fake_run(cmd, **kwargs):
            captured.append(cmd)
            return SimpleNamespace(stdout=_SAMPLE_HELP)

        monkeypatch.setattr(subprocess, "run", _fake_run)
        _run_mlflow_server_help()
        assert captured, "subprocess.run must be called"
        cmd = captured[0]
        assert cmd[0] == sys.executable
        assert "mlflow" in cmd
        assert "server" in cmd
        assert "--help" in cmd


# ===========================================================================
# _extract_long_flags
# ===========================================================================


class TestExtractLongFlags:
    """Tests for _extract_long_flags."""

    def test_extracts_simple_long_flags(self) -> None:
        """Standard --flag-name lines are captured."""
        text = "--host TEXT\n--port INTEGER\n--backend-store-uri TEXT"
        flags = _extract_long_flags(text)
        assert "--host" in flags
        assert "--port" in flags
        assert "--backend-store-uri" in flags

    def test_click_short_and_long_format(self) -> None:
        """Click renders '-h, --host TEXT'; long flag must be captured."""
        flags = _extract_long_flags(_SAMPLE_HELP)
        assert "--host" in flags
        assert "--port" in flags
        assert "--serve-artifacts" in flags
        assert "--no-serve-artifacts" in flags

    def test_short_flags_not_captured(self) -> None:
        """Short flags like -h or -p must NOT appear in the result."""
        flags = _extract_long_flags("-h TEXT\n-p INTEGER")
        # Only single-dash tokens; nothing starts with '--'
        assert not any(f.startswith("--") for f in flags)

    def test_hyphenated_flag_names(self) -> None:
        """Multi-hyphen flag names like --backend-store-uri are captured fully."""
        flags = _extract_long_flags("  --backend-store-uri TEXT\n  --default-artifact-root TEXT")
        assert "--backend-store-uri" in flags
        assert "--default-artifact-root" in flags

    def test_empty_text_returns_empty_frozenset(self) -> None:
        """Empty help text yields an empty frozenset."""
        assert _extract_long_flags("") == frozenset()

    def test_returns_frozenset(self) -> None:
        """Result type is frozenset."""
        result = _extract_long_flags("--host TEXT")
        assert isinstance(result, frozenset)

    def test_deduplicates_flags(self) -> None:
        """Repeated flags are deduplicated by the frozenset."""
        flags = _extract_long_flags("--host TEXT\n--host INTEGER\n--host BOOL")
        assert flags.count("--host") if hasattr(flags, "count") else "--host" in flags
        # frozenset has no count, but element appears only once
        assert len([f for f in flags if f == "--host"]) == 1

    def test_ignores_non_flag_words(self) -> None:
        """Words like TEXT, INTEGER, Options: are not captured."""
        flags = _extract_long_flags("Options:\n  --host TEXT\n  INTEGER\n  host")
        bad = {"TEXT", "INTEGER", "Options:", "host"}
        assert not any(b in flags for b in bad)


# ===========================================================================
# get_mlflow_server_cli_caps (lru_cache)
# ===========================================================================


class TestGetMlflowServerCliCaps:
    """Tests for get_mlflow_server_cli_caps including lru_cache semantics."""

    def test_returns_mlflow_server_cli_caps_instance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_m, "_run_mlflow_server_help", lambda: _SAMPLE_HELP)
        caps = get_mlflow_server_cli_caps()
        assert isinstance(caps, MlflowServerCliCaps)

    def test_flags_are_frozenset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_m, "_run_mlflow_server_help", lambda: _SAMPLE_HELP)
        caps = get_mlflow_server_cli_caps()
        assert isinstance(caps.flags, frozenset)

    def test_parses_click_style_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_m, "_run_mlflow_server_help", lambda: _SAMPLE_HELP)
        caps = get_mlflow_server_cli_caps()
        assert "--host" in caps.flags
        assert "--port" in caps.flags
        assert "--serve-artifacts" in caps.flags

    def test_lru_cache_returns_same_instance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """lru_cache must return the same object on repeated calls (maxsize=1)."""
        call_count = {"n": 0}

        def _help():
            call_count["n"] += 1
            return _SAMPLE_HELP

        monkeypatch.setattr(_m, "_run_mlflow_server_help", _help)
        c1 = get_mlflow_server_cli_caps()
        c2 = get_mlflow_server_cli_caps()
        assert c1 is c2, "lru_cache must return the same instance"
        assert call_count["n"] == 1, "_run_mlflow_server_help must be called only once"

    def test_cache_clear_triggers_re_parse(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After cache_clear(), the next call must invoke _run_mlflow_server_help again."""
        call_count = {"n": 0}

        def _help():
            call_count["n"] += 1
            return _SAMPLE_HELP

        monkeypatch.setattr(_m, "_run_mlflow_server_help", _help)
        get_mlflow_server_cli_caps()
        get_mlflow_server_cli_caps.cache_clear()
        get_mlflow_server_cli_caps()
        assert call_count["n"] == 2

    def test_empty_help_propagates_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If _run_mlflow_server_help raises, get_mlflow_server_cli_caps propagates it."""

        def _bad_help():
            raise RuntimeError("Failed to capture MLflow CLI help output (empty output).")

        monkeypatch.setattr(_m, "_run_mlflow_server_help", _bad_help)
        with pytest.raises(RuntimeError, match="empty output"):
            get_mlflow_server_cli_caps()


# ===========================================================================
# ensure_flags_supported
# ===========================================================================


class TestEnsureFlagsSupported:
    """Tests for ensure_flags_supported."""

    _SUPPORTED = frozenset({"--host", "--port", "--backend-store-uri", "--serve-artifacts"})

    def test_valid_flags_do_not_raise(self) -> None:
        """All long flags present in supported_flags must pass silently."""
        ensure_flags_supported(
            ["--host", "127.0.0.1", "--port", "5000"],
            supported_flags=self._SUPPORTED,
            context="test",
        )

    def test_unknown_flag_raises(self) -> None:
        """A single unknown long flag must raise MlflowCliIncompatibleError."""
        with pytest.raises(MlflowCliIncompatibleError, match="--unknown-flag"):
            ensure_flags_supported(
                ["--host", "127.0.0.1", "--unknown-flag"],
                supported_flags=self._SUPPORTED,
                context="test",
            )

    def test_equals_form_is_parsed_correctly(self) -> None:
        """--flag=value must extract --flag before the '=' for validation."""
        ensure_flags_supported(
            ["--host=127.0.0.1", "--port=5000"],
            supported_flags=self._SUPPORTED,
            context="test",
        )

    def test_equals_form_unknown_raises(self) -> None:
        """--unknown=value must raise even though it uses equals-form."""
        with pytest.raises(MlflowCliIncompatibleError, match="--no-such"):
            ensure_flags_supported(
                ["--no-such=value"],
                supported_flags=self._SUPPORTED,
                context="test",
            )

    def test_short_flags_are_skipped(self) -> None:
        """Short flags (-h, -p) are not validated and must not raise."""
        ensure_flags_supported(
            ["-h", "127.0.0.1", "-p", "5000"],
            supported_flags=frozenset(),  # empty supported set
            context="test",
        )

    def test_empty_args_list_does_not_raise(self) -> None:
        """An empty argument list is valid."""
        ensure_flags_supported(
            [],
            supported_flags=self._SUPPORTED,
            context="test",
        )

    def test_positional_values_not_treated_as_flags(self) -> None:
        """Plain values (no -- prefix) are ignored, not validated."""
        ensure_flags_supported(
            ["--host", "127.0.0.1", "some-extra-value"],
            supported_flags=self._SUPPORTED,
            context="test",
        )

    def test_error_message_includes_context(self) -> None:
        """Error message must include the context string for actionable diagnostics."""
        with pytest.raises(MlflowCliIncompatibleError, match="my-context"):
            ensure_flags_supported(
                ["--bad-flag"],
                supported_flags=frozenset({"--good-flag"}),
                context="my-context",
            )

    def test_error_message_includes_flag_name(self) -> None:
        """Error message must contain the unsupported flag name."""
        with pytest.raises(MlflowCliIncompatibleError, match="--missing-flag"):
            ensure_flags_supported(
                ["--missing-flag"],
                supported_flags=frozenset(),
                context="ctx",
            )

    def test_first_unsupported_flag_raises_immediately(self) -> None:
        """Fail-fast: raises on the first unsupported flag, does not continue."""
        raised_flags: list[str] = []
        try:
            ensure_flags_supported(
                ["--unknown-a", "--unknown-b"],
                supported_flags=frozenset(),
                context="ctx",
            )
        except MlflowCliIncompatibleError as exc:
            msg = str(exc)
            raised_flags = [f for f in ["--unknown-a", "--unknown-b"] if f in msg]

        # Only the first flag should be mentioned in the error
        assert "--unknown-a" in raised_flags
        assert "--unknown-b" not in raised_flags

    def test_single_supported_flag_passes(self) -> None:
        """A single valid flag list passes without error."""
        ensure_flags_supported(
            ["--serve-artifacts"],
            supported_flags=frozenset({"--serve-artifacts"}),
            context="ctx",
        )


# ===========================================================================
# MlflowServerCliCaps dataclass
# ===========================================================================


class TestMlflowServerCliCaps:
    """Tests for the MlflowServerCliCaps frozen dataclass."""

    def test_default_flags_is_empty_frozenset(self) -> None:
        caps = MlflowServerCliCaps()
        assert caps.flags == frozenset()

    def test_custom_flags_stored(self) -> None:
        flags = frozenset({"--host", "--port"})
        caps = MlflowServerCliCaps(flags=flags)
        assert caps.flags == flags

    def test_immutability(self) -> None:
        """Frozen dataclass must not allow mutation."""
        caps = MlflowServerCliCaps(flags=frozenset({"--host"}))
        with pytest.raises((AttributeError, TypeError)):
            caps.flags = frozenset()  # type: ignore[misc]
