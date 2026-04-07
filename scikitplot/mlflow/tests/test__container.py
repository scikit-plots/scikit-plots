# scikitplot/mlflow/tests/test__container.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Canonical tests for scikitplot.mlflow._container.

Naming convention: test__<module_name>.py

Covers
------
- running_in_docker : returns False when /.dockerenv absent,
                      returns True when /.dockerenv present,
                      return type is always bool,
                      result independent of cwd

Notes
-----
/.dockerenv is mocked via monkeypatch on pathlib.Path.exists to prevent
test results from depending on whether the test runner itself is in Docker.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from scikitplot.mlflow._container import running_in_docker


# ===========================================================================
# running_in_docker
# ===========================================================================


class TestRunningInDocker:
    """Tests for running_in_docker()."""

    def test_returns_false_when_no_dockerenv(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When /.dockerenv does not exist, must return False."""
        monkeypatch.setattr(Path, "exists", lambda self: False)
        assert running_in_docker() is False

    def test_returns_true_when_dockerenv_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When /.dockerenv exists, must return True."""
        monkeypatch.setattr(Path, "exists", lambda self: True)
        assert running_in_docker() is True

    def test_return_type_is_always_bool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Return value must be exactly bool, not a truthy/falsy object."""
        for exists_val in (True, False):
            monkeypatch.setattr(Path, "exists", lambda self, v=exists_val: v)
            result = running_in_docker()
            assert isinstance(result, bool), (
                f"running_in_docker() must return bool, got {type(result)}"
            )

    def test_checks_correct_path(self) -> None:
        """Must query exactly /.dockerenv and not some other path."""
        checked_paths: list[str] = []

        def _recording_exists(self: Path) -> bool:
            checked_paths.append(str(self))
            return False

        with patch.object(Path, "exists", _recording_exists):
            running_in_docker()

        assert any("/.dockerenv" in p or p == "/.dockerenv" for p in checked_paths), (
            f"Expected /.dockerenv to be checked, got: {checked_paths}"
        )

    def test_no_side_effects_on_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """running_in_docker() must not mutate os.environ or raise."""
        import os
        env_before = dict(os.environ)
        monkeypatch.setattr(Path, "exists", lambda self: False)
        running_in_docker()
        assert dict(os.environ) == env_before

    def test_exported_in_all(self) -> None:
        """running_in_docker must appear in module __all__."""
        import scikitplot.mlflow._container as m
        assert "running_in_docker" in m.__all__
