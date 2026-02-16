from __future__ import annotations

"""Tests for .env loading and environment application.

All environment mutations are isolated with pytest's monkeypatch.
"""

from pathlib import Path
import os

from scikitplot.mlflow._env import EnvSnapshot, apply_env


def test_envsnapshot_restore_roundtrip(monkeypatch) -> None:
    snap = EnvSnapshot.capture()
    monkeypatch.setenv("SP_TEST_ENV", "1")
    snap.restore()
    assert os.environ.get("SP_TEST_ENV") is None


def test_apply_env_defaults_only(tmp_path: Path, monkeypatch) -> None:
    envf = tmp_path / ".env"
    envf.write_text("FOO=fromfile\nBAR=fromfile\n", encoding="utf-8")

    monkeypatch.setenv("FOO", "existing")
    monkeypatch.delenv("BAR", raising=False)

    apply_env(env_file=str(envf), extra_env=None, set_defaults_only=True)
    assert os.environ["FOO"] == "existing"
    assert os.environ["BAR"] == "fromfile"


def test_apply_env_extra_overrides(tmp_path: Path, monkeypatch) -> None:
    envf = tmp_path / ".env"
    envf.write_text("FOO=fromfile\n", encoding="utf-8")

    monkeypatch.setenv("FOO", "existing")

    apply_env(env_file=str(envf), extra_env={"FOO": "override", "BAZ": "1"}, set_defaults_only=True)
    assert os.environ["FOO"] == "override"
    assert os.environ["BAZ"] == "1"
