from __future__ import annotations

"""Tests for test_cli_caps.py."""

import pytest

from scikitplot.mlflow._cli_caps import get_mlflow_server_cli_caps, ensure_flags_supported
from scikitplot.mlflow import MlflowCliIncompatibleError


def test_get_mlflow_server_cli_caps_parses_flags_click_style(monkeypatch) -> None:
    # Click can format options like: "-h, --host TEXT"
    sample_help = '''
Usage: mlflow server [OPTIONS]

Options:
  -h, --host TEXT
  -p, --port INTEGER
  --backend-store-uri TEXT
  --serve-artifacts
  --no-serve-artifacts
'''
    import scikitplot.mlflow._cli_caps as m
    monkeypatch.setattr(m, "_run_mlflow_server_help", lambda: sample_help)
    m.get_mlflow_server_cli_caps.cache_clear()
    caps = get_mlflow_server_cli_caps()
    assert "--host" in caps.flags
    assert "--port" in caps.flags
    assert "--serve-artifacts" in caps.flags


def test_ensure_flags_supported_raises() -> None:
    with pytest.raises(MlflowCliIncompatibleError):
        ensure_flags_supported(
            ["--host", "x", "--no-such-flag"],
            supported_flags=frozenset({"--host"}),
            context="test",
        )
