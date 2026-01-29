from __future__ import annotations

"""Tests for test_public_ui.py."""

import importlib
import pytest

from scikitplot.mlflow import SessionConfig
from scikitplot.mlflow._session import session


def test_public_tracking_uri_requires_http(monkeypatch, dummy_mlflow) -> None:
    s = importlib.import_module("scikitplot.mlflow._session")
    monkeypatch.setattr(s, "import_mlflow", lambda: dummy_mlflow)

    cfg = SessionConfig(
        tracking_uri="http://127.0.0.1:5000",
        public_tracking_uri="file:/tmp/x",
    )
    with pytest.raises(ValueError):
        with session(config=cfg, start_server=False):
            pass


def test_public_tracking_uri_sets_ui_url(monkeypatch, dummy_mlflow) -> None:
    s = importlib.import_module("scikitplot.mlflow._session")
    monkeypatch.setattr(s, "import_mlflow", lambda: dummy_mlflow)

    cfg = SessionConfig(
        tracking_uri="http://127.0.0.1:5000",
        public_tracking_uri="http://localhost:15000",
    )
    with session(config=cfg, start_server=False) as h:
        assert h.ui_url == "http://localhost:15000"
