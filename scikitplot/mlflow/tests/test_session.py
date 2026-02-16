from __future__ import annotations

"""Tests for the MLflow session context manager.

These tests use a dummy MLflow module fixture (see conftest.py) to avoid depending on the real MLflow package.
"""

import os
import pytest
import importlib

from scikitplot.mlflow import SessionConfig
from scikitplot.mlflow._session import session


def test_session_sets_experiment_and_defaults(monkeypatch, dummy_mlflow) -> None:
    s = importlib.import_module("scikitplot.mlflow._session")

    monkeypatch.setattr(s, "import_mlflow", lambda: dummy_mlflow)
    monkeypatch.setattr(s, "wait_tracking_ready", lambda *a, **k: None)

    cfg = SessionConfig(
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="exp1",
        create_experiment_if_missing=True,
        default_run_name="train",
        default_run_tags={"pipeline": "train"},
    )
    with session(config=cfg, start_server=False) as h:
        assert h.ui_url == "http://127.0.0.1:5000"
        assert h.experiment_name == "exp1"
        with h.start_run() as run:
            assert run.info.run_id == "run-123"
        assert dummy_mlflow._set_tags_calls == [{"pipeline": "train"}]
        assert dummy_mlflow._start_run_calls[-1]["kwargs"]["run_name"] == "train"


def test_session_strict_missing_experiment_raises(monkeypatch, dummy_mlflow) -> None:
    s = importlib.import_module("scikitplot.mlflow._session")
    monkeypatch.setattr(s, "import_mlflow", lambda: dummy_mlflow)

    cfg = SessionConfig(
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="missing",
        create_experiment_if_missing=False,
    )

    with pytest.raises(KeyError):
        with session(config=cfg, start_server=False):
            pass


def test_session_env_restored(monkeypatch, dummy_mlflow) -> None:
    s = importlib.import_module("scikitplot.mlflow._session")
    monkeypatch.setattr(s, "import_mlflow", lambda: dummy_mlflow)

    monkeypatch.delenv("SP_MLFLOW_TEST", raising=False)
    cfg = SessionConfig(tracking_uri="http://127.0.0.1:5000", extra_env={"SP_MLFLOW_TEST": "1"})
    with session(config=cfg, start_server=False):
        assert os.environ.get("SP_MLFLOW_TEST") == "1"
    assert os.environ.get("SP_MLFLOW_TEST") is None


def test_session_ensure_reachable_requires_http(monkeypatch, dummy_mlflow) -> None:
    s = importlib.import_module("scikitplot.mlflow._session")
    monkeypatch.setattr(s, "import_mlflow", lambda: dummy_mlflow)

    cfg = SessionConfig(tracking_uri="file:/tmp/mlruns", ensure_reachable=True, startup_timeout_s=0.1)
    with pytest.raises(ValueError):
        with session(config=cfg, start_server=False):
            pass
