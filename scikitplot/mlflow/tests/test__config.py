from __future__ import annotations

"""Tests for test_config.py."""

import pytest

from scikitplot.mlflow import SessionConfig


def test_sessionconfig_timeout_positive() -> None:
    with pytest.raises(ValueError):
        SessionConfig(startup_timeout_s=0)


def test_sessionconfig_extra_env_must_be_mapping() -> None:
    with pytest.raises(ValueError):
        SessionConfig(extra_env=["a"])  # type: ignore[arg-type]


def test_sessionconfig_extra_env_keys_values_str() -> None:
    with pytest.raises(ValueError):
        SessionConfig(extra_env={1: "x"})  # type: ignore[dict-item]
    with pytest.raises(ValueError):
        SessionConfig(extra_env={"A": 1})  # type: ignore[dict-item]


def test_sessionconfig_default_run_tags_strict() -> None:
    with pytest.raises(ValueError):
        SessionConfig(default_run_tags=["x"])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SessionConfig(default_run_tags={1: "x"})  # type: ignore[dict-item]
    with pytest.raises(ValueError):
        SessionConfig(default_run_tags={"A": 1})  # type: ignore[dict-item]


def test_sessionconfig_valid() -> None:
    cfg = SessionConfig(
        tracking_uri="http://127.0.0.1:5000",
        extra_env={"A": "1"},
        default_run_tags={"k": "v"},
        default_run_name="train",
        experiment_name="exp",
    )
    assert cfg.tracking_uri == "http://127.0.0.1:5000"
