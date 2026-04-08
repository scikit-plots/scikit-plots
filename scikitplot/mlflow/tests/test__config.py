# scikitplot/mlflow/tests/test__config.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Canonical tests for scikitplot.mlflow._config.

Naming convention: test__<module_name>.py

Covers
------
- SessionConfig  : startup_timeout_s validation (must be positive),
                   extra_env type and key/value str enforcement,
                   default_run_tags type and key/value str enforcement,
                   valid construction with all fields
- ServerConfig   : validate() port boundaries (1-65535),
                   validate() empty host raises,
                   validate() workers must be positive,
                   mutually exclusive flags (serve/no_serve_artifacts,
                   gunicorn/uvicorn/waitress opts, dev+opts, allowed_hosts+opts),
                   artifacts_only incompatible with managed tracking,
                   secrets_cache_ttl must be positive,
                   secrets_cache_max_size non-negative,
                   extra_args no empty strings,
                   valid boundary values pass

Notes
-----
Pure-Python; no external dependencies.
"""

from __future__ import annotations

import pytest

from scikitplot.mlflow._config import ServerConfig, SessionConfig


# ===========================================================================
# SessionConfig
# ===========================================================================


class TestSessionConfig:
    """Tests for SessionConfig validation."""

    def test_valid_minimal(self) -> None:
        cfg = SessionConfig()
        assert cfg.startup_timeout_s == 30.0

    def test_valid_all_fields(self) -> None:
        cfg = SessionConfig(
            tracking_uri="http://127.0.0.1:5000",
            extra_env={"A": "1"},
            default_run_tags={"k": "v"},
            default_run_name="train",
            experiment_name="exp",
        )
        assert cfg.tracking_uri == "http://127.0.0.1:5000"

    @pytest.mark.parametrize("timeout", [0, -1, -0.001])
    def test_startup_timeout_non_positive_raises(self, timeout: float) -> None:
        with pytest.raises(ValueError):
            SessionConfig(startup_timeout_s=timeout)

    def test_extra_env_must_be_mapping(self) -> None:
        with pytest.raises(ValueError):
            SessionConfig(extra_env=["a"])  # type: ignore[arg-type]

    def test_extra_env_int_key_raises(self) -> None:
        with pytest.raises(ValueError):
            SessionConfig(extra_env={1: "x"})  # type: ignore[dict-item]

    def test_extra_env_int_value_raises(self) -> None:
        with pytest.raises(ValueError):
            SessionConfig(extra_env={"A": 1})  # type: ignore[dict-item]

    def test_default_run_tags_non_mapping_raises(self) -> None:
        with pytest.raises(ValueError):
            SessionConfig(default_run_tags=["x"])  # type: ignore[arg-type]

    def test_default_run_tags_non_str_key_raises(self) -> None:
        with pytest.raises(ValueError):
            SessionConfig(default_run_tags={1: "x"})  # type: ignore[dict-item]

    def test_default_run_tags_non_str_value_raises(self) -> None:
        with pytest.raises(ValueError):
            SessionConfig(default_run_tags={"A": 1})  # type: ignore[dict-item]

    def test_extra_env_empty_key_raises(self) -> None:
        with pytest.raises(ValueError):
            SessionConfig(extra_env={"": "val"})

    def test_extra_env_none_is_valid(self) -> None:
        cfg = SessionConfig(extra_env=None)
        assert cfg.extra_env is None

    def test_default_run_tags_none_is_valid(self) -> None:
        cfg = SessionConfig(default_run_tags=None)
        assert cfg.default_run_tags is None

    def test_create_experiment_if_missing_default_true(self) -> None:
        cfg = SessionConfig()
        assert cfg.create_experiment_if_missing is True

    def test_custom_timeout_accepted(self) -> None:
        cfg = SessionConfig(startup_timeout_s=0.001)
        assert cfg.startup_timeout_s == pytest.approx(0.001)


# ===========================================================================
# ServerConfig
# ===========================================================================


class TestServerConfig:
    """Tests for ServerConfig.validate() exhaustive matrix."""

    def test_valid_defaults(self) -> None:
        cfg = ServerConfig()
        cfg.validate(for_managed_tracking=False)

    @pytest.mark.parametrize("port", [0, -1, 65536, 99999])
    def test_invalid_port_raises(self, port: int) -> None:
        with pytest.raises(ValueError, match="port"):
            ServerConfig(port=port).validate(for_managed_tracking=False)

    def test_valid_port_minimum(self) -> None:
        ServerConfig(port=1).validate(for_managed_tracking=False)

    def test_valid_port_maximum(self) -> None:
        ServerConfig(port=65535).validate(for_managed_tracking=False)

    def test_empty_host_raises(self) -> None:
        with pytest.raises(ValueError, match="host"):
            ServerConfig(host="").validate(for_managed_tracking=False)

    def test_workers_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="workers"):
            ServerConfig(workers=0).validate(for_managed_tracking=False)

    def test_workers_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="workers"):
            ServerConfig(workers=-1).validate(for_managed_tracking=False)

    def test_serve_and_no_serve_artifacts_conflict(self) -> None:
        with pytest.raises(ValueError):
            ServerConfig(
                serve_artifacts=True, no_serve_artifacts=True
            ).validate(for_managed_tracking=False)

    def test_gunicorn_and_uvicorn_opts_conflict(self) -> None:
        with pytest.raises(ValueError):
            ServerConfig(
                gunicorn_opts="--workers 2", uvicorn_opts="--workers 2"
            ).validate(for_managed_tracking=False)

    def test_three_server_opts_conflict(self) -> None:
        with pytest.raises(ValueError):
            ServerConfig(
                gunicorn_opts="x", uvicorn_opts="y", waitress_opts="z"
            ).validate(for_managed_tracking=False)

    def test_security_flags_with_gunicorn_conflict(self) -> None:
        with pytest.raises(ValueError):
            ServerConfig(
                allowed_hosts="localhost", gunicorn_opts="--workers 2"
            ).validate(for_managed_tracking=False)

    def test_dev_with_gunicorn_raises(self) -> None:
        with pytest.raises(ValueError):
            ServerConfig(dev=True, gunicorn_opts="x").validate(
                for_managed_tracking=False
            )

    def test_dev_with_uvicorn_raises(self) -> None:
        with pytest.raises(ValueError):
            ServerConfig(dev=True, uvicorn_opts="x").validate(
                for_managed_tracking=False
            )

    def test_artifacts_only_managed_tracking_raises(self) -> None:
        with pytest.raises(ValueError, match="artifacts_only"):
            ServerConfig(artifacts_only=True).validate(for_managed_tracking=True)

    def test_secrets_cache_ttl_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="secrets_cache_ttl"):
            ServerConfig(secrets_cache_ttl=0).validate(for_managed_tracking=False)

    def test_secrets_cache_max_size_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="secrets_cache_max_size"):
            ServerConfig(secrets_cache_max_size=-5).validate(
                for_managed_tracking=False
            )

    def test_extra_args_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="extra_args"):
            ServerConfig(extra_args=["", "--port"]).validate(
                for_managed_tracking=False
            )

    def test_no_serve_artifacts_flag_is_valid(self) -> None:
        ServerConfig(no_serve_artifacts=True).validate(for_managed_tracking=False)

    def test_artifacts_only_not_managed_tracking_is_valid(self) -> None:
        ServerConfig(artifacts_only=True).validate(for_managed_tracking=False)


# ===========================================================================
# Backward-compatible module-level tests
# ===========================================================================


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
