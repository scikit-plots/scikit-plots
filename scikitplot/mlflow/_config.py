# scikitplot/mlflow/_config.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
_config.
"""

from __future__ import annotations

from collections.abc import Mapping as _AbcMapping
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SessionConfig:
    """
    Session-level configuration for `scikitplot.mlflow.session`.

    Parameters
    ----------
    tracking_uri : str or None, default=None
        MLflow tracking URI to use inside the session.
    public_tracking_uri : str or None, default=None
        Optional UI URL to display to users (e.g., when running in Docker/remote notebooks).
        This does NOT change the internal tracking URI used by MLflow calls; it is only for
        human-facing navigation.
        If None, the session reads `MLFLOW_TRACKING_URI` from the environment (after
        loading `env_file`, if provided). If `start_server=True` and no URI is provided,
        it will be constructed as `http://{tracking_host}:{server.port}`.
    registry_uri : str or None, default=None
        Optional MLflow registry URI. If None, the session reads `MLFLOW_REGISTRY_URI`
        from the environment (after loading `env_file`, if provided).
    env_file : str or None, default=None
        Optional path to a `.env` file. Keys are loaded only if missing from `os.environ`,
        matching MLflow CLI `--env-file` behavior.
    extra_env : Mapping[str, str] or None, default=None
        Additional environment variables to set for the duration of the session.
        These override current env vars and are restored on exit.
    startup_timeout_s : float, default=30.0
        Maximum seconds to wait for MLflow server readiness when `start_server=True`.
    ensure_reachable : bool, default=False
        If True, verify the configured tracking URI is reachable even when `start_server=False`.
        This performs the same readiness check used for managed servers.

    experiment_name : str or None, default=None
        If provided, set the active experiment on session entry (before any runs).
    create_experiment_if_missing : bool, default=True
        Controls behavior when `experiment_name` does not exist:
        - True: create the experiment (via `mlflow.set_experiment`)
        - False: raise a KeyError (strict fail)
    default_run_name : str or None, default=None
        Default `run_name` applied by :meth:`MlflowHandle.start_run` if the caller did not
        provide an explicit run name.
    default_run_tags : Mapping[str, str] or None, default=None
        Default tags applied by :meth:`MlflowHandle.start_run` when a run begins.

    Raises
    ------
    ValueError
        If `startup_timeout_s` is not positive.

    Notes
    -----
    Precedence order is strict:
    1) explicit arguments (`tracking_uri`, `registry_uri`, `extra_env`)
    2) existing environment variables
    3) `.env` file (fills missing keys only)
    4) defaults
    """

    tracking_uri: str | None = None
    public_tracking_uri: str | None = None
    registry_uri: str | None = None
    env_file: str | None = None
    extra_env: Mapping[str, str] | None = None
    startup_timeout_s: float = 30.0
    ensure_reachable: bool = False

    experiment_name: str | None = None
    create_experiment_if_missing: bool = True
    default_run_name: str | None = None
    default_run_tags: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive.")
        if self.extra_env is not None:
            if not isinstance(self.extra_env, _AbcMapping):
                raise ValueError(
                    f"extra_env must be a mapping, got {type(self.extra_env).__name__}."
                )
            for k, v in self.extra_env.items():
                if not isinstance(k, str) or not k:
                    raise ValueError(
                        f"extra_env keys must be non-empty str, got {k!r}."
                    )
                if not isinstance(v, str):
                    raise ValueError(
                        f"extra_env[{k!r}] must be str, got {type(v).__name__}."
                    )
        if self.default_run_tags is not None:
            if not isinstance(self.default_run_tags, _AbcMapping):
                raise ValueError(
                    f"default_run_tags must be a mapping, got {type(self.default_run_tags).__name__}."
                )
            for k, v in self.default_run_tags.items():
                if not isinstance(k, str) or not k:
                    raise ValueError(
                        f"default_run_tags keys must be non-empty str, got {k!r}."
                    )
                if not isinstance(v, str):
                    raise ValueError(
                        f"default_run_tags[{k!r}] must be str, got {type(v).__name__}."
                    )


@dataclass(frozen=True)
class ServerConfig:
    """
    Configuration that maps directly to `mlflow server` CLI flags.

    Parameters
    ----------
    host : str, default="127.0.0.1"
        Host interface to bind.
    port : int, default=5000
        Port to bind.
    auto_host_in_docker : bool, default=False
        If True and running inside Docker (``/.dockerenv`` exists), then a configured
        host of ``127.0.0.1`` will be overridden to ``docker_host`` when spawning the
        server. This makes the UI reachable with container port publishing.
    docker_host : str, default="0.0.0.0"
        Host value to use when ``auto_host_in_docker`` triggers.
    workers : int or None, default=None
        Optional number of worker processes.

    backend_store_uri : str or None, default=None
        `--backend-store-uri`.
    registry_store_uri : str or None, default=None
        `--registry-store-uri`.
    default_artifact_root : str or None, default=None
        `--default-artifact-root`.

    serve_artifacts : bool, default=False
        If True, adds `--serve-artifacts`.
    no_serve_artifacts : bool, default=False
        If True, adds `--no-serve-artifacts`.
    artifacts_destination : str or None, default=None
        `--artifacts-destination`.
    artifacts_only : bool, default=False
        If True, adds `--artifacts-only`.

    allowed_hosts : str or None, default=None
        `--allowed-hosts`.
    cors_allowed_origins : str or None, default=None
        `--cors-allowed-origins`.
    x_frame_options : str or None, default=None
        `--x-frame-options`.
    disable_security_middleware : bool, default=False
        `--disable-security-middleware`.

    static_prefix : str or None, default=None
        `--static-prefix`.
    uvicorn_opts : str or None, default=None
        `--uvicorn-opts`.
    gunicorn_opts : str or None, default=None
        `--gunicorn-opts`.
    waitress_opts : str or None, default=None
        `--waitress-opts`.

    expose_prometheus : str or None, default=None
        `--expose-prometheus` (directory to store metrics, enables `/metrics`).
    app_name : str or None, default=None
        `--app-name`.
    dev : bool, default=False
        `--dev` (debug + auto-reload; unsupported on Windows).

    secrets_cache_ttl : int or None, default=None
        `--secrets-cache-ttl` (seconds; MLflow enforces ranges at runtime).
    secrets_cache_max_size : int or None, default=None
        `--secrets-cache-max-size` (entries; MLflow enforces ranges at runtime).

    strict_cli_compat : bool, default=True
        If True, validate that every configured CLI flag is supported by the installed MLflow,
        using `mlflow server --help`. Unknown flags raise :class:`MlflowCliIncompatibleError`.
    extra_args : Sequence[str] or None, default=None
        Extra CLI args appended verbatim. With `strict_cli_compat=True`, flags inside `extra_args`
        are also checked for support.

    Raises
    ------
    ValueError
        If configuration violates generic constraints (e.g., invalid port).
    """

    host: str = "127.0.0.1"
    port: int = 5000
    auto_host_in_docker: bool = False
    docker_host: str = "0.0.0.0"  # noqa: S104
    workers: int | None = None

    backend_store_uri: str | None = None
    registry_store_uri: str | None = None
    default_artifact_root: str | None = None

    serve_artifacts: bool = False
    no_serve_artifacts: bool = False
    artifacts_destination: str | None = None
    artifacts_only: bool = False

    allowed_hosts: str | None = None
    cors_allowed_origins: str | None = None
    x_frame_options: str | None = None
    disable_security_middleware: bool = False

    static_prefix: str | None = None
    uvicorn_opts: str | None = None
    gunicorn_opts: str | None = None
    waitress_opts: str | None = None

    expose_prometheus: str | None = None
    app_name: str | None = None
    dev: bool = False

    secrets_cache_ttl: int | None = None
    secrets_cache_max_size: int | None = None

    strict_cli_compat: bool = True
    extra_args: Sequence[str] | None = None

    def validate(self, *, for_managed_tracking: bool) -> None:  # noqa: PLR0912
        """
        Validate configuration against generic MLflow constraints.

        Parameters
        ----------
        for_managed_tracking : bool
            If True, validation assumes a tracking server will be spawned and readiness
            will be checked via tracking REST endpoints.

        Raises
        ------
        ValueError
            If validation fails.

        Notes
        -----
        - This method validates generic invariants and known mutual-exclusivity rules.
        - Exact per-version validation is delegated to MLflow itself; unsupported flags are
          caught separately when `strict_cli_compat=True`.
        """
        if not self.host:
            raise ValueError("ServerConfig.host must be non-empty.")
        if not (1 <= int(self.port) <= 65535):  # noqa: PLR2004
            raise ValueError(
                f"ServerConfig.port must be in 1..65535, got {self.port!r}."
            )
        if self.workers is not None and self.workers < 1:
            raise ValueError(
                f"ServerConfig.workers must be >= 1, got {self.workers!r}."
            )

        if self.serve_artifacts and self.no_serve_artifacts:
            raise ValueError(
                "Only one of serve_artifacts and no_serve_artifacts may be True."
            )

        overrides = [
            bool(self.uvicorn_opts),
            bool(self.gunicorn_opts),
            bool(self.waitress_opts),
        ]
        if sum(overrides) > 1:
            raise ValueError(
                "Only one of uvicorn_opts, gunicorn_opts, waitress_opts may be set."
            )

        security_flags = any(
            [
                self.allowed_hosts is not None,
                self.cors_allowed_origins is not None,
                self.x_frame_options is not None,
                self.disable_security_middleware,
            ]
        )
        if security_flags and (
            self.gunicorn_opts is not None or self.waitress_opts is not None
        ):
            raise ValueError(
                "Security middleware options are not supported when using gunicorn_opts or waitress_opts."
            )

        if self.dev and (
            self.gunicorn_opts is not None or self.uvicorn_opts is not None
        ):
            raise ValueError(
                "dev=True cannot be used with gunicorn_opts or uvicorn_opts."
            )

        if for_managed_tracking and self.artifacts_only:
            raise ValueError(
                "artifacts_only=True cannot satisfy managed tracking server readiness/usage contract."
            )

        if self.secrets_cache_ttl is not None and self.secrets_cache_ttl <= 0:
            raise ValueError("secrets_cache_ttl must be a positive integer.")
        if self.secrets_cache_max_size is not None and self.secrets_cache_max_size <= 0:
            raise ValueError("secrets_cache_max_size must be a positive integer.")

        if self.extra_args is not None:
            for i, a in enumerate(self.extra_args):
                if not isinstance(a, str) or not a:
                    raise ValueError(
                        f"extra_args[{i}] must be a non-empty str, got {a!r}."
                    )
