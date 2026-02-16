# scikitplot/mlflow/_session.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
_session.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Iterator, Mapping
from urllib.parse import urlparse

from ._compat import import_mlflow
from ._config import ServerConfig, SessionConfig
from ._env import EnvSnapshot, apply_env
from ._facade import ArtifactsFacade, ModelsFacade
from ._readiness import wait_tracking_ready
from ._server import SpawnedServer, spawn_server
from ._utils import mlflow_version


def _parse_http_uri(uri: str) -> tuple[str, str, int]:
    """
    Parse an HTTP(S) URI into (scheme, host, port) for strict comparisons.

    Parameters
    ----------
    uri : str
        URI string.

    Returns
    -------
    tuple[str, str, int]
        Parsed scheme, hostname, and port.

    Raises
    ------
    ValueError
        If the URI is invalid or not HTTP(S).
    """
    u = uri.rstrip("/")
    p = urlparse(u)
    if p.scheme not in {"http", "https"}:
        raise ValueError(
            f"tracking_uri must be http(s) when start_server=True, got {uri!r}."
        )
    if not p.hostname:
        raise ValueError(f"Invalid HTTP(S) tracking_uri: {uri!r}.")
    port = p.port
    if port is None:
        port = 443 if p.scheme == "https" else 80
    return p.scheme, p.hostname, int(port)


def _hosts_equivalent(server_bind_host: str, tracking_host: str) -> bool:
    """
    Determine if a tracking host is compatible with a server bind host.

    Notes
    -----
    - If bind host is wildcard ("0.0.0.0" or "::"), allow localhost/loopback tracking host.
    - Treat "localhost" and "127.0.0.1" as equivalent for local access.
    """
    bind = (server_bind_host or "").strip().lower()
    th = (tracking_host or "").strip().lower()

    if bind in {"0.0.0.0", "::"}:  # noqa: S104
        return th in {"localhost", "127.0.0.1", "::1"}
    if bind in {"127.0.0.1", "localhost"}:
        return th in {"127.0.0.1", "localhost"}
    if bind == "::1":
        return th in {"::1", "localhost"}
    return bind == th


def _default_tracking_host_for_bind(bind_host: str) -> str:
    """
    Choose a sensible client hostname for a given bind host.

    Notes
    -----
    If you bind to all interfaces (0.0.0.0 / ::), clients should typically connect via
    loopback (127.0.0.1 / ::1) on the same machine.
    """
    b = (bind_host or "").strip()
    if b in {"0.0.0.0", "::"}:  # noqa: S104
        return "127.0.0.1"
    return b or "127.0.0.1"


def _set_experiment_strict(
    mlflow_mod: Any,
    *,
    experiment_name: str,
    create_if_missing: bool,
) -> None:
    """
    Set the active MLflow experiment with strict behavior.

    Parameters
    ----------
    mlflow_mod : module
        Imported `mlflow` module.
    experiment_name : str
        Experiment name to set.
    create_if_missing : bool
        If False, raise if experiment does not exist.

    Raises
    ------
    KeyError
        If create_if_missing is False and the experiment is missing.

    Notes
    -----
    This uses MLflow's canonical experiment APIs:
    - `get_experiment_by_name`
    - `set_experiment` (creates if missing)
    """
    if not create_if_missing:
        get_by_name = getattr(mlflow_mod, "get_experiment_by_name", None)
        if get_by_name is None:
            # Old MLflow would still create on set_experiment; strict mode can't be enforced.
            raise KeyError(
                "Strict experiment existence check requires mlflow.get_experiment_by_name, "
                "but it is unavailable in this MLflow version."
            )
        exp = get_by_name(experiment_name)
        if exp is None:
            raise KeyError(f"Experiment does not exist: {experiment_name!r}")
    mlflow_mod.set_experiment(experiment_name)


@dataclass
class MlflowHandle:
    """
    A handle that proxies the upstream `mlflow` module while adding session context.

    Attributes
    ----------
    mlflow_module : module
        Imported `mlflow` module.
    tracking_uri : str
        Resolved tracking URI used for this session.
    registry_uri : str or None
        Optional registry URI used for this session.
    ui_url : str
        Human-facing URL for opening the MLflow UI. This may differ from the internal tracking URI
        when running in containers or remote notebook environments.
    client : mlflow.tracking.MlflowClient
        MLflow client bound to the session.
    artifacts : ArtifactsFacade
        Artifact helper facade.
    models : ModelsFacade
        Model helper facade.
    server : SpawnedServer or None
        Spawned server handle if `start_server=True`.
    version : MlflowVersion or None
        Parsed installed MLflow version, if available.
    experiment_name : str or None
        Active experiment name configured for this session (if any).
    default_run_name : str or None
        Default run name used by :meth:`start_run` wrapper.
    default_run_tags : Mapping[str, str] or None
        Default tags applied by :meth:`start_run` wrapper.

    Notes
    -----
    Attribute access falls back to the underlying `mlflow` module. This allows users
    to omit `import mlflow` while keeping identical API usage.
    """

    _mlflow_module: Any
    _tracking_uri: str
    _registry_uri: str | None
    _ui_url: str
    _client: Any
    _artifacts: ArtifactsFacade
    _models: ModelsFacade
    server: SpawnedServer | None = None
    version: Any | None = None

    experiment_name: str | None = None
    default_run_name: str | None = None
    default_run_tags: Mapping[str, str] | None = None

    @property
    def mlflow_module(self) -> Any:
        """Imported `mlflow` module for this session."""
        return self._mlflow_module

    @property
    def tracking_uri(self) -> str:
        """Resolved tracking URI used for this session."""
        return self._tracking_uri

    @property
    def registry_uri(self) -> str | None:
        """Optional registry URI used for this session."""
        return self._registry_uri

    @property
    def ui_url(self) -> str:
        """Human-facing URL for opening the MLflow UI."""
        return self._ui_url

    @property
    def client(self) -> Any:
        """
        MLflow client bound to the session.

        Returns
        -------
        Any
            Instance compatible with ``mlflow.tracking.MlflowClient``.
        """
        return self._client

    @property
    def artifacts(self) -> ArtifactsFacade:
        """
        Artifact helper facade.

        Returns
        -------
        ArtifactsFacade
            Facade providing artifact helpers (log/list/download, etc.).
        """
        return self._artifacts

    @property
    def models(self) -> ModelsFacade:
        """
        Model helper facade.

        Returns
        -------
        ModelsFacade
            Facade providing model registry helpers.
        """
        return self._models

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying `mlflow` module."""
        return getattr(self.mlflow_module, name)

    @contextlib.contextmanager
    def start_run(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        """
        Start an MLflow run and apply session defaults.

        Parameters
        ----------
        *args : Any
            Positional args forwarded to `mlflow.start_run`.
        **kwargs : Any
            Keyword args forwarded to `mlflow.start_run`.

        Returns
        -------
        Iterator[Any]
            Context manager yielding the active run object.

        Raises
        ------
        Exception
            Propagates underlying MLflow errors.

        Notes
        -----
        Strict behavior:
        - If `run_name` is not provided and `default_run_name` is set, we pass it.
        - After run starts, if `default_run_tags` is set, apply them to the active run.

        This wrapper does not modify MLflow global state beyond the active run tags.
        """
        if "run_name" not in kwargs and self.default_run_name is not None:
            kwargs["run_name"] = self.default_run_name

        with self.mlflow_module.start_run(*args, **kwargs) as run:
            if self.default_run_tags:
                # Prefer public API if present.
                set_tags = getattr(self.mlflow_module, "set_tags", None)
                if callable(set_tags):
                    set_tags(dict(self.default_run_tags))
                else:
                    # Fallback to client.set_tag per key.
                    for k, v in self.default_run_tags.items():
                        self.client.set_tag(run.info.run_id, k, v)
            yield run


@contextlib.contextmanager
def session(  # noqa: PLR0912
    *,
    config: SessionConfig | None = None,
    server: ServerConfig | None = None,
    start_server: bool = False,
) -> Iterator[MlflowHandle]:
    """
    Create a strict, context-managed MLflow session.

    Parameters
    ----------
    config : SessionConfig or None, default=None
        Session-level configuration (URIs, `.env`, extra env, timeouts, defaults).
    server : ServerConfig or None, default=None
        Server configuration for managed server mode.
    start_server : bool, default=False
        If True, spawns an ephemeral `mlflow server` subprocess and tears it down on exit.

    Returns
    -------
    Iterator[MlflowHandle]
        A handle that proxies `mlflow` and exposes session-bound helpers.

    Raises
    ------
    ImportError
        If MLflow is not installed.
    FileNotFoundError
        If `config.env_file` is provided but missing.
    ValueError
        If configuration is invalid.
    RuntimeError
        If the server fails to start or readiness is not reached within the timeout.
    TimeoutError
        If readiness is not reached within the timeout.
    KeyError
        If `experiment_name` is provided and `create_experiment_if_missing=False`
        but the experiment does not exist (strict fail).

    Notes
    -----
    Environment is restored exactly on exit.
    """
    cfg = config or SessionConfig()
    srv = server or ServerConfig()

    if start_server:
        srv.validate(for_managed_tracking=True)

    snapshot = EnvSnapshot.capture()
    spawned: SpawnedServer | None = None

    try:
        # Apply env-file + explicit overrides (mirrors MLflow CLI semantics).
        apply_env(
            env_file=cfg.env_file, extra_env=cfg.extra_env, set_defaults_only=True
        )

        import os  # noqa: PLC0415

        tracking_uri = cfg.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        registry_uri = cfg.registry_uri or os.environ.get("MLFLOW_REGISTRY_URI")

        if start_server:
            default_host = _default_tracking_host_for_bind(srv.host)
            expected_tracking_uri = f"http://{default_host}:{srv.port}"
            if tracking_uri is None:
                tracking_uri = expected_tracking_uri
            else:
                a_scheme, a_host, a_port = _parse_http_uri(tracking_uri)
                b_scheme, _b_host, b_port = _parse_http_uri(expected_tracking_uri)
                if (
                    a_scheme != b_scheme
                    or a_port != b_port
                    or not _hosts_equivalent(srv.host, a_host)
                ):
                    raise ValueError(
                        "start_server=True requires tracking_uri to match the spawned server. "
                        f"Got tracking_uri={tracking_uri!r} but server bind={srv.host!r}:{srv.port!r} "
                        f"(expected like {expected_tracking_uri!r})."
                    )

        ui_url = cfg.public_tracking_uri or tracking_uri
        # Validate ui_url is usable for browsers when provided explicitly.
        if cfg.public_tracking_uri is not None:
            u = urlparse(ui_url)
            if u.scheme not in {"http", "https"}:
                raise ValueError(
                    f"public_tracking_uri must be http(s) when provided, got {ui_url!r}."
                )

        if not tracking_uri:
            raise RuntimeError(
                "No tracking URI resolved. Provide SessionConfig.tracking_uri or set MLFLOW_TRACKING_URI."
            )

        # Import MLflow lazily with a friendly error before spawning.
        mlflow_mod = import_mlflow()

        # Ensure subprocess and MLflow inherit resolved URIs.
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        if registry_uri is not None:
            os.environ["MLFLOW_REGISTRY_URI"] = registry_uri

        if start_server:
            spawned = spawn_server(srv)
            wait_tracking_ready(
                tracking_uri, timeout_s=cfg.startup_timeout_s, server=spawned
            )
        elif cfg.ensure_reachable:
            # ensure_reachable requires http(s) URI
            u = urlparse(tracking_uri)
            if u.scheme not in {"http", "https"}:
                raise ValueError(
                    f"ensure_reachable=True requires an http(s) tracking_uri, got {tracking_uri!r}."
                )
            wait_tracking_ready(
                tracking_uri, timeout_s=cfg.startup_timeout_s, server=None
            )

        # Bind URIs explicitly (avoid reliance on env alone).
        mlflow_mod.set_tracking_uri(tracking_uri)
        if registry_uri is not None and hasattr(mlflow_mod, "set_registry_uri"):
            mlflow_mod.set_registry_uri(registry_uri)

        # Construct client (registry_uri may be ignored if unsupported in older versions).
        try:
            client = mlflow_mod.tracking.MlflowClient(
                tracking_uri=tracking_uri, registry_uri=registry_uri
            )
        except TypeError:
            client = mlflow_mod.tracking.MlflowClient(tracking_uri=tracking_uri)

        # Set experiment if requested (strict behavior configurable).
        if cfg.experiment_name:
            _set_experiment_strict(
                mlflow_mod,
                experiment_name=cfg.experiment_name,
                create_if_missing=cfg.create_experiment_if_missing,
            )

        artifacts = ArtifactsFacade(mlflow_module=mlflow_mod, client=client)
        models = ModelsFacade(mlflow_module=mlflow_mod, client=client)

        yield MlflowHandle(
            _mlflow_module=mlflow_mod,
            _tracking_uri=tracking_uri,
            _registry_uri=registry_uri,
            _ui_url=ui_url,
            _client=client,
            _artifacts=artifacts,
            _models=models,
            server=spawned,
            version=mlflow_version(),
            experiment_name=cfg.experiment_name,
            default_run_name=cfg.default_run_name,
            default_run_tags=cfg.default_run_tags,
        )

    finally:
        if spawned is not None:
            try:  # noqa: SIM105
                spawned.terminate()
            except Exception:
                pass
        snapshot.restore()
