# scikitplot/mlflow/_project.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Project configuration helpers for :py:mod:`~scikitplot.mlflow`.

This module provides two distinct, deterministic responsibilities:

1) **Project root discovery** via marker files/directories (e.g., ``pyproject.toml`` or ``.git``).
2) **Project-level MLflow config I/O** (TOML/YAML) that normalizes local paths so that
   multiple scripts (train/hpo/predict) behave consistently regardless of current working directory.

Notes
-----
This module intentionally exposes a *small* public surface for marker customization, while keeping
the underlying mutable default private. Users should prefer:
- :func:`get_project_markers`
- :func:`set_project_markers`
- :func:`project_markers` (context manager)
- Environment override via ``SCIKITPLOT_PROJECT_MARKERS`` (strict JSON list of strings)
- Config override via a TOML file containing ``[project].markers = [...]``

Examples
--------
Option A — temporary (best for automation pipelines)

>>> from scikitplot.mlflow._project import project_markers, find_project_root
>>>
>>> with project_markers(["pyproject.toml", ".git", "configs/mlflow.toml"]):
...     root = find_project_root()

Option B — environment (best for CI)

>>> import os
>>>
>>> # Default marker file-folder for auto detection
>>> # Walk upward from `start` until a directory containing any marker is found.
>>> # export SCIKITPLOT_PROJECT_MARKERS='[".git","pyproject.toml","README.txt","configs/mlflow.toml"]'
>>> os.environ["SCIKITPLOT_PROJECT_MARKERS"] = (
...     '[".git","pyproject.toml","README.txt","configs/mlflow.toml"]'
... )

Option C — config-driven (best for teams)

>>> [project]
>>> markers = ["pyproject.toml", ".git", "configs/mlflow.toml"]
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import json
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

from ._config import ServerConfig, SessionConfig

# ---------------------------------------------------------------------------
# Marker configuration (public API + private mutable default)
# ---------------------------------------------------------------------------

#: Public, stable default markers documented for users.
DEFAULT_PROJECT_MARKERS: tuple[str, ...] = (
    ".git",
    "configs",
    "configs/mlflow.toml",
    "Makefile",
    "pyproject.toml",
    "README",
    "README.txt",
    "README.md",
    "README.rst",
)

# Private mutable default used when caller does not pass markers and no config/env override exists.
_PROJECT_MARKERS: tuple[str, ...] = DEFAULT_PROJECT_MARKERS

_ENV_PROJECT_MARKERS = "SCIKITPLOT_PROJECT_MARKERS"

# Default objects used as dataclass defaults (kept immutable: SessionConfig is frozen).
# NOTE: This avoids Ruff RUF009 (function call in dataclass defaults) while also ensuring
# class-level attributes exist for Sphinx/linkcode introspection.
_DEFAULT_SESSION_CONFIG: SessionConfig = SessionConfig()


def _validate_markers(markers: Sequence[str]) -> tuple[str, ...]:
    """
    Validate and normalize marker sequences.

    Parameters
    ----------
    markers : sequence[str]
        Marker file/directory names.

    Returns
    -------
    tuple[str, ...]
        Validated marker tuple.

    Raises
    ------
    TypeError
        If `markers` is not a sequence of non-empty strings.
    ValueError
        If `markers` is empty.
    """
    if not isinstance(markers, Sequence) or isinstance(markers, (str, bytes)):
        raise TypeError("markers must be a sequence of strings.")
    out: list[str] = []
    for m in markers:
        if not isinstance(m, str) or not m.strip():
            raise TypeError("Each marker must be a non-empty string.")
        out.append(m)
    if not out:
        raise ValueError("markers must be non-empty.")
    return tuple(out)


def _markers_from_env(env_var: str = _ENV_PROJECT_MARKERS) -> tuple[str, ...] | None:
    """
    Parse markers from an environment variable.

    The env var must be a JSON list of strings, e.g.::

        export SCIKITPLOT_PROJECT_MARKERS='["pyproject.toml",".git","configs/mlflow.toml"]'

    Parameters
    ----------
    env_var : str, default="SCIKITPLOT_PROJECT_MARKERS"
        Environment variable name.

    Returns
    -------
    tuple[str, ...] or None
        Parsed markers or None if not set.

    Raises
    ------
    ValueError
        If the value is present but not valid JSON.
    TypeError
        If the JSON does not parse to a list of strings.
    """
    raw = os.environ.get(env_var)
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{env_var} must be a JSON list of strings, e.g. "
            f'["pyproject.toml",".git","configs/mlflow.toml"]. Got: {raw!r}'
        ) from e
    return _validate_markers(parsed)


def _markers_from_toml(config_path: Path) -> tuple[str, ...] | None:
    """
    Parse markers from a TOML file's ``[project].markers`` table.

    Parameters
    ----------
    config_path : pathlib.Path
        TOML file path.

    Returns
    -------
    tuple[str, ...] or None
        Parsed markers or None if not present.
    """
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    project = data.get("project")
    if not isinstance(project, dict):
        return None
    markers = project.get("markers")
    if markers is None:
        return None
    return _validate_markers(markers)


def get_project_markers(*, config_path: Path | None = None) -> tuple[str, ...]:
    """
    Resolve project markers deterministically.

    Resolution order
    ----------------
    1) If `config_path` provided and contains ``[project].markers``, use that.
    2) Else if env var ``SCIKITPLOT_PROJECT_MARKERS`` is set, use that (strict JSON list).
    3) Else use current module default set via :func:`set_project_markers` (or built-in default).

    Parameters
    ----------
    config_path : pathlib.Path or None, default=None
        Optional TOML file to read ``[project].markers`` from.

    Returns
    -------
    tuple[str, ...]
        Effective marker tuple.
    """
    if config_path is not None:
        cfg = _markers_from_toml(Path(config_path))
        if cfg is not None:
            return cfg
    env = _markers_from_env()
    if env is not None:
        return env
    return _PROJECT_MARKERS


def set_project_markers(markers: Sequence[str] | None) -> None:
    """
    Set the module default markers.

    Parameters
    ----------
    markers : sequence[str] or None
        New default markers. If None, resets to :data:`DEFAULT_PROJECT_MARKERS`.

    Notes
    -----
    This affects only calls that do not pass `markers=` and do not override via config/env.
    Prefer the context manager :func:`project_markers` in automation pipelines and tests.
    """
    global _PROJECT_MARKERS  # noqa: PLW0603
    _PROJECT_MARKERS = (
        DEFAULT_PROJECT_MARKERS if markers is None else _validate_markers(markers)
    )


@contextlib.contextmanager
def project_markers(markers: Sequence[str] | None) -> Iterator[None]:
    """
    Temporarily override module default markers for a block.

    Parameters
    ----------
    markers : sequence[str] or None
        Temporary markers. If None, resets to :data:`DEFAULT_PROJECT_MARKERS` for the block.

    Yields
    ------
    None

    Notes
    -----
    This is deterministic and exception-safe. It is the preferred way to alter marker behavior
    for a single workflow step (train/hpo/predict) without mutating global state permanently.
    """
    global _PROJECT_MARKERS  # noqa: PLW0603
    old = _PROJECT_MARKERS
    try:
        set_project_markers(markers)
        yield
    finally:
        _PROJECT_MARKERS = old


def find_project_root(
    start: Path | None = None,
    *,
    markers: Sequence[str] | None = None,
    config_path: Path | None = None,
) -> Path:
    """
    Find a project root directory deterministically.

    Parameters
    ----------
    start : pathlib.Path or None, default=None
        Starting directory. If None, uses the current working directory.
    markers : sequence[str] or None, default=None
        Marker files/directories that define a project root. If None, resolved via
        :func:`get_project_markers` (using `config_path` if provided).
    config_path : pathlib.Path or None, default=None
        Optional TOML file to read ``[project].markers`` from.

    Returns
    -------
    pathlib.Path
        Project root path.

    Raises
    ------
    FileNotFoundError
        If no project root can be found by walking to filesystem root.

    Notes
    -----
    Strict rule (no heuristics):
    - Walk upward from `start` until a directory containing any marker is found.
    - If none are found, raise.
    """
    resolved_markers = (
        _validate_markers(markers)
        if markers is not None
        else get_project_markers(config_path=config_path)
    )

    cur = (start or Path.cwd()).resolve()
    for p in (cur, *cur.parents):
        for m in resolved_markers:
            if (p / m).exists():
                return p
    raise FileNotFoundError(
        f"Project root not found from {cur} using markers {resolved_markers!r}."
    )


# ---------------------------------------------------------------------------
# Local/remote store normalization helpers
# ---------------------------------------------------------------------------

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+\-.]*:")


def _is_windows_drive_path(value: str) -> bool:
    r"""
    Return True if `value` is a Windows drive path such as ``C:\...`` or ``C:/...``.
    """
    if len(value) < 2:  # noqa: PLR2004
        return False
    if not value[0].isalpha() or value[1] != ":":  # noqa: SIM103
        return False
    # Accept "C:" and "C:\", "C:/"
    return True


def _is_probably_local_path(value: str) -> bool:
    r"""
    Return True if `value` is a local filesystem path (not an MLflow URI).

    Deterministic rule
    ------------------
    - If it contains '://' => URI (remote or file-style), not a local path
    - Else if it matches '<scheme>:' (e.g., 's3:', 'gs:', 'dbfs:', 'file:') => URI
      except Windows drive paths like 'C:\...'
    - Else => local path
    """
    if "://" in value:
        return False
    if _SCHEME_RE.match(value) and not _is_windows_drive_path(value):  # noqa: SIM103
        return False
    return True


def _normalize_sqlite_uri(uri: str, *, base_dir: Path) -> str:
    """
    Normalize sqlite URIs containing relative filesystem paths.

    Parameters
    ----------
    uri : str
        sqlite URI (e.g., "sqlite:///./.mlflow/mlflow.db").
    base_dir : pathlib.Path
        Base directory used to resolve relative paths.

    Returns
    -------
    str
        Normalized sqlite URI with an absolute filesystem path.

    Raises
    ------
    ValueError
        If the URI does not start with "sqlite:///".
    """
    if not uri.startswith("sqlite:///"):
        raise ValueError(f"Not a sqlite:/// URI: {uri!r}")
    fs_path = uri[len("sqlite:///") :]
    p = Path(fs_path).expanduser()
    if not p.is_absolute():  # noqa: SIM108
        p = (base_dir / p).resolve()
    else:
        p = p.resolve()
    return "sqlite:///" + p.as_posix()


def _normalize_path_like(value: str, *, base_dir: Path) -> str:
    """
    Normalize a path-like string to an absolute POSIX path.

    Notes
    -----
    - Relative paths are resolved against `base_dir`.
    - Returns POSIX style for cross-platform MLflow path/URI friendliness.
    """
    p = Path(value).expanduser()
    if not p.is_absolute():  # noqa: SIM108
        p = (base_dir / p).resolve()
    else:
        p = p.resolve()
    return p.as_posix()


def normalize_mlflow_store_values(
    *,
    backend_store_uri: str | None,
    default_artifact_root: str | None,
    base_dir: Path,
) -> tuple[str | None, str | None]:
    """
    Normalize local store values for consistent multi-script usage.

    Deterministic normalization rules
    --------------------------------
    - If backend_store_uri starts with "sqlite:///" => normalize the filesystem path to absolute.
    - Else if backend_store_uri is a local path => normalize to absolute.
    - For default_artifact_root: if it is a local path => normalize to absolute.
      Otherwise, leave as-is (remote schemes like s3://, gs://, dbfs:/, http(s)://).

    This ensures `train.py`, `hpo.py`, `predict.py` behave consistently regardless of CWD.

    Parameters
    ----------
    backend_store_uri : str or None
        Backend store URI or path.
    default_artifact_root : str or None
        Default artifact root URI or path.
    base_dir : pathlib.Path
        Base directory used to resolve relative paths.

    Returns
    -------
    tuple[str or None, str or None]
        Normalized (backend_store_uri, default_artifact_root).
    """
    b = backend_store_uri
    a = default_artifact_root

    if b:
        if b.startswith("sqlite:///"):
            b = _normalize_sqlite_uri(b, base_dir=base_dir)
        elif _is_probably_local_path(b):
            b = _normalize_path_like(b, base_dir=base_dir)

    if a and _is_probably_local_path(a):
        a = _normalize_path_like(a, base_dir=base_dir)

    return b, a


def ensure_local_store_layout(
    *,
    backend_store_uri: str | None,
    default_artifact_root: str | None,
) -> None:
    """
    Ensure local backend/artifact directories exist.

    Notes
    -----
    This function only creates directories for local filesystem locations.
    Remote stores (s3://, gs://, dbfs:/, http(s)://) are not touched.

    Parameters
    ----------
    backend_store_uri : str or None
        Backend store URI or path.
    default_artifact_root : str or None
        Default artifact root URI or path.

    Returns
    -------
    None
    """
    if backend_store_uri:
        if backend_store_uri.startswith("sqlite:///"):
            fs_path = backend_store_uri[len("sqlite:///") :]
            Path(fs_path).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        elif _is_probably_local_path(backend_store_uri):
            Path(backend_store_uri).expanduser().resolve().mkdir(
                parents=True, exist_ok=True
            )

    if default_artifact_root and _is_probably_local_path(default_artifact_root):
        Path(default_artifact_root).expanduser().resolve().mkdir(
            parents=True, exist_ok=True
        )


# ---------------------------------------------------------------------------
# ProjectConfig IO (TOML/YAML)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectConfig:
    """
    Project-level configuration for MLflow usage across multiple scripts.

    Attributes
    ----------
    profile : str
        Named profile (e.g., "local", "remote", "ci").
    session : SessionConfig
        Session configuration.
    server : ServerConfig or None
        Server configuration (if this profile starts a server).
    start_server : bool
        Whether to start a managed server for this profile.

    Notes
    -----
    This provides a single, shared configuration for:
    - train.py
    - hpo.py
    - predict.py

    It prevents drift between scripts and makes runs reproducible.
    """

    profile: str = "local"
    # Class-level attribute exists for Sphinx/linkcode; value is immutable (SessionConfig is frozen).
    session: SessionConfig = _DEFAULT_SESSION_CONFIG
    server: ServerConfig | None = None
    start_server: bool = False


def _construct_dataclass(cls: type[Any], /, **kwargs: Any) -> Any:
    """
    Construct a (data)class instance, filtering kwargs to supported fields/parameters.

    This is a deterministic compatibility layer to tolerate *internal* config evolution
    (e.g., adding new optional fields) without breaking older config readers.

    Parameters
    ----------
    cls : type
        Class to construct (typically a dataclass).
    **kwargs : Any
        Candidate keyword arguments.

    Returns
    -------
    Any
        Constructed instance.

    Raises
    ------
    TypeError
        If the class cannot be constructed with the filtered arguments.
    """
    if dataclasses.is_dataclass(cls):
        allowed = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return cls(**filtered)
    # Fallback for non-dataclass classes.
    sig = inspect.signature(cls)
    allowed = set(sig.parameters)
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return cls(**filtered)


def _norm_str(v: Any) -> str | None:
    """
    Normalize optional string values.

    Parameters
    ----------
    v : Any
        Candidate value.

    Returns
    -------
    str or None
        Returns None for None or empty string; otherwise returns the string.

    Raises
    ------
    TypeError
        If v is not None and not a string.
    """
    if v is None:
        return None
    if v == "":
        return None
    if not isinstance(v, str):
        raise TypeError(f"Expected a string or None, got {type(v).__name__}.")
    return v


def _coerce_mapping(v: Any, *, name: str) -> dict[str, Any]:
    """
    Coerce a value to a mapping (or raise).

    Parameters
    ----------
    v : Any
        Candidate value.
    name : str
        Name for error messages.

    Returns
    -------
    dict[str, Any]
        The mapping.

    Raises
    ------
    TypeError
        If v is not a mapping.
    """
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise TypeError(f"Expected a mapping for {name}, got {type(v).__name__}.")
    return v


def _build_project_config_from_mapping(
    mapping: dict[str, Any],
    *,
    profile: str,
    project_root: Path,
) -> ProjectConfig:
    """
    Build a ProjectConfig from a parsed mapping (TOML or YAML).

    Parameters
    ----------
    mapping : dict[str, Any]
        Parsed config mapping.
    profile : str
        Profile name.
    project_root : pathlib.Path
        Project root for path normalization.

    Returns
    -------
    ProjectConfig
        Built config.

    Raises
    ------
    KeyError
        If expected sections are missing.
    TypeError
        If types are invalid.
    ValueError
        If validation fails.
    """
    profiles = mapping.get("profiles")
    if not isinstance(profiles, dict):
        raise KeyError("Missing 'profiles' table in config.")

    p = profiles.get(profile)
    if not isinstance(p, dict):
        raise KeyError(f"Missing 'profiles.{profile}' table in config.")

    start_server = bool(p.get("start_server", False))
    session_tbl = _coerce_mapping(p.get("session"), name=f"profiles.{profile}.session")
    server_tbl = _coerce_mapping(p.get("server"), name=f"profiles.{profile}.server")

    # Validate and normalize session mappings.
    extra_env = session_tbl.get("extra_env")
    if extra_env is not None and not isinstance(extra_env, dict):
        raise TypeError(
            f"profiles.{profile}.session.extra_env must be a mapping, got {type(extra_env).__name__}."
        )
    default_run_tags = session_tbl.get("default_run_tags")
    if default_run_tags is not None and not isinstance(default_run_tags, dict):
        raise TypeError(
            f"profiles.{profile}.session.default_run_tags must be a mapping, got {type(default_run_tags).__name__}."
        )

    sess = SessionConfig(
        tracking_uri=_norm_str(session_tbl.get("tracking_uri")),
        registry_uri=_norm_str(session_tbl.get("registry_uri")),
        env_file=_norm_str(session_tbl.get("env_file")),
        extra_env=extra_env,
        startup_timeout_s=float(session_tbl.get("startup_timeout_s", 30.0)),
        ensure_reachable=bool(session_tbl.get("ensure_reachable", False)),
        experiment_name=_norm_str(session_tbl.get("experiment_name")),
        create_experiment_if_missing=bool(
            session_tbl.get("create_experiment_if_missing", True)
        ),
        default_run_name=_norm_str(session_tbl.get("default_run_name")),
        default_run_tags=default_run_tags,
        public_tracking_uri=_norm_str(session_tbl.get("public_tracking_uri")),
    )

    srv: ServerConfig | None = None
    if start_server:
        srv = ServerConfig(
            host=str(server_tbl.get("host", "127.0.0.1")),
            port=int(server_tbl.get("port", 5000)),
            auto_host_in_docker=bool(server_tbl.get("auto_host_in_docker", False)),
            docker_host=str(server_tbl.get("docker_host", "0.0.0.0")),  # noqa: S104
            workers=server_tbl.get("workers"),
            backend_store_uri=_norm_str(server_tbl.get("backend_store_uri")),
            registry_store_uri=_norm_str(server_tbl.get("registry_store_uri")),
            default_artifact_root=_norm_str(server_tbl.get("default_artifact_root")),
            serve_artifacts=bool(server_tbl.get("serve_artifacts", False)),
            no_serve_artifacts=bool(server_tbl.get("no_serve_artifacts", False)),
            artifacts_destination=_norm_str(server_tbl.get("artifacts_destination")),
            artifacts_only=bool(server_tbl.get("artifacts_only", False)),
            allowed_hosts=_norm_str(server_tbl.get("allowed_hosts")),
            cors_allowed_origins=_norm_str(server_tbl.get("cors_allowed_origins")),
            x_frame_options=_norm_str(server_tbl.get("x_frame_options")),
            disable_security_middleware=bool(
                server_tbl.get("disable_security_middleware", False)
            ),
            static_prefix=_norm_str(server_tbl.get("static_prefix")),
            uvicorn_opts=_norm_str(server_tbl.get("uvicorn_opts")),
            gunicorn_opts=_norm_str(server_tbl.get("gunicorn_opts")),
            waitress_opts=_norm_str(server_tbl.get("waitress_opts")),
            expose_prometheus=_norm_str(server_tbl.get("expose_prometheus")),
            app_name=_norm_str(server_tbl.get("app_name")),
            dev=bool(server_tbl.get("dev", False)),
            secrets_cache_ttl=server_tbl.get("secrets_cache_ttl"),
            secrets_cache_max_size=server_tbl.get("secrets_cache_max_size"),
            strict_cli_compat=bool(server_tbl.get("strict_cli_compat", True)),
            extra_args=server_tbl.get("extra_args"),
        )

        # Normalize local store values relative to project root
        b, a = normalize_mlflow_store_values(
            backend_store_uri=srv.backend_store_uri,
            default_artifact_root=srv.default_artifact_root,
            base_dir=project_root,
        )
        object.__setattr__(srv, "backend_store_uri", b)
        object.__setattr__(srv, "default_artifact_root", a)

        ensure_local_store_layout(backend_store_uri=b, default_artifact_root=a)
        srv.validate(for_managed_tracking=True)

    return ProjectConfig(
        profile=profile, session=sess, server=srv, start_server=start_server
    )


def load_project_config_toml(
    path: Path,
    *,
    profile: str = "local",
    project_root: Path | None = None,
) -> ProjectConfig:
    """
    Load project MLflow config from a TOML file.

    Parameters
    ----------
    path : pathlib.Path
        TOML config file path.
    profile : str, default="local"
        Profile name.
    project_root : pathlib.Path or None, default=None
        Project root used to resolve relative paths. If None, discovered via :func:`find_project_root`.

    Returns
    -------
    ProjectConfig
        Loaded project configuration.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    Notes
    -----
    TOML reading uses stdlib ``tomllib`` (Python 3.11+).
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    root = project_root or find_project_root(start=path.parent)
    mapping = tomllib.loads(path.read_text(encoding="utf-8"))
    return _build_project_config_from_mapping(
        mapping, profile=profile, project_root=root
    )


def load_project_config_yaml(
    path: Path,
    *,
    profile: str = "local",
    project_root: Path | None = None,
) -> ProjectConfig:
    """
    Load project MLflow config from a YAML file.

    Parameters
    ----------
    path : pathlib.Path
        Path to YAML config file (.yaml or .yml).
    profile : str, default="local"
        Profile name.
    project_root : pathlib.Path or None, default=None
        Project root used to resolve relative paths. If None, discovered via :func:`find_project_root`.

    Returns
    -------
    ProjectConfig
        Loaded project configuration.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ImportError
        If PyYAML is not installed.
    ValueError
        If the YAML does not parse to a mapping.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    try:
        import yaml  # type: ignore[]  # noqa: PLC0415
    except Exception as e:
        raise ImportError(
            "YAML support requires PyYAML. Install via: pip install pyyaml"
        ) from e

    root = project_root or find_project_root(start=path.parent)

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("YAML config must parse to a mapping at the document root.")
    return _build_project_config_from_mapping(
        loaded, profile=profile, project_root=root
    )


def load_project_config(
    path: Path,
    *,
    profile: str = "local",
    project_root: Path | None = None,
) -> ProjectConfig:
    """
    Load project MLflow config from TOML or YAML based on file extension.

    Parameters
    ----------
    path : pathlib.Path
        Path to a config file. Supported extensions: .toml, .yaml, .yml
    profile : str, default="local"
        Profile name.
    project_root : pathlib.Path or None, default=None
        Optional project root override for path normalization.

    Returns
    -------
    ProjectConfig
        Loaded config.

    Raises
    ------
    ValueError
        If file extension is unsupported.
    """
    suf = path.suffix.lower()
    if suf == ".toml":
        return load_project_config_toml(
            path, profile=profile, project_root=project_root
        )
    if suf in {".yaml", ".yml"}:
        return load_project_config_yaml(
            path, profile=profile, project_root=project_root
        )
    raise ValueError(
        f"Unsupported config extension {suf!r}; expected .toml, .yaml, or .yml."
    )


def dump_project_config_yaml(
    cfg: ProjectConfig | None = None,
    path: Path | None = None,
    *,
    profile: str = "local",
    source_config_path: Path | None = None,
) -> Path:
    """
    Write a ProjectConfig to a YAML file.

    This function supports two strict modes:

    1) Explicit mode (library-level):
       ``dump_project_config_yaml(cfg, path)``

    2) Project convenience mode (newbie-friendly):
       ``dump_project_config_yaml(profile="local")``
       - Loads config from ``<project_root>/configs/mlflow.toml`` by default
       - Writes YAML to ``<project_root>/configs/mlflow.yaml`` by default

    Parameters
    ----------
    cfg : ProjectConfig or None, default=None
        Configuration to write. If None, `source_config_path` (or default project TOML)
        is loaded and used.
    path : pathlib.Path or None, default=None
        Output YAML file path. If None in convenience mode, uses the default project YAML path.
    profile : str, default="local"
        Profile name used when loading the source configuration in convenience mode.
    source_config_path : pathlib.Path or None, default=None
        Source config path to load in convenience mode. If None, defaults to
        ``<project_root>/configs/mlflow.toml``.

    Returns
    -------
    pathlib.Path
        The YAML path written.

    Raises
    ------
    ImportError
        If PyYAML is not installed.
    FileNotFoundError
        If convenience mode cannot find a source config.
    ValueError
        If arguments are inconsistent (e.g., cfg provided but path missing).
    """
    if cfg is not None and path is None:
        raise ValueError("Explicit mode requires both cfg and path.")
    if cfg is not None and path is not None:
        _dump_project_config_yaml_explicit(cfg, path)
        return path

    root = find_project_root()
    src = source_config_path or (root / "configs" / "mlflow.toml")
    if not src.exists():
        raise FileNotFoundError(
            f"Source config not found: {src}. "
            "Create it first (e.g., copy from package demo or call export_builtin_config)."
        )
    out_path = path or (root / "configs" / "mlflow.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg2 = load_project_config(src, profile=profile)
    _dump_project_config_yaml_explicit(cfg2, out_path)
    return out_path


def _dump_project_config_yaml_explicit(cfg: ProjectConfig, path: Path) -> None:
    """
    Help implementation for writing ProjectConfig to YAML (explicit mode).

    Parameters
    ----------
    cfg : ProjectConfig
        Configuration to serialize.
    path : pathlib.Path
        Output YAML path.

    Returns
    -------
    None
    """
    try:
        import yaml  # type: ignore[]  # noqa: PLC0415
    except Exception as e:
        raise ImportError(
            "YAML support requires PyYAML. Install via: pip install pyyaml"
        ) from e

    out: dict[str, Any] = {
        "profiles": {
            cfg.profile: {
                "start_server": cfg.start_server,
                "session": {
                    "tracking_uri": cfg.session.tracking_uri,
                    "public_tracking_uri": getattr(
                        cfg.session, "public_tracking_uri", None
                    ),
                    "registry_uri": cfg.session.registry_uri,
                    "env_file": cfg.session.env_file,
                    "extra_env": (
                        dict(cfg.session.extra_env) if cfg.session.extra_env else None
                    ),
                    "startup_timeout_s": cfg.session.startup_timeout_s,
                    "ensure_reachable": cfg.session.ensure_reachable,
                    "experiment_name": cfg.session.experiment_name,
                    "create_experiment_if_missing": (
                        cfg.session.create_experiment_if_missing
                    ),
                    "default_run_name": cfg.session.default_run_name,
                    "default_run_tags": (
                        dict(cfg.session.default_run_tags)
                        if cfg.session.default_run_tags
                        else None
                    ),
                },
                "server": None,
            }
        }
    }

    if cfg.server is not None:
        s = cfg.server
        out["profiles"][cfg.profile]["server"] = {
            "host": s.host,
            "port": s.port,
            "auto_host_in_docker": s.auto_host_in_docker,
            "docker_host": s.docker_host,
            "workers": s.workers,
            "backend_store_uri": s.backend_store_uri,
            "registry_store_uri": s.registry_store_uri,
            "default_artifact_root": s.default_artifact_root,
            "serve_artifacts": s.serve_artifacts,
            "no_serve_artifacts": s.no_serve_artifacts,
            "artifacts_destination": s.artifacts_destination,
            "artifacts_only": s.artifacts_only,
            "allowed_hosts": s.allowed_hosts,
            "cors_allowed_origins": s.cors_allowed_origins,
            "x_frame_options": s.x_frame_options,
            "disable_security_middleware": s.disable_security_middleware,
            "static_prefix": s.static_prefix,
            "uvicorn_opts": s.uvicorn_opts,
            "gunicorn_opts": s.gunicorn_opts,
            "waitress_opts": s.waitress_opts,
            "expose_prometheus": s.expose_prometheus,
            "app_name": s.app_name,
            "dev": s.dev,
            "secrets_cache_ttl": s.secrets_cache_ttl,
            "secrets_cache_max_size": s.secrets_cache_max_size,
            "strict_cli_compat": s.strict_cli_compat,
            "extra_args": list(s.extra_args) if s.extra_args else None,
        }

    path.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
