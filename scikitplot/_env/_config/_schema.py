# scikitplot/_env/_config/_schema.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
From configuration documents, presets and arguments to one request.

A configuration file (``scikitplot-env.toml``) holds the request::

    schema = "scikitplot.env/v1"
    name = "analysis"              # or: prefix = ".venv"
    python = "3.12"
    backend = "auto"               # auto, uv, venv, micromamba, conda, mamba, pipenv
    packages = ["numpy>=2", "pandas"]
    conda_packages = []
    channels = ["conda-forge"]
    sources = ["requirements.txt", { path = "pyproject.toml", groups = ["test"] }]
    target = "host"                # host, emscripten-wasm32, pyodide, android, ios
    profile = "ci"                 # a preset, see below

    [policy]
    interaction = "auto"           # never, auto, always
    network = "online"             # online, offline
    bootstrap = "never"            # never, if-missing
    conversion = "exact"           # exact, allow-approximate

    [verify]
    imports = ["numpy", "pandas"]

Precedence, lowest first: built-in defaults, the preset (``profile``), the
document, then the arguments of the call or command line. Every field of the
resulting request records which layer set it.

Notes
-----
**User notes.** Relative paths (``prefix``, ``sources``, ``find_links``,
``root_prefix``) are relative to the document's directory, so a project
config works from any working directory. Presets: ``ci`` (no questions),
``agent`` (no questions, exact conversions), ``airgap`` (offline, no
bootstrap), ``interactive`` (always ask).

**Developer notes.** Keys that would grant authority (``approve``, ``yes``,
``assume_yes`` ...) are refused with their own message: approval belongs to
the invocation, never to a file someone else can commit. Unknown keys are
refused (Rule 106).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .._core._errors import ConfigError
from .._core._request import (
    REQUEST_SCHEMA,
    EnvironmentRequest,
    Policy,
    SourceRef,
    infer_source_kind,
)
from . import _formats

__all__ = ["CONFIG_NAMES", "PRESETS", "find_config", "load_batch", "load_request"]

#: File names looked for, in order, by :py:func:`find_config`.
CONFIG_NAMES = (
    "scikitplot-env.toml",
    "scikitplot-env.json",
    "scikitplot-env.yaml",
    "scikitplot-env.yml",
)

#: Named overlays applied below the document.
PRESETS: Mapping[str, Mapping[str, Any]] = {
    "ci": {"policy": {"interaction": "never"}},
    "agent": {"policy": {"interaction": "never", "conversion": "exact"}},
    "airgap": {"policy": {"network": "offline", "bootstrap": "never"}},
    "interactive": {"policy": {"interaction": "always"}},
}

#: Schema identifier of a batch line.
BATCH_SCHEMA = "scikitplot.env.request/v1"

_TOP = {
    "schema",
    "name",
    "prefix",
    "python",
    "backend",
    "packages",
    "conda_packages",
    "sources",
    "channels",
    "index_url",
    "extra_index_urls",
    "find_links",
    "target",
    "policy",
    "verify",
    "root_prefix",
    "profile",
}
_POLICY = {"interaction", "network", "bootstrap", "conversion"}
_VERIFY = {"imports"}
_AUTHORITY = {"approve", "approvals", "yes", "assume_yes", "trust", "allow"}
_PATHS = ("prefix", "root_prefix")


def _check_keys(where: str, mapping: Mapping[str, Any], allowed: set[str]) -> None:
    if not isinstance(mapping, Mapping):
        raise ConfigError(f"{where} must be a table/object")
    for key in mapping:
        if not isinstance(key, str):
            raise ConfigError(f"{where} keys must be strings, got {key!r}")
    authority = sorted(set(mapping) & _AUTHORITY)
    if authority:
        raise ConfigError(
            f"{where} sets {authority}: a configuration expresses intent and cannot"
            " grant approvals",
            code="config_authority",
            hint="approve on the command line (--yes / --approve KIND) or with"
            " approve= in Python",
        )
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(
            f"{where} has unknown key(s) {unknown}; allowed: {sorted(allowed)}"
        )


def _resolve(base: Path, value: str) -> str:
    path = Path(os.path.expanduser(value))
    return str(path if path.is_absolute() else (base / path).resolve())


def _normalise(  # noqa: PLR0912 (one branch per documented key)
    doc: Mapping[str, Any], base: Path, where: str
) -> dict[str, Any]:
    """
    Turn one document layer into request keyword arguments.

    Parameters
    ----------
    doc : mapping
        The layer (a document, a preset or the invocation).
    base : pathlib.Path
        Directory relative paths are resolved against.
    where : str
        Name of the layer, used in error messages.

    Returns
    -------
    dict
        Request fields of this layer (``policy`` as a mapping).

    Raises
    ------
    ConfigError
        For an unknown key, an authority key or a value of the wrong type.
    """
    _check_keys(where, doc, _TOP)
    if "schema" in doc and doc["schema"] != REQUEST_SCHEMA:
        raise ConfigError(
            f"{where}: schema {doc['schema']!r} is not {REQUEST_SCHEMA!r}",
            code="config_schema",
        )
    out: dict[str, Any] = {}
    for key in ("name", "python", "backend", "index_url", "profile"):
        if key in doc:
            value = doc[key]
            if not isinstance(value, str):
                raise ConfigError(f"{where}.{key} must be a string, got {value!r}")
            out[key] = value
    for key in _PATHS:
        if key in doc:
            if not isinstance(doc[key], str) or not doc[key]:
                raise ConfigError(f"{where}.{key} must be a non-empty string")
            out[key] = _resolve(base, doc[key])
    for key in ("packages", "conda_packages", "channels", "extra_index_urls"):
        if key in doc:
            out[key] = _string_list(f"{where}.{key}", doc[key])
    if "find_links" in doc:
        out["find_links"] = tuple(
            _resolve(base, p) if "://" not in p else p
            for p in _string_list(f"{where}.find_links", doc["find_links"])
        )
    if "target" in doc:
        target = doc["target"]
        if isinstance(target, Mapping):
            _check_keys(f"{where}.target", target, {"kind"})
            target = target.get("kind")
        if not isinstance(target, str):
            raise ConfigError(f"{where}.target must be a string or {{kind = ...}}")
        out["target"] = target
    if "sources" in doc:
        out["sources"] = tuple(
            _source(f"{where}.sources[{i}]", s, base)
            for i, s in enumerate(_list(f"{where}.sources", doc["sources"]))
        )
    if "policy" in doc:
        _check_keys(f"{where}.policy", doc["policy"], _POLICY)
        out["policy"] = dict(doc["policy"])
    if "verify" in doc:
        _check_keys(f"{where}.verify", doc["verify"], _VERIFY)
        if "imports" in doc["verify"]:
            out["verify_imports"] = _string_list(
                f"{where}.verify.imports", doc["verify"]["imports"]
            )
    return out


def _list(where: str, value: Any) -> list[Any]:
    if isinstance(value, (str, Mapping)):
        return [value]
    if not isinstance(value, list):
        raise ConfigError(f"{where} must be a list")
    return value


def _string_list(where: str, value: Any) -> tuple[str, ...]:
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not all(isinstance(v, str) for v in items):
        raise ConfigError(f"{where} must be a list of strings")
    return tuple(items)


def _source(where: str, value: Any, base: Path) -> SourceRef:
    if isinstance(value, str):
        value = {"path": value}
    _check_keys(where, value, {"path", "type", "groups"})
    path = value.get("path")
    if not isinstance(path, str) or not path:
        raise ConfigError(f"{where} needs a path")
    absolute = _resolve(base, path)
    kind = value.get("type") or infer_source_kind(absolute)
    groups = value.get("groups", ())
    if isinstance(groups, str):
        groups = (groups,)
    if not isinstance(groups, (list, tuple)):
        raise ConfigError(f"{where}.groups must be a list of strings")
    return SourceRef(path=absolute, kind=kind, groups=tuple(groups))


def _merge(
    layers: list[tuple[str, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, str]]:
    merged: dict[str, Any] = {}
    origins: dict[str, str] = {}
    policy: dict[str, Any] = {}
    for origin, layer in layers:
        for key, value in layer.items():
            if key == "policy":
                for pkey, pvalue in value.items():
                    policy[pkey] = pvalue
                    origins[f"policy.{pkey}"] = origin
                continue
            if key in ("name", "prefix"):  # one location: a later layer replaces both
                merged.pop("name", None)
                merged.pop("prefix", None)
                origins.pop("name", None)
                origins.pop("prefix", None)
            merged[key] = value
            origins[key] = origin
    merged["policy"] = Policy(**policy)
    return merged, origins


def find_config(start: Any = None) -> Path | None:
    """
    Find the configuration of a project, walking up from ``start``.

    Parameters
    ----------
    start : str, os.PathLike or None, default=None
        Directory to start from; the working directory when None.

    Returns
    -------
    pathlib.Path or None
        The first ``scikitplot-env.{toml,json,yaml,yml}`` found, else a
        ``pyproject.toml`` with a ``[tool.scikitplot.env]`` table, else None.

    Raises
    ------
    ConfigError
        For a ``pyproject.toml`` that is not valid TOML.

    Notes
    -----
    The walk stops at the project root: a directory with a ``pyproject.toml``
    (with or without the table) or a ``.git``/``.hg`` entry. A configuration
    above it (a parent directory, ``/tmp``) belongs to someone else and is
    never adopted.
    """
    here = Path(os.fspath(start) if start is not None else os.getcwd()).resolve()
    for directory in (here, *here.parents):
        for name in CONFIG_NAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
        pyproject = directory / "pyproject.toml"
        if pyproject.is_file():
            tree = _formats.loads(_formats.read_text(pyproject), "toml")
            table = ((tree.get("tool") or {}).get("scikitplot") or {}).get("env")
            return pyproject if table is not None else None
        if (directory / ".git").exists() or (directory / ".hg").exists():
            return None
    return None


def load_request(
    config: Any = None,
    *,
    base_dir: Any = None,
    **overrides: Any,
) -> EnvironmentRequest:
    """
    Build an :py:class:`~.EnvironmentRequest` from a document and/or arguments.

    Parameters
    ----------
    config : str, os.PathLike, mapping or None, default=None
        A configuration file, or an already parsed mapping.
    base_dir : str, os.PathLike or None, default=None
        Directory relative paths are resolved against: the file's directory
        for a file, else the working directory.
    **overrides
        Request fields given by the caller; they win over the document.
        ``policy`` may be a mapping; ``profile`` picks a preset.

    Returns
    -------
    EnvironmentRequest
        The request, with ``origins`` and ``config_digest``.

    Raises
    ------
    ConfigError
        For an invalid document, an unknown preset or an authority key.

    Examples
    --------
    >>> from scikitplot._env import load_request
    >>> request = load_request(
    ...     {"name": "demo", "packages": ["requests"], "profile": "ci"},
    ...     base_dir="/tmp",
    ... )
    >>> request.policy.interaction, dict(request.origins)["policy.interaction"]
    ('never', 'preset:ci')
    """
    digest = None
    if config is None:
        doc: Mapping[str, Any] = {}
        base = Path(os.fspath(base_dir) if base_dir is not None else os.getcwd())
        source = "config:<none>"
    elif isinstance(config, Mapping):
        doc = config
        base = Path(os.fspath(base_dir) if base_dir is not None else os.getcwd())
        source = "config:<mapping>"
    else:
        path = Path(os.fspath(config)).expanduser().resolve()
        doc, digest = _formats.read_document(path)
        base = Path(os.fspath(base_dir)) if base_dir is not None else path.parent
        source = f"config:{path}"
    document = _normalise(doc, base, "config")
    overrides = dict(overrides)
    if overrides.get("verify_imports") is not None:  # Python-side spelling
        overrides["verify"] = {"imports": list(overrides.pop("verify_imports"))}
    overrides.pop("verify_imports", None)
    invocation = (
        _normalise(
            {k: v for k, v in overrides.items() if v is not None and k != "policy"}
            | ({"policy": overrides["policy"]} if overrides.get("policy") else {}),
            Path(os.getcwd()),
            "argument",
        )
        if overrides
        else {}
    )
    profile = invocation.get("profile", document.get("profile"))
    layers: list[tuple[str, dict[str, Any]]] = []
    if profile is not None:
        if profile not in PRESETS:
            raise ConfigError(f"profile {profile!r} is not one of {sorted(PRESETS)}")
        layers.append(
            (f"preset:{profile}", _normalise(PRESETS[profile], base, "preset"))
        )
    layers.append((source, document))
    layers.append(("invocation", invocation))
    merged, origins = _merge(layers)
    for field in ("backend", "target"):
        origins.setdefault(field, "default")
    return EnvironmentRequest(
        **merged,
        origins=tuple(sorted(origins.items())),
        config_digest=digest,
    )


def load_batch(path: Any) -> list[tuple[str, EnvironmentRequest]]:
    """
    Read a JSON Lines batch of requests.

    Each line is ``{"schema": "scikitplot.env.request/v1", "id": "...",
    "request": {...}}``; ``request`` is a configuration mapping, relative
    paths resolved against the batch file's directory.

    Parameters
    ----------
    path : str or os.PathLike
        The ``.jsonl`` file.

    Returns
    -------
    list of (str, EnvironmentRequest)
        Requests by id, in file order.

    Raises
    ------
    ConfigError
        For a malformed line or duplicate ids, naming the line.
    """
    path = Path(os.fspath(path)).resolve()
    out: list[tuple[str, EnvironmentRequest]] = []
    seen: set[str] = set()
    for lineno, record in _formats.iter_jsonl(path):
        where = f"{path}:{lineno}"
        _check_keys(where, record, {"schema", "id", "request"})
        if record.get("schema") != BATCH_SCHEMA:
            raise ConfigError(f"{where}: schema must be {BATCH_SCHEMA!r}")
        rid = record.get("id")
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ConfigError(f"{where}: id must be a unique non-empty string")
        seen.add(rid)
        if not isinstance(record.get("request"), Mapping):
            raise ConfigError(f"{where}: request must be an object")
        try:
            out.append((rid, load_request(record["request"], base_dir=path.parent)))
        except ConfigError as exc:
            raise ConfigError(f"{where}: {exc}", code=exc.code, hint=exc.hint) from None
    return out
