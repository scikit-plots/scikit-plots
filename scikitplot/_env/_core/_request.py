# scikitplot/_env/_core/_request.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The normalised environment request: *what* to build, never *how*.

Notes
-----
**User notes.** Build a request with
:py:func:`~._config.load_request` (from a file and/or keyword arguments)
rather than by hand; it applies presets and records where every value came
from. A request is immutable and has a :py:attr:`~.EnvironmentRequest.digest`
that is equal for equal requests, whatever format they were written in.

**Developer notes.** Validation happens in ``__post_init__`` so a request
cannot exist in an invalid state, and every container is a tuple so the
request is hashable and picklable (Rule 112).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ._errors import ConfigError
from ._types import Bootstrap, Conversion, Interaction, Network, choice

__all__ = ["EnvironmentRequest", "Policy", "SourceRef"]

#: Schema identifier of a request document.
REQUEST_SCHEMA = "scikitplot.env/v1"

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_PYTHON = re.compile(r"^3\.\d{1,2}(\.\d{1,3})?$")
_BACKEND = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_SOURCE_KINDS = ("requirements", "environment", "pyproject", "pipfile")


def _strings(field: str, value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{field} must be a list of strings")
    out = []
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{field}[{i}] must be a non-empty string, got {item!r}")
        out.append(item.strip())
    return tuple(out)


_URL_IN_TEXT = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s;,]+")
_CONDA_TOKEN = re.compile(r"/t/[^/\s]+/")


def _no_credentials(field: str, url: str) -> None:
    for candidate in _URL_IN_TEXT.findall(url) or [url]:
        parts = urlsplit(candidate)
        if parts.username or parts.password or _CONDA_TOKEN.search(parts.path):
            _credential_error(field)


def _credential_error(field: str) -> None:
    raise ConfigError(
        f"{field} holds a credential in a URL (user info or a /t/<token>/ path);"
        " a request is logged and recorded in receipts",
        hint="give the URL without it and configure the credential in the tool"
        " (pip/uv keyring, ~/.netrc, PIP_INDEX_URL, micromamba auth)",
    )


def infer_source_kind(path: str) -> str:
    """
    Return the source kind implied by a dependency file's name.

    Parameters
    ----------
    path : str
        File path.

    Returns
    -------
    str
        ``"requirements"``, ``"environment"``, ``"pyproject"`` or ``"pipfile"``.

    Raises
    ------
    ConfigError
        If the name implies no kind; give ``type`` explicitly then.
    """
    name = Path(path).name.lower()
    if name == "pyproject.toml":
        return "pyproject"
    if name == "pipfile":
        return "pipfile"
    if name.endswith((".yml", ".yaml")):
        return "environment"
    if name.endswith((".txt", ".in")):
        return "requirements"
    raise ConfigError(
        f"cannot tell what kind of dependency file {path!r} is",
        hint=f"give type = one of {_SOURCE_KINDS}",
    )


@dataclasses.dataclass(frozen=True)
class SourceRef:  # numpydoc ignore=PR02
    """
    A dependency file the environment is built from.

    Parameters
    ----------
    path : str
        Absolute path of the file.
    kind : {"requirements", "environment", "pyproject", "pipfile"}
        What the file is.
    groups : tuple of str, default=()
        For ``pyproject``: optional-dependency extras and dependency groups
        to include besides ``[project.dependencies]``. For ``pipfile``:
        ``"dev"`` adds ``[dev-packages]``.
    """

    path: str
    kind: str
    groups: tuple[str, ...] = ()

    def __post_init__(self) -> None:  # noqa: D105
        if not isinstance(self.path, str) or not self.path:
            raise ConfigError("a source needs a path")
        if not os.path.isabs(self.path):
            raise ConfigError(f"source path {self.path!r} must be absolute")
        if self.kind not in _SOURCE_KINDS:
            raise ConfigError(
                f"source type {self.kind!r} is not one of {_SOURCE_KINDS}"
            )
        object.__setattr__(self, "groups", _strings("groups", self.groups))

    def to_dict(self) -> dict[str, Any]:
        """
        Return the source as a mapping.

        Returns
        -------
        dict
            Plain data (JSON-serialisable).
        """
        return {"path": self.path, "type": self.kind, "groups": list(self.groups)}


@dataclasses.dataclass(frozen=True)
class Policy:  # numpydoc ignore=PR02
    """
    How the builder may behave.

    Parameters
    ----------
    interaction : {"never", "auto", "always"}, default="auto"
        Whether a person may be asked.
    network : {"online", "offline"}, default="online"
        Network policy.
    bootstrap : {"never", "if-missing"}, default="never"
        Whether a missing micromamba or uv may be installed (still needs an
        approval).
    conversion : {"exact", "allow-approximate"}, default="exact"
        Whether a dependency may change ecosystem by name.
    """

    interaction: str = Interaction.AUTO.value
    network: str = Network.ONLINE.value
    bootstrap: str = Bootstrap.NEVER.value
    conversion: str = Conversion.EXACT.value

    def __post_init__(self) -> None:  # noqa: D105
        set_ = object.__setattr__
        set_(
            self,
            "interaction",
            choice(self.interaction, Interaction, field="policy.interaction").value,
        )
        set_(
            self, "network", choice(self.network, Network, field="policy.network").value
        )
        set_(
            self,
            "bootstrap",
            choice(self.bootstrap, Bootstrap, field="policy.bootstrap").value,
        )
        set_(
            self,
            "conversion",
            choice(self.conversion, Conversion, field="policy.conversion").value,
        )

    @property
    def offline(self) -> bool:  # numpydoc ignore=RT01
        """bool: The network policy is ``offline``."""
        return self.network == Network.OFFLINE.value

    def to_dict(self) -> dict[str, str]:
        """
        Return the policy as a mapping.

        Returns
        -------
        dict
            Plain data (JSON-serialisable).
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class EnvironmentRequest:  # numpydoc ignore=PR02
    """
    What environment to build.

    Parameters
    ----------
    name : str or None, default=None
        Environment name, placed under the environments home (see
        :py:func:`~._host.env_home`); exactly one of ``name`` and ``prefix``.
    prefix : str or None, default=None
        Absolute directory of the environment (``.venv`` of a project).
    python : str or None, default=None
        Python version (``"3.12"``); None uses the backend's default (the
        running interpreter for ``venv``).
    backend : str, default="auto"
        ``"auto"`` or a backend name (``"uv"``, ``"venv"``, ``"micromamba"``,
        ``"conda"``, ``"mamba"``, ``"pipenv"``).
    packages : tuple of str, default=()
        PyPI requirement strings (``"numpy>=2"``).
    conda_packages : tuple of str, default=()
        Conda match specs (``"numpy>=2"``, ``"conda-forge::pandas"``).
    sources : tuple of SourceRef, default=()
        Dependency files.
    channels : tuple of str, default=()
        Conda channels, in priority order.
    index_url : str or None, default=None
        PyPI index; credentials in the URL are refused.
    extra_index_urls, find_links : tuple of str, default=()
        Extra indexes and local wheel directories (offline installs).
    target : str, default="host"
        Target kind (see :py:mod:`~._targets`).
    policy : Policy, default=Policy()
        Behaviour policy.
    verify_imports : tuple of str, default=()
        Modules imported by the environment's Python after building it.
    root_prefix : str or None, default=None
        Root of a Conda-family backend (micromamba ``--root-prefix``).
    profile : str or None, default=None
        Preset that contributed values (recorded, not re-applied).
    origins : tuple of (str, str), default=()
        Where each field's value came from (``"default"``, ``"preset:ci"``,
        ``"config:<path>"``, ``"invocation"``); not part of the digest.
    config_digest : str or None, default=None
        SHA-256 of the configuration document; not part of the digest.

    Raises
    ------
    ConfigError
        For an invalid value.
    """

    name: str | None = None
    prefix: str | None = None
    python: str | None = None
    backend: str = "auto"
    packages: tuple[str, ...] = ()
    conda_packages: tuple[str, ...] = ()
    sources: tuple[SourceRef, ...] = ()
    channels: tuple[str, ...] = ()
    index_url: str | None = None
    extra_index_urls: tuple[str, ...] = ()
    find_links: tuple[str, ...] = ()
    target: str = "host"
    policy: Policy = dataclasses.field(default_factory=Policy)
    verify_imports: tuple[str, ...] = ()
    root_prefix: str | None = None
    profile: str | None = None
    origins: tuple[tuple[str, str], ...] = ()
    config_digest: str | None = None

    def __post_init__(self) -> None:  # noqa: D105, PLR0912 (one check per field)
        set_ = object.__setattr__
        if (self.name is None) == (self.prefix is None):
            raise ConfigError(
                "give exactly one of name and prefix",
                hint="name='analysis' (under the environments home) or"
                " prefix='/path/to/.venv'",
            )
        if self.name is not None and (
            not isinstance(self.name, str) or not _NAME.match(self.name)
        ):
            raise ConfigError(
                f"name={self.name!r} must be 1-100 letters, digits, '.', '_' or '-'"
                " and start with a letter or digit"
            )
        for field in ("prefix", "root_prefix"):
            value = getattr(self, field)
            if value is not None:
                if not isinstance(value, str) or not os.path.isabs(value):
                    raise ConfigError(f"{field}={value!r} must be an absolute path")
                value = os.path.normpath(value)
                if os.path.dirname(value) == value:  # "/", "C:\\"
                    raise ConfigError(f"{field}={value!r} is a file-system root")
                set_(self, field, value)
        if self.python is not None and (
            not isinstance(self.python, str) or not _PYTHON.match(self.python)
        ):
            raise ConfigError(
                f"python={self.python!r} must look like '3.12' or '3.12.4'"
            )
        if not isinstance(self.backend, str) or not _BACKEND.match(self.backend):
            raise ConfigError(f"backend={self.backend!r} is not a backend name")
        for field in (
            "packages",
            "conda_packages",
            "channels",
            "extra_index_urls",
            "find_links",
            "verify_imports",
        ):
            set_(self, field, _strings(field, getattr(self, field)))
        if self.index_url is not None:
            if not isinstance(self.index_url, str):
                raise ConfigError("index_url must be a string")
            _no_credentials("index_url", self.index_url)
        for field in (
            "extra_index_urls",
            "channels",
            "find_links",
            "packages",
            "conda_packages",
        ):
            for value in getattr(self, field):
                _no_credentials(field, value)
        sources = tuple(self.sources)
        if not all(isinstance(s, SourceRef) for s in sources):
            raise ConfigError("sources must be SourceRef objects")
        set_(self, "sources", sources)
        if not isinstance(self.policy, Policy):
            raise ConfigError("policy must be a Policy")
        if not isinstance(self.target, str) or not self.target:
            raise ConfigError("target must be a non-empty string")
        for module in self.verify_imports:
            if not re.match(
                r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$", module
            ):
                raise ConfigError(f"verify import {module!r} is not a module name")
        set_(self, "origins", tuple((str(k), str(v)) for k, v in self.origins))

    def to_dict(self, *, provenance: bool = False) -> dict[str, Any]:
        """
        Return the request as a mapping of the shared data model.

        Parameters
        ----------
        provenance : bool, default=False
            Include ``origins`` and ``config_digest``.

        Returns
        -------
        dict
            The request; ``schema`` is :data:`REQUEST_SCHEMA`.
        """
        data: dict[str, Any] = {"schema": REQUEST_SCHEMA}
        for field in dataclasses.fields(self):
            if field.name in ("origins", "config_digest"):
                continue
            value = getattr(self, field.name)
            if field.name == "policy":
                value = value.to_dict()
            elif field.name == "sources":
                value = [s.to_dict() for s in value]
            elif isinstance(value, tuple):
                value = list(value)
            data[field.name] = value
        if provenance:
            data["origins"] = dict(self.origins)
            data["config_digest"] = self.config_digest
        return data

    @property
    def digest(self) -> str:  # numpydoc ignore=RT01
        """str: SHA-256 of the canonical request (provenance excluded)."""
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()
