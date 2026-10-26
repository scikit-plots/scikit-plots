# scikitplot/_env/_backends/_base.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The backend contract: a tool turned into plan steps, never run directly.

Notes
-----
**User notes.** A backend says which ecosystems it installs from (``pypi``,
``conda``), which targets it can build, and which commands create, update,
remove and run inside an environment. It never executes anything and never
prompts: the executor runs the steps it returns.

**Developer notes.** Add a backend by subclassing :py:class:`Backend` in its
own module and listing it in :py:data:`~._backends.BACKENDS`; the registry
imports the module only when that backend is asked for.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Sequence

from .._core._host import HostInfo
from .._core._plan import Step
from .._core._request import EnvironmentRequest
from .._sources._readers import Dependency, DependencySet

__all__ = ["Backend", "PlanContext"]


@dataclasses.dataclass(frozen=True)
class PlanContext:  # numpydoc ignore=PR02
    """
    Everything a backend needs to produce steps.

    Parameters
    ----------
    request : EnvironmentRequest
        The request.
    deps : DependencySet
        Dependencies after ecosystem conversion.
    host : HostInfo
        The host.
    prefix : pathlib.Path
        Environment directory.
    exe : str
        Path of the backend's tool (planned path when a bootstrap provides it).
    python : str or None
        Python version to install, if any.
    channels : tuple of str
        Conda channels in priority order.
    conda_subdir : str or None
        Platform to build for, when it differs from the host (``"emscripten-wasm32"``).
    offline : bool
        The network policy is offline.
    relocate_prefix : str or None, default=None
        Prefix the built files should refer to instead of ``prefix``
        (``""`` for the browser, where the environment is mounted at ``/``).
    """

    request: EnvironmentRequest
    deps: DependencySet
    host: HostInfo
    prefix: Path
    exe: str
    python: str | None
    channels: tuple[str, ...]
    conda_subdir: str | None
    offline: bool
    relocate_prefix: str | None = None

    @property
    def network(self) -> bool:  # numpydoc ignore=RT01
        """bool: Steps of this context may use the network."""
        return not self.offline


class Backend:
    """
    Base class of every backend.

    Attributes
    ----------
    name : str
        Registry name.
    tool : str or None
        Tool name looked up on the host (None: the running Python).
    ecosystems : frozenset of str
        Ecosystems it installs from.
    bootstrappable : bool
        Whether the builder can install the tool.
    cross_builds : bool
        Whether it can build for another platform (``--platform``).
    summary : str
        One line for listings.
    """

    name = ""
    tool: str | None = None
    ecosystems: frozenset = frozenset()
    bootstrappable = False
    cross_builds = False
    summary = ""

    def python_in(self, prefix: Path, host: HostInfo) -> Path:
        """
        Return the Python executable inside an environment.

        Parameters
        ----------
        prefix : pathlib.Path
            Environment directory.
        host : HostInfo
            The host.

        Returns
        -------
        pathlib.Path
            ``bin/python``, or ``Scripts/python.exe`` on Windows.
        """
        return prefix / ("Scripts/python.exe" if host.is_windows else "bin/python")

    def check(self, ctx: PlanContext) -> None:
        """
        Refuse a request this backend cannot satisfy.

        Parameters
        ----------
        ctx : PlanContext
            Planning context.

        Raises
        ------
        BackendUnavailableError
            When it cannot.
        """

    def create(self, ctx: PlanContext) -> list[Step]:
        """
        Return the steps that create and fill a new environment.

        Parameters
        ----------
        ctx : PlanContext
            Planning context.

        Returns
        -------
        list of Step
            Steps in order.
        """
        raise NotImplementedError

    def update(self, ctx: PlanContext) -> list[Step]:
        """
        Return the steps that bring an existing environment up to the request.

        Parameters
        ----------
        ctx : PlanContext
            Planning context.

        Returns
        -------
        list of Step
            Steps in order.
        """
        raise NotImplementedError

    def remove(self, ctx: PlanContext) -> list[Step]:
        """
        Return the steps that delete the environment.

        Parameters
        ----------
        ctx : PlanContext
            Planning context.

        Returns
        -------
        list of Step
            Steps in order.
        """
        return [
            Step(
                id="remove",
                title=f"delete {ctx.prefix}",
                kind="action",
                action="remove_tree",
                params=(("path", str(ctx.prefix)),),
                writes=(str(ctx.prefix),),
                approval="remove",
            )
        ]

    def run_argv(
        self, prefix: Path, host: HostInfo, exe: str | None, command: Sequence[str]
    ) -> list[str]:
        """
        Return the argv that runs ``command`` inside the environment.

        Parameters
        ----------
        prefix : pathlib.Path
            Environment directory.
        host : HostInfo
            The host.
        exe : str or None
            Tool path.
        command : sequence of str
            Command and arguments.

        Returns
        -------
        list of str
            Argv; the executor prepends the environment's ``bin`` to ``PATH``.
        """
        return list(command)


def pip_index_args(
    request: EnvironmentRequest, *, offline: bool, flavour: str
) -> list[str]:
    """
    Return index arguments for pip (``flavour="pip"``) or uv (``"uv"``).

    Parameters
    ----------
    request : EnvironmentRequest
        The request.
    offline : bool
        Offline policy: pip gets ``--no-index``, uv ``--offline``.
    flavour : {"pip", "uv"}
        Installer.

    Returns
    -------
    list of str
        Arguments.
    """
    args: list[str] = []
    if request.index_url:
        args += ["--index-url", request.index_url]
    for url in request.extra_index_urls:
        args += ["--extra-index-url", url]
    for link in request.find_links:
        args += ["--find-links", link]
    if offline:
        args.append("--no-index" if flavour == "pip" else "--offline")
    return args


def pypi_install_args(deps: DependencySet) -> list[str]:
    """
    Return the requirement arguments of a pip/uv install.

    Parameters
    ----------
    deps : DependencySet
        Dependencies.

    Returns
    -------
    list of str
        Inline PyPI specs not read from a requirements file (``-e PATH`` for
        an editable entry), then ``-r FILE`` for each requirements file.
    """
    # Everything read from a requirements tree (the file and its -r includes)
    # reaches the installer through ``-r FILE`` only, with its options.
    files = set(deps.requirement_files) | set(deps.requirement_origins)
    args: list[str] = []
    for spec in dict.fromkeys(d.spec for d in deps.of("pypi") if d.origin not in files):
        args += ["-e", spec[3:]] if spec.startswith("-e ") else [spec]
    for path in deps.requirement_files:
        args += ["-r", path]
    return args


def has_pypi(deps: DependencySet) -> bool:
    """
    Return whether anything must be installed from PyPI.

    Parameters
    ----------
    deps : DependencySet
        Dependencies.

    Returns
    -------
    bool
        True when there is a PyPI dependency or a requirements file.
    """
    return bool(deps.of("pypi") or deps.requirement_files)


def specs_of(deps: Sequence[Dependency]) -> list[str]:
    """
    Return the specs of ``deps`` once each, in order.

    Parameters
    ----------
    deps : sequence of Dependency
        Dependencies.

    Returns
    -------
    list of str
        Specs.
    """
    return list(dict.fromkeys(d.spec for d in deps))
