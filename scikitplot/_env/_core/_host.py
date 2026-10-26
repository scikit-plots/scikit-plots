# scikitplot/_env/_core/_host.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
What the current host is and which environment tools it has, found at call time.

Notes
-----
**User notes.** :py:func:`detect_host` is what ``scikitplot env doctor``
prints. Tools are looked up on ``PATH``, in the builder's own tools directory
(where a bootstrap puts micromamba and uv) and through ``MAMBA_EXE`` /
``CONDA_EXE``.

**Developer notes.** Every input (``sys.platform``, ``platform.machine()``,
the environment, ``shutil.which``) is a keyword argument that defaults to the
real one, so each branch is tested by passing values, never by patching the
process (Rule 39). Nothing is cached: a tool installed after import is found
(ENV-002).
"""

from __future__ import annotations

import dataclasses
import os
import platform as _platform
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

__all__ = ["HostInfo", "detect_host", "env_home"]

#: Executables the builder knows, by tool name.
TOOLS: Mapping[str, str] = {
    "uv": "uv",
    "micromamba": "micromamba",
    "mamba": "mamba",
    "conda": "conda",
    "pipenv": "pipenv",
}

_SUBDIRS = {
    ("linux", "x86_64"): "linux-64",
    ("linux", "amd64"): "linux-64",
    ("linux", "aarch64"): "linux-aarch64",
    ("linux", "arm64"): "linux-aarch64",
    ("linux", "ppc64le"): "linux-ppc64le",
    ("darwin", "x86_64"): "osx-64",
    ("darwin", "arm64"): "osx-arm64",
    ("windows", "amd64"): "win-64",
    ("windows", "x86_64"): "win-64",
    ("windows", "arm64"): "win-arm64",
}

_CI_VARIABLES = (
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "BUILDKITE",
    "TF_BUILD",
    "JENKINS_URL",
    "CIRCLECI",
    "TRAVIS",
)


def env_home(environ: Mapping[str, str] | None = None) -> Path:
    """
    Return the directory under which named environments and tools live.

    Parameters
    ----------
    environ : mapping or None, default=None
        Environment variables; :py:data:`os.environ` when None.

    Returns
    -------
    pathlib.Path
        ``$SKPLT_ENV_HOME`` when set, else ``~/.scikitplot/env``.
    """
    environ = os.environ if environ is None else environ
    value = environ.get("SKPLT_ENV_HOME")
    return Path(value).expanduser() if value else Path.home() / ".scikitplot" / "env"


@dataclasses.dataclass(frozen=True)
class HostInfo:  # numpydoc ignore=PR02
    """
    The host the builder runs on.

    Parameters
    ----------
    system : str
        ``"linux"``, ``"darwin"``, ``"windows"``, ``"emscripten"``,
        ``"android"``, ``"ios"`` ... (lower case).
    machine : str
        CPU architecture, lower case.
    runtime : {"native", "wasm", "android", "ios"}
        Kind of Python runtime.
    conda_subdir : str or None
        Conda platform of the host (``"linux-64"``); None when unknown.
    python : str
        Version of the running Python (``"3.12.4"``).
    python_executable : str
        Path of the running Python.
    can_subprocess : bool
        Whether the runtime can start processes (false in a browser).
    in_ci, in_container : bool
        Detected from the usual environment variables and marker files.
    tools : tuple of (str, str)
        Found tools, ``(name, path)``.
    home : str
        Environments home (see :py:func:`env_home`).
    """

    system: str
    machine: str
    runtime: str
    conda_subdir: str | None
    python: str
    python_executable: str
    can_subprocess: bool
    in_ci: bool
    in_container: bool
    tools: tuple[tuple[str, str], ...]
    home: str

    def tool(self, name: str) -> str | None:
        """
        Return the path of tool ``name``, or None when it was not found.

        Parameters
        ----------
        name : str
            Tool name (``"uv"``, ``"micromamba"`` ...).

        Returns
        -------
        str or None
            Executable path.
        """
        return dict(self.tools).get(name)

    @property
    def is_windows(self) -> bool:  # numpydoc ignore=RT01
        """bool: The host runs Windows."""
        return self.system == "windows"

    @property
    def tools_dir(self) -> Path:  # numpydoc ignore=RT01
        """pathlib.Path: Where a bootstrap installs tools (``<home>/tools``)."""
        return Path(self.home) / "tools"

    def to_dict(self) -> dict[str, Any]:
        """
        Return the host description as a mapping.

        Returns
        -------
        dict
            Plain data (JSON-serialisable).
        """
        data = dataclasses.asdict(self)
        data["tools"] = dict(self.tools)
        return data


def _runtime(system: str) -> tuple[str, bool]:
    if system in ("emscripten", "wasi") or system.startswith("wasm"):
        return "wasm", False
    if system == "android":
        return "android", True
    if system in ("ios", "ipados", "tvos", "watchos"):
        return "ios", False
    return "native", True


def detect_host(
    *,
    sys_platform: str | None = None,
    machine: str | None = None,
    environ: Mapping[str, str] | None = None,
    which: Callable[..., str | None] | None = None,
    python_version: str | None = None,
    python_executable: str | None = None,
    marker_exists: Callable[[str], bool] | None = None,
) -> HostInfo:
    """
    Describe the current host and find its environment tools.

    Parameters
    ----------
    sys_platform : str or None, default=None
        Value of :py:data:`sys.platform` (``"linux"``, ``"win32"``,
        ``"emscripten"``, ``"android"``, ``"ios"``).
    machine : str or None, default=None
        Value of :py:func:`platform.machine`.
    environ : mapping or None, default=None
        Environment variables.
    which : callable or None, default=None
        ``which(name, path=None)``; :py:func:`shutil.which` when None.
    python_version, python_executable : str or None, default=None
        The running Python's version and path.
    marker_exists : callable or None, default=None
        ``marker_exists(path)`` used for container marker files.

    Returns
    -------
    HostInfo
        The description.

    Examples
    --------
    >>> from scikitplot._env import detect_host
    >>> host = detect_host(
    ...     sys_platform="linux",
    ...     machine="x86_64",
    ...     environ={"CI": "true"},
    ...     which=lambda name, path=None: None,
    ... )
    >>> host.conda_subdir, host.in_ci, host.tools
    ('linux-64', True, ())
    """
    plat = (sys.platform if sys_platform is None else sys_platform).lower()
    system = {"win32": "windows", "cygwin": "windows", "linux2": "linux"}.get(
        plat, plat
    )
    arch = (_platform.machine() if machine is None else machine).lower()
    environ = os.environ if environ is None else environ
    which = shutil.which if which is None else which
    marker_exists = os.path.exists if marker_exists is None else marker_exists
    runtime, can_subprocess = _runtime(system)
    home = env_home(environ)
    tools_bin = home / "tools" / "bin"
    search = os.pathsep.join(p for p in (str(tools_bin), environ.get("PATH", "")) if p)
    found = []
    if can_subprocess:
        for name, exe in TOOLS.items():
            path = which(exe, path=search)
            if path is None and name in ("micromamba", "conda"):
                hinted = environ.get(
                    "MAMBA_EXE" if name == "micromamba" else "CONDA_EXE"
                )
                if hinted and Path(hinted).name.lower().startswith(exe):
                    path = hinted if marker_exists(hinted) else None
            if path:
                found.append((name, str(path)))
    in_ci = any(
        environ.get(v, "").strip().lower() not in ("", "0", "false", "no")
        for v in _CI_VARIABLES
    )
    in_container = marker_exists("/.dockerenv") or marker_exists("/run/.containerenv")
    return HostInfo(
        system=system,
        machine=arch,
        runtime=runtime,
        conda_subdir=_SUBDIRS.get((system, arch)),
        python=python_version or _platform.python_version(),
        python_executable=python_executable or sys.executable,
        can_subprocess=can_subprocess,
        in_ci=in_ci,
        in_container=bool(in_container),
        tools=tuple(found),
        home=str(home),
    )
