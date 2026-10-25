# scikitplot/_env/_targets/__init__.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Targets: where the environment will run, which decides how it can be built.

=====================  ===========================================================
Target                 Meaning
=====================  ===========================================================
``host``               runs here; every backend; imports are verified
``<conda subdir>``     cross-build for another Conda platform (``linux-aarch64``,
                       ``osx-arm64``, ``win-64`` ...): micromamba/mamba solve for
                       it, nothing runs here, PyPI packages are refused
``emscripten-wasm32``  browser cross-build (JupyterLite / Xeus kernel):
                       micromamba/mamba with emscripten-forge, pure-Python
                       PyPI packages added by the Xeus installer
``pyodide``            plan only: the plan's artifact ``micropip.py`` installs
                       the PyPI packages inside Pyodide in the browser
``android``, ``ios``   refused with the reason (see below)
=====================  ===========================================================

Notes
-----
**User notes.** Phones and tablets: an Android or iOS app embeds its own
Python, built by a mobile packaging toolchain together with the app, so
there is no environment to create on a host and the builder refuses rather
than produce something that cannot run there. The browser targets
(``pyodide``, ``emscripten-wasm32``) do run on phones and tablets, in the
browser.

**Developer notes.** A target is data (:py:class:`Target`); the planner
reads its fields and never branches on the name, so adding a target is one
entry in :py:func:`get_target`.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType

from .._core._errors import ConfigError, TargetUnsupportedError

__all__ = ["CONDA_SUBDIRS", "EMSCRIPTEN_CHANNELS", "TARGETS", "Target", "get_target"]

#: Conda platforms accepted as cross-build targets.
CONDA_SUBDIRS = (
    "linux-64",
    "linux-aarch64",
    "linux-ppc64le",
    "osx-64",
    "osx-arm64",
    "win-64",
    "win-arm64",
)

#: Default channels of the browser cross-build (as the Xeus add-on used).
EMSCRIPTEN_CHANNELS = (
    "https://prefix.dev/emscripten-forge-4x",
    "https://prefix.dev/conda-forge",
)


@dataclasses.dataclass(frozen=True)
class Target:  # numpydoc ignore=PR02
    """
    How an environment for one kind of runtime is built.

    Parameters
    ----------
    name : str
        Target name.
    conda_subdir : str or None
        Platform to solve for; None for the host's own.
    backends : tuple of str or None
        Backends allowed; None allows every backend.
    channels : tuple of str
        Default Conda channels when the request names none (empty: the
        backend default).
    runs_here : bool
        Whether the environment runs on the host (imports can be verified,
        ``run`` works).
    executable : bool
        Whether the plan is executed here; False for a plan-only target.
    pypi : {"install", "pure-python", "micropip", "refuse"}
        What happens to PyPI packages.
    relocate_prefix : str or None
        Prefix the files refer to at run time (``""``: mounted at ``/`` in the
        browser); None keeps the build prefix, so a cross-build is copied to
        the same path on the target machine.
    summary : str
        One line for listings.
    """

    name: str
    conda_subdir: str | None = None
    backends: tuple[str, ...] | None = None
    channels: tuple[str, ...] = ()
    runs_here: bool = True
    executable: bool = True
    pypi: str = "install"
    relocate_prefix: str | None = None
    summary: str = ""

    def to_dict(self) -> dict:
        """
        Return the target as a mapping.

        Returns
        -------
        dict
            Plain data (JSON-serialisable).
        """
        return dataclasses.asdict(self)


_CROSS = ("micromamba", "mamba")

#: Named targets (Conda subdirs are added by :py:func:`get_target`).
TARGETS = MappingProxyType(
    {
        "host": Target("host", summary="runs on this machine"),
        "emscripten-wasm32": Target(
            "emscripten-wasm32",
            conda_subdir="emscripten-wasm32",
            backends=_CROSS,
            channels=EMSCRIPTEN_CHANNELS,
            runs_here=False,
            pypi="pure-python",
            relocate_prefix="",
            summary="browser cross-build for JupyterLite / Xeus kernels",
        ),
        "pyodide": Target(
            "pyodide",
            backends=("micropip",),
            runs_here=False,
            executable=False,
            pypi="micropip",
            summary="plan only: micropip install script for Pyodide",
        ),
    }
)

_MOBILE = {
    "android": "Android",
    "ios": "iOS",
}


def get_target(name: str, *, host_subdir: str | None = None) -> Target:
    """
    Return the target called ``name``.

    Parameters
    ----------
    name : str
        A name of :py:data:`TARGETS`, a Conda subdir, ``"android"`` or ``"ios"``.
    host_subdir : str or None, default=None
        Conda subdir of the host: a subdir equal to it is the ``host`` target.

    Returns
    -------
    Target
        The target.

    Raises
    ------
    TargetUnsupportedError
        For ``android`` and ``ios``, with the reason.
    ConfigError
        For an unknown name.

    Examples
    --------
    >>> from scikitplot._env._targets import get_target
    >>> get_target("linux-aarch64", host_subdir="linux-64").backends
    ('micromamba', 'mamba')
    >>> get_target("linux-64", host_subdir="linux-64").name
    'host'
    """
    if name in TARGETS:
        return TARGETS[name]
    if name in CONDA_SUBDIRS:
        if name == host_subdir:
            return TARGETS["host"]
        return Target(
            name,
            conda_subdir=name,
            backends=_CROSS,
            runs_here=False,
            pypi="refuse",
            summary=f"cross-build for {name}",
        )
    if name in _MOBILE:
        raise TargetUnsupportedError(
            f"target {name!r}: an {_MOBILE[name]} app embeds its own Python, built with"
            " the app by a mobile packaging toolchain; there is no environment to"
            " create on a host",
            hint="package the app with a mobile toolchain, or use target='pyodide' or"
            " 'emscripten-wasm32', which run in the browser of a phone or tablet",
        )
    raise ConfigError(
        f"unknown target {name!r}",
        hint=f"choose one of {sorted([*TARGETS, *CONDA_SUBDIRS, *_MOBILE])}",
    )
