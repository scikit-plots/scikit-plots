# scikitplot/_env/_backends/_conda.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Conda-family backends: ``micromamba`` (single binary), ``mamba`` 2.x and ``conda``.

Notes
-----
**User notes.** Conda packages come from ``channels`` (``conda-forge`` when
none is given: micromamba has no default channel). PyPI packages are then
installed with the environment's own pip, so ``pip`` is added to the Conda
specs. ``environment.yml`` files are read by the builder (see
:py:mod:`.._sources._readers`) and reach the tool as specs and channels, so
one file means one environment whichever backend builds it.

For the ``emscripten-wasm32`` target the environment is a cross-build for
the browser: the tool solves for that platform, nothing in it runs on the
host, and PyPI packages are added by the Xeus pure-Python installer.

Removing an environment deletes its directory (the same action for every
backend), so the result does not depend on the tool's own bookkeeping.

**Developer notes.** micromamba 2.9.0's ``create/install/run --help`` list
every flag used here (``-y -p -r -c --override-channels --platform --offline --no-pyc
--relocate-prefix``). ``conda`` is the one backend not executed in the tests
(no conda installation in the build sandbox); its argv follow the conda 24
command reference and are covered by plan tests only.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Sequence

from .._core._errors import BackendUnavailableError
from .._core._host import HostInfo
from .._core._plan import Step
from .._core._types import Ecosystem
from ._base import (
    Backend,
    PlanContext,
    has_pypi,
    pip_index_args,
    pypi_install_args,
    specs_of,
)
from ._python import PIP_ENV

__all__ = ["DEFAULT_CHANNELS", "CondaBackend", "MambaBackend", "MicromambaBackend"]

#: Channels used when a request names none.
DEFAULT_CHANNELS = ("conda-forge",)

_QUIET = (("MAMBA_NO_BANNER", "1"), ("CONDA_YES", "true"))
_NAME = re.compile(r"^\s*(?:[\w.-]+::)?([A-Za-z0-9_.-]+)")


def spec_name(spec: str) -> str:
    """
    Return the lower-cased package name of a Conda match spec.

    Parameters
    ----------
    spec : str
        Match spec such as ``"numpy>=2"``, ``"conda-forge::python=3.12"``.

    Returns
    -------
    str
        Package name, or ``""`` when the spec has none.

    Examples
    --------
    >>> spec_name("conda-forge::Python=3.12"), spec_name("numpy >=2")
    ('python', 'numpy')
    """
    match = _NAME.match(spec)
    return match.group(1).lower() if match else ""


@dataclasses.dataclass(frozen=True)
class CondaArgs:  # numpydoc ignore=PR02
    """
    The parts of a Conda solve: options and specs.

    Parameters
    ----------
    options : tuple of str
        ``-c CHANNEL`` pairs, ``--platform``, ``--offline``.
    specs : tuple of str
        Match specs.
    """

    options: tuple[str, ...]
    specs: tuple[str, ...]


def conda_args(ctx: PlanContext) -> CondaArgs:
    """
    Split a planning context into Conda solve arguments.

    Parameters
    ----------
    ctx : PlanContext
        Planning context.

    Returns
    -------
    CondaArgs
        ``python=<version>`` is added when a version is requested and no
        spec names ``python``; ``pip`` is added when PyPI packages follow
        on a host build.
    """
    specs = specs_of(ctx.deps.of(Ecosystem.CONDA.value))
    names = {spec_name(s) for s in specs}
    if ctx.python and "python" not in names:
        specs.insert(0, f"python={ctx.python}")
    if has_pypi(ctx.deps) and ctx.conda_subdir is None and "pip" not in names:
        specs.append("pip")
    # Only the planned channels: a user's ~/.mambarc / .condarc channels (or
    # conda's "defaults") would make the same request solve differently.
    options: list[str] = ["--override-channels"]
    for channel in ctx.channels:
        options += ["-c", channel]
    if ctx.conda_subdir is not None:
        options += ["--platform", ctx.conda_subdir]
    if ctx.offline:
        options.append("--offline")
    return CondaArgs(tuple(options), tuple(specs))


class MicromambaBackend(Backend):
    """``micromamba``: one static binary, no base environment needed."""

    name = "micromamba"
    tool = "micromamba"
    ecosystems = frozenset({Ecosystem.CONDA.value, Ecosystem.PYPI.value})
    bootstrappable = True
    cross_builds = True
    summary = "micromamba create; Conda packages + pip layer; cross-builds (emscripten)"

    def python_in(self, prefix: Path, host: HostInfo) -> Path:  # noqa: D102
        return prefix / ("python.exe" if host.is_windows else "bin/python")

    def root_prefix(self, ctx: PlanContext) -> str:
        """
        Return the root prefix (package cache) of the tool.

        Parameters
        ----------
        ctx : PlanContext
            Planning context.

        Returns
        -------
        str
            ``request.root_prefix`` or ``<home>/micromamba``.
        """
        return ctx.request.root_prefix or str(Path(ctx.host.home) / "micromamba")

    def _env(self, ctx: PlanContext) -> tuple[tuple[str, str], ...]:
        return (*_QUIET, ("MAMBA_ROOT_PREFIX", self.root_prefix(ctx)))

    def _pip_layer(self, ctx: PlanContext) -> list[Step]:
        if not has_pypi(ctx.deps):
            return []
        if ctx.conda_subdir is not None:  # cross-build: pure-Python wheels, host-side
            return [
                Step(
                    id="install-pypi",
                    title="add pure-Python PyPI packages to the cross-build",
                    kind="action",
                    action="xeus_pip_layer",
                    params=(
                        ("prefix", str(ctx.prefix)),
                        ("requirements", tuple(pypi_install_args(ctx.deps))),
                        (
                            "index_args",
                            tuple(
                                pip_index_args(
                                    ctx.request, offline=ctx.offline, flavour="pip"
                                )
                            ),
                        ),
                    ),
                    network=ctx.network,
                    writes=(str(ctx.prefix),),
                )
            ]
        python = str(self.python_in(ctx.prefix, ctx.host))
        argv = (
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            *pip_index_args(ctx.request, offline=ctx.offline, flavour="pip"),
            *pypi_install_args(ctx.deps),
        )
        return [
            Step(
                id="install-pypi",
                title="install PyPI packages with the environment's pip",
                argv=argv,
                env=PIP_ENV,
                network=ctx.network,
                writes=(str(ctx.prefix),),
            )
        ]

    def create(self, ctx: PlanContext) -> list[Step]:  # noqa: D102
        parts = conda_args(ctx)
        argv = [
            ctx.exe,
            "create",
            "--yes",
            "--prefix",
            str(ctx.prefix),
            "--root-prefix",
            self.root_prefix(ctx),
        ]
        if ctx.conda_subdir is not None:  # nothing built for another platform runs here
            argv.append("--no-pyc")
        if ctx.relocate_prefix is not None:
            argv += ["--relocate-prefix", ctx.relocate_prefix]
        argv += [*parts.options, *parts.specs]
        title = "create a Conda environment" + (
            f" for {ctx.conda_subdir}" if ctx.conda_subdir else ""
        )
        create = Step(
            id="create",
            title=title,
            argv=tuple(argv),
            env=self._env(ctx),
            network=ctx.network,
            writes=(str(ctx.prefix), self.root_prefix(ctx)),
        )
        return [create, *self._pip_layer(ctx)]

    def update(self, ctx: PlanContext) -> list[Step]:  # noqa: D102
        parts = conda_args(ctx)
        steps: list[Step] = []
        if parts.specs:
            argv = (
                ctx.exe,
                "install",
                "--yes",
                "--prefix",
                str(ctx.prefix),
                "--root-prefix",
                self.root_prefix(ctx),
                *parts.options,
                *parts.specs,
            )
            steps.append(
                Step(
                    id="install-conda",
                    title="install Conda packages",
                    argv=argv,
                    env=self._env(ctx),
                    network=ctx.network,
                    writes=(str(ctx.prefix), self.root_prefix(ctx)),
                )
            )
        return steps + self._pip_layer(ctx)

    def run_argv(
        self, prefix: Path, host: HostInfo, exe: str | None, command: Sequence[str]
    ) -> list[str]:  # noqa: D102
        # ``run`` activates the environment (activation scripts, PATH), which
        # matters for Conda packages that set variables on activation.
        return [str(exe), "run", "--prefix", str(prefix), *command]


class MambaBackend(MicromambaBackend):
    """``mamba`` 2.x, which shares micromamba's command line."""

    name = "mamba"
    tool = "mamba"
    bootstrappable = False
    summary = "mamba 2.x (micromamba command line)"


class CondaBackend(MicromambaBackend):
    """Classic ``conda``."""

    name = "conda"
    tool = "conda"
    bootstrappable = False
    cross_builds = False
    summary = "conda create / conda install (plan-tested; argv from the conda 24 docs)"

    def _env(self, ctx: PlanContext) -> tuple[tuple[str, str], ...]:
        return _QUIET

    def check(self, ctx: PlanContext) -> None:  # noqa: D102
        if ctx.exe.lower().endswith((".bat", ".cmd")):
            # cmd.exe re-parses a batch file's arguments: "x&calc" in a spec runs.
            raise BackendUnavailableError(
                f"conda was found as the batch file {ctx.exe}; its arguments would be"
                " parsed by cmd.exe",
                hint="point CONDA_EXE at conda.exe (<conda>/Scripts/conda.exe), or use"
                " micromamba",
            )

    def create(self, ctx: PlanContext) -> list[Step]:  # noqa: D102
        parts = conda_args(ctx)
        create = Step(
            id="create",
            title="create a Conda environment",
            argv=(
                ctx.exe,
                "create",
                "--yes",
                "--prefix",
                str(ctx.prefix),
                *parts.options,
                *parts.specs,
            ),
            env=self._env(ctx),
            network=ctx.network,
            writes=(str(ctx.prefix),),
        )
        return [create, *self._pip_layer(ctx)]

    def update(self, ctx: PlanContext) -> list[Step]:  # noqa: D102
        parts = conda_args(ctx)
        steps: list[Step] = []
        if parts.specs:
            steps.append(
                Step(
                    id="install-conda",
                    title="install Conda packages",
                    argv=(
                        ctx.exe,
                        "install",
                        "--yes",
                        "--prefix",
                        str(ctx.prefix),
                        *parts.options,
                        *parts.specs,
                    ),
                    env=self._env(ctx),
                    network=ctx.network,
                    writes=(str(ctx.prefix),),
                )
            )
        return steps + self._pip_layer(ctx)
