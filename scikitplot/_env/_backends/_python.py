# scikitplot/_env/_backends/_python.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
PyPI backends: ``venv`` + pip (always present) and ``uv`` (fast, installs Pythons).

Notes
-----
**User notes.** ``venv`` uses the Python that runs the builder, so it cannot
give another Python version; ask for ``uv`` (or ``micromamba``) when
``python`` differs. ``uv`` can download a missing Python itself; under the
``offline`` policy it is told not to (``--no-python-downloads``).

**Developer notes.** Both install with ``-r FILE`` for requirements files, so
the file's own options (``--index-url``, ``-c``) keep their meaning. The
flags used are checked against the installed pip and uv (``--help``) by
``test_planned_flags_exist_in_the_installed_tool`` (pip 24.0, uv 0.11.32 here).
"""

from __future__ import annotations

from .._core._errors import BackendUnavailableError
from .._core._plan import Step
from .._core._types import Ecosystem
from ._base import Backend, PlanContext, has_pypi, pip_index_args, pypi_install_args

__all__ = ["UvBackend", "VenvBackend"]

#: Environment variables that keep pip silent and non-interactive.
PIP_ENV = (("PIP_NO_INPUT", "1"), ("PIP_DISABLE_PIP_VERSION_CHECK", "1"))


def _major_minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


class VenvBackend(Backend):
    """Standard-library ``venv`` and pip of the running Python."""

    name = "venv"
    tool = None
    ecosystems = frozenset({Ecosystem.PYPI.value})
    summary = "python -m venv + pip; always available, running Python only"

    def check(self, ctx: PlanContext) -> None:  # noqa: D102
        if ctx.python and _major_minor(ctx.python) != _major_minor(ctx.host.python):
            raise BackendUnavailableError(
                f"venv can only create Python {_major_minor(ctx.host.python)} (the"
                f" running Python), not {ctx.python}",
                hint="use backend='uv' or 'micromamba', which install other Pythons",
            )

    def _install(self, ctx: PlanContext) -> list[Step]:
        if not has_pypi(ctx.deps):
            return []
        python = str(self.python_in(ctx.prefix, ctx.host))
        argv = [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            *pip_index_args(ctx.request, offline=ctx.offline, flavour="pip"),
            *pypi_install_args(ctx.deps),
        ]
        return [
            Step(
                id="install-pypi",
                title="install PyPI packages with pip",
                argv=tuple(argv),
                env=PIP_ENV,
                network=ctx.network,
                writes=(str(ctx.prefix),),
            )
        ]

    def create(self, ctx: PlanContext) -> list[Step]:  # noqa: D102
        create = Step(
            id="create",
            title=f"create a venv with Python {ctx.host.python}",
            argv=(ctx.host.python_executable, "-m", "venv", str(ctx.prefix)),
            writes=(str(ctx.prefix),),
        )
        return [create, *self._install(ctx)]

    def update(self, ctx: PlanContext) -> list[Step]:  # noqa: D102
        return self._install(ctx)


class UvBackend(Backend):
    """Astral's ``uv``: ``uv venv`` + ``uv pip install``."""

    name = "uv"
    tool = "uv"
    ecosystems = frozenset({Ecosystem.PYPI.value})
    bootstrappable = True
    summary = "uv venv + uv pip; fast, can install any CPython"

    def _install(self, ctx: PlanContext) -> list[Step]:
        if not has_pypi(ctx.deps):
            return []
        python = str(self.python_in(ctx.prefix, ctx.host))
        argv = [
            ctx.exe,
            "pip",
            "install",
            "--python",
            python,
            *pip_index_args(ctx.request, offline=ctx.offline, flavour="uv"),
            *pypi_install_args(ctx.deps),
        ]
        return [
            Step(
                id="install-pypi",
                title="install PyPI packages with uv",
                argv=tuple(argv),
                env=(("UV_NO_PROGRESS", "1"),),
                network=ctx.network,
                writes=(str(ctx.prefix),),
            )
        ]

    def create(self, ctx: PlanContext) -> list[Step]:  # noqa: D102
        argv = [ctx.exe, "venv", str(ctx.prefix), "--seed"]
        version = ctx.python or ctx.host.python
        argv += ["--python", version]
        if ctx.offline:
            argv += ["--offline", "--no-python-downloads"]
        create = Step(
            id="create",
            title=f"create a uv environment with Python {version}",
            argv=tuple(argv),
            env=(("UV_NO_PROGRESS", "1"),),
            network=ctx.network,
            writes=(str(ctx.prefix),),
        )
        return [create, *self._install(ctx)]

    def update(self, ctx: PlanContext) -> list[Step]:  # noqa: D102
        return self._install(ctx)
