# scikitplot/_env/_backends/_pipenv.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
``pipenv`` backend: install a project's ``Pipfile.lock`` exactly.

Notes
-----
**User notes.** Use it for a project that already has a ``Pipfile.lock``:
``pipenv sync`` installs the locked versions and checks their hashes. The
environment is a plain venv at the planned prefix (not pipenv's own
location), so ``scikitplot env run`` and every other backend's tooling work
on it. Without a lock file, read the ``Pipfile`` with any other backend
(``uv`` resolves it in seconds) or run ``pipenv lock`` first: the builder
never writes a lock file into a project on its own.

**Developer notes.** pipenv installs into the virtual environment named by
``VIRTUAL_ENV`` when that is set (its "courtesy notice" behaviour), and reads
the project from ``PIPENV_PIPFILE``; ``PIP_NO_INDEX`` reaches its pip calls
for the offline policy.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from .._core._errors import BackendUnavailableError
from .._core._plan import Step
from .._core._types import Ecosystem
from .._sources._readers import DependencySet
from ._base import PlanContext
from ._python import PIP_ENV, VenvBackend

__all__ = ["PipenvBackend"]


class PipenvBackend(VenvBackend):
    """``venv`` + ``pipenv sync`` of one locked Pipfile."""

    name = "pipenv"
    tool = "pipenv"
    ecosystems = frozenset({Ecosystem.PYPI.value})
    summary = "venv + pipenv sync (Pipfile.lock installed exactly)"

    def _pipfile(self, ctx: PlanContext) -> Path:
        if len(ctx.deps.pipfiles) != 1:
            raise BackendUnavailableError(
                f"pipenv needs exactly one Pipfile source, got {len(ctx.deps.pipfiles)}",
                hint="add the project's Pipfile to sources, or use backend='uv'",
            )
        pipfile = Path(ctx.deps.pipfiles[0])
        if not (pipfile.parent / "Pipfile.lock").is_file():
            raise BackendUnavailableError(
                f"{pipfile.parent / 'Pipfile.lock'} does not exist",
                hint="run 'pipenv lock' in the project, or use backend='uv' to resolve"
                " the Pipfile",
            )
        return pipfile

    def check(self, ctx: PlanContext) -> None:  # noqa: D102
        super().check(ctx)
        self._pipfile(ctx)

    def _install(self, ctx: PlanContext) -> list[Step]:
        pipfile = self._pipfile(ctx)
        dev = any(
            s.path == str(pipfile) and "dev" in s.groups for s in ctx.request.sources
        )
        env = (
            *PIP_ENV,
            ("VIRTUAL_ENV", str(ctx.prefix)),
            ("PIPENV_PIPFILE", str(pipfile)),
            ("PIPENV_YES", "1"),
            ("PIPENV_NOSPIN", "1"),
            ("PIPENV_VERBOSITY", "-1"),
        )
        if ctx.offline:
            env = (*env, ("PIP_NO_INDEX", "1"))
        steps = [
            Step(
                id="pipenv-sync",
                title=f"install {pipfile.parent.name}/Pipfile.lock",
                argv=(ctx.exe, "sync", *(["--dev"] if dev else [])),
                env=env,
                cwd=str(pipfile.parent),
                network=ctx.network,
                writes=(str(ctx.prefix),),
            )
        ]
        others = [
            d
            for d in ctx.deps.of(Ecosystem.PYPI.value)
            if not d.origin.startswith(str(pipfile) + ":")
        ]
        if others or ctx.deps.requirement_files:
            rest = DependencySet(
                dependencies=tuple(others), requirement_files=ctx.deps.requirement_files
            )
            steps += super()._install(dataclasses.replace(ctx, deps=rest))
        return steps
