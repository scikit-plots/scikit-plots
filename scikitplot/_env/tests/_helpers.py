# scikitplot/_env/tests/_helpers.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Shared test helpers: fake hosts, fake runners, small file writers."""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .._core._host import HostInfo, detect_host


def fake_host(
    home: Path,
    *,
    tools: Mapping[str, str] | None = None,
    system: str = "linux",
    machine: str = "x86_64",
    python: str = "3.12.4",
    environ: Mapping[str, str] | None = None,
) -> HostInfo:
    """Return a host with exactly ``tools`` (``name -> path``) and home ``home``."""
    tools = dict(tools or {})
    env = {"SKPLT_ENV_HOME": str(home), "PATH": "", **(environ or {})}
    return detect_host(
        sys_platform=system,
        machine=machine,
        environ=env,
        which=lambda name, path=None: tools.get(name),
        python_version=python,
        python_executable="/usr/bin/python3" if system != "win32" else r"C:\Python\python.exe",
        marker_exists=lambda path: False,
    )


@dataclasses.dataclass
class FakeRunner:
    """Record commands; fail the ones whose argv contains ``fail_on``."""

    fail_on: str | None = None
    output: str = "line one\nline two\n"
    calls: list = dataclasses.field(default_factory=list)

    def __call__(self, argv: Sequence[str], *, cwd: Any, env: Mapping[str, str]):
        self.calls.append({"argv": list(argv), "cwd": cwd, "env": dict(env)})
        code = 1 if self.fail_on and any(self.fail_on in a for a in argv) else 0
        return subprocess.CompletedProcess(list(argv), code, stdout=self.output)


def write(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` (parents created) and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_venv_like(prefix: Path, *, windows: bool = False) -> Path:
    """Create a directory that looks like a built venv; return its Python."""
    python = prefix / ("Scripts/python.exe" if windows else "bin/python")
    write(python, "")
    python.chmod(0o755)
    write(prefix / "pyvenv.cfg", "home = /usr/bin\n")
    return python
