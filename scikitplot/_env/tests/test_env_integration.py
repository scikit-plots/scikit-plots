# scikitplot/_env/tests/test_env_integration.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Real environments, built with the real tools.

Notes
-----
The ``venv`` test needs nothing but this Python and always runs. The others
download packages and run only with ``SKPLT_ENV_NETWORK_TESTS=1``; they use
the tools found on ``PATH`` (``uv``, ``pipenv``) and bootstrap micromamba
from conda-forge into a temporary home, which also tests the pinned hash.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

from .. import _api
from .._core._receipt import read_receipt

NETWORK = os.environ.get("SKPLT_ENV_NETWORK_TESTS") == "1"
needs_network = pytest.mark.skipif(not NETWORK, reason="set SKPLT_ENV_NETWORK_TESTS=1")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SKPLT_ENV_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def _python(result):
    return result.plan.python_executable


def test_venv_end_to_end(home, tmp_path):
    prefix = str(tmp_path / "venv")
    result = _api.create(prefix=prefix, backend="venv", verify_imports=["json"])
    assert result.ok
    assert [s.id for s in result.steps] == ["create", "verify"]
    out = subprocess.run([_python(result), "-c", "import sys; print(sys.prefix)"],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert os.path.realpath(out) == os.path.realpath(prefix)
    assert read_receipt(prefix)["backend"] == "venv"
    assert not _api.create(prefix=prefix, backend="venv", verify_imports=["json"]).changed
    code = _api.run(["python", "-c",
                     "import os, sys; sys.exit(0 if os.environ['VIRTUAL_ENV'] else 1)"],
                    prefix=prefix)
    assert code == 0
    assert _api.remove(prefix=prefix, approve=["remove"]).ok
    assert not os.path.exists(prefix)


@needs_network
@pytest.mark.slow
@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_uv_with_requirements(home, tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("six\n", encoding="utf-8")
    prefix = str(tmp_path / "uvenv")
    result = _api.create(prefix=prefix, backend="uv", sources=[str(req)],
                         packages=["tomli-w"], verify_imports=["six", "tomli_w"])
    assert result.ok
    req.write_text("six\nidna\n", encoding="utf-8")
    again = _api.create(prefix=prefix, backend="uv", sources=[str(req)],
                        packages=["tomli-w"], verify_imports=["six", "tomli_w", "idna"])
    assert [s.id for s in again.steps] == ["install-pypi", "verify"]


@needs_network
@pytest.mark.slow
def test_micromamba_bootstrap_conda_and_pip(home, tmp_path, monkeypatch):
    for name in ("MAMBA_EXE", "CONDA_EXE"):
        monkeypatch.delenv(name, raising=False)
    if shutil.which("micromamba") or shutil.which("mamba") or shutil.which("conda"):
        pytest.skip("a Conda-family tool is installed; nothing to bootstrap")
    prefix = str(tmp_path / "mm")
    fields = {"prefix": prefix, "conda_packages": ["python=3.12", "tomli-w"],
              "packages": ["six"], "verify_imports": ["tomli_w", "six"],
              "policy": {"bootstrap": "if-missing"}}
    result = _api.create(approve=["bootstrap"], **fields)
    assert result.ok
    assert result.steps[0].id == "bootstrap-micromamba"
    assert (home / "tools" / "bin" / "micromamba").exists()
    version = subprocess.run([_python(result), "-c", "import sys; print(sys.version)"],
                             capture_output=True, text=True, check=True).stdout
    assert version.startswith("3.12")
    assert not _api.create(**{**fields, "policy": {"network": "offline"}}).changed
    code = _api.run(["python", "-c", "import tomli_w"], prefix=prefix)
    assert code == 0


@needs_network
@pytest.mark.slow
def test_micromamba_cross_build(home, tmp_path):
    prefix = tmp_path / "arm"
    result = _api.create(prefix=str(prefix), target="linux-aarch64",
                         conda_packages=["python=3.12"],
                         policy={"bootstrap": "if-missing"}, approve=["bootstrap"])
    assert result.ok
    assert result.plan.python_executable is None
    meta = list((prefix / "conda-meta").glob("python-3.12*.json"))
    assert json.loads(meta[0].read_text())["subdir"] == "linux-aarch64"


@needs_network
@pytest.mark.slow
@pytest.mark.skipif(shutil.which("pipenv") is None, reason="pipenv not on PATH")
def test_pipenv_lock_sync(home, tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "Pipfile").write_text('[packages]\nsix = "*"\n', encoding="utf-8")
    subprocess.run(["pipenv", "lock"], cwd=project, check=True, capture_output=True,
                   env={**os.environ, "PIPENV_VERBOSITY": "-1", "PIPENV_YES": "1"})
    prefix = str(tmp_path / "penv")
    result = _api.create(prefix=prefix, backend="pipenv",
                         sources=[str(project / "Pipfile")], verify_imports=["six"])
    assert result.ok
    assert [s.id for s in result.steps] == ["create", "pipenv-sync",
                                                          "verify"]


@needs_network
@pytest.mark.slow
def test_uv_bootstrap_real(tmp_path):
    from .._bootstrap import UV_REQUIREMENT  # noqa: PLC0415
    from .._core import _executor  # noqa: PLC0415

    _executor._bootstrap_uv(str(tmp_path / "tools"), UV_REQUIREMENT, sys.executable)
    uv = next((tmp_path / "tools" / "bin").iterdir())
    out = subprocess.run([str(uv), "--version"], capture_output=True, text=True,
                         check=True).stdout
    assert out.split()[1] == UV_REQUIREMENT.split("==")[1]


def _micromamba():
    return shutil.which("micromamba") or os.environ.get("MAMBA_EXE")


@pytest.mark.parametrize("tool", ["uv", "micromamba", "pip", "pipenv"])
def test_planned_flags_exist_in_the_installed_tool(tool, tmp_path):
    """Every ``--flag`` a plan passes is listed by the installed tool's ``--help``."""
    from .._core._planner import plan_request  # noqa: PLC0415
    from .._core._request import EnvironmentRequest, Policy, SourceRef  # noqa: PLC0415
    from ._helpers import fake_host  # noqa: PLC0415

    exe = {"uv": shutil.which("uv"), "micromamba": _micromamba(),
           "pip": sys.executable, "pipenv": shutil.which("pipenv")}[tool]
    if not exe:
        pytest.skip(f"{tool} not installed")
    if tool == "pip" and subprocess.run([exe, "-m", "pip", "--version"],
                                        capture_output=True, check=False).returncode:
        pytest.skip("this Python has no pip")
    host = fake_host(tmp_path, tools={} if tool == "pip" else {tool: exe})
    pipfile = tmp_path / "p" / "Pipfile"
    pipfile.parent.mkdir()
    pipfile.write_text('[packages]\nsix = "*"\n', encoding="utf-8")
    (pipfile.parent / "Pipfile.lock").write_text("{}", encoding="utf-8")
    fields = {
        "uv": {"packages": ("six",), "index_url": "https://pypi.org/simple"},
        "micromamba": {"conda_packages": ("python=3.12",), "packages": ("six",)},
        "pip": {"packages": ("six",), "backend": "venv", "find_links": (str(tmp_path),)},
        "pipenv": {"sources": (SourceRef(str(pipfile), "pipfile"),)},
    }[tool]
    requests = [EnvironmentRequest(prefix=str(tmp_path / "e"), policy=Policy(network=n),
                                   **fields) for n in ("online", "offline")]
    if tool == "micromamba":
        requests.append(EnvironmentRequest(prefix=str(tmp_path / "x"),
                                           target="linux-aarch64",
                                           conda_packages=("python=3.12",)))
    steps = [s for r in requests for s in plan_request(r, host=host).steps]
    if tool == "pip":
        commands = [s.argv for s in steps if s.argv[1:4] == ("-m", "pip", "install")]
    else:
        commands = [s.argv for s in steps if s.argv[:1] == (exe,)]
    assert commands
    for argv in commands:
        if tool == "pip":
            help_argv = [sys.executable, "-m", "pip", "install", "--help"]
        else:
            words = [a for a in argv[1:3] if not a.startswith("-")]
            help_argv = [exe, *(words[:2] if tool == "uv" else words[:1]), "--help"]
        text = subprocess.run(help_argv, capture_output=True, text=True,
                              check=False).stdout
        for flag in (a.split("=")[0] for a in argv if a.startswith("--")):
            assert flag in text, (tool, flag, help_argv)
