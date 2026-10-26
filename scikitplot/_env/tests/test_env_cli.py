# scikitplot/_env/tests/test_env_cli.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""The ``scikitplot env`` command line and the public functions behind it."""

from __future__ import annotations

import io
import json
import subprocess
import sys

import pytest

from ..._cli import registry
from .. import _api, _cli
from .._core._errors import ConfigError
from ._helpers import fake_host, make_venv_like, write


def _main(*argv):
    out, err = io.StringIO(), io.StringIO()
    code = _cli.main(list(argv), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def host(tmp_path, monkeypatch):
    """Make every API call plan for a fake host with uv."""
    fake = fake_host(tmp_path, tools={"uv": "/opt/uv"})
    real = _api.plan_request
    monkeypatch.setattr(_api, "detect_host", lambda: fake)
    monkeypatch.setattr(
        _api, "plan_request",
        lambda request, operation="ensure", host=None: real(
            request, operation=operation, host=fake))
    return fake


def test_registered_as_delegated_command():
    spec = registry.BY_NAME["env"]
    assert spec.delegate == f"{_cli.__name__}:main"


def test_doctor_json(host):
    code, out, _ = _main("doctor", "--format", "json")
    data = json.loads(out)
    assert code == 0
    assert data["backends"]["uv"]["available"]
    assert "android" in data["targets"]
    assert "linux-64" in data["bootstrap_platforms"]


def test_plan_text_and_json(host, tmp_path):
    code, out, _ = _main("plan", "-n", "demo", "--package", "six")
    assert code == 0
    assert "backend  uv" in out
    assert "/opt/uv pip install" in out
    code, out, _ = _main("--format", "json", "plan", "-n", "demo", "--package", "six")
    assert json.loads(out)["backend"] == "uv"
    code, out, _ = _main("plan", "-n", "demo", "--format", "json", "-v")
    assert json.loads(out)["steps"][0]["id"] == "create"


def test_pyodide_artifact_printed(host):
    code, out, _ = _main("plan", "-n", "b", "--target", "pyodide", "--package", "six")
    assert code == 0
    assert "--- micropip.py" in out
    assert "await micropip.install" in out


@pytest.mark.parametrize(("argv", "code", "error"), [
    (("plan", "-n", "a", "--target", "android"), 3, "target_unsupported"),
    (("plan", "-n", "a", "--backend", "conda"), 3, "backend_unavailable"),
    (("plan", "-n", "a b"), 2, "config_invalid"),
    (("plan", "-n", "a", "--group", "dev"), 2, "config_invalid"),
    (("remove", "-n", "built"), 4, "approval_required"),
])
def test_errors_have_codes_and_exit_status(host, tmp_path, argv, code, error):
    make_venv_like(tmp_path / "envs" / "built")
    status, out, err = _main(*argv)
    assert status == code
    assert f"error[{error}]" in err
    status, out, _ = _main("--format", "json", *argv)
    data = json.loads(out)
    assert status == code
    assert data["ok"] is False
    assert data["error"]["code"] == error


def test_create_and_remove_with_approval(host, tmp_path, monkeypatch):
    def runner(argv, *, cwd, env):
        if list(argv[1:2]) == ["venv"]:
            make_venv_like(tmp_path / "envs" / "c")
        return subprocess.CompletedProcess(list(argv), 0, stdout="")

    real = _api.execute
    monkeypatch.setattr(_api, "execute",
                        lambda plan, **kw: real(plan, **{**kw, "runner": runner}))
    events = tmp_path / "events.jsonl"
    code, out, err = _main("create", "-n", "c", "--events", str(events))
    assert code == 0, err
    assert "done with uv" in out
    assert events.read_text().count("\n") >= 4
    code, out, _ = _main("create", "-n", "c")
    assert code == 0
    assert "up to date" in out
    code, out, _ = _main("remove", "-n", "c", "--yes")
    assert code == 0
    assert not (tmp_path / "envs" / "c").exists()


def test_validate_and_config_discovery(host, tmp_path, monkeypatch):
    write(tmp_path / "proj" / "scikitplot-env.toml", 'name = "fromfile"\n')
    monkeypatch.chdir(tmp_path / "proj")
    code, out, _ = _main("--format", "json", "validate")
    assert code == 0
    assert json.loads(out)["name"] == "fromfile"
    monkeypatch.chdir(tmp_path)
    code, _, err = _main("validate")
    assert code == 2
    assert "no environment given" in err


def test_run_uses_the_environment(host, tmp_path):
    prefix = tmp_path / "envs" / "r"
    make_venv_like(prefix)
    calls = []
    code = _api.run(["python", "-V"], name="r", host=host,
                    environ={"PATH": "/usr/bin", "PYTHONPATH": "/x"},
                    runner=lambda argv, env: calls.append((argv, env)) or 7)
    argv, env = calls[0]
    assert code == 7
    assert argv == [str(prefix / "bin" / "python"), "-V"]  # resolved in the env
    assert env["PATH"].split(":")[0] == str(prefix / "bin")
    assert env["VIRTUAL_ENV"] == str(prefix)
    assert "PYTHONPATH" not in env
    with pytest.raises(ConfigError, match="no command"):
        _api.run([], name="r", host=host)


def test_shell_init_print(host):
    code, out, _ = _main("shell-init", "bash")
    assert code == 0
    assert out.startswith("# >>> scikitplot env >>>")


def test_batch_dry_run(host, tmp_path):
    lines = [{"schema": "scikitplot.env.request/v1", "id": i,
              "request": {"name": i, "packages": ["six"]}} for i in ("a", "b")]
    lines.append({"schema": "scikitplot.env.request/v1", "id": "bad",
                  "request": {"name": "c", "target": "ios"}})
    path = write(tmp_path / "b.jsonl", "\n".join(json.dumps(x) for x in lines))
    code, out, _ = _main("--format", "json", "batch", str(path), "--dry-run",
                         "--keep-going")
    data = json.loads(out)
    assert code == 1
    assert [r["ok"] for r in data["results"]] == [True, True, False]
    assert data["results"][2]["error"]["code"] == "target_unsupported"


def test_python_m_entry_point():
    done = subprocess.run([sys.executable, "-m", _cli.__name__.rsplit(".", 1)[0],
                           "--help"], capture_output=True, text=True, check=False)
    assert done.returncode == 0
    assert "shell-init" in done.stdout


def test_api_request_forms(host, tmp_path):
    from .._core._request import EnvironmentRequest  # noqa: PLC0415

    request = EnvironmentRequest(name="x")
    assert _api.plan(request).prefix == str(tmp_path / "envs" / "x")
    with pytest.raises(ConfigError, match="not both"):
        _api.plan(request, name="y")


def test_import_is_light():
    package = _cli.__name__.rsplit(".", 1)[0]
    code = (f"import sys, {package} as e\n"
            "bad = [m for m in sys.modules if m.startswith(e.__name__ + '.')"
            " or m.split('.')[0] in ('yaml', 'tomli', 'jupyterlite_core', 'empack')]\n"
            "print(bad)\n"
            "e.plan, e.EnvironmentRequest\n"
            "assert 'yaml' not in sys.modules\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().splitlines()[-1] == "[]"



def test_run_missing_command_is_a_clean_error(host, tmp_path):
    make_venv_like(tmp_path / "envs" / "r")
    code, out, err = _main("run", "-n", "r", "--", "no-such-command-xyz")
    assert code == 127
    assert "error[command_not_found]" in err
    code, out, _ = _main("--format", "json", "run", "-n", "r", "--", "no-such-xyz")
    assert json.loads(out)["error"]["code"] == "command_not_found"


def test_root_prefix_refused(host):
    code, _, err = _main("remove", "-p", "/", "--yes")
    assert code == 2
    assert "file-system root" in err
