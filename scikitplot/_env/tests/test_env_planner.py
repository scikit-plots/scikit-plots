# scikitplot/_env/tests/test_env_planner.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""The planner: backend rules, targets, policies, existing environments."""

from __future__ import annotations

import json
from urllib.parse import urlsplit

import pytest

from .._backends import BACKENDS, get_backend
from .._bootstrap import micromamba_archive
from .._core import _planner
from .._core._errors import (
    BackendUnavailableError,
    ConfigError,
    ConversionRefusedError,
    NetworkForbiddenError,
    PlanningError,
    TargetUnsupportedError,
)
from .._core._planner import plan_request
from .._core._receipt import RECEIPT_NAME, RECEIPT_SCHEMA
from .._core._request import EnvironmentRequest, Policy, SourceRef
from ._helpers import fake_host, make_venv_like, write

UV = "/opt/bin/uv"
MM = "/opt/bin/micromamba"


def _plan(host, operation="ensure", **fields):
    return plan_request(EnvironmentRequest(**fields), operation=operation, host=host)


def _ids(plan):
    return [s.id for s in plan.steps]


class TestBackendRules:
    def test_pypi_prefers_uv(self, tmp_path):
        plan = _plan(fake_host(tmp_path, tools={"uv": UV}), name="a", packages=["six"])
        assert plan.backend == "uv"
        assert "uv found" in plan.backend_reason
        assert plan.steps[0].argv[:4] == (UV, "venv", str(tmp_path / "envs" / "a"), "--seed")
        assert plan.steps[1].argv[:3] == (UV, "pip", "install")

    def test_pypi_falls_back_to_venv(self, tmp_path):
        plan = _plan(fake_host(tmp_path), name="a", packages=["six"])
        assert plan.backend == "venv"
        assert plan.steps[0].argv == ("/usr/bin/python3", "-m", "venv",
                                      str(tmp_path / "envs" / "a"))

    def test_other_python_needs_a_python_installer(self, tmp_path):
        host = fake_host(tmp_path)
        with pytest.raises(BackendUnavailableError, match="differs") as info:
            _plan(host, name="a", python="3.10")
        assert "bootstrap" in info.value.hint
        plan = _plan(host, name="a", python="3.10",
                     policy=Policy(bootstrap="if-missing"))
        assert _ids(plan)[:2] == ["bootstrap-uv", "create"]
        assert plan.steps[0].approval == "bootstrap"
        assert plan.steps[1].argv[0] == str(tmp_path / "tools" / "bin" / "uv")

    def test_other_python_uses_micromamba_when_uv_cannot_be_bootstrapped(self, tmp_path):
        plan = _plan(fake_host(tmp_path, tools={"micromamba": MM}), name="a",
                     python="3.10")
        assert plan.backend == "micromamba"

    def test_conda_packages_choose_conda_family(self, tmp_path):
        host = fake_host(tmp_path, tools={"uv": UV, "conda": "/c/conda", "micromamba": MM})
        plan = _plan(host, name="a", conda_packages=["numpy"])
        assert plan.backend == "micromamba"
        argv = plan.steps[0].argv
        assert argv[:2] == (MM, "create")
        assert list(argv[argv.index("-c"):argv.index("-c") + 2]) == ["-c", "conda-forge"]
        assert "numpy" in argv
        assert "--platform" not in argv

    def test_conda_bootstrap(self, tmp_path):
        host = fake_host(tmp_path)
        with pytest.raises(BackendUnavailableError, match="bootstrap policy"):
            _plan(host, name="a", conda_packages=["numpy"])
        plan = _plan(host, name="a", conda_packages=["numpy"],
                     policy=Policy(bootstrap="if-missing"))
        step = plan.steps[0]
        assert step.action == "bootstrap_micromamba"
        assert step.network
        assert step.params_dict["subdir"] == "linux-64"
        url = step.params_dict["url"]
        assert url == micromamba_archive("linux-64")["url"]
        assert urlsplit(url).hostname == "conda.anaconda.org"
        assert url in plan.render()  # the exact URL is shown before approval

    def test_unpinned_platform_cannot_bootstrap(self, tmp_path):
        host = fake_host(tmp_path, machine="ppc64le")
        with pytest.raises(BackendUnavailableError, match="no micromamba archive"):
            _plan(host, name="a", conda_packages=["x"],
                  policy=Policy(bootstrap="if-missing"))

    def test_named_backend(self, tmp_path):
        host = fake_host(tmp_path, tools={"uv": UV})
        assert _plan(host, name="a", backend="venv").backend == "venv"
        with pytest.raises(BackendUnavailableError, match="pipenv"):
            _plan(host, name="a", backend="pipenv")
        with pytest.raises(ConfigError, match="unknown backend"):
            _plan(host, name="a", backend="poetry")

    def test_pipenv_rule(self, tmp_path):
        project = tmp_path / "proj"
        write(project / "Pipfile", '[packages]\nsix = "*"\n')
        source = SourceRef(str(project / "Pipfile"), "pipfile", groups=("dev",))
        host = fake_host(tmp_path, tools={"uv": UV, "pipenv": "/p/pipenv"})
        assert _plan(host, name="a", sources=(source,)).backend == "uv"  # no lock
        with pytest.raises(BackendUnavailableError, match="Pipfile.lock"):
            _plan(host, name="a", sources=(source,), backend="pipenv")
        write(project / "Pipfile.lock", "{}")
        plan = _plan(host, name="a", sources=(source,), packages=["tomli-w"])
        assert plan.backend == "pipenv"
        sync = plan.steps[1]
        assert sync.argv == ("/p/pipenv", "sync", "--dev")
        env = dict(sync.env)
        assert env["VIRTUAL_ENV"] == plan.prefix
        assert env["PIPENV_PIPFILE"] == str(project / "Pipfile")
        assert plan.steps[2].argv[-1] == "tomli-w"  # non-Pipfile package by pip

    def test_registry_is_complete(self):
        assert set(BACKENDS) == {"uv", "venv", "pipenv", "micromamba", "mamba", "conda"}
        for name in BACKENDS:
            assert get_backend(name).name == name


class TestConversion:
    def test_exact_refuses(self, tmp_path):
        with pytest.raises(ConversionRefusedError, match="numpy"):
            _plan(fake_host(tmp_path, tools={"uv": UV}), name="a", backend="uv",
                  conda_packages=["numpy>=2"])

    def test_approximate_maps_names(self, tmp_path):
        plan = _plan(fake_host(tmp_path, tools={"uv": UV}), name="a", backend="uv",
                     conda_packages=["numpy>=2", "python=3.12"],
                     policy=Policy(conversion="allow-approximate"))
        assert plan.steps[1].argv[-1] == "numpy"
        assert any("approximate conversion" in n for n in plan.notes)


class TestTargets:
    def test_cross_build(self, tmp_path):
        host = fake_host(tmp_path, tools={"micromamba": MM, "uv": UV})
        plan = _plan(host, prefix=str(tmp_path / "arm"), target="linux-aarch64",
                     conda_packages=["python=3.12"], verify_imports=["json"])
        argv = plan.steps[0].argv
        assert "--platform" in argv
        assert "linux-aarch64" in argv
        assert "--no-pyc" in argv
        assert "--relocate-prefix" not in argv
        assert plan.python_executable is None
        assert any("verify_imports skipped" in n for n in plan.notes)
        with pytest.raises(ConversionRefusedError, match="cross-build"):
            _plan(host, prefix=str(tmp_path / "arm"), target="linux-aarch64",
                  packages=["six"])

    def test_own_subdir_is_host(self, tmp_path):
        plan = _plan(fake_host(tmp_path, tools={"uv": UV}), name="a", target="linux-64")
        assert plan.target == "host"

    def test_browser_build(self, tmp_path):
        host = fake_host(tmp_path, tools={"micromamba": MM})
        plan = _plan(host, prefix=str(tmp_path / "w"), target="emscripten-wasm32",
                     conda_packages=["xeus-python"], packages=["py2vega"])
        create, pip = plan.steps
        assert create.argv[create.argv.index("--relocate-prefix") + 1] == ""
        assert "https://prefix.dev/emscripten-forge-4x" in create.argv
        assert "pip" not in create.argv
        assert pip.action == "xeus_pip_layer"
        assert pip.params_dict["requirements"] == ("py2vega",)
        with pytest.raises(BackendUnavailableError, match="cannot be built with"):
            _plan(host, prefix=str(tmp_path / "w"), target="emscripten-wasm32",
                  backend="uv")

    def test_pyodide_is_plan_only(self, tmp_path):
        plan = _plan(fake_host(tmp_path), name="a", target="pyodide",
                     packages=["six"], index_url="https://pypi.example/simple")
        assert not plan.executable
        assert plan.steps == ()
        script = dict(plan.artifacts)["micropip.py"]
        assert "await micropip.install(['six'], index_urls=" in script
        with pytest.raises(ConversionRefusedError, match="local path"):
            _plan(fake_host(tmp_path), name="a", target="pyodide",
                  packages=[str(tmp_path / "pkg")])

    @pytest.mark.parametrize("target", ["android", "ios"])
    def test_mobile_refused_with_reason(self, tmp_path, target):
        with pytest.raises(TargetUnsupportedError, match="embeds its own Python") as info:
            _plan(fake_host(tmp_path), name="a", target=target)
        assert "pyodide" in info.value.hint

    def test_unknown_target(self, tmp_path):
        with pytest.raises(ConfigError, match="unknown target"):
            _plan(fake_host(tmp_path), name="a", target="tablet")

    def test_browser_host_cannot_execute(self, tmp_path):
        host = fake_host(tmp_path, system="emscripten", machine="wasm32")
        with pytest.raises(PlanningError, match="cannot start processes"):
            _plan(host, name="a")
        assert not _plan(host, name="a", target="pyodide").executable


class TestPolicies:
    def test_offline_flags(self, tmp_path):
        plan = _plan(fake_host(tmp_path, tools={"uv": UV, "micromamba": MM}), name="a",
                     packages=["six"], policy=Policy(network="offline"))
        assert "--offline" in plan.steps[0].argv
        assert "--no-python-downloads" in \
            plan.steps[0].argv
        assert "--offline" in plan.steps[1].argv
        assert not plan.uses_network
        venv = _plan(fake_host(tmp_path), name="b", packages=["six"],
                     policy=Policy(network="offline"))
        assert "--no-index" in venv.steps[1].argv

    def test_offline_bootstrap_needs_cached_archive(self, tmp_path):
        host = fake_host(tmp_path)
        policy = Policy(network="offline", bootstrap="if-missing")
        with pytest.raises(BackendUnavailableError, match="not cached"):
            _plan(host, name="a", conda_packages=["x"], policy=policy)
        write(tmp_path / "tools" / "cache" / "linux-64-micromamba-2.9.0-0.tar.bz2", "x")
        plan = _plan(host, name="a", conda_packages=["x"], policy=policy)
        assert plan.steps[0].action == "bootstrap_micromamba"
        assert not plan.steps[0].network

    def test_network_forbidden_guard(self, tmp_path, monkeypatch):
        host = fake_host(tmp_path)
        real = _planner._bootstrap_step

        def online_step(tool, host, offline):
            step, exe = real(tool, host, offline)
            return step.__class__(**{**step.__dict__, "network": True}), exe

        monkeypatch.setattr(_planner, "_bootstrap_step", online_step)
        write(tmp_path / "tools" / "cache" / "linux-64-micromamba-2.9.0-0.tar.bz2", "x")
        with pytest.raises(NetworkForbiddenError):
            _plan(host, name="a", conda_packages=["x"],
                  policy=Policy(network="offline", bootstrap="if-missing"))

    def test_python_conflict(self, tmp_path):
        pytest.importorskip("yaml")
        env = write(tmp_path / "environment.yml", "dependencies:\n  - python=3.11\n")
        with pytest.raises(ConfigError, match="conflicts"):
            _plan(fake_host(tmp_path, tools={"micromamba": MM}), name="a",
                  python="3.12", sources=(SourceRef(str(env), "environment"),))


class TestExisting:
    def _receipt(self, prefix, plan, **over):
        data = {"schema": RECEIPT_SCHEMA, "state_digest": plan.state_digest,
                "backend": plan.backend, "target": plan.target,
                "request": {"python": None}, **over}
        write(prefix / RECEIPT_NAME, json.dumps(data))

    def test_up_to_date_and_update(self, tmp_path):
        host = fake_host(tmp_path, tools={"uv": UV})
        prefix = tmp_path / "envs" / "a"
        fresh = _plan(host, name="a", packages=["six"])
        make_venv_like(prefix)
        self._receipt(prefix, fresh)
        again = _plan(host, name="a", packages=["six"],
                      policy=Policy(interaction="never", network="offline"))
        assert again.up_to_date
        assert again.steps == ()
        changed = _plan(host, name="a", packages=["six", "tomli-w"])
        assert not changed.up_to_date
        assert _ids(changed) == ["install-pypi"]
        assert changed.state_digest != fresh.state_digest

    def test_dependency_file_edit_breaks_up_to_date(self, tmp_path):
        host = fake_host(tmp_path, tools={"uv": UV})
        req = write(tmp_path / "r.txt", "six\n")
        source = (SourceRef(str(req), "requirements"),)
        first = _plan(host, name="a", sources=source)
        make_venv_like(tmp_path / "envs" / "a")
        self._receipt(tmp_path / "envs" / "a", first)
        assert _plan(host, name="a", sources=source).up_to_date
        write(req, "six\ntomli-w\n")
        assert not _plan(host, name="a", sources=source).up_to_date

    def test_built_with_other_backend_or_python(self, tmp_path):
        host = fake_host(tmp_path, tools={"uv": UV})
        prefix = tmp_path / "envs" / "a"
        plan = _plan(host, name="a")
        make_venv_like(prefix)
        self._receipt(prefix, plan, backend="micromamba")
        with pytest.raises(PlanningError, match="backend='micromamba'") as info:
            _plan(host, name="a")
        assert "recreate" in info.value.hint
        recreate = _plan(host, "recreate", name="a")
        assert _ids(recreate) == ["remove", "create"]
        assert recreate.approvals == ("remove",)

    def test_broken_environment_is_recreated(self, tmp_path):
        prefix = tmp_path / "envs" / "a"
        write(prefix / "pyvenv.cfg", "")
        plan = _plan(fake_host(tmp_path, tools={"uv": UV}), name="a")
        assert _ids(plan) == ["remove", "create"]
        assert any("broken" in n for n in plan.notes)

    def test_unknown_directory_refused(self, tmp_path):
        write(tmp_path / "data" / "notes.txt", "keep me")
        with pytest.raises(PlanningError, match="not an environment"):
            _plan(fake_host(tmp_path, tools={"uv": UV}), prefix=str(tmp_path / "data"))

    def test_remove(self, tmp_path):
        host = fake_host(tmp_path)
        missing = _plan(host, "remove", name="a", sources=(
            SourceRef(str(tmp_path / "gone.txt"), "requirements"),))
        assert missing.steps == ()
        assert "nothing to remove" in missing.notes[0]
        make_venv_like(tmp_path / "envs" / "a")
        plan = _plan(host, "remove", name="a")
        assert plan.steps[0].action == "remove_tree"
        assert plan.approvals == ("remove",)

    def test_windows_layout(self, tmp_path):
        host = fake_host(tmp_path, system="win32", machine="AMD64", tools={"uv": "uv.exe"})
        plan = _plan(host, name="a", packages=["six"])
        assert plan.python_executable.endswith("Scripts/python.exe") or \
            plan.python_executable.endswith("Scripts\\python.exe")
        conda = _plan(fake_host(tmp_path, system="win32", machine="AMD64",
                                tools={"micromamba": "mm.exe"}),
                      name="b", conda_packages=["x"])
        assert conda.python_executable.endswith("python.exe")
        assert "Scripts" not in conda.python_executable


def test_plan_document_round_trip(tmp_path):
    plan = _plan(fake_host(tmp_path, tools={"uv": UV}), name="a", packages=["six"])
    data = json.loads(json.dumps(plan.to_dict()))
    assert data["schema"] == "scikitplot.env.plan/v1"
    assert data["digest"] == plan.digest
    assert data["state_digest"] == plan.state_digest
    assert data["approvals"] == []
    assert data["uses_network"]
    assert "plan" in plan.render()
    assert plan.render().count("$ ") == 2


class TestReviewFindings:
    """Regression tests of the independent review (one per finding)."""

    def test_plain_python_install_is_not_an_environment(self, tmp_path):
        system = tmp_path / "usr"
        write(system / "bin" / "python", "")
        with pytest.raises(PlanningError, match="not an environment"):
            _plan(fake_host(tmp_path), prefix=str(system), backend="venv",
                  packages=["six"])
        assert _plan(fake_host(tmp_path), "remove", prefix=str(system)).steps == ()

    def test_python_change_from_a_dependency_file_is_not_applied_in_place(
        self, tmp_path
    ):
        pytest.importorskip("yaml")
        host = fake_host(tmp_path, tools={"micromamba": MM})
        env = write(tmp_path / "environment.yml", "dependencies:\n  - python=3.11\n")
        source = (SourceRef(str(env), "environment"),)
        first = _plan(host, prefix=str(tmp_path / "e"), sources=source)
        assert first.python == "3.11"
        write(tmp_path / "e" / "bin" / "python", "")
        write(tmp_path / "e" / RECEIPT_NAME, json.dumps({
            "schema": RECEIPT_SCHEMA, "state_digest": first.state_digest,
            "backend": "micromamba", "target": "host", "python": "3.11"}))
        (tmp_path / "e" / "conda-meta").mkdir()
        write(env, "dependencies:\n  - python=3.12\n")
        with pytest.raises(PlanningError, match="python='3.11'"):
            _plan(host, prefix=str(tmp_path / "e"), sources=source)

    def test_dropped_dependency_recreates(self, tmp_path):
        host = fake_host(tmp_path, tools={"uv": UV})
        both = _plan(host, name="a", packages=["six", "attrs"])
        make_venv_like(tmp_path / "envs" / "a")
        write(tmp_path / "envs" / "a" / RECEIPT_NAME, json.dumps({
            "schema": RECEIPT_SCHEMA, "state_digest": both.state_digest,
            "backend": "uv", "target": "host", "python": None,
            "dependencies": [dict(d) for d in both.dependencies]}))
        plan = _plan(host, name="a", packages=["six"])
        assert _ids(plan)[:2] == ["remove", "create"]
        assert any("attrs" in n for n in plan.notes)

    def test_micromamba_ignores_user_rc_channels(self, tmp_path):
        plan = _plan(fake_host(tmp_path, tools={"micromamba": MM}), name="a",
                     conda_packages=["x"])
        assert "--override-channels" in plan.steps[0].argv

    def test_conda_batch_file_refused(self, tmp_path):
        host = fake_host(tmp_path, tools={"conda": r"C:\c\condabin\conda.bat"})
        with pytest.raises(BackendUnavailableError, match="batch file"):
            _plan(host, name="a", conda_packages=["x"], backend="conda")

    def test_browser_pip_layer_gets_index_and_offline(self, tmp_path):
        host = fake_host(tmp_path, tools={"micromamba": MM})
        write(tmp_path / "tools" / "x", "")
        plan = _plan(host, prefix=str(tmp_path / "w"), target="emscripten-wasm32",
                     conda_packages=["python"], packages=["py2vega"],
                     index_url="https://corp.example/simple",
                     policy=Policy(network="offline"))
        args = plan.steps[1].params_dict["index_args"]
        assert "--no-index" in args
        assert "https://corp.example/simple" in args

    def test_included_requirements_never_inline(self, tmp_path):
        write(tmp_path / "base.txt", "numpy==2.0.0 --hash=sha256:" + "a" * 64 + "\n")
        req = write(tmp_path / "requirements.txt", "-r base.txt\n")
        plan = _plan(fake_host(tmp_path, tools={"uv": UV}), name="a",
                     sources=(SourceRef(str(req), "requirements"),))
        argv = plan.steps[1].argv
        assert not any("numpy" in a for a in argv)
        assert argv[-2:] == ("-r", str(req))

    def test_pipfile_lock_is_an_input_and_python_gates_rule(self, tmp_path):
        project = tmp_path / "p"
        write(project / "Pipfile", '[packages]\nsix = "*"\n')
        write(project / "Pipfile.lock", '{"v": 1}')
        source = (SourceRef(str(project / "Pipfile"), "pipfile"),)
        host = fake_host(tmp_path, tools={"uv": UV, "pipenv": "/p/pipenv"})
        first = _plan(host, name="a", sources=source)
        assert first.backend == "pipenv"
        assert str(project / "Pipfile.lock") in [p for p, _ in first.inputs]
        write(project / "Pipfile.lock", '{"v": 2}')
        assert _plan(host, name="a", sources=source).state_digest != first.state_digest
        write(project / "Pipfile", '[packages]\nsix = "*"\n[requires]\n'
                                   'python_version = "3.10"\n')
        assert _plan(host, name="a", sources=source).backend == "uv"

    def test_pipfile_private_index_is_not_silently_dropped(self, tmp_path):
        pipfile = write(tmp_path / "Pipfile", (
            '[[source]]\nurl = "https://corp.example/simple"\nname = "corp"\n'
            '[packages]\ncorp-internal = "*"\n'))
        source = (SourceRef(str(pipfile), "pipfile"),)
        host = fake_host(tmp_path, tools={"uv": UV})
        with pytest.raises(PlanningError, match="public PyPI"):
            _plan(host, name="a", sources=source)
        plan = _plan(host, name="a", sources=source,
                     index_url="https://corp.example/simple")
        assert "https://corp.example/simple" in plan.steps[1].argv

    def test_credentials_never_reach_plan_documents(self, tmp_path):
        req = write(tmp_path / "r.txt",
                    "tool @ git+https://ghp_TOKEN123@github.com/o/tool\n")
        plan = _plan(fake_host(tmp_path, tools={"uv": UV}), name="a",
                     sources=(SourceRef(str(req), "requirements"),))
        text = json.dumps(plan.to_dict()) + plan.render()
        assert "ghp_TOKEN123" not in text
        for field, value in (
            ("packages", ["corp @ https://deploy:S3CRET@h/p.whl"]),
            ("packages", ["tool @ git+https://ghp_X@github.com/o/r"]),
            ("find_links", ["https://u:pw@host/wheels"]),
            ("channels", ["https://conda.anaconda.org/t/tk-123/corp"]),
        ):
            with pytest.raises(ConfigError, match="credential"):
                EnvironmentRequest(name="a", **{field: value})
