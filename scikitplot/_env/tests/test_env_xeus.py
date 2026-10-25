# scikitplot/_env/tests/test_env_xeus.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""The Xeus lane: pure-Python layer for browser prefixes, JupyterLite add-on fixes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from .._core._errors import ExecutionError
from .._xeus import _pip
from ._helpers import write


def _prefix(tmp_path, version="3.13.1"):
    prefix = tmp_path / "wasm"
    write(prefix / "conda-meta" / f"python-{version}-h0.json",
          json.dumps({"name": "python", "version": version}))
    return prefix


def _dist(staging, name, files, *, purelib=True):
    info = staging / f"{name}-1.0.dist-info"
    rows = []
    for rel, body in files.items():
        local = rel[6:] if rel.startswith("../../") else rel
        if ".." not in Path(local).parts:  # never write outside the staging directory
            write(staging / local, body)
        rows.append(f"{rel},,")
    write(info / "WHEEL", f"Wheel-Version: 1.0\nRoot-Is-Purelib: {str(purelib).lower()}\n")
    rows += [f"{name}-1.0.dist-info/WHEEL,,", f"{name}-1.0.dist-info/RECORD,,"]
    write(info / "RECORD", "\n".join(rows) + "\n")


def _runner(build):
    def run(argv, **kwargs):
        staging = Path(argv[argv.index("--target") + 1])
        build(staging)
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n")

    return run


class TestPurePython:
    def test_version_of_prefix(self, tmp_path):
        assert _pip.prefix_python_version(_prefix(tmp_path)) == "3.13"
        with pytest.raises(ExecutionError, match="0 python package records"):
            _pip.prefix_python_version(tmp_path / "none")

    def test_installs_into_site_packages_and_prefix(self, tmp_path):
        prefix = _prefix(tmp_path)
        seen = {}

        def build(staging):
            _dist(staging, "py2vega", {"py2vega/__init__.py": "x = 1\n",
                                       "../../bin/py2vega": "#!python\n"})

        run = _runner(build)

        def spy(argv, **kwargs):
            seen["argv"] = argv
            return run(argv, **kwargs)

        names = _pip.install_pure_python(prefix, ["py2vega", "-e", "/src/pkg"], run=spy)
        site = prefix / "lib" / "python3.13" / "site-packages"
        assert names == ["py2vega-1.0.dist-info"]
        assert (site / "py2vega" / "__init__.py").read_text() == "x = 1\n"
        assert (prefix / "bin" / "py2vega").exists()
        record = (site / "py2vega-1.0.dist-info" / "RECORD").read_text()
        assert "../../../bin/py2vega" in record
        argv = seen["argv"]
        assert argv[argv.index("--python-version") + 1] == "3.13"
        assert "--no-deps" in argv
        assert "--no-compile" in argv
        assert "-e" not in argv
        assert "/src/pkg" in argv

    @pytest.mark.parametrize(("files", "purelib", "match"), [
        ({"pkg/_speed.so": "\x7fELF"}, True, "compiled file"),
        ({"pkg/x.py": ""}, False, "not a pure-Python"),
        ({"../../../etc/passwd": "x"}, True, "leaves the prefix"),
        ({"pkg\\..\\..\\..\\x.py": "x"}, True, "leaves the prefix"),
    ])
    def test_refused_before_anything_is_copied(self, tmp_path, files, purelib, match):
        prefix = _prefix(tmp_path)

        def build(staging):
            _dist(staging, "good", {"good/__init__.py": ""})
            _dist(staging, "bad", files, purelib=purelib)

        with pytest.raises(ExecutionError, match=match):
            _pip.install_pure_python(prefix, ["good", "bad"], run=_runner(build))
        assert not (prefix / "lib").exists()

    def test_pip_failure(self, tmp_path):
        def run(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 1, stdout="no matching dist\n")

        with pytest.raises(ExecutionError, match="no matching dist") as info:
            _pip.install_pure_python(_prefix(tmp_path), ["nope"], run=run)
        assert info.value.returncode == 1


class TestAddonApprovals:
    def test_approval_comes_from_the_invocation_only(self):
        pytest.importorskip("jupyterlite_core")
        pytest.importorskip("empack")
        from .._xeus import _addon  # noqa: PLC0415

        assert "bootstrap" not in _addon.XeusAddon.class_trait_names()
        assert _addon._approvals_from_invocation({}) == set()
        assert _addon._approvals_from_invocation(
            {"SKPLT_ENV_APPROVE": "bootstrap"}) == {"bootstrap"}
        with pytest.raises(ValueError, match="unknown"):
            _addon._approvals_from_invocation({"SKPLT_ENV_APPROVE": "everything"})


class TestAddon:
    """Regression tests of the defects of the original ``add_on.py``."""

    @pytest.fixture
    def addon_module(self):
        pytest.importorskip("jupyterlite_core")
        pytest.importorskip("empack")
        from .._xeus import _addon  # noqa: PLC0415

        return _addon

    def _addon(self, module, tmp_path):
        addon = module.XeusAddon.__new__(module.XeusAddon)
        addon.cwd_name = str(tmp_path / "cwd")
        addon.specs, addon.channels = {}, {}
        return addon

    def test_env_004_path_prefix(self, addon_module, tmp_path):
        prefix = tmp_path / "p"
        write(prefix / "conda-meta" / "history", (
            "# cmd: micromamba create -c https://ch/one --channel=https://ch/two\n"
            "# update specs: ['python', 'xeus-python']\n"))
        addon = self._addon(addon_module, tmp_path)
        assert addon.get_environment_specs(prefix) == ["python", "xeus-python"]
        assert addon.get_environment_channels(prefix) == ["https://ch/one",
                                                          "https://ch/two"]
        assert addon.get_environment_specs(str(prefix)) == ["python", "xeus-python"]

    def test_malformed_history_raises(self, addon_module, tmp_path):
        prefix = tmp_path / "p"
        write(prefix / "conda-meta" / "history", "# update specs: [oops\n")
        with pytest.raises(ValueError, match="history:1"):
            self._addon(addon_module, tmp_path).get_environment_specs(prefix)

    def test_env_005_002_create_prefix_uses_the_builder(self, addon_module, tmp_path):
        env_file = write(tmp_path / "kernel-env.yml", "dependencies:\n  - xeus-python\n")
        addon = self._addon(addon_module, tmp_path)
        addon.default_channels = []
        fake_plan = mock.Mock(render=mock.Mock(return_value="plan"))
        with mock.patch.object(addon_module, "plan_request", autospec=True,
                               return_value=fake_plan) as planned, \
                mock.patch.object(addon_module, "execute", autospec=True) as executed, \
                mock.patch.object(addon_module.XeusAddon, "log",
                                  new_callable=mock.PropertyMock):
            name, _prefix = addon.create_prefix(env_file)
        assert name == "kernel-env"  # no `name:` key: the file stem (ENV-005)
        request = planned.call_args.args[0]
        assert request.target == "emscripten-wasm32"
        assert request.channels[0] == "https://prefix.dev/emscripten-forge-4x"
        assert request.policy.interaction == "never"
        assert executed.call_args.args[0] is fake_plan
        assert addon.specs["kernel-env"] == ["xeus-python"]

    def test_environment_name_cannot_escape(self, addon_module, tmp_path):
        env_file = write(tmp_path / "e.yml", "name: ../../../../tmp/x\ndependencies: []\n")
        addon = self._addon(addon_module, tmp_path)
        addon.default_channels = []
        with pytest.raises(ValueError, match="names a directory"):
            addon.create_prefix(env_file)

    def test_env_006_config_download_has_timeout_and_status(self, addon_module):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.read.return_value = b"packages: {}\n"
        with mock.patch.object(addon_module.urllib.request, "urlopen", autospec=True,
                               return_value=response) as urlopen:
            assert addon_module._download_text("https://x/c.yaml") == b"packages: {}\n"
        assert urlopen.call_args.kwargs["timeout"] == 60.0
        response.status = 500
        with mock.patch.object(addon_module.urllib.request, "urlopen", autospec=True,
                               return_value=response):
            with pytest.raises(RuntimeError, match="500"):
                addon_module._download_text("https://x/c.yaml")



def test_index_args_reach_pip(tmp_path):
    seen = {}

    def run(argv, **kwargs):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, stdout="")

    _pip.install_pure_python(_prefix(tmp_path), ["six"], run=run,
                             index_args=["--index-url", "https://corp/simple",
                                         "--no-index"])
    assert "--no-index" in seen["argv"]
    assert "https://corp/simple" in seen["argv"]
