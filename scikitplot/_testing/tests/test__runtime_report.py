# scikitplot/_testing/tests/test__runtime_report.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of ``scikitplot/_testing/_runtime_report.py``."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

from .. import _runtime_report as rr


class TestHubCacheDir:
    def test_hf_hub_cache_wins(self, tmp_path):
        env = {"HF_HUB_CACHE": str(tmp_path / "a"), "HF_HOME": str(tmp_path / "b")}
        assert rr.hub_cache_dir(env) == tmp_path / "a"

    def test_the_legacy_name_comes_next(self, tmp_path):
        env = {
            "HUGGINGFACE_HUB_CACHE": str(tmp_path / "a"),
            "HF_HOME": str(tmp_path / "b"),
        }
        assert rr.hub_cache_dir(env) == tmp_path / "a"

    def test_hf_home_holds_the_hub(self, tmp_path):
        assert rr.hub_cache_dir({"HF_HOME": str(tmp_path)}) == tmp_path / "hub"

    def test_xdg_then_home(self, tmp_path):
        env = {"XDG_CACHE_HOME": str(tmp_path)}
        assert rr.hub_cache_dir(env) == tmp_path / "huggingface" / "hub"
        default = Path("~").expanduser() / ".cache" / "huggingface" / "hub"
        assert rr.hub_cache_dir({}) == default

    def test_agrees_with_huggingface_hub(self, tmp_path):
        """The rule is huggingface_hub's own (Rule 64): ask it in a child."""
        if importlib.util.find_spec("huggingface_hub") is None:
            pytest.skip("huggingface_hub is not installed")
        cases = [
            {"HF_HOME": str(tmp_path / "home")},
            {"XDG_CACHE_HOME": str(tmp_path / "xdg")},
            {"HF_HUB_CACHE": str(tmp_path / "hub"), "HF_HOME": str(tmp_path / "x")},
        ]
        for env in cases:
            child_env = {
                key: value
                for key, value in __import__("os").environ.items()
                if key
                not in (
                    "HF_HOME",
                    "HF_HUB_CACHE",
                    "HUGGINGFACE_HUB_CACHE",
                    "XDG_CACHE_HOME",
                )
            }
            child_env.update(env)
            done = subprocess.run(  # noqa: S603 - our interpreter, our code
                [
                    sys.executable,
                    "-c",
                    "from huggingface_hub import constants; print(constants.HF_HUB_CACHE)",
                ],
                capture_output=True,
                text=True,
                env=child_env,
                check=False,
                timeout=120,
            )
            assert done.returncode == 0, done.stderr
            assert Path(done.stdout.strip()) == rr.hub_cache_dir(env), env


class TestCachedModels:
    def test_models_are_listed_as_owner_and_name(self, tmp_path):
        (tmp_path / "models--sentence-transformers--all-MiniLM-L6-v2").mkdir()
        (tmp_path / "models--Systran--faster-whisper-base").mkdir()
        (tmp_path / "datasets--x--y").mkdir()
        (tmp_path / "models--not-a-directory").write_text("x", encoding="utf-8")
        assert rr.cached_models(tmp_path) == [
            "Systran/faster-whisper-base",
            "sentence-transformers/all-MiniLM-L6-v2",
        ]

    def test_a_missing_cache_is_empty(self, tmp_path):
        assert rr.cached_models(tmp_path / "absent") == []


class TestDescribeRuntime:
    def test_two_lines_naming_cache_offline_and_session(self, tmp_path):
        cache = tmp_path / "hub"
        cache.mkdir()
        (cache / "models--a--b").mkdir()
        stack, models = rr.describe_runtime(
            {
                "HF_HUB_CACHE": str(cache),
                "HF_HUB_OFFLINE": "1",
                rr.TENSORFLOW_SESSION_ENV: "1",
            }
        )
        assert stack.startswith("ML runtime: ")
        assert "1 model(s): a/b" in models
        assert "Hugging Face offline: yes" in models
        assert f"TensorFlow session ({rr.TENSORFLOW_SESSION_ENV}): yes" in models

    @pytest.mark.parametrize(
        ("env", "offline"),
        [
            ({}, "no"),
            ({"HF_HUB_OFFLINE": "0"}, "no"),
            ({"HF_HUB_OFFLINE": "true"}, "yes"),
            ({"TRANSFORMERS_OFFLINE": "YES"}, "yes"),
        ],
    )
    def test_offline_is_read_as_huggingface_hub_reads_it(self, tmp_path, env, offline):
        lines = rr.describe_runtime({"HF_HUB_CACHE": str(tmp_path), **env})
        assert f"Hugging Face offline: {offline}" in lines[1]

    def test_a_long_model_list_is_shortened(self, tmp_path):
        for index in range(10):
            (tmp_path / f"models--o--m{index}").mkdir()
        line = rr.describe_runtime({"HF_HUB_CACHE": str(tmp_path)})[1]
        assert "10 model(s): " in line
        assert line.count("o/m") == rr._SHOWN_MODELS  # noqa: SLF001
        assert ", ..." in line

    def test_versions_come_from_metadata_without_importing(self):
        before = set(sys.modules)
        rr.describe_runtime({})
        assert not {"tensorflow", "torch", "triton"} & (set(sys.modules) - before)


class TestTensorflowConflict:
    @staticmethod
    def spec(installed):
        return lambda name: object() if name in installed else None

    def test_tensorflow_loaded_triton_installed_not_loaded(self):
        assert rr.tensorflow_conflict({"tensorflow": 1}, {}, self.spec({"triton"}))

    @pytest.mark.parametrize(
        ("modules", "environ", "installed"),
        [
            ({}, {}, {"triton"}),
            ({"tensorflow": 1, "triton._C.libtriton": 1}, {}, {"triton"}),
            ({"tensorflow": 1}, {rr.TENSORFLOW_SESSION_ENV: "1"}, {"triton"}),
            ({"tensorflow": 1}, {}, set()),
        ],
    )
    def test_no_conflict(self, modules, environ, installed):
        assert not rr.tensorflow_conflict(modules, environ, self.spec(installed))


class _Reporter:
    def __init__(self):
        self.lines = []
        self.sections = []

    def write_line(self, text, **markup):
        self.lines.append(text)

    def section(self, title):
        self.sections.append(title)


class _Config:
    def __init__(self, reporter):
        self._plugins = {"terminalreporter": reporter}
        self.pluginmanager = self

    def get_plugin(self, name):
        return self._plugins.get(name)

    def register(self, plugin, name):
        self._plugins[name] = plugin


class TestRuntimeReport:
    def report(self, monkeypatch, environ=None):
        reporter = _Reporter()
        plugin = rr.register(_Config(reporter), environ or {})
        monkeypatch.setattr(rr, "tensorflow_conflict", lambda environ=None: True)
        return plugin, reporter

    def test_the_test_that_loaded_tensorflow_is_named(self, monkeypatch):
        plugin, reporter = self.report(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorflow", types.ModuleType("tensorflow"))
        plugin.pytest_runtest_logreport(
            types.SimpleNamespace(when="call", nodeid="t::a")
        )
        assert plugin.loaded_by is None  # only after the teardown report
        plugin.pytest_runtest_logreport(
            types.SimpleNamespace(when="teardown", nodeid="t::a")
        )
        assert plugin.loaded_by == "t::a"
        assert any("by t::a" in line and "SIGSEGV" in line for line in reporter.lines)
        plugin.pytest_runtest_logreport(
            types.SimpleNamespace(when="teardown", nodeid="t::b")
        )
        assert plugin.loaded_by == "t::a"  # the first one stays named
        summary = _Reporter()
        plugin.pytest_terminal_summary(summary)
        assert summary.sections == ["native runtime"]
        assert "by t::a" in summary.lines[0]

    def test_collection_is_named_too(self, monkeypatch):
        plugin, reporter = self.report(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorflow", types.ModuleType("tensorflow"))
        plugin.pytest_collection_finish(session=None)
        assert plugin.loaded_by == "collection"
        assert "during collection" in reporter.lines[-1]

    def test_nothing_is_said_without_tensorflow(self, monkeypatch):
        plugin, reporter = self.report(monkeypatch)
        monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
        plugin.pytest_runtest_logreport(
            types.SimpleNamespace(when="teardown", nodeid="t::a")
        )
        summary = _Reporter()
        plugin.pytest_terminal_summary(summary)
        assert (plugin.loaded_by, reporter.lines, summary.sections) == (None, [], [])

    def test_register_is_idempotent(self):
        config = _Config(_Reporter())
        assert rr.register(config) is rr.register(config)

    def test_the_header_is_describe_runtime(self, tmp_path):
        plugin = rr.RuntimeReport({"HF_HUB_CACHE": str(tmp_path)})
        assert plugin.pytest_report_header(config=None) == rr.describe_runtime(
            {"HF_HUB_CACHE": str(tmp_path)}
        )
