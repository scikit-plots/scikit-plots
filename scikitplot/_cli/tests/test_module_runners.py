"""Standalone `python -m` runners on the library modules expose --mode."""
import io
import json
import sys

import pytest


def _call_main(module_name, argv, monkeypatch):
    import importlib
    mod = importlib.import_module(module_name)
    assert hasattr(mod, "main"), f"{module_name} has no main()"
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    code = mod.main(argv)
    return code, buf.getvalue()


def test_show_versions_module_dict(monkeypatch):
    code, out = _call_main("scikitplot.utils._show_versions", ["-m", "dict"], monkeypatch)
    assert code == 0
    # assert "scikitplot" in json.loads(out)


def test_config_module_dicts(monkeypatch):
    code, out = _call_main("scikitplot.config.__config__", ["--mode", "dicts"], monkeypatch)
    assert code == 0
    json.loads(out)  # valid JSON on stdout


def test_module_default_mode_runs(monkeypatch):
    code, _ = _call_main("scikitplot.config.__config__", [], monkeypatch)
    assert code == 0
