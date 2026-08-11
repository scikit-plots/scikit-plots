"""Invariant 3.1/3.5: stdlib-only bootstrap; help imports no handlers."""
import builtins
import importlib
import io
import json
import sys

import pytest

_BLOCKED = {"click", "rich", "yaml"}


def _block_tier2(monkeypatch):
    real_import = builtins.__import__

    def guard(name, *a, **k):
        if name.split(".")[0] in _BLOCKED:
            raise ImportError(f"blocked {name}")
        return real_import(name, *a, **k)

    for mod in list(sys.modules):
        if mod.split(".")[0] in _BLOCKED:
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(builtins, "__import__", guard)


def test_import_cli_without_click(monkeypatch):
    _block_tier2(monkeypatch)
    for mod in list(sys.modules):
        if mod.startswith("scikitplot._cli"):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    importlib.import_module("scikitplot._cli")  # must not import click/rich/yaml


def test_registry_imports_no_handler(monkeypatch):
    for mod in list(sys.modules):
        if mod.startswith("scikitplot._cli._commands"):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    importlib.import_module("scikitplot._cli.registry")
    assert not any(
        m.startswith("scikitplot._cli._commands.") for m in sys.modules
    ), "registry import pulled in a handler module"


def test_argparse_runs_without_click(monkeypatch):
    _block_tier2(monkeypatch)
    from scikitplot._cli._frontends import _argparse
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    code = _argparse.run(["info", "--format", "json"])
    assert code == 0
    json.loads(buf.getvalue())
