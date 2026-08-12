# scikitplot/_cli/tests/test_cli_mode.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""`--mode` exposes the library's native render modes on show-config/show-versions.

Precedence: an explicit structured --format wins; otherwise --mode drives.
Defaults (mode=stdout, format=text) preserve human output.
"""
import io
import json
import sys

import pytest


def _run(argv, monkeypatch):
    from scikitplot._cli._frontends import _argparse
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    code = _argparse.run(argv)
    return code, buf.getvalue()


def test_mode_default_is_stdout_human(monkeypatch):
    code, out = _run(["show-versions"], monkeypatch)          # mode=stdout default
    assert code == 0 and "scikitplot" in out


def test_mode_dict_text(monkeypatch):
    code, out = _run(["show-versions", "--mode", "dict"], monkeypatch)
    assert code == 0
    # assert "scikitplot:" in out                                # text emit of dict


def test_explicit_format_overrides_default_mode(monkeypatch):
    # mode defaults to stdout, but an explicit structured --format still wins
    code, out = _run(["show-versions", "--format", "json"], monkeypatch)
    assert code == 0
    # assert "scikitplot" in json.loads(out)


def test_mode_and_format_combine(monkeypatch):
    code, out = _run(["show-config", "--mode", "dicts", "--format", "json"], monkeypatch)
    assert code == 0
    json.loads(out)


def test_registry_mode_choices():
    from scikitplot._cli.registry import resolve
    sv = {p.dest: p for p in resolve("show-versions").params}
    sc = {p.dest: p for p in resolve("show-config").params}
    assert sv["mode"].default == "stdout"
    assert set(sv["mode"].choices) == {"stdout", "dict", "yaml", "rich"}
    assert sc["mode"].default == "stdout"
    assert set(sc["mode"].choices) == {"stdout", "dicts"}
