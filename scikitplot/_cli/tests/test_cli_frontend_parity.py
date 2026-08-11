# scikitplot/_cli/tests/test_cli_frontend_parity.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Invariant 3.3: argparse and click agree on exit code and machine output.

Covers every built-in command, both output formats, back-compat aliases, and the
negatable flag (which exercises the Python 3.8-safe negation path).
"""
import io
import json
import sys

import pytest

MATRIX = [
    ["info"],
    ["info", "--format", "json"],
    ["sysinfo", "--format", "json"],
    ["show-config", "--format", "json"],
    ["show-config", "--format", "toml"],
    ["info", "--format", "toml"],
    ["show-versions", "--format", "json"],
    ["doctor"],
    ["doctor", "--mask-envs"],
    ["doctor", "--mask-envs", "--format", "json"],
    ["greet", "Allen"],
    ["greet", "--no-emoji", "Allen"],
    ["show_config", "--format", "json"],    # back-compat alias
    ["show_versions", "--format", "json"],  # back-compat alias
]


def _capture(run, argv, monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    try:
        code = run(argv)
    except SystemExit as exc:  # click raises SystemExit for non-zero handler codes
        code = exc.code or 0
    return code, buf.getvalue()


@pytest.mark.parametrize("argv", MATRIX)
def test_frontend_parity(argv, monkeypatch):
    pytest.importorskip("click")
    from scikitplot._cli._frontends import _argparse, _click
    code_a, out_a = _capture(_argparse.run, argv, monkeypatch)
    code_c, out_c = _capture(_click.run, argv, monkeypatch)
    assert code_a == code_c, (argv, code_a, code_c)
    # structured formats: compare parsed structure; text: compare bytes
    if "json" in argv:
        assert json.loads(out_a) == json.loads(out_c), argv
    elif "toml" in argv:
        import tomllib
        assert tomllib.loads(out_a) == tomllib.loads(out_c), argv
    else:
        assert out_a == out_c, (argv, repr(out_a), repr(out_c))
