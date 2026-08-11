# scikitplot/_cli/tests/test_cli_verbosity.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Verbosity (-v/-q) is accepted at root and on subcommands, in any position,
combines additively, and resolves identically across both frontends.
"""  # noqa: D205, D400

import io
import logging as _logging
import sys

import pytest

from scikitplot._cli import logging as cli_logging


def _verbosity_seen(run, argv, monkeypatch):
    """Return the net verbosity that reaches the Context for ``argv``."""
    import scikitplot._cli.context as ctx_mod
    seen = {}
    orig = ctx_mod.Context

    class Probe(orig):
        def __init__(self, *a, **k):
            seen["v"] = k.get("verbosity", 0)
            super().__init__(*a, **k)

    from scikitplot._cli._frontends import _argparse, _click
    monkeypatch.setattr(_argparse, "Context", Probe)
    monkeypatch.setattr(_click, "Context", Probe)
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    try:
        run(argv)
    except SystemExit:
        pass
    return seen.get("v")


CASES = [
    (["doctor"], 0),
    (["doctor", "-v"], 1),
    (["doctor", "-vv"], 2),
    (["-v", "doctor"], 1),
    (["-v", "doctor", "-v"], 2),      # combines before + after the command
    (["-vv", "doctor", "-q"], 1),     # -q cancels one -v
    (["doctor", "-qq"], -2),
    (["-vvv", "info"], 3),
    (["show-config", "-v"], 1),
]


@pytest.mark.parametrize("argv,expected", CASES)
def test_argparse_verbosity(argv, expected, monkeypatch):
    from scikitplot._cli._frontends import _argparse
    assert _verbosity_seen(_argparse.run, argv, monkeypatch) == expected


@pytest.mark.parametrize("argv,expected", CASES)
def test_click_verbosity_parity(argv, expected, monkeypatch):
    pytest.importorskip("click")
    from scikitplot._cli._frontends import _click
    assert _verbosity_seen(_click.run, argv, monkeypatch) == expected


def test_resolve_and_level():
    assert cli_logging.resolve(3, 1) == 2
    assert cli_logging.level_for(0) == _logging.WARNING
    assert cli_logging.level_for(1) == _logging.INFO
    assert cli_logging.level_for(2) == _logging.DEBUG
    assert cli_logging.level_for(5) == _logging.DEBUG      # clamped
    assert cli_logging.level_for(-5) == _logging.CRITICAL  # clamped


def test_vv_sets_debug_level_and_stays_on_stderr(monkeypatch, capsys):
    import logging as pylogging
    from scikitplot._cli._frontends import _argparse
    _argparse.run(["doctor", "-vv"])
    # It is set to the absolute name of the module as imported.
    assert pylogging.getLogger("scikitplot").level == pylogging.DEBUG
    captured = capsys.readouterr()
    assert "DEBUG" in captured.err       # diagnostics -> stderr
    assert "DEBUG" not in captured.out    # never stdout


def test_default_level_is_warning(monkeypatch):
    import io
    import logging as pylogging
    from scikitplot._cli._frontends import _argparse
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    _argparse.run(["doctor"])
    # It is set to the absolute name of the module as imported.
    assert pylogging.getLogger("scikitplot").level == pylogging.WARNING


def test_debug_does_not_contaminate_json(monkeypatch):
    import io
    import json
    from scikitplot._cli._frontends import _argparse
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    code = _argparse.run(["doctor", "-vv", "--format", "json"])
    assert code == 0
    json.loads(buf.getvalue())  # stdout stays clean JSON even at debug verbosity
