"""Invariant 3.6: --format json yields clean JSON on stdout only."""
import io
import json
import sys


def test_json_is_clean(monkeypatch):
    from scikitplot._cli._frontends import _argparse
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    code = _argparse.run(["info", "--format", "json"])
    assert code == 0
    payload = json.loads(buf.getvalue())  # no diagnostics on stdout
    assert "scikitplot" in payload
