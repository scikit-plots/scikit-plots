"""TOML output renders valid, parseable TOML on stdout."""
import io
import sys

import pytest


def test_toml_roundtrips(monkeypatch):
    pytest.importorskip("tomli_w")
    tomllib = pytest.importorskip("tomllib")  # py311+; read-only parser
    from scikitplot._cli._frontends import _argparse
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    code = _argparse.run(["info", "--format", "toml"])
    assert code == 0
    parsed = tomllib.loads(buf.getvalue())
    assert "scikitplot" in parsed


def test_toml_missing_writer_is_actionable(monkeypatch):
    import builtins
    from scikitplot._cli.context import Context
    from scikitplot._cli.errors import CapabilityMissingError
    from scikitplot._cli import output

    real_import = builtins.__import__

    def guard(name, *a, **k):
        if name in {"tomli_w", "toml"}:
            raise ImportError(f"blocked {name}")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", guard)
    ctx = Context(fmt="toml")
    with pytest.raises(CapabilityMissingError):
        output.emit(ctx, {"a": 1})
