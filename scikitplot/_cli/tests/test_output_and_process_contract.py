"""
Machine-format and process-error regressions (slices S-27, S-29, S-31).

Three contracts are pinned here. The structured formats must agree with each
other and with their own specification: ``json.dump`` defaults to
``allow_nan=True`` and emitted bare ``NaN``, which a strict reader rejects,
while ``json`` sorted keys and ``yaml`` preserved them, so one machine format
reordered the caller's data and the other did not. An unusable option value
must be rejected rather than silently resolved to a default. And the process
must end predictably when its output goes nowhere or when something
unexpected fails.

See Also
--------
scikitplot._cli.output.emit
scikitplot._cli.app.main
"""

import io
import json
import subprocess
import sys
import textwrap

import pytest

from .. import app, exit_codes, output
from .._frontends import _argparse
from ..context import Context


def _render(data, fmt):
    """Render ``data`` through the public emitter and return stdout."""
    ctx = Context(stdout=io.StringIO(), stderr=io.StringIO(), fmt=fmt)
    output.emit(ctx, data)
    return ctx.stdout.getvalue()


def _reject(value):
    raise AssertionError(f"non-standard JSON constant reached the reader: {value}")


# -- S-27: strict, mutually consistent machine formats ---------------------


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_are_refused_not_emitted(value):
    """A value JSON cannot express is reported, not written as a bare token."""
    with pytest.raises(ValueError) as excinfo:
        _render({"score": value}, "json")
    assert "score" in str(excinfo.value)


def test_json_output_parses_under_a_strict_reader():
    """Whatever is emitted is valid JSON by the specification, not by extension."""
    text = _render({"b": 2, "a": 1, "nested": {"z": 1, "y": 2}}, "json")
    assert json.loads(text, parse_constant=_reject) == {
        "b": 2, "a": 1, "nested": {"z": 1, "y": 2}
    }


def test_machine_formats_agree_on_ordering():
    """The caller's key order survives every structured format identically."""
    data = {"zebra": 1, "alpha": 2, "middle": 3}
    json_keys = list(json.loads(_render(data, "json")))
    yaml_text = _render(data, "yaml")
    yaml_keys = [line.split(":")[0] for line in yaml_text.splitlines() if ":" in line]
    assert json_keys == list(data)
    assert yaml_keys == list(data)


def test_nested_ordering_is_preserved_too():
    """Ordering is a property of the document, not only of its top level."""
    data = {"outer": {"zebra": 1, "alpha": 2}}
    assert list(json.loads(_render(data, "json"))["outer"]) == ["zebra", "alpha"]


# -- S-29: an unusable option value is rejected ---------------------------


@pytest.mark.parametrize("value", ["clikc", "bash", "argparse2", "CLICKY"])
def test_an_unknown_frontend_is_rejected(value):
    """A typo is a mistake to report, not a request for the default."""
    with pytest.raises(SystemExit) as excinfo:
        app._select_frontend({"SCIKITPLOT_CLI_FRONTEND": value})
    message = str(excinfo.value)
    assert value in message
    assert "argparse" in message and "click" in message


@pytest.mark.parametrize("value", ["argparse", "click", "CLICK", " argparse "])
def test_known_frontends_are_accepted(value):
    """Accepted spellings, including case and surrounding space, still work."""
    assert app._select_frontend({"SCIKITPLOT_CLI_FRONTEND": value}) in {
        "argparse", "click"
    }


def test_an_unset_frontend_uses_the_default():
    """Absence is not a mistake; only a value that cannot be honoured is."""
    assert app._select_frontend({}) in {"argparse", "click"}
    assert app._select_frontend({"SCIKITPLOT_CLI_FRONTEND": ""}) in {
        "argparse", "click"
    }


# -- S-31: the process-error contract -------------------------------------


def test_a_closed_reader_exits_cleanly():
    """
    ``scikitplot ... | head`` is a normal invocation, not a crash.

    Notes
    -----
    The handling lives in :func:`app.main`, so the probe has to go through
    ``main`` rather than through the emitter: catching the error in the test
    script itself would prove nothing about the product.
    """
    script = textwrap.dedent(
        """
        import sys
        from scikitplot._cli import app
        from scikitplot._cli._frontends import _argparse

        def flood(argv=None):
            for _ in range(20000):
                sys.stdout.write("x" * 200 + "\\n")
            sys.stdout.flush()
            return 0

        app._select_frontend = lambda env=None: "argparse"
        _argparse.run = flood
        code = app.main([])
        sys.stderr.write("exit=%d\\n" % code)
        """
    )
    producer = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    producer.stdout.close()
    _, stderr = producer.communicate()
    text = stderr.decode("utf-8", "replace")
    assert "Traceback" not in text, text
    assert "exit=0" in text, text


def test_an_unexpected_failure_maps_to_the_defined_exit_code(monkeypatch):
    """``exit_codes.SOFTWARE`` exists for this; it is now assigned."""
    def explode(argv=None):
        raise RuntimeError("internal invariant broken")

    monkeypatch.setattr(app, "_select_frontend", lambda env=None: "argparse")
    monkeypatch.setattr(_argparse, "run", explode)
    assert app.main([]) == exit_codes.SOFTWARE


def test_an_unexpected_failure_is_reported_on_stderr(monkeypatch, capsys):
    """The operator is told something failed, without a raw traceback on stdout."""
    def explode(argv=None):
        raise RuntimeError("internal invariant broken")

    monkeypatch.setattr(app, "_select_frontend", lambda env=None: "argparse")
    monkeypatch.setattr(_argparse, "run", explode)
    app.main([])
    captured = capsys.readouterr()
    assert "internal" in captured.err.lower() or "error" in captured.err.lower()
    assert captured.out == ""
