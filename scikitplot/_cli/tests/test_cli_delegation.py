"""
Delegated (pass-through) commands forward all args to a submodule.

Covers run_delegate (function form), exit-code propagation, --help passthrough,
missing-submodule -> actionable error, and argparse<->click routing parity.
"""
import io
import json
import sys

import pytest

from scikitplot._cli import loader
from scikitplot._cli.errors import CapabilityMissingError, HandlerLoadError

_FAKE = "scikitplot._cli.tests._fake_delegate:main"


def _cap(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    return buf


def test_run_delegate_forwards_and_returns_zero(monkeypatch):
    buf = _cap(monkeypatch)
    code = loader.run_delegate(_FAKE, ["--name", "allen"])
    assert code == 0
    assert json.loads(buf.getvalue())["name"] == "allen"


def test_run_delegate_propagates_exit_code(monkeypatch):
    _cap(monkeypatch)
    assert loader.run_delegate(_FAKE, ["--fail"]) == 3


def test_run_delegate_help_is_exit_zero(monkeypatch):
    _cap(monkeypatch)
    # argparse --help raises SystemExit(0); run_delegate converts it to 0.
    assert loader.run_delegate(_FAKE, ["--help"]) == 0


def test_run_delegate_missing_module_is_actionable():
    with pytest.raises(CapabilityMissingError) as excinfo:
        loader.run_delegate("scikitplot.__no_such_sub__.x:main", ["--help"],
                            install_hint="pip install scikit-plots[x]")
    assert "pip install scikit-plots[x]" in (excinfo.value.hint or "")


def test_run_delegate_malformed_colon_target():
    # A colon target with an empty module or attr is malformed.
    with pytest.raises(HandlerLoadError):
        loader.run_delegate("mod:", [])
    with pytest.raises(HandlerLoadError):
        loader.run_delegate(":main", [])


def test_run_delegate_bare_module_missing_is_actionable():
    # A colon-less target is the bare-module ("python -m") form; a missing one
    # is an actionable capability error, not a traceback.
    with pytest.raises(CapabilityMissingError):
        loader.run_delegate("scikitplot.__no_such_bare_module__", [])


def test_registry_mcp_is_delegated():
    from scikitplot._cli.registry import resolve
    spec = resolve("mcp")
    assert spec is not None
    assert spec.delegate == "scikitplot.mcp.__main__:main"
    assert spec.handler == ""          # delegated: no native handler
    assert spec.params == ()           # delegated: no declared params
    assert spec.install_hint


@pytest.mark.parametrize("argv", [
    ["mcp", "--transport", "stdio", "--print-effective-config"],
    ["mcp", "--help"],
    ["mcp"],
])
def test_frontends_forward_identical_argv(argv, monkeypatch):
    """Both frontends strip the command name and forward the rest verbatim."""
    pytest.importorskip("click")
    seen = {}

    def fake_run_delegate(target, forwarded, install_hint=None):
        seen.setdefault("targets", []).append(target)
        seen.setdefault("argvs", []).append(list(forwarded))
        return 0

    monkeypatch.setattr(loader, "run_delegate", fake_run_delegate)
    from scikitplot._cli._frontends import _argparse, _click
    assert _argparse.run(argv) == 0
    assert _click.run(argv) == 0
    # identical target and forwarded argv across both frontends
    assert seen["targets"][0] == seen["targets"][1] == "scikitplot.mcp.__main__:main"
    assert seen["argvs"][0] == seen["argvs"][1] == argv[1:]


def test_global_verbosity_before_delegated_command(monkeypatch):
    seen = {}
    monkeypatch.setattr(loader, "run_delegate",
                        lambda t, a, install_hint=None: seen.update(argv=list(a)) or 0)
    from scikitplot._cli._frontends import _argparse
    assert _argparse.run(["-v", "mcp", "--print-effective-config"]) == 0
    assert seen["argv"] == ["--print-effective-config"]  # -v consumed as global


# ---------------------------------------------------------------------------
# A delegated part that is not installed (partial distributions)
# ---------------------------------------------------------------------------
#
# With a partial distribution the submodule a command delegates to may simply
# not be installed. The command's own ``install_hint`` describes its optional
# dependencies and is the wrong advice then; the distribution map names the
# package that ships the part.


def _refuse_import(monkeypatch, missing):
    """Make importing through the loader fail as if ``missing`` were not installed."""
    from .. import loader as _loader

    def import_module(name, *args, **kwargs):
        raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)

    monkeypatch.setattr(_loader.importlib, "import_module", import_module)
    return _loader


@pytest.mark.parametrize(
    ("target", "missing", "distribution"),
    [
        ("scikitplot.mcp.__main__:main", "scikitplot.mcp", "scikit-plots-mcp"),
        ("scikitplot.mcp.__main__:main", "scikitplot.mcp.__main__", "scikit-plots-mcp"),
        ("scikitplot.cleanprompt.__main__:main", "scikitplot.cleanprompt",
         "scikit-plots-cleanprompt"),
    ],
)
def test_missing_part_names_its_distribution(monkeypatch, target, missing, distribution):
    _loader = _refuse_import(monkeypatch, missing)
    with pytest.raises(CapabilityMissingError) as excinfo:
        _loader.run_delegate(target, [], install_hint="pip install scikit-plots[extra]")
    hint = excinfo.value.hint
    assert f"pip install {distribution}" in hint
    # The command's own hint is about optional dependencies of a part that is
    # present; it must not be shown for a part that is absent.
    assert "scikit-plots[extra]" not in hint
    assert excinfo.value.exit_code == 69


def test_missing_third_party_dependency_keeps_the_commands_hint(monkeypatch):
    _loader = _refuse_import(monkeypatch, "pydantic")
    with pytest.raises(CapabilityMissingError) as excinfo:
        _loader.run_delegate("scikitplot.mcp.__main__:main", [],
                             install_hint="pip install scikit-plots[mcp]")
    assert excinfo.value.hint == "pip install scikit-plots[mcp]"


def test_missing_part_no_partial_distribution_ships_keeps_the_commands_hint(monkeypatch):
    # ``scikitplot.utils`` belongs to the full distribution only; the map has
    # nothing more specific to say than the command does.
    _loader = _refuse_import(monkeypatch, "scikitplot.utils")
    with pytest.raises(CapabilityMissingError) as excinfo:
        _loader.run_delegate("scikitplot.utils.__main__:main", [], install_hint="the hint")
    assert excinfo.value.hint == "the hint"


def test_import_error_that_is_not_a_missing_module_keeps_the_commands_hint(monkeypatch):
    from .. import loader as _loader

    def import_module(name, *args, **kwargs):
        raise ImportError("cannot import name 'x' from 'scikitplot.mcp'", name="scikitplot.mcp")

    monkeypatch.setattr(_loader.importlib, "import_module", import_module)
    with pytest.raises(CapabilityMissingError) as excinfo:
        _loader.run_delegate("scikitplot.mcp.__main__:main", [], install_hint="the hint")
    assert excinfo.value.hint == "the hint"


def test_bare_module_form_names_the_distribution_too(monkeypatch):
    from .. import loader as _loader

    def run_module(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'scikitplot.mcp'", name="scikitplot.mcp")

    monkeypatch.setattr(_loader.runpy, "run_module", run_module)
    old_argv = list(sys.argv)
    with pytest.raises(CapabilityMissingError) as excinfo:
        _loader.run_delegate("scikitplot.mcp", ["--help"], install_hint="the hint")
    assert "pip install scikit-plots-mcp" in excinfo.value.hint
    assert sys.argv == old_argv  # restored even though the run failed
