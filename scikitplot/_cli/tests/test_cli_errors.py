"""Invariant 3.7: broken handlers/capabilities raise actionable errors."""
import pytest

from scikitplot._cli.errors import HandlerLoadError
from scikitplot._cli.loader import load_handler


def test_malformed_target():
    with pytest.raises(HandlerLoadError):
        load_handler("no-colon-here")


def test_missing_module():
    with pytest.raises(HandlerLoadError):
        load_handler("scikitplot._cli._commands.does_not_exist:run")


def test_missing_attribute():
    with pytest.raises(HandlerLoadError):
        load_handler("scikitplot._cli._commands.info:not_a_real_attr")


def test_unknown_command_exit_code():
    from scikitplot._cli.registry import resolve
    assert resolve("totally-unknown") is None


# ---------------------------------------------------------------------------
# A native command whose library part is not installed (partial distributions)
# ---------------------------------------------------------------------------


def _spec_calling(monkeypatch, handler):
    """Return a native command whose handler is ``handler``."""
    from .. import loader
    from .._spec import CommandSpec

    monkeypatch.setattr(loader, "load_handler", lambda target: handler)
    return loader, CommandSpec(name="x", summary="s", handler="pkg.mod:run")


def _context():
    from ..context import Context

    return Context(fmt="text", verbosity=0)


@pytest.mark.parametrize(
    ("missing", "command"),
    [
        ("scikitplot.utils", "pip install scikit-plots"),
        ("scikitplot.config.__config__", "pip install scikit-plots"),
        ("scikitplot.corpus", "pip install scikit-plots-corpus"),
    ],
)
def test_handler_needing_an_absent_part_is_an_unavailable_capability(
    monkeypatch, missing, command
):
    from ..errors import CapabilityMissingError

    def handler(ctx, **params):
        raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)

    loader, spec = _spec_calling(monkeypatch, handler)
    with pytest.raises(CapabilityMissingError) as excinfo:
        loader.dispatch(spec, {}, _context())
    assert excinfo.value.exit_code == 69
    assert missing in str(excinfo.value)
    assert excinfo.value.hint.endswith(command)
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)


def test_handler_missing_a_third_party_module_is_left_to_the_handler(monkeypatch):
    def handler(ctx, **params):
        raise ModuleNotFoundError("No module named 'yaml'", name="yaml")

    loader, spec = _spec_calling(monkeypatch, handler)
    with pytest.raises(ModuleNotFoundError) as excinfo:
        loader.dispatch(spec, {}, _context())
    assert excinfo.value.name == "yaml"


def test_handler_result_is_returned_as_an_exit_code(monkeypatch):
    loader, spec = _spec_calling(monkeypatch, lambda ctx, **params: 3)
    assert loader.dispatch(spec, {}, _context()) == 3


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ModuleNotFoundError("m", name="scikitplot"), "scikitplot"),
        (ModuleNotFoundError("m", name="scikitplot.utils._x"), "scikitplot.utils._x"),
        (ModuleNotFoundError("m", name="scikitplotx"), None),  # another project
        (ModuleNotFoundError("m", name="numpy"), None),
        (ModuleNotFoundError("m"), None),  # no module name recorded
        (ImportError("cannot import name", name="scikitplot.mcp"), None),
    ],
)
def test_missing_part_classification(exc, expected):
    from ..loader import _missing_part

    assert _missing_part(exc) == expected


def test_show_config_reads_from_the_config_package_not_the_root(monkeypatch):
    """The root re-exports ``show_config`` only in the full distribution."""
    import sys
    import types

    from .._commands import show_config as command

    calls = []
    stub = types.ModuleType("scikitplot.config")
    stub.show_config = lambda mode="stdout": calls.append(mode)
    monkeypatch.setitem(sys.modules, "scikitplot.config", stub)
    assert command.run(_context(), mode="stdout", fmt="text") == 0
    assert calls == ["stdout"]
