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
