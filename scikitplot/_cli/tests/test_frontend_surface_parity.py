"""
Structural parity between the two frontends (slice S-45).

``test_cli_frontend_parity`` compares behaviour across a hand-written matrix of
invocations. That is the right shape for behaviour and the wrong shape for
coverage: a command added to the registry tomorrow is simply absent from the
matrix, and nothing says so.

These checks build each frontend's real surface -- an ``argparse`` parser and a
``click.Group`` -- from the shared registry and compare them to it and to each
other. Keeping two frontends is defensible only while they cannot drift apart
silently; this is what makes that true.

See Also
--------
scikitplot._cli._frontends._argparse.build_parser
scikitplot._cli._frontends._click.build_group
"""

import pytest

from .._frontends import _argparse, _click
from ..registry import BUILTIN_COMMANDS

EXPECTED = {command.name for command in BUILTIN_COMMANDS}


def _argparse_commands():
    """Return the subcommand names the argparse parser exposes."""
    parser = _argparse.build_parser()
    for action in parser._subparsers._group_actions:
        if hasattr(action, "choices"):
            return set(action.choices)
    raise AssertionError("no subparser action found")


def _click_commands():
    """Return the command names the click group exposes."""
    return set(_click.build_group().commands)


def test_argparse_exposes_every_registered_command():
    """A command in the registry that argparse cannot invoke is a drifted surface."""
    assert EXPECTED <= _argparse_commands()


def test_click_exposes_every_registered_command():
    """The same, for the opt-in frontend nobody exercises by default."""
    assert EXPECTED <= _click_commands()


def test_the_two_frontends_expose_the_same_commands():
    """Whatever either adds beyond the registry, both must add."""
    assert _argparse_commands() == _click_commands()


@pytest.mark.parametrize("spec", BUILTIN_COMMANDS, ids=lambda s: s.name)
def test_every_declared_option_reaches_both_frontends(spec):
    """Each flag the shared spec declares is accepted by both parsers."""
    parser = _argparse.build_parser()
    sub = next(a for a in parser._subparsers._group_actions if hasattr(a, "choices"))
    argparse_flags = {
        flag
        for action in sub.choices[spec.name]._actions
        for flag in action.option_strings
    }
    click_flags = {
        flag
        for param in _click.build_group().commands[spec.name].params
        for flag in getattr(param, "opts", ())
    }
    for prm in spec.params:
        for flag in prm.flags:
            assert flag in argparse_flags, f"argparse lacks {spec.name} {flag}"
            assert flag in click_flags, f"click lacks {spec.name} {flag}"


def test_the_behavioural_matrix_covers_every_command():
    """
    The hand-written matrix cannot silently lag the registry.

    Notes
    -----
    Comparing exit codes and machine output across a matrix is the right way to
    compare behaviour, but a command added tomorrow is absent from it and
    nothing complains. This is what tells whoever adds one.
    """
    from . import test_cli_frontend_parity as behavioural

    # A delegated command hands control to another program -- `mcp` starts a
    # server -- so it cannot appear in a matrix that compares captured output.
    # Exempting it explicitly is the point: an exemption someone wrote down is
    # different from a command nobody noticed was missing.
    delegated = {c.name for c in BUILTIN_COMMANDS if getattr(c, "delegate", None)}
    covered = {argv[0].replace("_", "-") for argv in behavioural.MATRIX}
    uncovered = sorted(EXPECTED - covered - delegated)
    assert uncovered == [], (
        f"commands with no behavioural parity case: {uncovered}. Add them to "
        "MATRIX in test_cli_frontend_parity.py, or mark them delegated."
    )


def test_the_delegated_exemption_is_narrow():
    """Only a command that hands off control may skip the behavioural matrix."""
    delegated = {c.name for c in BUILTIN_COMMANDS if getattr(c, "delegate", None)}
    assert delegated == {"mcp"}, (
        f"delegated commands are exempt from behavioural parity; {delegated} is "
        "more than expected, so the exemption has widened without review."
    )
