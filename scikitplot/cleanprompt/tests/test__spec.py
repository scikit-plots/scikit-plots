"""Tests for :mod:`scikitplot.cleanprompt._spec`."""

from __future__ import annotations

import pytest

from .._spec import Command, Param


class TestParam:
    """Validation of the neutral parameter description."""

    def test_option_defaults(self):
        param = Param(dest="fmt", flags=("--format",))
        assert param.kind == "option"
        assert param.multiple is False

    def test_argument_must_not_declare_flags(self):
        with pytest.raises(ValueError, match="must not declare flags"):
            Param(dest="path", flags=("--path",), kind="argument")

    @pytest.mark.parametrize("kind", ["option", "flag"])
    def test_non_argument_needs_a_flag(self, kind):
        with pytest.raises(ValueError, match="at least one flag"):
            Param(dest="x", kind=kind)

    def test_flag_cannot_have_choices(self):
        with pytest.raises(ValueError, match="cannot declare choices"):
            Param(dest="x", flags=("--x",), kind="flag", choices=("a", "b"))

    def test_long_flags_use_hyphens(self):
        """Underscores would read wrong in a shell and split the two frontends."""
        with pytest.raises(ValueError, match="hyphens"):
            Param(dest="word_boundary", flags=("--word_boundary",), kind="flag")

    def test_short_flags_are_exempt(self):
        assert Param(dest="v", flags=("-v", "--verbose"), kind="flag") is not None

    def test_is_frozen(self):
        with pytest.raises(Exception):
            Param(dest="x", flags=("--x",)).dest = "y"


class TestCommand:
    """Validation of the neutral command description."""

    def test_minimal(self):
        command = Command(name="x", summary="s", handler="_cmd_x")
        assert command.params == ()
        assert command.help_text == "s"

    def test_description_overrides_summary_in_help(self):
        command = Command(name="x", summary="s", handler="h", description="longer")
        assert command.help_text == "longer"

    def test_duplicate_dest_is_refused(self):
        with pytest.raises(ValueError, match="twice"):
            Command(
                name="x",
                summary="s",
                handler="h",
                params=(
                    Param(dest="a", flags=("--a",)),
                    Param(dest="a", flags=("--b",)),
                ),
            )

    def test_defaults_covers_every_param(self):
        command = Command(
            name="x",
            summary="s",
            handler="h",
            params=(
                Param(dest="a", flags=("--a",), default=1),
                Param(dest="b", flags=("--b",), kind="flag", default=False),
            ),
        )
        assert command.defaults() == {"a": 1, "b": False}


class TestDeclaredSurface:
    """The real command table must satisfy its own rules."""

    def test_every_command_resolves_to_a_handler(self):
        from .. import _cli

        for command in _cli.COMMANDS:
            assert callable(getattr(_cli, command.handler))

    def test_names_and_aliases_are_unique(self):
        from .. import _cli

        seen: set[str] = set()
        for command in _cli.COMMANDS:
            for name in (command.name, *command.aliases):
                assert name not in seen, name
                seen.add(name)

    def test_every_command_has_a_summary(self):
        from .. import _cli

        for command in _cli.COMMANDS:
            assert command.summary.endswith(".")

    def test_detection_options_are_identical_everywhere(self):
        """One definition, so the dry run cannot diverge from the real run."""
        from .. import _cli

        shared = {"profile", "kinds", "hide", "allow", "ignore_case", "ner"}
        carriers = ("redact", "inspect", "scan", "cli", "flask")
        for command in _cli.COMMANDS:
            if command.name in carriers:
                assert shared <= {param.dest for param in command.params}, command.name
