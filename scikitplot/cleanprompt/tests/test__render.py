"""Tests for :mod:`scikitplot.cleanprompt._render`."""

from __future__ import annotations

import io


from .. import (
    DEFAULT_POLICY,
    RestorationResult,
    TagStyle,
    highlight_placeholders,
    restoration_note,
    summary_table,
)
from .._render import ANSI, should_colorize


class _Tty(io.StringIO):
    """A stream that claims to be a terminal."""

    def isatty(self):
        return True


class _Pipe(io.StringIO):
    """A stream that is explicit about not being a terminal."""

    def isatty(self):
        return False


class TestShouldColorize:
    """The three signals, in order."""

    def test_explicit_true_wins(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        assert should_colorize(stream=_Pipe(), color=True) is True

    def test_explicit_false_wins(self):
        assert should_colorize(stream=_Tty(), color=False) is False

    def test_no_color_is_honoured(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "")
        assert should_colorize(stream=_Tty()) is False

    def test_terminal_enables_colour(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        assert should_colorize(stream=_Tty()) is True

    def test_pipe_disables_colour(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        assert should_colorize(stream=_Pipe()) is False

    def test_stream_without_isatty_is_not_a_terminal(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)

        class Bare:
            pass

        assert should_colorize(stream=Bare()) is False

    def test_stream_whose_isatty_raises_is_not_a_terminal(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)

        class Hostile:
            def isatty(self):
                raise OSError("detached")

        assert should_colorize(stream=Hostile()) is False


class TestHighlight:
    """Decoration never changes the underlying text."""

    def test_no_colour_returns_the_input_unchanged(self):
        text = "a [EMAIL-1] b"
        assert highlight_placeholders(text, color=False) is text

    def test_colour_wraps_only_the_placeholders(self):
        out = highlight_placeholders("a [EMAIL-1] b", color=True)
        assert out == "a {0}[EMAIL-1]{1} b".format(ANSI.GREEN, ANSI.RESET)

    def test_stripping_the_colour_recovers_the_text(self):
        import re

        text = "x [EMAIL-1] y [URL-2] z"
        painted = highlight_placeholders(text, color=True)
        assert re.sub(r"\033\[[0-9;]*m", "", painted) == text

    def test_a_foreign_grammar_is_not_highlighted(self):
        policy = DEFAULT_POLICY.evolve(tag_style=TagStyle(prefix="<<", suffix=">>"))
        assert highlight_placeholders("a [EMAIL-1] b", policy, color=True) == (
            "a [EMAIL-1] b"
        )

    def test_custom_grammar_is_highlighted(self):
        policy = DEFAULT_POLICY.evolve(
            tag_style=TagStyle(prefix="<<", suffix=">>", separator="_")
        )
        assert ANSI.GREEN in highlight_placeholders("a <<EMAIL_1>> b", policy, color=True)

    def test_text_without_placeholders_is_untouched(self):
        assert highlight_placeholders("plain text", color=True) == "plain text"


class TestSummaryTable:
    """The what-was-removed table."""

    def test_empty_result(self, redactor):
        assert summary_table(redactor.redact("plain"), color=False) == (
            "no sensitive values detected"
        )

    def test_headers_and_rows(self, redactor):
        table = summary_table(redactor.redact("a@x.com and b@x.com"), color=False)
        lines = table.splitlines()
        assert "placeholder" in lines[0]
        assert len(lines) == 4  # header, rule, two entries

    def test_values_are_hidden_by_default(self, redactor):
        assert "topsecret" not in summary_table(
            redactor.redact("mail topsecret@x.com"), color=False
        )

    def test_reveal_is_explicit(self, redactor):
        table = summary_table(
            redactor.redact("mail topsecret@x.com"), reveal=True, color=False
        )
        assert "topsecret@x.com" in table
        assert "original" in table.splitlines()[0]

    def test_columns_are_aligned(self, redactor):
        text = "a@x.com and averyveryverylongaddress@example.org"
        lines = summary_table(redactor.redact(text), color=False).splitlines()
        assert len({len(line) for line in lines}) == 1

    def test_colour_is_opt_in(self, redactor):
        result = redactor.redact("mail a@x.com")
        assert ANSI.CYAN not in summary_table(result, color=False)
        assert ANSI.CYAN in summary_table(result, color=True)


class TestRestorationNote:
    """The one-line restoration summary."""

    def test_counts_only(self):
        assert restoration_note(RestorationResult("x", ("[A-1]", "[A-2]"))) == (
            "restored 2 placeholder(s)"
        )

    def test_unknown_labels_are_named(self):
        note = restoration_note(RestorationResult("x", (), ("[A-9]",)))
        assert "1 unknown: [A-9]" in note

    def test_unused_entries_are_counted(self):
        note = restoration_note(RestorationResult("x", (), (), ("[A-2]",)))
        assert "1 vault entr(y/ies) unused" in note

    def test_never_names_a_secret(self, redactor):
        from .. import restore

        result = redactor.redact("mail topsecret@x.com")
        assert "topsecret" not in restoration_note(restore(result.text, result.vault))


class TestSeparation:
    """Presentation must not reach back into the data."""

    def test_rendering_does_not_mutate_the_result(self, redactor):
        result = redactor.redact("mail a@x.com")
        before = result.text
        summary_table(result, reveal=True, color=True)
        highlight_placeholders(result.text, color=True)
        assert result.text == before

    def test_render_module_imports_no_engine_state(self):
        """The renderer reads values; it never runs the pipeline."""
        import ast
        import pathlib

        path = pathlib.Path(__file__).resolve().parent.parent / "_render.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_level = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
        }
        assert "._engine" not in module_level
