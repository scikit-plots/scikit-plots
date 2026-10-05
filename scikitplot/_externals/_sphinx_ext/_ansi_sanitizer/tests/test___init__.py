"""
Tests for :mod:`_ansi_sanitizer`.

Notes
-----
**Developer notes.** Every case names the terminal sequence it carries. The
cases under ``TestNoVisibleTextIsLost`` failed against the one-pattern
expression this module started with: it treated every escape as a control
sequence, so a two-byte escape consumed the character after it and a string
sequence left most of its payload in the document.
"""

from __future__ import annotations

import types

import pytest
from docutils import nodes

from .. import _sanitize_latex_text, setup, strip_terminal_controls

ESC = "\x1b"


class TestControlSequences:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (ESC + "[31mred" + ESC + "[0m", "red"),
            (ESC + "[1;38;5;208mbold" + ESC + "[m", "bold"),
            (ESC + "[2K" + ESC + "[1Gline", "line"),
            (ESC + "[?25lhidden cursor" + ESC + "[?25h", "hidden cursor"),
            ("\x9b31mred", "red"),
        ],
    )
    def test_a_control_sequence_is_removed_whole(self, text, expected):
        assert strip_terminal_controls(text) == expected


class TestNoVisibleTextIsLost:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (ESC + "MHello", "Hello"),
            (ESC + "7Hello" + ESC + "8", "Hello"),
            (ESC + "cReset", "Reset"),
            (ESC + "=keypad", "keypad"),
        ],
    )
    def test_a_two_byte_escape_takes_nothing_after_it(self, text, expected):
        assert strip_terminal_controls(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (ESC + "]0;window title\x07body", "body"),
            (ESC + "]8;;https://example.com/a" + ESC + "\\link" + ESC + "]8;;" + ESC + "\\", "link"),
            (ESC + "Pq#0;2;0;0;0" + ESC + "\\after", "after"),
            (ESC + "_application command" + ESC + "\\after", "after"),
            ("\x9d0;title\x9cbody", "body"),
        ],
    )
    def test_a_string_sequence_is_removed_with_its_payload(self, text, expected):
        assert strip_terminal_controls(text) == expected

    def test_an_intermediate_byte_escape_is_removed_whole(self):
        assert strip_terminal_controls(ESC + "(Bplain") == "plain"

    def test_an_unterminated_string_loses_only_its_introducer(self):
        text = ESC + "]0;title never closed\nnext line"
        assert strip_terminal_controls(text) == "0;title never closed\nnext line"

    def test_a_terminator_on_a_later_line_does_not_close_the_string(self):
        text = ESC + "]0;open\nkept text\x07tail"
        assert "kept text" in strip_terminal_controls(text)


class TestWhatIsKept:
    def test_tab_newline_and_carriage_return_survive(self):
        assert strip_terminal_controls("a\tb\nc\r\nd") == "a\tb\nc\r\nd"

    def test_printable_text_is_unchanged(self):
        text = "x = [1, 2]  # 50% of ~/path \\ {ok} ünïcödé 日本語"
        assert strip_terminal_controls(text) == text

    def test_brackets_without_an_escape_are_text(self):
        assert strip_terminal_controls("[31m is not a colour here") == "[31m is not a colour here"

    @pytest.mark.parametrize("char", ["\x00", "\x07", "\x08", "\x0b", "\x0c", "\x7f", "\x85", "\x9f"])
    def test_a_bare_control_character_is_removed(self, char):
        assert strip_terminal_controls("a" + char + "b") == "ab"

    def test_an_escape_character_at_the_end_is_removed(self):
        assert strip_terminal_controls("tail" + ESC) == "tail"

    def test_empty_text(self):
        assert strip_terminal_controls("") == ""

    def test_the_result_is_stable(self):
        once = strip_terminal_controls(ESC + "[1m" + ESC + "]0;t\x07x" + ESC + "M")
        assert strip_terminal_controls(once) == once == "x"

    @pytest.mark.parametrize("value", [None, b"bytes", 3])
    def test_a_non_string_is_refused(self, value):
        with pytest.raises(TypeError, match="text must be str"):
            strip_terminal_controls(value)


def _doctree(*texts):
    document = nodes.section()
    for text in texts:
        document += nodes.paragraph("", "", nodes.Text(text))
    return document


def _app(fmt):
    return types.SimpleNamespace(builder=types.SimpleNamespace(format=fmt))


class TestDoctreeHandler:
    def test_latex_text_nodes_are_cleaned(self):
        tree = _doctree(ESC + "[31mred" + ESC + "[0m", "plain")
        _sanitize_latex_text(_app("latex"), tree, "index")
        assert [node.astext() for node in tree.findall(nodes.Text)] == ["red", "plain"]

    def test_an_untouched_node_keeps_its_identity(self):
        tree = _doctree("plain")
        before = list(tree.findall(nodes.Text))
        _sanitize_latex_text(_app("latex"), tree, "index")
        assert [id(node) for node in tree.findall(nodes.Text)] == [id(node) for node in before]

    @pytest.mark.parametrize("fmt", ["html", "text", "epub"])
    def test_other_builders_are_left_alone(self, fmt):
        raw = ESC + "[31mred"
        tree = _doctree(raw)
        _sanitize_latex_text(_app(fmt), tree, "index")
        assert [node.astext() for node in tree.findall(nodes.Text)] == [raw]


def test_setup_connects_one_handler_and_declares_parallel_safety():
    connected = []
    app = types.SimpleNamespace(connect=lambda event, handler: connected.append((event, handler)))
    assert setup(app) == {"parallel_read_safe": True, "parallel_write_safe": True}
    assert connected == [("doctree-resolved", _sanitize_latex_text)]
