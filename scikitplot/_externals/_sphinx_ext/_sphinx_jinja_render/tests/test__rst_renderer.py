"""
Tests for ``_rst_renderer``: rendering, and how failures are reported.

Notes
-----
**Developer notes.** Log records are read through the ``captured_logger``
fixture (:mod:`scikitplot.logging._capture`), not ``caplog.at_level``: the
``"scikitplot"`` logger does not propagate, and pytest before 9.1 (the
release Python 3.9 resolves to) attaches ``caplog`` to the root logger only,
so ``caplog.at_level`` saw no record there and passed vacuously elsewhere.
"""

from __future__ import annotations

import logging

import pytest

from .. import _rst_renderer
from .._rst_renderer import render_rst_templates

GOOD = "Hello {{ name }}!\n"
UNDEFINED = "Hello {{ missing }}!\n"
SYNTAX = "Hello {% if %}\n"


def _write(directory, name, text):
    path = directory / f"{name}.rst.template"
    path.write_text(text, encoding="utf-8")
    return path


class TestRendering:
    def test_a_template_renders_beside_itself(self, tmp_path):
        _write(tmp_path, "index", GOOD)
        outputs = render_rst_templates(tmp_path, context={"name": "World"})
        assert outputs == [tmp_path / "index.rst"]
        assert (tmp_path / "index.rst").read_text(encoding="utf-8") == "Hello World!\n"

    def test_templates_are_rendered_in_sorted_order(self, tmp_path):
        for name in ("b", "a", "c"):
            _write(tmp_path, name, GOOD)
        outputs = render_rst_templates(tmp_path, context={"name": "x"})
        assert [p.name for p in outputs] == ["a.rst", "b.rst", "c.rst"]

    def test_values_are_written_as_given_not_html_escaped(self, tmp_path):
        """RST output: a URL's ``&`` and a ``<`` must survive unchanged."""
        _write(tmp_path, "index", "Guide <{{ link }}>\n")
        link = "https://example.invalid/a?x=1&y=<2>"
        render_rst_templates(tmp_path, context={"link": link}, strict=True)
        written = (tmp_path / "index.rst").read_text(encoding="utf-8")
        assert written == f"Guide <{link}>\n"

    def test_an_explicit_escape_filter_still_escapes(self, tmp_path):
        _write(tmp_path, "index", "{{ value | e }}\n")
        render_rst_templates(tmp_path, context={"value": "a&b"}, strict=True)
        assert (tmp_path / "index.rst").read_text(encoding="utf-8") == "a&amp;b\n"

    def test_recursive_discovery_is_opt_in(self, tmp_path):
        (tmp_path / "sub").mkdir()
        _write(tmp_path / "sub", "page", GOOD)
        assert render_rst_templates(tmp_path, context={"name": "x"}) == []
        outputs = render_rst_templates(tmp_path, context={"name": "x"}, recursive=True)
        assert outputs == [tmp_path / "sub" / "page.rst"]


class TestStrict:
    @pytest.mark.parametrize("text", [UNDEFINED, SYNTAX], ids=["undefined", "syntax"])
    def test_a_failing_template_raises_naming_it(self, tmp_path, text):
        template = _write(tmp_path, "index", text)
        with pytest.raises(RuntimeError, match="1 of 1 RST template") as info:
            render_rst_templates(tmp_path, strict=True)
        assert str(template) in str(info.value)
        assert info.value.__cause__ is not None

    def test_every_template_is_attempted_and_every_failure_listed(self, tmp_path):
        bad_a = _write(tmp_path, "a", UNDEFINED)
        _write(tmp_path, "b", GOOD)
        bad_c = _write(tmp_path, "c", SYNTAX)
        with pytest.raises(RuntimeError, match="2 of 3") as info:
            render_rst_templates(tmp_path, context={"name": "x"}, strict=True)
        message = str(info.value)
        assert str(bad_a) in message
        assert "UndefinedError" in message
        assert str(bad_c) in message
        assert "TemplateSyntaxError" in message
        # The good template between the two failures was still rendered.
        assert (tmp_path / "b.rst").read_text(encoding="utf-8") == "Hello x!\n"
        # The first failure is the cause, so its traceback is kept.
        assert type(info.value.__cause__).__name__ == "UndefinedError"

    def test_an_old_output_cannot_hide_a_failure(self, tmp_path):
        _write(tmp_path, "index", UNDEFINED)
        old = tmp_path / "index.rst"
        old.write_text("previous build\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match=r"index\.rst was not regenerated"):
            render_rst_templates(tmp_path, strict=True)
        assert old.read_text(encoding="utf-8") == "previous build\n"

    def test_a_first_build_failure_has_no_stale_note(self, tmp_path):
        _write(tmp_path, "index", UNDEFINED)
        with pytest.raises(RuntimeError) as info:
            render_rst_templates(tmp_path, strict=True)
        assert "not regenerated" not in str(info.value)

    def test_an_unwritable_output_is_reported(self, tmp_path):
        _write(tmp_path, "index", GOOD)
        (tmp_path / "index.rst").mkdir()  # a directory where the file goes
        with pytest.raises(RuntimeError, match=r"index\.rst\.template"):
            render_rst_templates(tmp_path, context={"name": "x"}, strict=True)


class TestBestEffort:
    def test_a_failure_is_skipped_and_logged(self, tmp_path, caplog, captured_logger):
        _write(tmp_path, "a", UNDEFINED)
        _write(tmp_path, "b", GOOD)
        with captured_logger(logging.WARNING, _rst_renderer.logger):
            outputs = render_rst_templates(tmp_path, context={"name": "x"})
        assert outputs == [tmp_path / "b.rst"]
        warnings = [r.getMessage() for r in caplog.records]
        assert len(warnings) == 1
        assert "a.rst.template" in warnings[0]
        assert "UndefinedError" in warnings[0]

    def test_no_failure_logs_nothing(self, tmp_path, caplog, captured_logger):
        _write(tmp_path, "a", GOOD)
        with captured_logger(logging.WARNING, _rst_renderer.logger):
            render_rst_templates(tmp_path, context={"name": "x"})
        assert caplog.records == []
