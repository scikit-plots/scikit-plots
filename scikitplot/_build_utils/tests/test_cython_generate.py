"""
Tests for ``cython_generate``: exact validation and atomic output.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("Cython.Tempita")

from .. import cython_generate as cg  # noqa: E402

TEMPLATE = (
    "{{py:\nNAMES = ['a', 'b']\n}}\n"
    "{{for n in NAMES}}\ndef get_{{n}}(): return {{repr(n)}}\n{{endfor}}\n"
    # The repository's Annoy template writes CSS braces like this:
    "STYLE = '''.x {{ '{'+'{' }} color: red; {{ '}'+'}' }}'''\n"
)


def _template(tmp_path, name="mod.pyx.in", text=TEMPLATE):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestValidate:
    def test_generated_output_with_literal_doubled_braces_passes(self, tmp_path):
        """CSS-like ``{{``/``}}`` in a generated file is not an unexpanded marker."""
        template = _template(tmp_path)
        cg.process_template(template, tmp_path / "mod.pyx")
        assert "{{ color: red; }}" in (tmp_path / "mod.pyx").read_text(encoding="utf-8")
        assert cg.validate_generated_files(tmp_path) is True

    def test_an_edited_output_is_reported_stale(self, tmp_path, capsys):
        template = _template(tmp_path)
        output = tmp_path / "mod.pyx"
        cg.process_template(template, output)
        output.write_text(
            output.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8"
        )
        assert cg.validate_generated_files(tmp_path) is False
        assert "stale or edited" in capsys.readouterr().err

    def test_a_missing_output_is_reported(self, tmp_path, capsys):
        _template(tmp_path)
        assert cg.validate_generated_files(tmp_path) is False
        assert "not generated" in capsys.readouterr().err

    def test_a_raw_copy_of_the_template_is_reported(self, tmp_path):
        template = _template(tmp_path)
        (tmp_path / "mod.pyx").write_text(
            template.read_text(encoding="utf-8"), encoding="utf-8"
        )
        assert cg.validate_generated_files(tmp_path) is False

    def test_files_without_a_template_are_not_checked(self, tmp_path):
        template = _template(tmp_path)
        cg.process_template(template, tmp_path / "mod.pyx")
        (tmp_path / "handwritten.pyx").write_text("x = '{{'\n", encoding="utf-8")
        assert cg.validate_generated_files(tmp_path) is True

    def test_no_template_is_a_failure(self, tmp_path):
        assert cg.validate_generated_files(tmp_path) is False


class TestAtomicOutput:
    def test_output_replaces_the_previous_file_and_leaves_no_temporary(self, tmp_path):
        template = _template(tmp_path)
        output = tmp_path / "mod.pyx"
        output.write_text("old\n", encoding="utf-8")
        cg.process_template(template, output)
        assert "def get_a()" in output.read_text(encoding="utf-8")
        assert sorted(p.name for p in tmp_path.iterdir()) == ["mod.pyx", "mod.pyx.in"]

    def test_a_failed_write_keeps_the_previous_output(self, tmp_path, monkeypatch):
        template = _template(tmp_path)
        output = tmp_path / "mod.pyx"
        output.write_text("old\n", encoding="utf-8")

        def failing_replace(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", failing_replace)
        with pytest.raises(RuntimeError, match="disk full"):
            cg.process_template(template, output)
        assert output.read_text(encoding="utf-8") == "old\n"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["mod.pyx", "mod.pyx.in"]

    def test_a_failing_template_writes_nothing(self, tmp_path):
        template = _template(tmp_path, text="{{for x in }}\n")
        with pytest.raises((ValueError, RuntimeError)):
            cg.process_template(template, tmp_path / "mod.pyx")
        assert not (tmp_path / "mod.pyx").exists()
