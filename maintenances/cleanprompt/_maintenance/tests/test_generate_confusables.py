"""Tests for ``tools/generate_confusables.py`` (CP-114), on synthetic UTS #39 lines."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
generator = runpy.run_path(str(TOOLS / "generate_confusables.py"))
build = generator["build"]
render = generator["render"]

_HEADER = [
    "# confusables.txt",
    "# Date: 2099-01-01, 00:00:00 GMT",
    "# Version: 99.0.0",
]


def _line(source, target, name="X"):
    return f"{source} ;\t{target} ;\tMA\t# {name}"


class TestBuild:
    def test_records_the_header(self):
        _, version, date = build(_HEADER)
        assert (version, date) == ("99.0.0", "2099-01-01, 00:00:00 GMT")

    def test_case_picks_the_member_of_the_prototype_class(self):
        # I and 1 share the prototype l; an upper-case look-alike is I.
        lines = [
            *_HEADER,
            _line("0049", "006C"),
            _line("0031", "006C"),
            _line("0406", "006C"),
            _line("0456", "0069"),
        ]
        table, _, _ = build(lines)
        assert table == {0x0406: "I", 0x0456: "i"}

    def test_digits_only_where_value_and_shape_agree(self):
        lines = [
            *_HEADER,
            _line("09E6", "004F"),
            _line("0030", "004F"),
            _line("09EA", "0038"),
        ]
        table, _, _ = build(lines)
        assert table == {0x09E6: "0"}

    def test_symbols_multi_character_targets_and_ascii_are_skipped(self):
        lines = [
            *_HEADER,
            _line("2223", "006C"),  # a symbol
            _line("0271", "0072 006E"),  # two characters
            _line("0041", "0041"),  # ASCII itself
        ]
        table, _, _ = build(lines)
        assert table == {}

    def test_a_malformed_record_is_refused(self):
        with pytest.raises(ValueError, match="line 4"):
            build([*_HEADER, "0430 0061 MA"])


class TestRender:
    def test_the_output_is_python_with_the_version(self, tmp_path):
        table, version, date = build([*_HEADER, _line("0430", "0061")])
        module = tmp_path / "generated.py"
        module.write_text(render(table, version, date), encoding="utf-8")
        namespace = runpy.run_path(str(module))
        assert namespace["CONFUSABLES"] == {0x0430: "a"}
        assert namespace["UNICODE_VERSION"] == "99.0.0"

    def test_rendering_is_deterministic(self):
        lines = [*_HEADER, _line("0430", "0061"), _line("0410", "0041")]
        assert render(*build(lines)) == render(
            *build([*reversed(lines[3:]), *lines[:3]])
        )
