# scikitplot/corpus/tests/test__serialize.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""One serialisation contract: strict JSON, JSON Lines, optional YAML and TOML."""

from __future__ import annotations

import enum
import math
import sys
from types import MappingProxyType

import numpy as np
import pytest

from .. import _serialize as s


class Colour(enum.Enum):
    RED = "red"


class WithToDict:
    def to_dict(self):
        return {"x": np.int64(1)}


class TestFormats:
    @pytest.mark.parametrize(
        ("path", "fmt"),
        [("a.json", "json"), ("a.JSONL", "jsonl"), ("a.ndjson", "jsonl"), ("a.yml", "yaml"), ("a.yaml", "yaml"), ("a.toml", "toml")],
    )
    def test_suffix(self, path, fmt):
        assert s.resolve_format(path) == fmt

    def test_explicit_format_wins(self):
        assert s.resolve_format("a.txt", "json") == "json"

    @pytest.mark.parametrize(
        ("args", "match"),
        [(("a.txt", None), "Cannot infer a format"), ((None, None), "Pass a format"), (("a.json", "xml"), "format must be one of")],
    )
    def test_refused(self, args, match):
        with pytest.raises(ValueError, match=match):
            s.resolve_format(*args)


class TestToBuiltin:
    def test_values(self):
        data = {
            "f": np.float32(0.5),
            "i": np.int64(3),
            "arr": np.arange(3),
            "tuple": (1, "a"),
            "enum": Colour.RED,
            "nested": WithToDict(),
            "none": None,
            "bool": True,
        }
        assert s.to_builtin(data) == {
            "f": 0.5, "i": 3, "arr": [0, 1, 2], "tuple": [1, "a"], "enum": "red",
            "nested": {"x": 1}, "none": None, "bool": True,
        }
        assert type(s.to_builtin(np.int64(3))) is int

    @pytest.mark.parametrize(
        ("value", "match"),
        [({1: "a"}, "keys must be str"), ({"a", "b"}, "no defined order"), (frozenset(), "no defined order"), (object(), "Cannot serialise")],
    )
    def test_refused(self, value, match):
        with pytest.raises(TypeError, match=match):
            s.to_builtin(value)


class TestJson:
    def test_nan_is_null_and_text_stays_unicode(self):
        text = s.dumps({"v": math.nan, "name": "Ærø 日本"}, "json", indent=None)
        assert text == '{"v": null, "name": "Ærø 日本"}\n'

    def test_infinity_is_refused(self):
        with pytest.raises(ValueError, match="infinite"):
            s.dumps({"v": math.inf}, "json")

    @pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
    def test_non_standard_tokens_are_refused_on_read(self, token):
        with pytest.raises(ValueError, match="Non-standard JSON token"):
            s.loads(f'{{"v": {token}}}', "json")

    def test_type_of_text(self):
        with pytest.raises(TypeError):
            s.loads(b"{}", "json")


class TestJsonLines:
    def test_rows(self):
        assert s.dumps([{"q": 1}, {"q": math.nan}], "jsonl") == '{"q":1}\n{"q":null}\n'
        assert s.dumps([], "jsonl") == ""
        assert s.loads('{"q":1}\n{"q":2}\n', "jsonl") == [{"q": 1}, {"q": 2}]
        assert s.loads("", "jsonl") == []

    def test_rows_must_be_mappings(self):
        with pytest.raises(TypeError, match="sequence of mappings"):
            s.dumps([1, 2], "jsonl")
        with pytest.raises(TypeError, match="sequence of mappings"):
            s.dumps({"q": 1}, "jsonl")

    def test_errors_name_the_line(self):
        with pytest.raises(ValueError, match="blank line at line 2"):
            s.loads('{"q":1}\n\n{"q":2}\n', "jsonl")
        with pytest.raises(ValueError, match="Invalid JSON at line 2"):
            s.loads('{"q":1}\n{q}\n', "jsonl")

    def test_rows_of(self):
        assert s.rows_of([{"a": np.int64(1)}]) == [{"a": 1}]
        with pytest.raises(TypeError):
            s.rows_of([1])


class TestOptionalFormats:
    DATA = MappingProxyType(
        {"name": "x", "values": (1.5, math.nan), "nested": {"k": None, "n": 2}, "rows": ({"a": 1}, {"a": 2})}
    )

    def test_yaml_round_trip(self):
        pytest.importorskip("yaml")
        back = s.loads(s.dumps(self.DATA, "yaml"), "yaml")
        assert back["nested"] == {"k": None, "n": 2}
        assert back["values"][0] == 1.5
        assert math.isnan(back["values"][1])

    def test_toml_round_trip_drops_none(self):
        pytest.importorskip("tomli_w")
        back = s.loads(s.dumps(self.DATA, "toml"), "toml")
        assert back["nested"] == {"n": 2}
        assert math.isnan(back["values"][1])
        assert back["rows"] == [{"a": 1}, {"a": 2}]

    def test_toml_refusals(self):
        pytest.importorskip("tomli_w")
        with pytest.raises(ValueError, match="no null value"):
            s.dumps({"a": [1, None]}, "toml")
        with pytest.raises(TypeError, match="mapping at the top level"):
            s.dumps([1], "toml")

    @pytest.mark.parametrize(
        ("module", "call", "match"),
        [
            ("yaml", lambda: s.dumps({}, "yaml"), "pip install pyyaml"),
            ("tomli_w", lambda: s.dumps({}, "toml"), "pip install tomli-w"),
        ],
    )
    def test_missing_library_names_the_install(self, monkeypatch, module, call, match):
        monkeypatch.setitem(sys.modules, module, None)
        with pytest.raises(ImportError, match=match):
            call()

    def test_missing_toml_reader(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "tomllib", None)
        monkeypatch.setitem(sys.modules, "tomli", None)
        with pytest.raises(ImportError, match="pip install tomli"):
            s.loads("a = 1", "toml")


def test_write_and_read(tmp_path):
    path = s.write(tmp_path / "sub" / "data.json", {"a": [1, 2]})
    assert path.read_text(encoding="utf-8") == '{\n  "a": [\n    1,\n    2\n  ]\n}\n'
    assert s.read(path) == {"a": [1, 2]}
    assert list(path.parent.iterdir()) == [path]  # no staging file left behind



LINE_BREAKS = ("\x85", "\u2028", "\u2029")


class TestLineBreakCharacters:
    """Review F4/F21: NEL, LS and PS inside strings."""

    @pytest.mark.parametrize("char", LINE_BREAKS)
    def test_yaml_round_trips(self, char):
        pytest.importorskip("yaml")
        data = {"query_id": f"intro{char}summary", "ids": [f"a{char}b"]}
        assert s.loads(s.dumps(data, "yaml"), "yaml") == data

    @pytest.mark.parametrize("char", LINE_BREAKS)
    def test_json_writes_them_escaped(self, char):
        text = s.dumps({"v": f"x{char}y"}, "json", indent=None)
        assert char not in text
        assert s.loads(text, "json") == {"v": f"x{char}y"}

    def test_jsonl_rows_survive_line_splitting_readers(self):
        rows = [{"v": f"x{char}y"} for char in LINE_BREAKS]
        text = s.dumps(rows, "jsonl")
        assert len(text.splitlines()) == len(rows)
        assert s.loads(text, "jsonl") == rows

    def test_yaml_fuzz_round_trips_every_string(self):
        pytest.importorskip("yaml")
        import random

        rng = random.Random(9)  # noqa: S311 - seeded test data
        alphabet = "ab \t\n\r\x00\x1f\x7f\x85\xa0\u2028\u2029\ufeff\U0001F600:#'\"-&*!|>%@`{}[],?"
        for _ in range(2000):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
            assert s.loads(s.dumps({"k": text, "l": [text]}, "yaml"), "yaml") == {"k": text, "l": [text]}, repr(text)


class TestUntrustedInput:
    """Review F7/F17: aliases, deep nesting and overflowing numbers."""

    def test_yaml_aliases_are_refused(self):
        pytest.importorskip("yaml")
        bomb = "a: &a [x, x]\nb: &b [*a, *a]\nc: [*b, *b]\n"
        with pytest.raises(ValueError, match="aliases are refused"):
            s.loads(bomb, "yaml")

    def test_yaml_anchor_without_alias_is_fine(self):
        pytest.importorskip("yaml")
        assert s.loads("a: &a [1]\n", "yaml") == {"a": [1]}

    def test_invalid_yaml_is_a_value_error(self):
        pytest.importorskip("yaml")
        with pytest.raises(ValueError, match="Invalid YAML"):
            s.loads("a: [1, 2\n", "yaml")

    @pytest.mark.parametrize("fmt", ["json", "yaml"])
    def test_deep_nesting_is_a_value_error(self, fmt):
        if fmt == "yaml":
            pytest.importorskip("yaml")
        deep = "[" * 100_000 + "]" * 100_000
        with pytest.raises(ValueError, match="nested too deeply"):
            s.loads(deep, fmt)

    def test_deep_data_on_write_is_a_value_error(self):
        data: list = []
        for _ in range(100_000):
            data = [data]
        with pytest.raises(ValueError, match="nested too deeply"):
            s.dumps(data, "json")

    @pytest.mark.parametrize("text", ['{"v": 1e400}', '{"v": -1e400}'])
    def test_overflowing_numbers_are_refused(self, text):
        with pytest.raises(ValueError, match="overflows a float"):
            s.loads(text, "json")


class TestNumpyScalars:
    """Review F8: numpy strings and booleans."""

    def test_to_builtin_normalises(self):
        out = s.to_builtin({"a": np.str_("x"), "b": np.bool_(True), "c": np.array(["u", "v"])})
        assert out == {"a": "x", "b": True, "c": ["u", "v"]}
        assert type(out["a"]) is str
        assert type(out["b"]) is bool

    def test_yaml_accepts_them(self):
        pytest.importorskip("yaml")
        data = {"a": np.str_("x"), "b": np.bool_(False)}
        assert s.loads(s.dumps(data, "yaml"), "yaml") == {"a": "x", "b": False}
