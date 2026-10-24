# scikitplot/feature_selection/tests/test_fs_serialize.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Round trips and refusals of the four text formats (Rules 95, 107)."""

from __future__ import annotations

import enum
import math

import numpy as np
import pytest

from scikitplot.feature_selection import _serialize as S

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

FORMATS = ["json", "jsonl", "yaml", "toml"]
SPECIAL = ["a b", "x\x85y", "nul\x00", " ", "\r\n", "tab\t", "é", "\U0001f600"]


def _wrap(value, fmt):
    return [{"k": value}] if fmt == "jsonl" else {"k": value}


def _available(fmt):
    if fmt == "yaml":
        pytest.importorskip("yaml")
    if fmt == "toml":
        pytest.importorskip("tomli_w")


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("text", SPECIAL)
def test_special_characters_round_trip(fmt, text):
    _available(fmt)
    data = _wrap(text, fmt)
    assert S.loads(S.dumps(data, fmt), fmt) == data


@pytest.mark.parametrize("fmt", ["json", "jsonl", "yaml"])
def test_lone_surrogates_round_trip_where_escaped(fmt):
    _available(fmt)
    data = _wrap("lone\ud800", fmt)
    assert S.loads(S.dumps(data, fmt), fmt) == data


def test_toml_refuses_a_lone_surrogate_at_write_time():
    _available("toml")
    with pytest.raises(ValueError, match="lone surrogate"):
        S.dumps({"k": "lone\ud800"}, "toml")


@settings(max_examples=150, deadline=None)
@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=40))
def test_text_fuzz_round_trips_in_every_format(text):
    for fmt in FORMATS:
        if fmt == "yaml":
            pytest.importorskip("yaml")
        if fmt == "toml":
            pytest.importorskip("tomli_w")
        data = _wrap(text, fmt)
        assert S.loads(S.dumps(data, fmt), fmt) == data


def test_toml_drops_none_values_and_refuses_none_in_arrays():
    _available("toml")
    assert S.loads(S.dumps({"a": None, "b": 1}, "toml"), "toml") == {"b": 1}
    with pytest.raises(ValueError, match="None inside an array"):
        S.dumps({"a": [1, None]}, "toml")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_numbers_are_refused(value):
    with pytest.raises(ValueError, match="non-finite"):
        S.dumps({"x": value}, "json")


def test_non_standard_json_constants_are_refused_on_read():
    with pytest.raises(ValueError, match="non-standard"):
        S.loads('{"x": NaN}', "json")


def test_values_outside_the_data_model_are_refused():
    with pytest.raises(TypeError, match="cannot serialise set"):
        S.dumps({"x": {1, 2}}, "json")
    with pytest.raises(TypeError, match="keys must be strings"):
        S.dumps({1: "a"}, "json")


def test_numpy_scalars_and_enums_become_builtins():
    class Colour(str, enum.Enum):
        RED = "red"

    tree = S.to_builtin(
        {
            "a": np.int64(3),
            "b": np.float32(1.5),
            "c": np.bool_(True),
            "d": Colour.RED,
            "e": (1, 2),
        }
    )
    assert tree == {"a": 3, "b": 1.5, "c": True, "d": "red", "e": [1, 2]}
    assert type(tree["a"]) is int and type(tree["c"]) is bool


def test_yaml_aliases_are_refused():
    pytest.importorskip("yaml")
    with pytest.raises(Exception, match="aliases are refused"):
        S.loads("a: &x [1, 2]\nb: *x\n", "yaml")


def test_deep_nesting_is_refused():
    deep = "[" * 200 + "]" * 200
    with pytest.raises(ValueError, match="nesting"):
        S.loads(deep, "json")


def test_jsonl_is_split_on_newlines_only():
    rows = [{"k": "a b\x85c"}, {"k": "d"}]
    text = S.dumps(rows, "jsonl")
    assert text.count("\n") == 2
    assert S.loads(text, "jsonl") == rows


def test_jsonl_needs_a_list_of_mappings():
    with pytest.raises(TypeError):
        S.dumps({"a": 1}, "jsonl")
    with pytest.raises(ValueError, match="JSON object"):
        S.loads("[1]\n", "jsonl")


def test_format_resolution(tmp_path):
    assert S.resolve_format("x.yml") == "yaml"
    assert S.resolve_format("x.NDJSON") == "jsonl"
    assert S.resolve_format("x.txt", "json") == "json"
    with pytest.raises(ValueError, match="cannot derive"):
        S.resolve_format("x.txt")
    with pytest.raises(ValueError, match="not one of"):
        S.resolve_format(format="xml")


@pytest.mark.parametrize("fmt", ["json", "yaml", "toml"])
def test_write_and_read_round_trip_atomically(tmp_path, fmt):
    _available(fmt)
    data = {"a": 1, "b": [1.5, "x"], "c": {"d": True}}
    path = S.write(data, tmp_path / f"doc.{fmt}")
    assert S.read(path) == data
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"doc.{fmt}"]


def test_a_failed_write_leaves_no_file(tmp_path):
    with pytest.raises(ValueError):
        S.write({"x": math.nan}, tmp_path / "bad.json")
    assert list(tmp_path.iterdir()) == []
