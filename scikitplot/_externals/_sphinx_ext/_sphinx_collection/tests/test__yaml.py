"""
Tests for ``_sphinx_collection._yaml``: bounded, safe YAML loading.

The loader promises limits on input bytes, alias count, nesting depth, parsed
value count and scalar text, and that every failure is a ``BoundedYAMLError``.
"""

from __future__ import annotations

import datetime as dt
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOST_ROOT = ROOT.parents[2]


def _yaml_module():
    externals = str(HOST_ROOT / "scikitplot" / "_externals")
    if externals not in sys.path:
        sys.path.insert(0, externals)
    return importlib.import_module("_sphinx_ext._sphinx_collection._yaml")


@pytest.fixture(scope="module")
def bounded():
    return _yaml_module()


def _nested(depth: int) -> str:
    return "[" * depth + "]" * depth


def _alias_doc(count: int) -> str:
    return "a: &a 1\nb: [" + ", ".join(["*a"] * count) + "]\n"


def _laughs(levels: int, width: int) -> str:
    lines = ["a0: &a0 [" + ", ".join(["x"] * width) + "]"]
    for level in range(1, levels):
        refs = ", ".join([f"*a{level - 1}"] * width)
        lines.append(f"a{level}: &a{level} [{refs}]")
    return "\n".join(lines) + "\n"


# -- constants ----------------------------------------------------------------


def test_documented_limits(bounded):
    assert bounded.MAX_YAML_BYTES == 8 * 1024 * 1024
    assert bounded.MAX_YAML_ALIASES == 100
    assert bounded.MAX_YAML_DEPTH == 32
    assert bounded.MAX_YAML_NODES == 100_000
    assert bounded.MAX_YAML_SCALAR_CHARS == 1_048_576
    assert bounded.MAX_COLLECTION_ITEMS == 5_000
    assert all(hasattr(bounded, name) for name in bounded.__all__)


def test_error_type_is_a_value_error(bounded):
    assert issubclass(bounded.BoundedYAMLError, ValueError)


# -- load_bounded_yaml: accepted input ----------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("", None, id="empty"),
        pytest.param("# only a comment\n", None, id="comment-only"),
        pytest.param("just text", "just text", id="scalar-string"),
        pytest.param("42", 42, id="scalar-int"),
        pytest.param("~", None, id="null"),
        pytest.param("- a\n- b\n", ["a", "b"], id="sequence"),
        pytest.param("a: 1\nb: [x, y]\n", {"a": 1, "b": ["x", "y"]}, id="mapping"),
        pytest.param("[]", [], id="empty-sequence"),
        pytest.param("{}", {}, id="empty-mapping"),
        pytest.param("t: Ünïcødé 日本語 🎉\n", {"t": "Ünïcødé 日本語 🎉"}, id="unicode"),
        pytest.param("d: 2024-05-01\n", {"d": dt.date(2024, 5, 1)}, id="date"),
        pytest.param("b: !!binary aGVsbG8=\n", {"b": b"hello"}, id="binary"),
        pytest.param("s: !!set {a}\n", {"s": {"a"}}, id="set"),
        pytest.param(
            "t: '<script>alert(1)</script>'\n",
            {"t": "<script>alert(1)</script>"},
            id="markup-is-plain-text",
        ),
        pytest.param(
            "base: &b {x: 1}\nuse: *b\n",
            {"base": {"x": 1}, "use": {"x": 1}},
            id="shared-alias",
        ),
        pytest.param(
            "base: &b {x: 1}\nmore:\n  <<: *b\n  y: 2\n",
            {"base": {"x": 1}, "more": {"x": 1, "y": 2}},
            id="merge-key",
        ),
    ],
)
def test_load_returns_the_parsed_payload(bounded, text, expected):
    assert bounded.load_bounded_yaml(text, "origin") == expected


def test_loading_is_deterministic(bounded):
    text = "items:\n  - {title: b, tags: [x, y]}\n  - {title: a}\n"
    first = bounded.load_bounded_yaml(text, "origin")
    assert bounded.load_bounded_yaml(text, "origin") == first
    assert list(first) == ["items"]
    assert [item["title"] for item in first["items"]] == ["b", "a"]


# -- load_bounded_yaml: rejected input ----------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("a: [", id="unterminated-flow"),
        pytest.param("a: 1\n  b: 2\n", id="bad-indent"),
        pytest.param("a: 'x\n", id="unterminated-quote"),
        pytest.param("a: 1\n---\nb: 2\n", id="multiple-documents"),
        pytest.param("a: *missing\n", id="undefined-alias"),
        pytest.param("? [1, 2]\n: x\n", id="unhashable-key"),
        pytest.param("\tbad: tab\n", id="tab-indent"),
        pytest.param("a: \x07\n", id="control-character"),
    ],
)
def test_invalid_yaml_is_reported_as_bounded_error(bounded, text):
    with pytest.raises(bounded.BoundedYAMLError, match="data/x.yaml: could not parse"):
        bounded.load_bounded_yaml(text, "data/x.yaml")


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            "!!python/object/apply:os.system ['echo pwned']", id="python-apply"
        ),
        pytest.param("!!python/name:os.system", id="python-name"),
        pytest.param("a: !!python/tuple [1, 2]", id="python-tuple"),
        pytest.param("a: !custom {x: 1}", id="unknown-application-tag"),
    ],
)
def test_object_construction_tags_are_refused(bounded, text):
    with pytest.raises(bounded.BoundedYAMLError, match="could not parse"):
        bounded.load_bounded_yaml(text, "origin")


def test_parse_error_chains_the_yaml_error(bounded):
    import yaml

    with pytest.raises(bounded.BoundedYAMLError) as info:
        bounded.load_bounded_yaml("a: [", "origin")
    assert isinstance(info.value.__cause__, yaml.YAMLError)


def test_alias_limit_boundary(bounded):
    limit = bounded.MAX_YAML_ALIASES
    payload = bounded.load_bounded_yaml(_alias_doc(limit), "origin")
    assert payload == {"a": 1, "b": [1] * limit}
    with pytest.raises(bounded.BoundedYAMLError, match="origin: YAML uses more than"):
        bounded.load_bounded_yaml(_alias_doc(limit + 1), "origin")


def test_depth_limit_boundary(bounded):
    limit = bounded.MAX_YAML_DEPTH
    payload = bounded.load_bounded_yaml(_nested(limit), "origin")
    depth = 0
    while payload:
        payload = payload[0]
        depth += 1
    assert depth == limit - 1
    with pytest.raises(bounded.BoundedYAMLError, match="nesting exceeds 32 levels"):
        bounded.load_bounded_yaml(_nested(limit + 1), "origin")


def test_depth_limit_applies_to_mappings_and_mixed_nesting(bounded):
    limit = bounded.MAX_YAML_DEPTH
    deep_map = "{a: " * (limit + 1) + "1" + "}" * (limit + 1)
    with pytest.raises(bounded.BoundedYAMLError, match="nesting exceeds"):
        bounded.load_bounded_yaml(deep_map, "origin")
    mixed = "{a: [" * (limit // 2 + 1) + "1" + "]}" * (limit // 2 + 1)
    with pytest.raises(bounded.BoundedYAMLError, match="nesting exceeds"):
        bounded.load_bounded_yaml(mixed, "origin")


def test_depth_counts_nesting_not_sibling_containers(bounded):
    text = "\n".join(f"- [{index}]" for index in range(200)) + "\n"
    payload = bounded.load_bounded_yaml(text, "origin")
    assert payload == [[index] for index in range(200)]


def test_pathological_depth_does_not_escape_as_recursion_error(bounded):
    with pytest.raises(bounded.BoundedYAMLError, match="nesting exceeds"):
        bounded.load_bounded_yaml(_nested(5_000), "origin")


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("&a [*a]", id="sequence-contains-itself"),
        pytest.param("&a {self: *a}", id="mapping-contains-itself"),
        pytest.param("x: &a [1, {y: *a}]", id="indirect-cycle"),
    ],
)
def test_recursive_aliases_are_rejected(bounded, text):
    with pytest.raises(bounded.BoundedYAMLError, match="recursive YAML aliases"):
        bounded.load_bounded_yaml(text, "origin")


def test_byte_limit_is_checked_before_parsing(bounded):
    oversized = "#" * (bounded.MAX_YAML_BYTES + 1)
    with pytest.raises(bounded.BoundedYAMLError, match=r"8,388,609 bytes; the limit"):
        bounded.load_bounded_yaml(oversized, "origin")


def test_byte_limit_counts_utf8_bytes_not_characters(bounded, monkeypatch):
    monkeypatch.setattr(bounded, "MAX_YAML_BYTES", 10)
    assert bounded.load_bounded_yaml("a: 1234567", "origin") == {"a": 1234567}
    with pytest.raises(bounded.BoundedYAMLError, match="12 bytes; the limit is 10"):
        bounded.load_bounded_yaml("é" * 6, "origin")


def test_node_limit_boundary(bounded, monkeypatch):
    monkeypatch.setattr(bounded, "MAX_YAML_NODES", 5)
    assert bounded.load_bounded_yaml("[1, 2, 3, 4]", "origin") == [1, 2, 3, 4]
    with pytest.raises(bounded.BoundedYAMLError, match="origin: parsed YAML exceeds 5"):
        bounded.load_bounded_yaml("[1, 2, 3, 4, 5]", "origin")


def test_node_limit_counts_mapping_keys(bounded, monkeypatch):
    monkeypatch.setattr(bounded, "MAX_YAML_NODES", 5)
    assert bounded.load_bounded_yaml("{a: 1, b: 2}", "origin") == {"a": 1, "b": 2}
    with pytest.raises(bounded.BoundedYAMLError, match="parsed YAML exceeds"):
        bounded.load_bounded_yaml("{a: 1, b: 2, c: 3}", "origin")


def test_scalar_text_limit_is_cumulative(bounded, monkeypatch):
    monkeypatch.setattr(bounded, "MAX_YAML_SCALAR_CHARS", 10)
    assert bounded.load_bounded_yaml("[aaaaa, bbbbb]", "origin") == ["aaaaa", "bbbbb"]
    with pytest.raises(bounded.BoundedYAMLError, match="scalar text exceeds 10"):
        bounded.load_bounded_yaml("[aaaaa, bbbbbb]", "origin")


def test_scalar_text_limit_counts_keys_and_binary(bounded, monkeypatch):
    monkeypatch.setattr(bounded, "MAX_YAML_SCALAR_CHARS", 10)
    with pytest.raises(bounded.BoundedYAMLError, match="scalar text exceeds"):
        bounded.load_bounded_yaml("{aaaaaa: bbbbb}", "origin")
    with pytest.raises(bounded.BoundedYAMLError, match="scalar text exceeds"):
        bounded.load_bounded_yaml("b: !!binary aGVsbG8gd29ybGQh", "origin")


def test_default_scalar_text_limit_rejects_one_huge_value(bounded):
    text = "t: " + "x" * (bounded.MAX_YAML_SCALAR_CHARS + 1)
    with pytest.raises(bounded.BoundedYAMLError, match=r"exceeds 1,048,576"):
        bounded.load_bounded_yaml(text, "origin")


def test_limits_are_restored_between_tests(bounded):
    assert bounded.MAX_YAML_NODES == 100_000
    assert bounded.MAX_YAML_SCALAR_CHARS == 1_048_576
    assert bounded.MAX_YAML_BYTES == 8 * 1024 * 1024


def test_alias_expansion_bomb_is_rejected(bounded):
    # 90 aliases (under the limit of 100) describe 10**10 values when expanded:
    # far beyond MAX_YAML_NODES, and any consumer that walks or serialises the
    # payload pays for every one of them.
    text = _laughs(levels=10, width=10)
    assert text.count("*") <= bounded.MAX_YAML_ALIASES
    with pytest.raises(bounded.BoundedYAMLError):
        bounded.load_bounded_yaml(text, "origin")


def test_depth_limit_applies_to_nesting_reached_through_aliases(bounded):
    text = "a: &a " + _nested(30) + "\nb: " + "[" * 30 + "*a" + "]" * 30 + "\n"
    with pytest.raises(bounded.BoundedYAMLError, match="nesting exceeds"):
        bounded.load_bounded_yaml(text, "origin")


# -- read_bounded_utf8 --------------------------------------------------------


def test_read_returns_utf8_text(bounded, tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text("title: Ünïcødé 日本語\n", encoding="utf-8")
    assert bounded.read_bounded_utf8(path, "data.yaml") == "title: Ünïcødé 日本語\n"


def test_read_empty_file(bounded, tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_bytes(b"")
    assert bounded.read_bounded_utf8(path, "empty.yaml") == ""


def test_read_size_limit_boundary(bounded, tmp_path, monkeypatch):
    monkeypatch.setattr(bounded, "MAX_YAML_BYTES", 16)
    path = tmp_path / "data.yaml"
    path.write_bytes(b"a" * 16)
    assert bounded.read_bounded_utf8(path, "data.yaml") == "a" * 16
    path.write_bytes(b"a" * 17)
    with pytest.raises(
        bounded.BoundedYAMLError, match="data.yaml: YAML input is 17 bytes"
    ):
        bounded.read_bounded_utf8(path, "data.yaml")


def test_read_rejects_oversized_file_without_reading_it(bounded, tmp_path, monkeypatch):
    monkeypatch.setattr(bounded, "MAX_YAML_BYTES", 4)
    path = tmp_path / "big.yaml"
    path.write_bytes(b"0123456789")

    def forbidden(self, *args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("oversized file was read")

    monkeypatch.setattr(type(path), "read_text", forbidden)
    with pytest.raises(bounded.BoundedYAMLError, match="the limit is 4 bytes"):
        bounded.read_bounded_utf8(path, "big.yaml")


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"title: caf\xe9\n", id="latin-1"),
        pytest.param("title: x\n".encode("utf-16"), id="utf-16"),
        pytest.param(b"\xff\xfe\xfd", id="garbage"),
    ],
)
def test_read_rejects_non_utf8(bounded, tmp_path, data):
    path = tmp_path / "data.yaml"
    path.write_bytes(data)
    with pytest.raises(
        bounded.BoundedYAMLError, match="could not read the-origin: .*UTF-8"
    ) as info:
        bounded.read_bounded_utf8(path, "the-origin")
    assert isinstance(info.value.__cause__, UnicodeDecodeError)


def test_read_missing_file(bounded, tmp_path):
    with pytest.raises(bounded.BoundedYAMLError, match="could not read gone") as info:
        bounded.read_bounded_utf8(tmp_path / "gone.yaml", "gone")
    assert isinstance(info.value.__cause__, OSError)


def test_read_directory_is_an_error(bounded, tmp_path):
    with pytest.raises(bounded.BoundedYAMLError, match="could not read a-dir"):
        bounded.read_bounded_utf8(tmp_path, "a-dir")


def test_read_then_load_round_trip(bounded, tmp_path):
    path = tmp_path / "items.yaml"
    path.write_text("- title: One\n- title: Two\n", encoding="utf-8")
    text = bounded.read_bounded_utf8(path, "items.yaml")
    assert bounded.load_bounded_yaml(text, "items.yaml") == [
        {"title": "One"},
        {"title": "Two"},
    ]
