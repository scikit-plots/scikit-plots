# scikitplot/corpus/_readers/tests/test__structured.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the JSON, JSON Lines, YAML and TOML readers (CX-50)."""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from unittest import mock

import pytest

from ... import _readers  # noqa: F401  (registers the readers)
from ..._base import DocumentReader
from ..._schema import SourceType
from .._structured import JSONReader, TOMLReader, YAMLReader, _key_step, _lines


def _write(tmp_path: Path, data: str | bytes, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return path


def _texts(reader) -> list[str]:
    return [line for c in reader.get_raw_chunks() for line in c["text"].split("\n")]


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("a.json", JSONReader),
        ("a.jsonl", JSONReader),
        ("a.ndjson", JSONReader),
        ("a.yaml", YAMLReader),
        ("a.YML", YAMLReader),
        ("a.toml", TOMLReader),
    ],
)
def test_extensions_create_the_readers(tmp_path, name, cls):
    assert type(DocumentReader.create(_write(tmp_path, "{}", name))) is cls


def test_the_shared_base_registers_no_extension():
    from .._text import TextReader

    assert DocumentReader._registry[".txt"] is TextReader


def test_expansion_meets_the_text_ratio_limit(tmp_path):
    pytest.importorskip("yaml")
    body = 'a: &a ["xxxxxxxx", "xxxxxxxx", "xxxxxxxx", "xxxxxxxx"]\n'
    for name, prev in zip("bcdefghij", "abcdefghi"):
        body += f"{name}: &{name} [" + ", ".join(["*" + prev] * 4) + "]\n"
    reader = YAMLReader(input_path=_write(tmp_path, body, "a.yaml"))
    with pytest.raises(ValueError, match="max_text_chars_per_byte"):
        list(reader.get_documents())


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

_DOC = {
    "title": "Guide",
    "users": [{"name": "Ada", "age": 36, "admin": True}, {"name": "Bob", "tags": []}],
    "odd key": {"a.b": None, "x": 1.5},
    "note": "two\nlines",
}


def test_every_value_becomes_a_path_line(tmp_path):
    path = _write(tmp_path, json.dumps(_DOC), "a.json")
    assert _texts(JSONReader(input_path=path)) == [
        "title: Guide",
        "users[0].name: Ada",
        "users[0].age: 36",
        "users[0].admin: true",
        "users[1].name: Bob",
        "users[1].tags: []",
        '["odd key"]["a.b"]: null',
        '["odd key"].x: 1.5',
        "note: two lines",
    ]


def test_text_only_keeps_strings(tmp_path):
    path = _write(tmp_path, json.dumps(_DOC), "a.json")
    assert _texts(JSONReader(input_path=path, values="text")) == [
        "title: Guide",
        "users[0].name: Ada",
        "users[1].name: Bob",
        "note: two lines",
    ]


def test_chunks_group_top_level_items(tmp_path):
    rows = [{"id": i, "text": f"row {i}"} for i in range(5)]
    path = _write(tmp_path, json.dumps(rows), "a.json")
    chunks = list(JSONReader(input_path=path, items_per_chunk=2).get_raw_chunks())
    assert [(c["record_start"], c["record_end"]) for c in chunks] == [(0, 1), (2, 3), (4, 4)]
    assert chunks[0]["text"] == "[0].id: 0\n[0].text: row 0\n[1].id: 1\n[1].text: row 1"
    assert {c["format"] for c in chunks} == {"json"}
    assert {c["source_type"] for c in chunks} == {SourceType.DATASET.value}


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ('"just text"', ["$: just text"]),
        ("42", ["$: 42"]),
        ("[]", []),
        ("{}", []),
        ('{"a": {}}', ["a: {}"]),
    ],
)
def test_scalar_and_empty_documents(tmp_path, data, expected):
    reader = JSONReader(input_path=_write(tmp_path, data, "a.json"))
    chunks = list(reader.get_raw_chunks())
    assert [line for c in chunks for line in c["text"].split("\n")] == expected


def test_bom_and_nonfinite_numbers(tmp_path):
    raw = b"\xef\xbb\xbf" + b'{"a": NaN, "b": -Infinity}'
    assert _texts(JSONReader(input_path=_write(tmp_path, raw, "a.json"))) == [
        "a: nan",
        "b: -inf",
    ]


def test_malformed_json_names_the_position(tmp_path):
    with pytest.raises(ValueError, match=r"JSONReader: a.json: not JSON at line 2 column"):
        list(JSONReader(input_path=_write(tmp_path, '{"a":\n  ]', "a.json")).get_raw_chunks())


def test_deep_nesting_never_crashes(tmp_path):
    # Whether json parses 200 000 levels depends on the Python version
    # (3.14+ bounds recursion by stack size, not a count): the contract is
    # a clean read or a ValueError naming the depth, never a crash.
    path = _write(tmp_path, "[" * 200_000 + "]" * 200_000, "a.json")
    try:
        list(JSONReader(input_path=path).get_raw_chunks())
    except ValueError as exc:
        assert "nests too deeply" in str(exc)


def test_a_parser_recursion_error_names_the_depth(tmp_path):
    path = _write(tmp_path, "[[1]]", "a.json")
    with mock.patch("json.loads", side_effect=RecursionError):
        with pytest.raises(ValueError, match="JSONReader: a.json nests too deeply"):
            list(JSONReader(input_path=path).get_raw_chunks())


def test_the_value_budget_refuses(tmp_path):
    path = _write(tmp_path, json.dumps(list(range(50))), "a.json")
    with pytest.raises(ValueError, match="max_values"):
        list(JSONReader(input_path=path, max_values=10).get_raw_chunks())


def test_json_lines_records_are_numbered_by_line(tmp_path):
    body = '{"a": 1}\n\n{"a": 2}\n["x"]\n'
    chunks = list(JSONReader(input_path=_write(tmp_path, body, "a.jsonl")).get_raw_chunks())
    assert chunks[0]["text"] == ".a: 1\n.a: 2\n[0]: x"
    assert (chunks[0]["record_start"], chunks[0]["record_end"]) == (1, 4)


def test_json_lines_bad_record_refuses_or_skips(tmp_path):
    path = _write(tmp_path, '{"a": 1}\nnot json\n{"a": 3}\n', "a.ndjson")
    with pytest.raises(ValueError, match="line 2 is not JSON"):
        list(JSONReader(input_path=path).get_raw_chunks())
    reader = JSONReader(input_path=path, on_error="skip")
    assert _texts(reader) == [".a: 1", ".a: 3"]
    assert reader.coverage.skipped == {"malformed": 1}
    assert reader.coverage.read == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"values": "numbers"},
        {"items_per_chunk": 0},
        {"max_values": True},
        {"on_error": "ignore"},
    ],
)
def test_bad_options_are_refused(tmp_path, kwargs):
    with pytest.raises(ValueError):
        JSONReader(input_path=_write(tmp_path, "{}", "a.json"), **kwargs)


def test_bad_yaml_tag_policy_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown_tags"):
        YAMLReader(input_path=_write(tmp_path, "a: 1", "a.yaml"), unknown_tags="run")


def test_documents_are_built(tmp_path):
    path = _write(tmp_path, json.dumps({"summary": "A longer sentence to keep."}), "a.json")
    docs = list(JSONReader(input_path=path).get_documents())
    assert docs and docs[0].source_type == SourceType.DATASET


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------


def test_yaml_values_dates_and_anchors(tmp_path):
    pytest.importorskip("yaml")
    body = (
        "base: &b {host: example.org, port: 80}\n"
        "prod:\n  <<: *b\n  port: 443\n"
        "released: 2026-11-25\n"
    )
    assert _texts(YAMLReader(input_path=_write(tmp_path, body, "a.yaml"))) == [
        "base.host: example.org",
        "base.port: 80",
        "prod.host: example.org",
        "prod.port: 443",
        "released: 2026-11-25",
    ]


def test_yaml_streams_prefix_the_document(tmp_path):
    pytest.importorskip("yaml")
    chunks = list(
        YAMLReader(input_path=_write(tmp_path, "a: 1\n---\na: 2\n", "a.yml")).get_raw_chunks()
    )
    assert chunks[0]["text"] == "[0].a: 1\n[1].a: 2"
    assert chunks[0]["format"] == "yaml"


def test_yaml_unknown_tags_are_read_as_data_never_run(tmp_path):
    pytest.importorskip("yaml")
    body = (
        "x: !!python/object/apply:os.system ['echo hacked']\n"
        "docs: !relative $config_dir/docs\n"
        "key: !ENV [API_KEY, none]\n"
        "fmt: !!python/name:material.extensions.emoji.to_svg\n"
    )
    path = _write(tmp_path, body, "a.yaml")
    with mock.patch("os.system") as system:
        lines = _texts(YAMLReader(input_path=path))
    system.assert_not_called()
    assert lines == [
        "x[0]: echo hacked",
        "docs: !relative $config_dir/docs",
        "key[0]: API_KEY",
        "key[1]: none",
        "fmt: tag:yaml.org,2002:python/name:material.extensions.emoji.to_svg",
    ]


def test_yaml_unknown_tags_can_be_refused(tmp_path):
    pytest.importorskip("yaml")
    path = _write(tmp_path, "docs: !relative x\n", "a.yaml")
    with pytest.raises(ValueError, match="not YAML"):
        list(YAMLReader(input_path=path, unknown_tags="refuse").get_raw_chunks())


def test_yaml_self_reference_is_written_once(tmp_path):
    pytest.importorskip("yaml")
    lines = _texts(YAMLReader(input_path=_write(tmp_path, "a: &x [1, *x]\n", "a.yaml")))
    assert lines == ["a[0]: 1", "a[1]: <cycle>"]


def test_yaml_alias_bomb_stops_at_the_budget(tmp_path):
    pytest.importorskip("yaml")
    body = 'a: &a ["x", "x", "x", "x", "x", "x", "x", "x", "x", "x"]\n'
    for i, prev in zip("bcdefgh", "abcdefg"):
        body += f"{i}: &{i} [{', '.join(['*' + prev] * 10)}]\n"
    path = _write(tmp_path, body, "a.yaml")  # 10**8 strings if expanded
    with pytest.raises(ValueError, match="max_values"):
        list(YAMLReader(input_path=path, max_values=100_000).get_raw_chunks())


def test_malformed_yaml_is_a_value_error(tmp_path):
    pytest.importorskip("yaml")
    with pytest.raises(ValueError, match="YAMLReader: a.yaml: not YAML"):
        list(YAMLReader(input_path=_write(tmp_path, "a: [1, 2\n", "a.yaml")).get_raw_chunks())


def test_missing_pyyaml_names_the_install(tmp_path):
    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("no yaml")
        return real(name, *args, **kwargs)

    reader = YAMLReader(input_path=_write(tmp_path, "a: 1\n", "a.yaml"))
    with mock.patch.object(builtins, "__import__", refuse):
        with pytest.raises(ImportError, match="pip install pyyaml"):
            list(reader.get_raw_chunks())


# ---------------------------------------------------------------------------
# TOML
# ---------------------------------------------------------------------------


def test_toml_tables_arrays_and_datetimes(tmp_path):
    body = (
        'title = "Config"\n'
        "[server]\nhost = \"example.org\"\nports = [80, 443]\n"
        "started = 1979-05-27T07:32:00Z\n"
        '[[users]]\nname = "Ada"\n[[users]]\nname = "Bob"\n'
    )
    chunks = list(TOMLReader(input_path=_write(tmp_path, body, "a.toml")).get_raw_chunks())
    assert chunks[0]["text"].split("\n") == [
        "title: Config",
        "server.host: example.org",
        "server.ports[0]: 80",
        "server.ports[1]: 443",
        "server.started: 1979-05-27T07:32:00+00:00",
        "users[0].name: Ada",
        "users[1].name: Bob",
    ]
    assert chunks[0]["format"] == "toml"


def test_malformed_toml_is_a_value_error(tmp_path):
    with pytest.raises(ValueError, match="TOMLReader: a.toml: not TOML"):
        list(TOMLReader(input_path=_write(tmp_path, "a = = 1\n", "a.toml")).get_raw_chunks())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "step"),
    [("name", ".name"), ("snake_case-2", ".snake_case-2"), ("a b", '["a b"]'),
     ("2x", '["2x"]'), ("é", '["é"]'), ('q"', '["q\\""]')],
)
def test_key_step(key, step):
    assert _key_step(key) == step


def test_lines_handle_deep_nesting_without_recursion():
    value: list = []
    for _ in range(100_000):
        value = [value]
    lines = list(_lines(value, "", "all", [10**6]))
    assert len(lines) == 1 and lines[0].endswith(": []")


# ---------------------------------------------------------------------------
# Paths, include / exclude and records mode
# ---------------------------------------------------------------------------

from .._structured import _parse_path, _select  # noqa: E402

_API = {
    "meta": {"page": 1, "token": "secret"},
    "data": {
        "articles": [
            {"title": "First", "body": "Alpha text.", "author": "Ada",
             "tags": ["a", "b"], "link": {"href": "https://e.x/1"}},
            {"title": "Second", "body": "Beta text.", "author": "Bob", "tags": []},
            {"title": "", "body": "", "author": "Cy"},
        ]
    },
}


@pytest.mark.parametrize(
    ("expr", "steps"),
    [
        ("$", ()),
        ("", ()),
        ("a", (("key", "a"),)),
        ("$.a.b", (("key", "a"), ("key", "b"))),
        ("a[0][*]", (("key", "a"), ("index", 0), ("any_index",))),
        ('["odd key"].*', (("key", "odd key"), ("any_key",))),
        ('*["q\\"x"]', (("any_key",), ("key", 'q"x'))),
    ],
)
def test_parse_path(expr, steps):
    assert _parse_path(expr) == steps


@pytest.mark.parametrize("expr", ["a..b", "a[x]", "a[-1]", 'a["open', "a b", "a.", 3])
def test_parse_path_refuses_bad_paths(expr):
    with pytest.raises(ValueError):
        _parse_path(expr)


def test_select_fans_out_and_skips_missing():
    assert _select(_API, _parse_path("data.articles[*].title")) == ["First", "Second", ""]
    assert _select(_API, _parse_path("data.articles[*].link.href")) == ["https://e.x/1"]
    assert _select(_API, _parse_path("data.articles[9].title")) == []
    assert _select({1: "int key"}, _parse_path('["1"]')) == ["int key"]


def test_include_and_exclude_choose_subtrees(tmp_path):
    path = _write(tmp_path, json.dumps(_API), "a.json")
    lines = _texts(JSONReader(input_path=path, include=["data.articles[*].title"]))
    assert lines == [
        "data.articles[0].title: First",
        "data.articles[1].title: Second",
        "data.articles[2].title: ",
    ]
    lines = _texts(JSONReader(input_path=path, exclude=["meta.token", "data.articles[*].tags"]))
    assert "meta.token: secret" not in lines and "meta.page: 1" in lines
    assert not any(".tags" in line for line in lines)


def test_records_with_a_template_and_metadata(tmp_path):
    path = _write(tmp_path, json.dumps(_API), "a.json")
    reader = JSONReader(
        input_path=path,
        record_path="data.articles[*]",
        template="# {title}\n\n{body} {{tags: {tags[*]}}}",
        metadata_fields={"author": "author", "url": "link.href", "tag": "tags[*]"},
    )
    chunks = list(reader.get_raw_chunks())
    assert [c["text"] for c in chunks] == [
        "# First\n\nAlpha text. {tags: a\nb}",
        "# Second\n\nBeta text. {tags: }",
        "# \n\n {tags: }",
    ]
    assert [c["record_index"] for c in chunks] == [0, 1, 2]
    assert chunks[0]["url"] == "https://e.x/1" and chunks[0]["tag"] == ["a", "b"]
    assert chunks[1]["author"] == "Bob" and "url" not in chunks[1] and "tag" not in chunks[1]


def test_records_with_text_fields_count_empty_records(tmp_path):
    path = _write(tmp_path, json.dumps(_API), "a.json")
    reader = JSONReader(input_path=path, record_path="data.articles[*]", text_fields=["title", "body"])
    assert [c["text"] for c in reader.get_raw_chunks()] == [
        "First\n\nAlpha text.",
        "Second\n\nBeta text.",
    ]
    assert reader.coverage.read == 2 and reader.coverage.empty == 1


def test_a_chat_dataset_in_json_lines(tmp_path):
    rows = [
        {"id": "c1", "messages": [{"role": "user", "content": "Hi?"}, {"role": "assistant", "content": "Hello."}]},
        {"id": "c2", "messages": [{"role": "user", "content": "Bye."}]},
    ]
    path = _write(tmp_path, "\n".join(json.dumps(r) for r in rows), "chats.jsonl")
    reader = JSONReader(
        input_path=path,
        text_fields=["messages[*].content"],
        field_separator="\n",
        metadata_fields={"conversation_id": "id"},
    )
    chunks = list(reader.get_raw_chunks())
    assert [(c["text"], c["conversation_id"], c["record_start"]) for c in chunks] == [
        ("Hi?\nHello.", "c1", 1),
        ("Bye.", "c2", 2),
    ]


def test_records_default_to_top_level_elements_with_path_lines(tmp_path):
    path = _write(tmp_path, json.dumps([{"a": 1}, {"a": 2, "b": [3]}]), "a.json")
    chunks = list(JSONReader(input_path=path, metadata_fields=["a"]).get_raw_chunks())
    assert [(c["text"], c["a"]) for c in chunks] == [(".a: 1", 1), (".a: 2\n.b[0]: 3", 2)]


def test_a_container_field_is_compact_json(tmp_path):
    path = _write(tmp_path, json.dumps({"x": {"k": [1, "é"]}}), "a.json")
    [chunk] = JSONReader(input_path=path, text_fields=["x"]).get_raw_chunks()
    assert chunk["text"] == '{"k":[1,"é"]}'


def test_documents_carry_promoted_metadata(tmp_path):
    rows = [{"text": "A sentence that is long enough to keep.", "link": "https://e.x/a"}]
    path = _write(tmp_path, json.dumps(rows), "a.json")
    reader = JSONReader(input_path=path, text_fields=["text"], metadata_fields={"url": "link"})
    [doc] = list(reader.get_documents())
    assert doc.url == "https://e.x/a"


def test_yaml_documents_are_records(tmp_path):
    pytest.importorskip("yaml")
    path = _write(tmp_path, "title: One\n---\ntitle: Two\n", "a.yaml")
    chunks = list(YAMLReader(input_path=path, text_fields=["title"]).get_raw_chunks())
    assert [(c["text"], c["record_start"]) for c in chunks] == [("One", 0), ("Two", 1)]


def test_the_value_budget_applies_in_records_mode(tmp_path):
    path = _write(tmp_path, json.dumps([{"t": list(range(100))}]), "a.json")
    with pytest.raises(ValueError, match="max_values"):
        list(JSONReader(input_path=path, text_fields=["t[*]"], max_values=10).get_raw_chunks())


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"text_fields": ["a"], "template": "{a}"}, ValueError, "not both"),
        ({"text_fields": []}, ValueError, "must not be empty"),
        ({"text_fields": "a"}, TypeError, "list of paths"),
        ({"include": ["a..b"]}, ValueError, "include: bad path"),
        ({"record_path": "a[x]"}, ValueError, "record_path: bad path"),
        ({"template": "{a..b}"}, ValueError, "template: bad path"),
        ({"template": "{a"}, ValueError, "unmatched brace"),
        ({"template": 3}, TypeError, "template must be a str"),
        ({"metadata_fields": {"text": "a"}}, ValueError, "reserved"),
        ({"metadata_fields": ["source_type"]}, ValueError, "reserved"),
        ({"metadata_fields": "a"}, TypeError, "list of paths or a mapping"),
        ({"field_separator": None}, TypeError, "field_separator"),
    ],
)
def test_bad_record_options_fail_at_construction(tmp_path, kwargs, error, message):
    with pytest.raises(error, match=message):
        JSONReader(input_path=_write(tmp_path, "{}", "a.json"), **kwargs)


def test_filters_stay_linear_on_deep_nesting():
    from .._structured import _Filter, _lines

    value: list = ["leaf"]
    for _ in range(50_000):
        value = [value]
    keep = _Filter((_parse_path("[0][0][*]"),), (_parse_path("[1]"),))
    lines = list(_lines(value, "", "all", [10**6], keep=keep))
    assert len(lines) == 1 and lines[0].endswith(": leaf")
