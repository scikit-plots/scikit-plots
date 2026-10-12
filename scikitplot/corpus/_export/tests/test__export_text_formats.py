# scikitplot/corpus/_export/tests/test__export_text_formats.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Document export to JSON, JSON Lines, YAML and TOML, and loading it back."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import pathlib

import numpy as np
import pytest

from ..._pipeline import _FORMAT_SUFFIX
from ..._schema import CorpusDocument, ExportFormat
from .._export import (
    EMBEDDING_DTYPE_KEY,
    TEXT_FORMATS,
    export_documents,
    load_documents,
)

FORMATS = ["json", "jsonl", "yaml", "toml"]


def _need(fmt):
    if fmt == "yaml":
        pytest.importorskip("yaml")
    if fmt == "toml":
        pytest.importorskip("tomli_w")
        try:
            import tomllib  # noqa: F401
        except ImportError:
            pytest.importorskip("tomli")


def _doc(i=0, **kwargs):
    defaults = {"metadata": {"source": "unit", "page": i}}
    defaults.update(kwargs)
    return CorpusDocument.create("notes.txt", i, f"Paragraph {i}.", **defaults)


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


@pytest.mark.parametrize("fmt", FORMATS)
class TestRoundTrip:
    def test_every_field_comes_back(self, fmt, tmp_path):
        _need(fmt)
        docs = [
            _doc(0, language="en", tokens=["Paragraph", "0"], bbox=(0.0, 1.0, 2.0, 3.0)),
            _doc(1, metadata={"nested": {"a": [1, 2.5, "x"]}, "flag": True, "empty": ""}),
        ]
        back = load_documents(export_documents(docs, tmp_path / f"c.{fmt}", ExportFormat(fmt)))
        assert [_same(a.to_dict(), b.to_dict()) for a, b in zip(docs, back)] == [True, True]

    @pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
    def test_embeddings_keep_dtype_values_and_nan(self, fmt, dtype, tmp_path):
        _need(fmt)
        vector = np.array([0.1, -2.5, np.nan, 1e-3], dtype=dtype)
        back = load_documents(export_documents([_doc(embedding=vector)], tmp_path / f"e.{fmt}", ExportFormat(fmt)))
        assert back[0].embedding.dtype == dtype
        assert np.array_equal(back[0].embedding, vector, equal_nan=True)

    def test_awkward_strings(self, fmt, tmp_path):
        _need(fmt)
        text = 'Line\nbreak \x85 NEL \u2028 LS \u2029 PS "quotes" \\ $x {y} # : - ~ null true 中 \U0001F600'
        doc = CorpusDocument.create("p.txt", 0, text, metadata={"k": text})
        back = load_documents(export_documents([doc], tmp_path / f"s.{fmt}", ExportFormat(fmt)))
        assert back[0].text == text
        assert back[0].metadata == {"k": text}

    def test_empty_corpus(self, fmt, tmp_path):
        _need(fmt)
        assert load_documents(export_documents([], tmp_path / f"n.{fmt}", ExportFormat(fmt))) == []

    def test_numpy_scalars_in_metadata(self, fmt, tmp_path):
        """F-56: a NumPy scalar in metadata made JSON export raise TypeError."""
        _need(fmt)
        doc = _doc(metadata={"n": np.int64(3), "x": np.float32(0.5), "ok": np.bool_(True)})
        back = load_documents(export_documents([doc], tmp_path / f"np.{fmt}", ExportFormat(fmt)))
        assert back[0].metadata == {"n": 3, "x": 0.5, "ok": True}

    def test_integrity_check_applies(self, fmt, tmp_path):
        _need(fmt)
        path = export_documents([_doc()], tmp_path / f"i.{fmt}", ExportFormat(fmt))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert len(load_documents(path, expected_sha256=digest)) == 1
        with pytest.raises(ValueError, match="does not match"):
            load_documents(path, expected_sha256="0" * 64)


class TestStrictness:
    @pytest.mark.parametrize("fmt", ["json", "jsonl"])
    def test_json_has_no_nan_token(self, fmt, tmp_path):
        """F-55: NaN was written as the non-standard token NaN."""
        path = export_documents([_doc(metadata={"score": float("nan")})], tmp_path / f"x.{fmt}", ExportFormat(fmt))
        text = path.read_text(encoding="utf-8")
        assert "NaN" not in text
        first = text if fmt == "json" else text.splitlines()[0]
        json.loads(first, parse_constant=lambda token: pytest.fail(f"non-standard token {token}"))
        assert load_documents(path)[0].metadata == {"score": None}  # documented: JSON has no NaN

    @pytest.mark.parametrize("fmt", ["yaml", "toml"])
    def test_yaml_and_toml_keep_metadata_nan(self, fmt, tmp_path):
        _need(fmt)
        path = export_documents([_doc(metadata={"score": float("nan")})], tmp_path / f"x.{fmt}", ExportFormat(fmt))
        assert math.isnan(load_documents(path)[0].metadata["score"])

    def test_json_lines_survive_line_splitting(self, tmp_path):
        doc = CorpusDocument.create("p.txt", 0, "a\u2028b\x85c")
        path = export_documents([doc, _doc(1)], tmp_path / "x.jsonl", ExportFormat.JSONL)
        assert len(path.read_text(encoding="utf-8").splitlines()) == 2

    @pytest.mark.parametrize("fmt", ["json", "jsonl"])
    def test_infinity_names_the_document(self, fmt, tmp_path):
        with pytest.raises(ValueError, match=r"document 1 .*\$\.metadata\.score is inf"):
            export_documents([_doc(0), _doc(1, metadata={"score": float("inf")})], tmp_path / f"x.{fmt}", ExportFormat(fmt))
        assert not (tmp_path / f"x.{fmt}").exists()

    @pytest.mark.parametrize(
        ("metadata", "match"),
        [({"note": None}, r"\$\.metadata\.note is None"), ({"tags": ["a", None]}, r"\$\.metadata\.tags\[1\] is None")],
    )
    def test_toml_refuses_what_it_would_drop(self, metadata, match, tmp_path):
        _need("toml")
        with pytest.raises(ValueError, match=match):
            export_documents([_doc(metadata=metadata)], tmp_path / "x.toml", ExportFormat.TOML)

    @pytest.mark.parametrize("value", [datetime.date(2026, 1, 2), {1, 2}, object()])
    def test_unserialisable_metadata_names_the_document(self, value, tmp_path):
        with pytest.raises(TypeError, match=r"document 0 \(doc_id="):
            export_documents([_doc(metadata={"v": value})], tmp_path / "x.json", ExportFormat.JSON)


class TestLoading:
    @pytest.mark.parametrize(
        ("suffix", "fmt"),
        [(".json", "json"), (".jsonl", "jsonl"), (".ndjson", "jsonl"), (".yaml", "yaml"), (".yml", "yaml"), (".toml", "toml")],
    )
    def test_suffix_inference(self, suffix, fmt, tmp_path):
        _need(fmt)
        path = export_documents([_doc()], tmp_path / f"c{suffix}", ExportFormat(fmt))
        assert len(load_documents(path)) == 1

    def test_unknown_suffix(self, tmp_path):
        path = tmp_path / "c.txt"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="cannot infer"):
            load_documents(path)
        assert load_documents(path, ExportFormat.JSON) == []

    @pytest.mark.parametrize(
        "fmt", [ExportFormat.CSV, ExportFormat.PANDAS, ExportFormat.PARQUET, ExportFormat.POLARS,
                ExportFormat.NUMPY, ExportFormat.HUGGINGFACE],
    )
    def test_lossy_formats_are_refused_not_empty(self, fmt, tmp_path):
        """F-57: these returned [] with a log warning."""
        path = tmp_path / "c.bin"
        path.write_bytes(b"x")
        with pytest.raises(ValueError, match="cannot be loaded back"):
            load_documents(path, fmt)

    def test_mlflow_points_to_the_artifact(self, tmp_path):
        with pytest.raises(ValueError, match=r"corpus_documents\.jsonl"):
            load_documents(tmp_path / "x", ExportFormat.MLFLOW)

    @pytest.mark.parametrize(
        ("name", "text", "match"),
        [
            ("a.json", '{"doc_id": "x"}', "expected a list"),
            ("b.json", "[1]", "record 0: it is int"),
            ("c.json", '[{"schema_version": "2.0", "doc_id": "x"}]', "record 0: missing required keys"),
            ("g.json", "[NaN]", r"not valid json.*earlier versions of export_documents"),
            ("h.json", "[1,", r"h\.json' is not valid json"),
        ],
    )
    def test_malformed_files(self, name, text, match, tmp_path):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_documents(path)

    @pytest.mark.parametrize(
        ("change", "match"),
        [
            ({"chunk_index": 1.7}, "chunk_index is float, expected an integer"),
            ({"chunk_index": True}, "chunk_index is bool, expected an integer"),
            ({"chunk_index": "7"}, "chunk_index is str, expected an integer"),
            ({"text": ["a"]}, "text is list, expected a string"),
            ({"tokens": "abc"}, "tokens is str, expected a list of strings"),
            ({"bbox": "1234"}, "bbox is str, expected a list of 4 numbers"),
            ({"bbox": [1, 2, 3]}, "bbox is list, expected a list of 4 numbers"),
            ({"metadata": ""}, "metadata is str, expected a mapping"),
            ({"url": ["x"]}, "url is list, expected a string"),
            ({"surprise": 1}, r"unknown keys \['surprise'\]"),
            ({"embedding": [1, 2], "embedding_dtype": "object"}, "is not one of"),
            ({"embedding": ["a"]}, "embedding is list, expected a list of numbers"),
            ({"embedding": [True]}, "embedding is list, expected a list of numbers"),
            ({"embedding_dtype": "float32"}, "embedding_dtype without an embedding"),
            ({"embedding": [70000.0], "embedding_dtype": "float16"}, "overflows float16"),
            ({"embedding": [10**400], "embedding_dtype": "float64"}, "overflows float64"),
        ],
    )
    def test_off_contract_records_are_refused(self, change, match, tmp_path):
        """Review of 2c #2, #3: these loaded as plausible documents, or crashed with another exception."""
        record = {**_doc().to_dict(), **change}
        path = tmp_path / "r.json"
        path.write_text(json.dumps([record]), encoding="utf-8")
        with pytest.raises(ValueError, match=f"record 0: .*{match}"):
            load_documents(path)

    @pytest.mark.parametrize(("fmt", "match"), [("yaml", r"source_date is a date"), ("toml", r"text is a date")])
    def test_yaml_and_toml_dates_are_refused(self, fmt, match, tmp_path):
        """Review of 2c #2, #3: an unquoted date loaded as datetime.date (or crashed)."""
        _need(fmt)
        record = _doc().to_dict()
        record.pop("source_date", None)
        record.pop("text", None)
        if fmt == "yaml":
            import yaml  # noqa: PLC0415

            body = yaml.safe_dump(record, default_flow_style=False, sort_keys=False)
            content = "- " + body.rstrip("\n").replace("\n", "\n  ") + "\n  source_date: 2026-01-02\n"
        else:
            import tomli_w  # noqa: PLC0415

            head, _, tail = tomli_w.dumps({k: v for k, v in record.items() if v is not None}).partition("[metadata]")
            content = "[[documents]]\n" + head + "text = 1979-05-27\n[documents.metadata]" + tail
        path = tmp_path / f"d.{fmt}"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_documents(path)

    def test_toml_needs_the_documents_table(self, tmp_path):
        pytest.importorskip("tomli_w")
        path = tmp_path / "c.toml"
        path.write_text('title = "not an export"\n', encoding="utf-8")
        with pytest.raises(ValueError, match="exactly one key"):
            load_documents(path)

    def test_records_without_a_dtype_load_as_float64(self, tmp_path):
        record = _doc().to_dict()
        record["embedding"] = [0.1, 0.2]
        path = tmp_path / "old.json"
        path.write_text(json.dumps([record]), encoding="utf-8")
        loaded = load_documents(path)[0].embedding
        assert loaded.dtype == np.float64
        assert loaded.tolist() == [0.1, 0.2]

    def test_format_may_be_its_string_value(self, tmp_path):
        """Review of 2c: export_documents(..., "json") raised AttributeError."""
        path = export_documents([_doc()], tmp_path / "c.data", "JSON")
        assert len(load_documents(path, "json")) == 1
        with pytest.raises(ValueError, match="unknown format 'xml'"):
            load_documents(path, "xml")
        with pytest.raises(TypeError, match="ExportFormat or its string value"):
            load_documents(path, 3)

    def test_text_formats_need_no_trust(self, tmp_path):
        path = export_documents([_doc()], tmp_path / "c.json", ExportFormat.JSON)
        assert load_documents(path, trusted=False)[0].text == "Paragraph 0."


def test_the_dtype_key_is_only_written_with_an_embedding(tmp_path):
    path = export_documents([_doc(), _doc(1, embedding=np.zeros(2, np.float32))], tmp_path / "c.json", ExportFormat.JSON)
    records = json.loads(path.read_text(encoding="utf-8"))
    assert EMBEDDING_DTYPE_KEY not in records[0]
    assert records[1][EMBEDDING_DTYPE_KEY] == "float32"
    without = export_documents([_doc(embedding=np.zeros(2))], tmp_path / "d.json", ExportFormat.JSON, include_embedding=False)
    assert "embedding" not in json.loads(without.read_text(encoding="utf-8"))[0]


def test_pipeline_knows_every_text_suffix():
    assert {fmt: _FORMAT_SUFFIX[fmt] for fmt in TEXT_FORMATS} == {
        ExportFormat.JSON: ".json", ExportFormat.JSONL: ".jsonl", ExportFormat.YAML: ".yaml", ExportFormat.TOML: ".toml",
    }


def test_toml_layout(tmp_path):
    _need("toml")
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    path = export_documents([_doc()], tmp_path / "c.toml", ExportFormat.TOML)
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    assert list(data) == ["documents"]
    assert data["documents"][0]["text"] == "Paragraph 0."
    assert isinstance(pathlib.Path(path), pathlib.Path)


class TestReviewFixes:
    """Increment 2c independent review."""

    @pytest.mark.parametrize(
        ("embedding", "match"),
        [
            (np.array([1, 2], dtype=np.int8), "dtype int8 is not one of"),
            (np.array([True, False]), "dtype bool is not one of"),
            (np.array([1 + 2j]), "dtype complex128 is not one of"),
            (np.zeros((2, 3), dtype=np.float32), r"shape \(2, 3\); text exports hold one vector"),
        ],
    )
    def test_embeddings_that_cannot_round_trip_are_refused(self, embedding, match, tmp_path):
        """Review #1, #5: these exported, then failed to load, or lost data (2-D silently dropped)."""
        with pytest.raises(ValueError, match=rf"document 0 .*{match}"):
            export_documents([_doc(embedding=embedding)], tmp_path / "x.json", ExportFormat.JSON)
        assert not (tmp_path / "x.json").exists()

    def test_a_list_of_numbers_is_float64(self, tmp_path):
        """Review #1: a Python list of ints recorded int64 and could not be loaded."""
        path = export_documents([_doc(embedding=[1, 2, 3])], tmp_path / "x.json", ExportFormat.JSON)
        back = load_documents(path)[0].embedding
        assert back.dtype == np.float64
        assert back.tolist() == [1.0, 2.0, 3.0]

    @pytest.mark.parametrize(("field", "value"), [("bbox", (0.0, float("nan"), 1.0, 1.0)), ("timecode_start", float("nan"))])
    @pytest.mark.parametrize("fmt", ["json", "jsonl"])
    def test_json_refuses_nan_outside_metadata(self, field, value, fmt, tmp_path):
        """Review #4: NaN in bbox could not be loaded; in timecodes it came back None."""
        with pytest.raises(ValueError, match=rf"\$\.{field}.* is nan"):
            export_documents([_doc(**{field: value})], tmp_path / f"x.{fmt}", ExportFormat(fmt))

    @pytest.mark.parametrize("fmt", ["yaml", "toml"])
    def test_yaml_and_toml_keep_nan_outside_metadata(self, fmt, tmp_path):
        _need(fmt)
        doc = _doc(bbox=(0.0, float("nan"), 1.0, 1.0), timecode_start=float("nan"))
        back = load_documents(export_documents([doc], tmp_path / f"x.{fmt}", ExportFormat(fmt)))[0]
        assert math.isnan(back.bbox[1])
        assert math.isnan(back.timecode_start)

    def test_deep_metadata_names_the_document(self, tmp_path):
        """Review #6: a bare RecursionError escaped."""
        deep: dict = {}
        node = deep
        for _ in range(5000):
            node["k"] = {}
            node = node["k"]
        with pytest.raises(ValueError, match=r"document 0 .*nested too deeply"):
            export_documents([_doc(metadata=deep)], tmp_path / "x.json", ExportFormat.JSON)

    @pytest.mark.parametrize("fmt", ["json", "jsonl", "toml"])
    def test_lone_surrogates_name_the_document(self, fmt, tmp_path):
        """Review #6: UnicodeEncodeError escaped from the write."""
        _need(fmt)
        with pytest.raises(ValueError, match=r"document 0 .*\$\.metadata\.k holds a lone surrogate"):
            export_documents([_doc(metadata={"k": "a\ud800b"})], tmp_path / f"x.{fmt}", ExportFormat(fmt))

    def test_yaml_keeps_lone_surrogates(self, tmp_path):
        _need("yaml")
        path = export_documents([_doc(metadata={"k": "a\ud800b"})], tmp_path / "x.yaml", ExportFormat.YAML)
        assert load_documents(path)[0].metadata == {"k": "a\ud800b"}

    def test_huge_integers_name_the_document(self, tmp_path):
        """Review #6: an int beyond Python's int-to-text limit raised an unnamed ValueError."""
        import sys  # noqa: PLC0415

        limit = getattr(sys, "get_int_max_str_digits", lambda: 0)()
        if not limit:
            pytest.skip("this Python has no int-to-text limit")
        with pytest.raises(ValueError, match=r"document 1 \(doc_id=.*cannot be written"):
            export_documents([_doc(0), _doc(1, metadata={"n": 10 ** (limit + 10)})], tmp_path / "x.json", ExportFormat.JSON)
        assert not (tmp_path / "x.json").exists()

    @pytest.mark.parametrize(("suffix", "fmt"), [(".csv", "csv"), (".parquet", "parquet"), (".npy", "numpy")])
    def test_lossy_suffixes_give_the_reason(self, suffix, fmt, tmp_path):
        """Review #8: these said 'cannot infer the format', then were refused anyway."""
        path = tmp_path / f"c{suffix}"
        path.write_bytes(b"x")
        with pytest.raises(ValueError, match=f"'{fmt}' exports cannot be loaded back"):
            load_documents(path)

    def test_record_contract_lists_every_field(self):
        """FIELD_KINDS must hold exactly the to_dict keys, kinds agreeing with the annotations."""
        import dataclasses  # noqa: PLC0415

        from .._records import FIELD_KINDS  # noqa: PLC0415

        keys = set(CorpusDocument.create("f", 0, "x", embedding=[0.0]).to_dict(include_embedding=True))
        assert keys | {EMBEDDING_DTYPE_KEY} == set(FIELD_KINDS)
        annotations = {f.name: str(f.type) for f in dataclasses.fields(CorpusDocument)}
        expected = {"str": "str", "int": "int", "float": "float", "bool": "bool"}
        for key, kind in FIELD_KINDS.items():
            base = kind.rstrip("?")
            enum_types = ("SectionType", "ChunkingStrategy", "SourceType", "Modality")
            if base in expected and key in annotations and not annotations[key].startswith(enum_types):
                assert annotations[key].startswith(expected[base]), (key, kind, annotations[key])
            if not kind.endswith("?") and key in annotations and key != "text":
                assert "None" not in annotations[key], (key, annotations[key])
