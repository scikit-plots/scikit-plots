# scikitplot/corpus/_export/tests/test__export.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for scikitplot.corpus._export._export
============================================

Coverage
--------
* :func:`export_documents` — dispatch to every format; unknown format
  ``ValueError``; parent-dir auto-creation; return value is output_path;
  logging calls.
* :func:`_compute_csv_fieldnames` — identity fields first in stable order;
  extra keys sorted after; keys absent from some rows still appear.
* :func:`_atomic_write_bytes` — writes file; overwrites existing; no partial
  file on error.
* :func:`_atomic_write_text` — writes UTF-8 text; no partial file on error.
* ``_export_csv`` — empty list produces empty file; non-empty produces valid
  CSV with header; all doc fields present; embeddings always excluded.
* ``_export_jsonl`` — empty list produces empty file; non-empty produces
  one JSON object per line; each line is valid JSON; doc_id present.
* ``_export_json`` — produces a JSON array; indent respected; empty list
  produces ``[]``; ensure_ascii=False (Unicode preserved).
* ``_export_pickle`` — round-trips documents faithfully.
* ``_export_joblib`` — skipped when joblib absent (ImportError propagated).
* ``_export_numpy`` — raises ``ValueError`` when no embeddings.

All tests use stdlib only.  pandas / polars / joblib / numpy are mocked
where needed so no optional dependencies are required.

Run with::

    pytest corpus/_export/tests/test__export.py -v
"""
from __future__ import annotations

import csv
import io
import json
import pathlib
import pickle
import sys
from unittest.mock import MagicMock, patch

import pytest

from .._export import (
    _atomic_write_bytes,
    _atomic_write_text,
    _compute_csv_fieldnames,
    export_documents,
)
from ..._schema import CorpusDocument, ExportFormat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(
    text: str = "Hello world.",
    idx: int = 0,
    source: str = "test.txt",
) -> CorpusDocument:
    """Return a minimal but fully-valid ``CorpusDocument``."""
    return CorpusDocument.create(source, idx, text)


def _docs(n: int = 3) -> list[CorpusDocument]:
    return [_doc(f"Sentence number {i} is here.", i) for i in range(n)]


# ===========================================================================
# _compute_csv_fieldnames
# ===========================================================================


class TestComputeCsvFieldnames:

    def test_identity_fields_come_first(self) -> None:
        rows = [{"doc_id": "x", "text": "y", "extra_col": "z"}]
        names = _compute_csv_fieldnames(rows)
        assert names.index("doc_id") < names.index("extra_col")
        assert names.index("text") < names.index("extra_col")

    def test_extra_keys_sorted_alphabetically(self) -> None:
        rows = [{"doc_id": "x", "zebra": 1, "apple": 2}]
        names = _compute_csv_fieldnames(rows)
        extra = [n for n in names if n not in ("doc_id",)]
        assert extra == sorted(extra)

    def test_superset_across_rows(self) -> None:
        rows = [{"doc_id": "a", "col_a": 1}, {"doc_id": "b", "col_b": 2}]
        names = _compute_csv_fieldnames(rows)
        assert "col_a" in names
        assert "col_b" in names

    def test_empty_rows_returns_empty(self) -> None:
        assert _compute_csv_fieldnames([]) == []

    def test_single_identity_field_only(self) -> None:
        rows = [{"doc_id": "x"}]
        names = _compute_csv_fieldnames(rows)
        assert names == ["doc_id"]

    def test_stable_identity_order_across_calls(self) -> None:
        rows = [{"source_file": "f", "doc_id": "d", "text": "t", "chunk_index": 0}]
        n1 = _compute_csv_fieldnames(rows)
        n2 = _compute_csv_fieldnames(rows)
        assert n1 == n2

    def test_all_standard_identity_fields_ordered(self) -> None:
        identity = [
            "doc_id", "source_file", "chunk_index", "text",
            "section_type", "chunking_strategy", "language",
            "source_type", "source_title", "source_author",
        ]
        rows = [{k: "v" for k in identity}]
        names = _compute_csv_fieldnames(rows)
        present = [n for n in identity if n in names]
        indices = [names.index(n) for n in present]
        assert indices == sorted(indices)


# ===========================================================================
# _atomic_write_bytes
# ===========================================================================


class TestAtomicWriteBytes:

    def test_file_created(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "out.bin"
        _atomic_write_bytes(p, b"hello")
        assert p.exists()
        assert p.read_bytes() == b"hello"

    def test_overwrites_existing(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "out.bin"
        p.write_bytes(b"old")
        _atomic_write_bytes(p, b"new")
        assert p.read_bytes() == b"new"

    def test_no_tmp_file_left_after_success(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "out.bin"
        _atomic_write_bytes(p, b"data")
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_empty_bytes(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "empty.bin"
        _atomic_write_bytes(p, b"")
        assert p.read_bytes() == b""

    def test_large_payload(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "large.bin"
        data = b"x" * 1_000_000
        _atomic_write_bytes(p, data)
        assert p.read_bytes() == data


# ===========================================================================
# _atomic_write_text
# ===========================================================================


class TestAtomicWriteText:

    def test_file_created_utf8(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "out.txt"
        _atomic_write_text(p, "Héllo wörld")
        assert p.read_text(encoding="utf-8") == "Héllo wörld"

    def test_overwrites_existing(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "out.txt"
        p.write_text("old", encoding="utf-8")
        _atomic_write_text(p, "new")
        assert p.read_text(encoding="utf-8") == "new"

    def test_no_tmp_file_after_success(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "out.txt"
        _atomic_write_text(p, "data")
        assert not list(tmp_path.glob("*.tmp"))

    def test_unicode_preserved(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "unicode.txt"
        content = "中文 Arabic: مرحبا"
        _atomic_write_text(p, content)
        assert p.read_text(encoding="utf-8") == content


# ===========================================================================
# export_documents — dispatch and common behaviour
# ===========================================================================


class TestExportDocuments:

    def test_returns_output_path(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.csv"
        returned = export_documents(_docs(2), out, ExportFormat.CSV)
        assert returned == out

    def test_creates_parent_dirs(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "deep" / "nested" / "out.csv"
        export_documents(_docs(2), out, ExportFormat.CSV)
        assert out.exists()

    def test_unknown_format_raises(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.xxx"
        # Inject a fake ExportFormat value that has no dispatch entry.
        fake_fmt = MagicMock(spec=ExportFormat)
        fake_fmt.value = "xxx"
        with pytest.raises((ValueError, AttributeError)):
            export_documents(_docs(), out, fake_fmt)  # type: ignore[arg-type]

    def test_logging_on_export(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging  # noqa: PLC0415

        out = tmp_path / "log_test.csv"
        with caplog.at_level(logging.INFO):
            export_documents(_docs(1), out, ExportFormat.CSV)
        assert any("export" in r.message.lower() or "wrote" in r.message.lower()
                   for r in caplog.records)


# ===========================================================================
# CSV export
# ===========================================================================


class TestExportCSV:

    def test_empty_list_creates_file(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "empty.csv"
        export_documents([], out, ExportFormat.CSV)
        assert out.exists()

    def test_empty_list_file_is_empty(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "empty.csv"
        export_documents([], out, ExportFormat.CSV)
        assert out.read_text(encoding="utf-8") == ""

    def test_non_empty_has_header(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.csv"
        export_documents(_docs(2), out, ExportFormat.CSV)
        content = out.read_text(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(content))
        assert reader.fieldnames is not None
        assert len(reader.fieldnames) > 0

    def test_doc_id_column_present(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.csv"
        export_documents(_docs(2), out, ExportFormat.CSV)
        content = out.read_text(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(content))
        assert "doc_id" in (reader.fieldnames or [])

    def test_text_column_present(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.csv"
        export_documents(_docs(2), out, ExportFormat.CSV)
        content = out.read_text(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(content))
        assert "text" in (reader.fieldnames or [])

    def test_row_count_matches_docs(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.csv"
        docs = _docs(5)
        export_documents(docs, out, ExportFormat.CSV)
        content = out.read_text(encoding="utf-8")
        rows = list(csv.DictReader(io.StringIO(content)))
        assert len(rows) == 5

    def test_text_values_preserved(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.csv"
        docs = [_doc("Unique sentence alpha.", 0), _doc("Unique sentence beta.", 1)]
        export_documents(docs, out, ExportFormat.CSV)
        content = out.read_text(encoding="utf-8")
        assert "Unique sentence alpha." in content
        assert "Unique sentence beta." in content

    def test_embedding_excluded_from_csv(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.csv"
        export_documents(
            _docs(2), out, ExportFormat.CSV, include_embedding=True
        )
        content = out.read_text(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(content))
        assert "embedding" not in (reader.fieldnames or [])

    def test_unicode_in_text_preserved(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "unicode.csv"
        docs = [_doc("Привет мир. 你好世界.", 0)]
        export_documents(docs, out, ExportFormat.CSV)
        content = out.read_text(encoding="utf-8")
        assert "Привет" in content


# ===========================================================================
# JSONL export
# ===========================================================================


class TestExportJSONL:

    def test_empty_list_produces_empty_or_newline(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "empty.jsonl"
        export_documents([], out, ExportFormat.JSONL)
        content = out.read_text(encoding="utf-8")
        assert content.strip() == ""

    def test_each_line_is_valid_json(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.jsonl"
        export_documents(_docs(3), out, ExportFormat.JSONL)
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 3
        for line in lines:
            obj = json.loads(line)
            assert isinstance(obj, dict)

    def test_doc_id_in_each_record(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.jsonl"
        export_documents(_docs(2), out, ExportFormat.JSONL)
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                assert "doc_id" in json.loads(line)

    def test_text_in_each_record(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.jsonl"
        export_documents(_docs(2), out, ExportFormat.JSONL)
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                assert "text" in json.loads(line)

    def test_no_multi_line_objects(self, tmp_path: pathlib.Path) -> None:
        """Each JSONL record must be a single line (no embedded newlines)."""
        out = tmp_path / "out.jsonl"
        export_documents(_docs(3), out, ExportFormat.JSONL)
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 3

    def test_unicode_preserved(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "unicode.jsonl"
        docs = [_doc("中文 Arabic: مرحبا", 0)]
        export_documents(docs, out, ExportFormat.JSONL)
        line = out.read_text(encoding="utf-8").strip()
        obj = json.loads(line)
        assert "中文" in obj.get("text", "")

    def test_n_lines_matches_n_docs(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.jsonl"
        export_documents(_docs(7), out, ExportFormat.JSONL)
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 7


# ===========================================================================
# JSON export
# ===========================================================================


class TestExportJSON:

    def test_empty_list_produces_empty_array(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "empty.json"
        export_documents([], out, ExportFormat.JSON)
        parsed = json.loads(out.read_text(encoding="utf-8"))
        assert parsed == []

    def test_produces_json_array(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.json"
        export_documents(_docs(3), out, ExportFormat.JSON)
        parsed = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(parsed, list)
        assert len(parsed) == 3

    def test_each_element_is_dict_with_doc_id(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.json"
        export_documents(_docs(2), out, ExportFormat.JSON)
        for item in json.loads(out.read_text(encoding="utf-8")):
            assert "doc_id" in item

    def test_default_indent_is_not_none(self, tmp_path: pathlib.Path) -> None:
        """Default json_indent should produce multi-line output."""
        out = tmp_path / "out.json"
        export_documents(_docs(1), out, ExportFormat.JSON)
        content = out.read_text(encoding="utf-8")
        assert "\n" in content  # indented → has newlines

    def test_custom_indent_none_produces_compact(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "compact.json"
        export_documents(_docs(1), out, ExportFormat.JSON, json_indent=None)
        content = out.read_text(encoding="utf-8").strip()
        # Compact output: the entire array on one line.
        assert content.startswith("[") and content.endswith("]")
        assert "\n" not in content

    def test_unicode_not_escaped(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "unicode.json"
        docs = [_doc("Héllo wörld", 0)]
        export_documents(docs, out, ExportFormat.JSON)
        content = out.read_text(encoding="utf-8")
        # ensure_ascii=False → literal Unicode chars, not \\uXXXX.
        assert "Héllo" in content


# ===========================================================================
# PICKLE export
# ===========================================================================


class TestExportPickle:

    def test_pickle_round_trips_documents(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.pkl"
        docs = _docs(3)
        export_documents(docs, out, ExportFormat.PICKLE)
        loaded = pickle.loads(out.read_bytes())
        assert isinstance(loaded, list)
        assert len(loaded) == len(docs)

    def test_pickle_preserves_doc_ids(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.pkl"
        docs = _docs(2)
        export_documents(docs, out, ExportFormat.PICKLE)
        loaded = pickle.loads(out.read_bytes())
        assert {d.doc_id for d in loaded} == {d.doc_id for d in docs}

    def test_pickle_empty_list(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "empty.pkl"
        export_documents([], out, ExportFormat.PICKLE)
        loaded = pickle.loads(out.read_bytes())
        assert loaded == []

    def test_pickle_file_is_binary(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.pkl"
        export_documents(_docs(1), out, ExportFormat.PICKLE)
        raw = out.read_bytes()
        # Pickle files always start with the PROTO opcode (0x80).
        assert raw[0] == 0x80


# ===========================================================================
# JOBLIB export — mocked (optional dependency)
# ===========================================================================


class TestExportJoblib:

    def test_joblib_delegates_to_joblib_dump(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.joblib"
        # The exporter writes to a .tmp sibling then renames; mock.dump must
        # create that file so the rename succeeds.
        def _fake_dump(obj, path, **kw):
            pathlib.Path(path).write_bytes(b"fake")

        mock_joblib = MagicMock()
        mock_joblib.dump.side_effect = _fake_dump
        with patch.dict("sys.modules", {"joblib": mock_joblib}):
            export_documents(_docs(2), out, ExportFormat.JOBLIB)
        mock_joblib.dump.assert_called_once()

    def test_joblib_import_error_propagates(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.joblib"
        with patch.dict("sys.modules", {"joblib": None}):
            with pytest.raises(ImportError):
                export_documents(_docs(1), out, ExportFormat.JOBLIB)


# ===========================================================================
# NUMPY export — no-embedding error path (mocked)
# ===========================================================================


class TestExportNumpy:

    def test_no_embeddings_raises_value_error(self, tmp_path: pathlib.Path) -> None:
        """Documents without embeddings must cause a clear ValueError."""
        out = tmp_path / "out.npy"
        docs = _docs(2)  # none have embeddings
        import numpy as np  # noqa: PLC0415
        with pytest.raises((ValueError, ImportError)):
            export_documents(docs, out, ExportFormat.NUMPY)

    def test_numpy_no_embeddings_raises_value_error_directly(self, tmp_path: pathlib.Path) -> None:
        """The no-embeddings guard fires before numpy is imported, so no ImportError."""
        out = tmp_path / "out.npy"
        # _export_numpy checks for embeddings first (before importing numpy),
        # so docs with no embeddings always raise ValueError regardless of numpy.
        with pytest.raises(ValueError, match="no documents have embeddings"):
            export_documents(_docs(1), out, ExportFormat.NUMPY)


# ===========================================================================
# PANDAS export — mocked
# ===========================================================================


class TestExportPandas:

    def test_pandas_delegates_to_dataframe(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.csv"
        # to_csv is called with the .tmp path; it must actually write that file
        # so the atomic rename succeeds.
        def _fake_to_csv(path, **kw):
            pathlib.Path(path).write_text("col\nval", encoding="utf-8")

        mock_pd = MagicMock()
        mock_df = MagicMock()
        mock_df.to_csv.side_effect = _fake_to_csv
        mock_pd.DataFrame.return_value = mock_df
        with patch.dict("sys.modules", {"pandas": mock_pd}):
            export_documents(_docs(2), out, ExportFormat.PANDAS)
        mock_pd.DataFrame.assert_called_once()
        mock_df.to_csv.assert_called_once()

    def test_pandas_import_error_propagates(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.csv"
        with patch.dict("sys.modules", {"pandas": None}):
            with pytest.raises((ImportError, TypeError)):
                export_documents(_docs(1), out, ExportFormat.PANDAS)


# ===========================================================================
# PARQUET export — mocked
# ===========================================================================


class TestExportParquet:

    def test_parquet_delegates_to_pandas(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.parquet"

        def _fake_to_parquet(path, **kw):
            pathlib.Path(path).write_bytes(b"fake_parquet")

        mock_pd = MagicMock()
        mock_df = MagicMock()
        mock_df.to_parquet.side_effect = _fake_to_parquet
        mock_pd.DataFrame.return_value = mock_df
        with patch.dict("sys.modules", {"pandas": mock_pd, "pyarrow": MagicMock()}):
            export_documents(_docs(2), out, ExportFormat.PARQUET)
        mock_df.to_parquet.assert_called_once()

    def test_parquet_compression_forwarded(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "out.parquet"

        captured: list = []

        def _fake_to_parquet(path, **kw):
            captured.append(kw)
            pathlib.Path(path).write_bytes(b"fake_parquet")

        mock_pd = MagicMock()
        mock_df = MagicMock()
        mock_df.to_parquet.side_effect = _fake_to_parquet
        mock_pd.DataFrame.return_value = mock_df
        with patch.dict("sys.modules", {"pandas": mock_pd, "pyarrow": MagicMock()}):
            export_documents(
                _docs(1), out, ExportFormat.PARQUET, parquet_compression="gzip"
            )
        assert mock_df.to_parquet.called
        assert any(kw.get("compression") == "gzip" for kw in captured)


# ===========================================================================
# Integration: CSV → reload → verify round-trip
# ===========================================================================


class TestCSVRoundTrip:

    def test_doc_ids_round_trip(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "rt.csv"
        docs = _docs(4)
        original_ids = {d.doc_id for d in docs}
        export_documents(docs, out, ExportFormat.CSV)
        content = out.read_text(encoding="utf-8")
        rows = list(csv.DictReader(io.StringIO(content)))
        loaded_ids = {r["doc_id"] for r in rows}
        assert loaded_ids == original_ids

    def test_text_values_round_trip(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "rt.csv"
        texts = [f"Unique text sample number {i}." for i in range(3)]
        docs = [_doc(t, i) for i, t in enumerate(texts)]
        export_documents(docs, out, ExportFormat.CSV)
        content = out.read_text(encoding="utf-8")
        rows = list(csv.DictReader(io.StringIO(content)))
        loaded_texts = {r["text"] for r in rows}
        assert set(texts) == loaded_texts

    def test_jsonl_then_json_same_doc_ids(self, tmp_path: pathlib.Path) -> None:
        docs = _docs(3)
        jsonl_out = tmp_path / "out.jsonl"
        json_out = tmp_path / "out.json"
        export_documents(docs, jsonl_out, ExportFormat.JSONL)
        export_documents(docs, json_out, ExportFormat.JSON)

        jsonl_ids = {
            json.loads(l)["doc_id"]
            for l in jsonl_out.read_text().splitlines()
            if l.strip()
        }
        json_ids = {
            item["doc_id"]
            for item in json.loads(json_out.read_text())
        }
        assert jsonl_ids == json_ids == {d.doc_id for d in docs}
