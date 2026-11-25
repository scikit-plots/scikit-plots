# scikitplot/corpus/_readers/tests/test__table.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :class:`~scikitplot.corpus._readers._table.CSVReader` (CX-50)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ... import _readers  # noqa: F401  (registers the readers)
from ..._base import DocumentReader
from ..._schema import SectionType, SourceType
from .._table import CSVReader, _sniff_delimiter


def _write(tmp_path: Path, data: bytes | str, name: str = "t.csv") -> Path:
    path = tmp_path / name
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return path


def _chunks(path: Path, **kwargs) -> list[dict]:
    return list(CSVReader(input_path=path, **kwargs).get_raw_chunks())


@pytest.mark.parametrize("name", ["t.csv", "t.tsv", "T.CSV"])
def test_extensions_create_the_reader(tmp_path, name):
    assert type(DocumentReader.create(_write(tmp_path, "a,b\n", name))) is CSVReader


def test_rows_become_tab_separated_lines_under_the_header(tmp_path):
    [chunk] = _chunks(_write(tmp_path, "name,age\nAda,36\nBob,41\n"))
    assert chunk["text"] == "name\tage\nAda\t36\nBob\t41"
    assert (chunk["row_start"], chunk["row_end"]) == (2, 3)
    assert chunk["header"] == "name\tage"
    assert chunk["delimiter"] == ","
    assert chunk["section_type"] == SectionType.TABLE.value
    assert chunk["source_type"] == SourceType.SPREADSHEET.value


def test_every_chunk_opens_with_the_header(tmp_path):
    body = "h1,h2\n" + "".join(f"v{i},w{i}\n" for i in range(5))
    chunks = _chunks(_write(tmp_path, body), rows_per_chunk=2)
    assert [c["text"].split("\n")[0] for c in chunks] == ["h1\th2"] * 3
    assert [(c["row_start"], c["row_end"]) for c in chunks] == [(2, 3), (4, 5), (6, 6)]


def test_without_a_header_rows_stand_alone(tmp_path):
    chunks = _chunks(_write(tmp_path, "a,b\nc,d\n"), header=False)
    assert chunks[0]["text"] == "a\tb\nc\td"
    assert chunks[0]["row_start"] == 1 and "header" not in chunks[0]


def test_quoting_bom_crlf_blank_rows_and_trailing_fields(tmp_path):
    raw = b'\xef\xbb\xbfname;note;x\r\nAda;"multi\r\nline; with ; semis";\r\n\r\nBob;"say ""hi""";1\r\n'
    [chunk] = _chunks(_write(tmp_path, raw))
    assert chunk["delimiter"] == ";"
    assert chunk["text"].split("\n") == [
        "name\tnote\tx",
        "Ada\tmulti line; with ; semis",
        'Bob\tsay "hi"\t1',
    ]
    assert (chunk["row_start"], chunk["row_end"]) == (2, 4)


def test_tsv_uses_tabs_without_sniffing(tmp_path):
    [chunk] = _chunks(_write(tmp_path, "a,1\tb\nc,2\td\n", "t.tsv"))
    assert chunk["delimiter"] == "\t"
    assert chunk["text"] == "a,1\tb\nc,2\td"


def test_an_explicit_delimiter_wins(tmp_path):
    [chunk] = _chunks(_write(tmp_path, "a|b,c\n1|2,3\n"), delimiter="|")
    assert chunk["text"] == "a\tb,c\n1\t2,3"


def test_a_header_only_table_is_kept(tmp_path):
    [chunk] = _chunks(_write(tmp_path, "only,columns\n"))
    assert chunk["text"] == "only\tcolumns" and chunk["row_start"] == 1


def test_an_empty_table_yields_nothing(tmp_path):
    assert _chunks(_write(tmp_path, "\n\n")) == []


def test_legacy_encodings_are_detected(tmp_path):
    [chunk] = _chunks(_write(tmp_path, "caf\xe9,x\n1,2\n".encode("cp1252")))
    assert chunk["encoding"] == "cp1252"
    assert chunk["text"].startswith("café")


def test_an_oversized_field_is_refused_with_its_line(tmp_path):
    path = _write(tmp_path, "a\n" + '"' + "x" * 200_000 + '"\n')
    with pytest.raises(ValueError, match="malformed table at line"):
        _chunks(path)


def test_binary_and_large_files_are_refused(tmp_path):
    with pytest.raises(ValueError, match="not a text file"):
        _chunks(_write(tmp_path, b"a,b\x00\n"))
    with pytest.raises(ValueError, match="max_file_bytes"):
        _chunks(_write(tmp_path, "a,b\n" * 10), max_file_bytes=5)


def test_the_text_limit_ends_a_chunk_early(tmp_path):
    body = "h\n" + "".join(f"{'x' * 50}\n" for _ in range(10))
    chunks = _chunks(_write(tmp_path, body), max_text_chars=120, on_text_limit="truncate")
    assert len(chunks) > 1


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"delimiter": ",,"}, ValueError),
        ({"delimiter": '"'}, ValueError),
        ({"delimiter": "\n"}, ValueError),
        ({"rows_per_chunk": 0}, ValueError),
        ({"rows_per_chunk": True}, ValueError),
        ({"header": "yes"}, TypeError),
    ],
)
def test_bad_options_are_refused(tmp_path, kwargs, error):
    with pytest.raises(error):
        CSVReader(input_path=_write(tmp_path, "a\n"), **kwargs)


def test_documents_are_built(tmp_path):
    docs = list(CSVReader(input_path=_write(tmp_path, "name,role\nAda,engineer\n")).get_documents())
    assert docs and docs[0].source_type == SourceType.SPREADSHEET


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a,b,c\n1,2,3\n", ","),
        ("a;b;c\n1;2;3\n", ";"),
        ("a\tb\n1\t2\n", "\t"),
        ("a|b\n1|2\n", "|"),
        ("single\ncolumn\n", ","),
        ("a;b\n" * 40_000, ";"),  # longer than the sniff window
    ],
)
def test_sniff_delimiter(text, expected):
    assert _sniff_delimiter(text) == expected
