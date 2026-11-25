# scikitplot/corpus/_readers/tests/test__code.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :class:`~scikitplot.corpus._readers._code.CodeReader` (CX-50)."""

from __future__ import annotations

import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

from ... import _readers  # noqa: F401  (registers the readers)
from ..._base import DocumentReader
from ..._schema import SectionType, SourceType
from .._code import LANGUAGES, CodeReader

_PY = "import os\n\n\ndef main():\n    return os.getcwd()\n"


def _write(tmp_path: Path, data: str | bytes, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return path


def _chunks(path: Path, **kwargs) -> list[dict]:
    return list(CodeReader(input_path=path, **kwargs).get_raw_chunks())


def test_every_listed_extension_creates_the_reader():
    for ext in LANGUAGES:
        assert DocumentReader._registry[ext] is CodeReader, ext


@pytest.mark.parametrize("ext", [".env", ".m", ".pl", ".v", ".fs", ".s"])
def test_secret_or_ambiguous_extensions_are_not_registered(ext):
    assert ext not in LANGUAGES
    assert DocumentReader._registry.get(ext) is not CodeReader


def test_languages_is_read_only():
    with pytest.raises(TypeError):
        LANGUAGES[".foo"] = "foo"  # type: ignore[index]


@pytest.mark.parametrize(
    ("name", "language"),
    [("a.py", "python"), ("A.PY", "python"), ("run.sh", "shell"), ("x.R", "r"),
     ("q.xq", "xquery"), ("setup.cfg", "ini"), ("main.tf", "terraform"),
     ("api.h", "c/c++")],
)
def test_the_language_follows_the_extension(tmp_path, name, language):
    [chunk] = _chunks(_write(tmp_path, "x = 1\n", name))
    assert chunk["language"] == language


def test_lines_are_kept_exactly_with_their_range(tmp_path):
    [chunk] = _chunks(_write(tmp_path, _PY.replace("\n", "\r\n"), "a.py"))
    assert chunk["text"] == _PY.rstrip("\n")
    assert (chunk["line_start"], chunk["line_end"]) == (1, 5)
    assert chunk["section_type"] == SectionType.CODE.value
    assert chunk["source_type"] == SourceType.CODE.value
    assert chunk["encoding"] == "utf-8"


def test_blocks_of_lines_and_blank_blocks_are_skipped(tmp_path):
    body = "a\nb\n\n\n\nc\n"
    chunks = _chunks(_write(tmp_path, body, "a.py"), lines_per_chunk=2)
    assert [(c["line_start"], c["line_end"], c["text"]) for c in chunks] == [
        (1, 2, "a\nb"),
        (5, 6, "\nc"),
    ]


def test_the_whole_file_in_one_chunk(tmp_path):
    body = "".join(f"line {i}\n" for i in range(500))
    [chunk] = _chunks(_write(tmp_path, body, "a.c"), lines_per_chunk=None)
    assert (chunk["line_start"], chunk["line_end"]) == (1, 500)


def test_a_final_line_without_a_break_is_kept(tmp_path):
    [chunk] = _chunks(_write(tmp_path, "a\nb", "a.go"))
    assert chunk["text"] == "a\nb" and chunk["line_end"] == 2


def test_the_text_limit_ends_a_block_early(tmp_path):
    body = "".join("x" * 60 + "\n" for _ in range(10))
    chunks = _chunks(
        _write(tmp_path, body, "a.js"), max_text_chars=130, on_text_limit="truncate"
    )
    assert [(c["line_start"], c["line_end"]) for c in chunks][:2] == [(1, 3), (4, 6)]


def test_an_explicit_language_wins(tmp_path):
    [chunk] = _chunks(_write(tmp_path, "x\n", "a.h"), language="cpp")
    assert chunk["language"] == "cpp"


def test_empty_and_binary_files(tmp_path):
    assert _chunks(_write(tmp_path, "\n\n", "a.py")) == []
    with pytest.raises(ValueError, match="not a text file"):
        _chunks(_write(tmp_path, b"\x7fELF\x00\x01", "b.py"))


def test_legacy_encodings_are_detected(tmp_path):
    [chunk] = _chunks(_write(tmp_path, "# caf\xe9\n".encode("cp1252"), "a.py"))
    assert chunk["encoding"] == "cp1252" and chunk["text"] == "# café"


@pytest.mark.parametrize(
    "kwargs",
    [{"lines_per_chunk": 0}, {"lines_per_chunk": True}, {"language": ""}, {"language": 3}],
)
def test_bad_options_are_refused(tmp_path, kwargs):
    with pytest.raises(ValueError):
        CodeReader(input_path=_write(tmp_path, "x\n", "a.py"), **kwargs)


def test_a_subclass_adds_a_language(tmp_path):
    @dataclass
    class FooReader(CodeReader):
        file_types: ClassVar[list[str] | None] = [".foo123"]
        languages: ClassVar[dict[str, str]] = {".foo123": "foo"}

    try:
        path = _write(tmp_path, "bar\n", "a.foo123")
        reader = DocumentReader.create(path)
        assert type(reader) is FooReader
        assert next(reader.get_raw_chunks())["language"] == "foo"
    finally:
        DocumentReader._registry.pop(".foo123", None)


def test_documents_are_built(tmp_path):
    docs = list(CodeReader(input_path=_write(tmp_path, _PY, "a.py")).get_documents())
    assert docs and docs[0].source_type == SourceType.CODE


@pytest.mark.parametrize(
    ("raw", "lines"),
    [
        (b"a\rb\rc", ["a", "b", "c"]),  # classic Mac line endings
        (b"a\r\nb\nc\r", ["a", "b", "c"]),  # mixed
        (b"a\x0cb\nc\n", ["a\x0cb", "c"]),  # a form feed does not end a line
        ("\ufeffa\nb\n".encode("utf-16"), ["a", "b"]),  # byte-order mark dropped
    ],
)
def test_line_numbers_follow_line_breaks_only(tmp_path, raw, lines):
    [chunk] = _chunks(_write(tmp_path, raw, "a.c"))
    assert chunk["text"].split("\n") == lines
    assert chunk["line_end"] == len(lines)


def test_a_large_file_is_read_as_a_stream(tmp_path):
    body = "".join(f"    value_{i} = compute(a, b)  # note\n" for i in range(200_000))
    path = _write(tmp_path, body, "big.py")  # about 8 MB
    reader = CodeReader(input_path=path, max_text_chars_per_byte=None)
    tracemalloc.start()
    try:
        ends = [c["line_end"] for c in reader.get_raw_chunks()]
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert (len(ends), ends[-1]) == (1000, 200_000)
    # The bytes are held once, to choose the codec; the text never whole.
    assert peak < 1.5 * path.stat().st_size
