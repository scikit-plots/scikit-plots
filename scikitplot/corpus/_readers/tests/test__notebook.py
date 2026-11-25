# scikitplot/corpus/_readers/tests/test__notebook.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :class:`~scikitplot.corpus._readers._notebook.NotebookReader` (CX-50)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from ... import _readers  # noqa: F401  (registers the readers)
from ..._base import DocumentReader
from ..._schema import SectionType, SourceType
from .._notebook import NotebookReader, _output_text


def _nb4(cells: list, language: str | None = "python") -> dict:
    meta = {"kernelspec": {"language": language, "name": "k"}} if language else {}
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": meta, "cells": cells}


def _md(text) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def _code(text, outputs=None, count=1) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": count,
        "source": text,
        "outputs": outputs or [],
    }


_OUTPUTS = [
    {"output_type": "stream", "name": "stdout", "text": ["hello\n", "world\n"]},
    {
        "output_type": "execute_result",
        "execution_count": 2,
        "metadata": {},
        "data": {"text/plain": ["42"], "text/html": ["<b>42</b>"]},
    },
    {
        "output_type": "display_data",
        "metadata": {},
        "data": {"image/png": "iVBORw0KGgo=" * 1000},
    },
    {"output_type": "error", "ename": "ValueError", "evalue": "bad", "traceback": []},
]


def _write(tmp_path: Path, notebook, name: str = "n.ipynb") -> Path:
    path = tmp_path / name
    data = notebook if isinstance(notebook, (str, bytes)) else json.dumps(notebook)
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return path


def _chunks(path: Path, **kwargs) -> list[dict]:
    return list(NotebookReader(input_path=path, **kwargs).get_raw_chunks())


def test_the_extension_creates_the_reader(tmp_path):
    assert type(DocumentReader.create(_write(tmp_path, _nb4([])))) is NotebookReader


def test_one_chunk_per_cell_in_order(tmp_path):
    notebook = _nb4([_md(["# Title\n", "Some *text*."]), _code("x = 1\nprint(x)\n", count=3)])
    md, code = _chunks(_write(tmp_path, notebook))
    assert (md["text"], md["cell_type"], md["section_type"]) == (
        "# Title\nSome *text*.",
        "markdown",
        SectionType.TEXT.value,
    )
    assert (code["text"], code["language"], code["execution_count"]) == ("x = 1\nprint(x)", "python", 3)
    assert code["section_type"] == SectionType.CODE.value
    assert [md["cell_index"], code["cell_index"]] == [0, 1]
    assert {md["source_type"], code["source_type"]} == {SourceType.CODE.value}
    assert "language" not in md


def test_outputs_are_text_only_and_off_by_default(tmp_path):
    path = _write(tmp_path, _nb4([_code("f()", _OUTPUTS)]))
    assert [c.get("is_output") for c in _chunks(path)] == [None]
    code, out = _chunks(path, outputs=True)
    assert out["is_output"] is True and out["cell_index"] == code["cell_index"]
    assert out["text"] == "hello\nworld\n42\nValueError: bad"
    assert "iVBOR" not in out["text"] and "<b>" not in out["text"]


def test_cell_types_select_cells_and_coverage_counts_them(tmp_path):
    raw = {"cell_type": "raw", "metadata": {}, "source": "raw text"}
    notebook = _nb4([_md("a"), _code(""), raw, "junk", {"cell_type": "widget"}])
    reader = NotebookReader(input_path=_write(tmp_path, notebook))
    assert [c["text"] for c in reader.get_raw_chunks()] == ["a"]
    assert reader.coverage.read == 1 and reader.coverage.empty == 1
    assert reader.coverage.skipped == {"raw": 1, "malformed": 1, "unknown": 1}
    reader = NotebookReader(input_path=_write(tmp_path, notebook, "b.ipynb"), cell_types=["raw"])
    assert [c["text"] for c in reader.get_raw_chunks()] == ["raw text"]


def test_language_from_language_info_or_none(tmp_path):
    nb = _nb4([_code("x")], language=None)
    assert _chunks(_write(tmp_path, nb))[0]["language"] is None
    nb["metadata"] = {"language_info": {"name": "R"}}
    assert _chunks(_write(tmp_path, nb, "b.ipynb"))[0]["language"] == "r"


def test_nbformat_3_worksheets(tmp_path):
    notebook = {
        "nbformat": 3,
        "metadata": {},
        "worksheets": [
            {
                "cells": [
                    {"cell_type": "markdown", "source": ["Old notes"]},
                    {
                        "cell_type": "code",
                        "input": ["print(1)"],
                        "prompt_number": 7,
                        "outputs": [{"output_type": "pyout", "text": ["1"]}],
                    },
                ]
            }
        ],
    }
    md, code, out = _chunks(_write(tmp_path, notebook), outputs=True)
    assert md["text"] == "Old notes"
    assert (code["text"], code["execution_count"]) == ("print(1)", 7)
    assert out["text"] == "1"


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ("{not json", "is not JSON"),
        ('{"nbformat": 4}', "not a notebook"),
        ("[1, 2]", "not a notebook"),
    ],
)
def test_files_that_are_refused(tmp_path, data, message):
    with pytest.raises(ValueError, match=message):
        _chunks(_write(tmp_path, data))


def test_deep_nesting_never_crashes(tmp_path):
    # Whether json parses 100 000 levels depends on the Python version
    # (3.14+ bounds recursion by stack size, not a count): the contract is
    # a ValueError either way (too deep, or not a notebook), never a crash.
    with pytest.raises(ValueError, match="nests too deeply|not a notebook"):
        _chunks(_write(tmp_path, "[" * 100_000 + "]" * 100_000))


def test_a_parser_recursion_error_names_the_depth(tmp_path):
    with mock.patch("json.loads", side_effect=RecursionError):
        with pytest.raises(ValueError, match="NotebookReader: n.ipynb nests too deeply"):
            _chunks(_write(tmp_path, _nb4([])))


def test_an_empty_file_yields_nothing(tmp_path):
    assert _chunks(_write(tmp_path, "")) == []


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"cell_types": "code"}, ValueError),
        ({"cell_types": ()}, ValueError),
        ({"cell_types": ("code", "widget")}, ValueError),
        ({"outputs": "yes"}, TypeError),
    ],
)
def test_bad_options_are_refused(tmp_path, kwargs, error):
    with pytest.raises(error):
        NotebookReader(input_path=_write(tmp_path, _nb4([])), **kwargs)


@pytest.mark.parametrize(
    ("output", "text"),
    [
        ({"output_type": "stream", "text": "a"}, "a"),
        ({"output_type": "execute_result", "data": {"text/markdown": "**b**", "text/plain": "b"}}, "**b**"),
        ({"output_type": "display_data", "data": {"image/png": "x"}}, ""),
        ({"output_type": "pyerr", "ename": "E", "evalue": ""}, "E"),
        ("not a dict", ""),
    ],
)
def test_output_text(output, text):
    assert _output_text(output) == text


def test_documents_are_built(tmp_path):
    path = _write(tmp_path, _nb4([_md("A paragraph long enough to keep as a document.")]))
    docs = list(NotebookReader(input_path=path).get_documents())
    assert docs and docs[0].source_type == SourceType.CODE
