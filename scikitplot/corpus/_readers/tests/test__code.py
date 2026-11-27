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


# ---------------------------------------------------------------------------
# Documents keep every line, as chunks do
# ---------------------------------------------------------------------------

SHORT = "x = 1\n# comment\ny = 2\n}\n42\n"


def _documents(path, **options):
    return list(CodeReader(input_path=path, **options).get_documents())


def test_documents_keep_short_blocks_of_code(tmp_path):
    # DefaultFilter (built for prose) dropped "y = 2", "}" and "42".
    path = tmp_path / "short.py"
    path.write_text(SHORT, encoding="utf-8")
    docs = _documents(path, lines_per_chunk=1)
    assert [d.text for d in docs] == ["x = 1", "# comment", "y = 2", "}", "42"]
    covered = [
        n
        for d in docs
        for n in range(d.metadata["line_start"], d.metadata["line_end"] + 1)
    ]
    assert covered == [1, 2, 3, 4, 5]


def test_a_filter_the_caller_chooses_still_applies(tmp_path):
    from ..._base import DefaultFilter  # noqa: PLC0415

    path = tmp_path / "short.py"
    path.write_text(SHORT, encoding="utf-8")
    docs = _documents(
        path, lines_per_chunk=1, filter_=DefaultFilter(min_words=1, min_chars=1)
    )
    assert [d.text for d in docs] == ["x = 1", "# comment", "y = 2"]  # no letters: dropped


def test_prose_is_filtered_as_before(tmp_path):
    # The default keeps code that is not blank and judges the rest as prose,
    # with DefaultFilter's rules unchanged.
    from ..._base import CodeAwareFilter, DefaultFilter  # noqa: PLC0415
    from .._text import TextReader  # noqa: PLC0415

    (tmp_path / "a.txt").write_text("ok\n\n42\n\nA sentence of prose here.\n", encoding="utf-8")
    reader = TextReader(input_path=tmp_path / "a.txt")
    assert isinstance(reader.filter_, CodeAwareFilter)
    assert isinstance(reader.filter_.prose, DefaultFilter)
    texts = [d.text for d in reader.get_documents()]
    explicit = [
        d.text
        for d in TextReader(
            input_path=tmp_path / "a.txt", filter_=DefaultFilter()
        ).get_documents()
    ]
    assert texts == explicit


def test_code_aware_filter_options():
    from ..._base import CodeAwareFilter, NonBlankFilter  # noqa: PLC0415

    keep_all = CodeAwareFilter(prose=NonBlankFilter())
    assert "NonBlankFilter" in repr(keep_all)
    with pytest.raises(TypeError, match="prose"):
        CodeAwareFilter(prose="strict")


def test_code_in_an_archive_keeps_its_short_blocks(tmp_path):
    import zipfile  # noqa: PLC0415

    from ..._base import DefaultFilter  # noqa: PLC0415
    from .._zip import ZipReader  # noqa: PLC0415

    archive = tmp_path / "src.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("pkg/short.py", SHORT)
    one_line = {".py": {"lines_per_chunk": 1}}  # each line a document of its own
    reader = ZipReader(input_path=archive, reader_kwargs=one_line)
    texts = [d.text for d in reader.get_documents()]
    assert texts == ["x = 1", "# comment", "y = 2", "}", "42"]
    chosen = ZipReader(
        input_path=archive,
        reader_kwargs=one_line,
        filter_=DefaultFilter(min_words=50, min_chars=1),
    )
    assert list(chosen.get_documents()) == []  # the caller's filter reaches members


def test_lean_end_of_namespace_is_kept(tmp_path):
    from .._lean import LeanReader  # noqa: PLC0415

    path = tmp_path / "A.lean"
    path.write_text(
        "namespace Foo\ntheorem t : True := trivial\nend Foo\n", encoding="utf-8"
    )
    texts = [d.text for d in LeanReader(input_path=path).get_documents()]
    assert "end Foo" in texts


# ---------------------------------------------------------------------------
# Every path that builds documents keeps short code
# ---------------------------------------------------------------------------


def test_a_pipeline_keeps_short_blocks_of_code(tmp_path):
    from ..._pipeline import CorpusPipeline  # noqa: PLC0415

    path = tmp_path / "short.py"
    path.write_text(SHORT, encoding="utf-8")
    result = CorpusPipeline(reader_kwargs={"lines_per_chunk": 1}).run(path)
    assert [d.text for d in result.documents] == ["x = 1", "# comment", "y = 2", "}", "42"]


def test_the_builders_filter_keeps_code_and_judges_prose_by_its_kwargs():
    from ..._base import CodeAwareFilter, DefaultFilter  # noqa: PLC0415
    from ..._corpus_builder import BuilderConfig, CorpusBuilder  # noqa: PLC0415

    built = CorpusBuilder(BuilderConfig(filter_kwargs={"min_words": 7}))._get_filter()
    assert isinstance(built, CodeAwareFilter)
    assert isinstance(built.prose, DefaultFilter) and built.prose.min_words == 7


def test_the_registry_names_every_built_in_filter():
    from ..._base import CodeAwareFilter, DefaultFilter, NonBlankFilter  # noqa: PLC0415
    from ..._registry._registry import ComponentRegistry  # noqa: PLC0415

    registry = ComponentRegistry()
    registry.register_builtins()
    assert isinstance(registry.build_filter("default", min_words=5), DefaultFilter)
    assert isinstance(registry.build_filter("code_aware"), CodeAwareFilter)
    assert isinstance(registry.build_filter("non_blank"), NonBlankFilter)


def test_a_hook_filter_without_a_base_keeps_code():
    from ..._base import CodeAwareFilter  # noqa: PLC0415
    from ..._custom_hooks import _CompositeHookFilter  # noqa: PLC0415

    assert isinstance(_CompositeHookFilter(None, lambda doc: True)._base, CodeAwareFilter)


def test_chunked_code_stays_code(tmp_path):
    # A chunker that splits a block keeps its section type, so the pieces
    # are still judged as code.
    from ..._chunkers._paragraph import ParagraphChunker  # noqa: PLC0415

    path = tmp_path / "a.py"
    path.write_text("def f():\n    return 1\n\n}\n\n42\n", encoding="utf-8")
    docs = list(CodeReader(input_path=path, chunker=ParagraphChunker()).get_documents())
    assert [d.text for d in docs] == ["def f():\n    return 1", "}", "42"]
