"""
tests/test__source.py
========================
Tests for scikitplot.corpus._sources._source.
"""
from __future__ import annotations

import pathlib
import textwrap
from unittest.mock import patch

import pytest

from .._source import CorpusSource, SourceEntry, SourceKind


class TestSourceKind:
    def test_values_are_strings(self) -> None:
        for member in SourceKind:
            assert isinstance(member.value, str)

    def test_file_value(self) -> None:
        assert SourceKind.FILE == "file"


class TestSourceEntry:
    def test_basic_construction(self) -> None:
        entry = SourceEntry(path_or_url="corpus.txt", kind=SourceKind.FILE)
        assert entry.kind == SourceKind.FILE
        assert entry.provenance == {}

    def test_is_url_false_for_file(self) -> None:
        entry = SourceEntry(path_or_url="file.txt", kind=SourceKind.FILE)
        assert entry.is_url is False

    def test_is_url_true_for_url(self) -> None:
        entry = SourceEntry(
            path_or_url="https://example.com", kind=SourceKind.URL
        )
        assert entry.is_url is True

    def test_as_path_returns_path_for_file(self) -> None:
        entry = SourceEntry(path_or_url="corpus.txt", kind=SourceKind.FILE)
        assert isinstance(entry.as_path, pathlib.Path)
        assert entry.as_path.name == "corpus.txt"

    def test_as_path_raises_for_url(self) -> None:
        entry = SourceEntry(
            path_or_url="https://example.com", kind=SourceKind.URL
        )
        with pytest.raises(ValueError, match="URL"):
            _ = entry.as_path

    def test_provenance_propagated(self) -> None:
        entry = SourceEntry(
            path_or_url="f.txt",
            kind=SourceKind.FILE,
            provenance={"source_author": "Tolstoy"},
        )
        assert entry.provenance["source_author"] == "Tolstoy"

    def test_frozen(self) -> None:
        entry = SourceEntry(path_or_url="f.txt", kind=SourceKind.FILE)
        with pytest.raises((AttributeError, TypeError)):
            entry.kind = SourceKind.URL  # type: ignore[misc]


class TestCorpusSourceFromFile:
    def test_construction(self, tmp_path: pathlib.Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        src = CorpusSource.from_file(f)
        assert src.kind == SourceKind.FILE
        assert src.root == f

    def test_iter_entries_yields_one(self, tmp_path: pathlib.Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        src = CorpusSource.from_file(f)
        entries = list(src.iter_entries())
        assert len(entries) == 1
        assert entries[0].kind == SourceKind.FILE
        assert entries[0].path_or_url == str(f)

    def test_provenance_propagated(self, tmp_path: pathlib.Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        src = CorpusSource.from_file(f, source_provenance={"source_author": "X"})
        entries = list(src.iter_entries())
        assert entries[0].provenance["source_author"] == "X"

    def test_missing_file_raises(self, tmp_path: pathlib.Path) -> None:
        src = CorpusSource.from_file(tmp_path / "missing.txt")
        with pytest.raises(FileNotFoundError, match="not found"):
            list(src.iter_entries())


class TestCorpusSourceFromDirectory:
    def test_yields_matching_files(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "c.py").write_text("c")
        src = CorpusSource.from_directory(tmp_path, pattern="*.txt", recursive=False)
        entries = list(src.iter_entries())
        names = {pathlib.Path(e.path_or_url).name for e in entries}
        assert names == {"a.txt", "b.txt"}

    def test_extension_filter(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.xml").write_text("b")
        src = CorpusSource.from_directory(
            tmp_path, pattern="*", recursive=False, extensions=[".xml"]
        )
        entries = list(src.iter_entries())
        assert all(e.path_or_url.endswith(".xml") for e in entries)

    def test_missing_directory_raises(self, tmp_path: pathlib.Path) -> None:
        src = CorpusSource.from_directory(tmp_path / "missing")
        with pytest.raises(FileNotFoundError):
            list(src.iter_entries())

    def test_count(self, tmp_path: pathlib.Path) -> None:
        for i in range(3):
            (tmp_path / f"f{i}.txt").write_text("x")
        src = CorpusSource.from_directory(tmp_path, pattern="*.txt", recursive=False)
        assert src.count() == 3


class TestCorpusSourceFromUrls:
    def test_yields_url_entries(self) -> None:
        urls = ["https://a.com", "https://b.com"]
        src = CorpusSource.from_urls(urls)
        entries = list(src.iter_entries())
        assert len(entries) == 2
        assert all(e.kind == SourceKind.URL for e in entries)

    def test_empty_urls_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            CorpusSource.from_urls([])

    def test_invalid_url_raises(self) -> None:
        with pytest.raises(ValueError, match="non-URL"):
            CorpusSource.from_urls(["not-a-url"])


class TestCorpusSourceFromManifest:
    def test_yields_url_and_file_entries(self, tmp_path: pathlib.Path) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text("hello")
        manifest = tmp_path / "manifest.txt"
        manifest.write_text(
            textwrap.dedent("""\
            # Comment line
            https://example.com/page1
            doc.txt
            """)
        )
        src = CorpusSource.from_manifest(manifest)
        entries = list(src.iter_entries())
        kinds = {e.kind for e in entries}
        assert SourceKind.URL in kinds
        assert SourceKind.FILE in kinds
        assert len(entries) == 2

    def test_missing_manifest_raises(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="not found"):
            CorpusSource.from_manifest(tmp_path / "missing.txt")

    def test_blank_and_comment_lines_skipped(self, tmp_path: pathlib.Path) -> None:
        manifest = tmp_path / "m.txt"
        manifest.write_text("# comment\n\nhttps://x.com\n")
        src = CorpusSource.from_manifest(manifest)
        entries = list(src.iter_entries())
        assert len(entries) == 1


class TestCorpusSourceValidate:
    def test_file_without_root_raises(self) -> None:
        src = CorpusSource(kind=SourceKind.FILE)
        with pytest.raises(ValueError, match="root"):
            src.validate()

    def test_url_without_urls_raises(self) -> None:
        src = CorpusSource(kind=SourceKind.URL)
        with pytest.raises(ValueError, match="urls"):
            src.validate()

    def test_invalid_extension_raises(self) -> None:
        src = CorpusSource(
            kind=SourceKind.DIRECTORY,
            root=pathlib.Path("/tmp"),
            extensions=["txt"],  # missing leading dot
        )
        with pytest.raises(ValueError, match="'\\."):
            src.validate()
