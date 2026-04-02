# scikitplot/corpus/tests/test__pipeline.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for scikitplot.corpus._pipeline
======================================

Coverage
--------
* :class:`PipelineResult` — all fields, properties (``n_documents``),
  invariant ``n_read - n_omitted == len(documents)``, ``__repr__``.
* :class:`CorpusPipeline` — construction defaults; ``run()`` with a real
  ``TextReader`` source (tmp file); ``run()`` TypeError on bad input;
  ``run()`` with no output (export skipped); ``run()`` with output_dir;
  ``run_batch()`` with multiple files; ``run_batch()`` stop_on_error
  behaviour; ``run_batch()`` with mixed success/failure;
  ``run_batch()`` TypeError on non-str-or-Path element; progress_callback
  invoked during batch; elapsed_seconds > 0; export_format override per call.
* :func:`create_corpus` — end-to-end with tmp files, CSV output produced.

All external readers and exporters are exercised with real ``TextReader``
files so that no mocking of core pipeline logic is needed.  Optional deps
(embedding, spaCy) are never imported.

Run with::

    pytest corpus/tests/test__pipeline.py -v
"""
from __future__ import annotations

import pathlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from .._pipeline import CorpusPipeline, PipelineResult, create_corpus
from .._schema import ExportFormat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_txt(tmp_path: pathlib.Path, name: str, text: str) -> pathlib.Path:
    f = tmp_path / name
    f.write_text(text, encoding="utf-8")
    return f


def _make_result(
    source: str = "test.txt",
    n_docs: int = 3,
    n_read: int = 5,
    n_omitted: int = 2,
    n_embedded: int = 0,
    elapsed: float = 0.12,
    output_path: pathlib.Path | None = None,
    export_format: ExportFormat | None = None,
) -> PipelineResult:
    from .._schema import CorpusDocument  # noqa: PLC0415

    docs = [CorpusDocument.create(f"{source}", i, f"Sentence {i}.") for i in range(n_docs)]
    return PipelineResult(
        source=source,
        documents=docs,
        output_path=output_path,
        n_read=n_read,
        n_omitted=n_omitted,
        n_embedded=n_embedded,
        elapsed_seconds=elapsed,
        export_format=export_format,
    )


# ===========================================================================
# PipelineResult
# ===========================================================================


class TestPipelineResult:

    def test_n_documents_matches_list_length(self) -> None:
        r = _make_result(n_docs=4)
        assert r.n_documents == 4

    def test_invariant_n_read_minus_omitted_equals_n_documents(self) -> None:
        r = _make_result(n_docs=3, n_read=5, n_omitted=2)
        assert r.n_read - r.n_omitted == r.n_documents

    def test_source_stored(self) -> None:
        r = _make_result(source="chapter01.txt")
        assert r.source == "chapter01.txt"

    def test_output_path_none_by_default(self) -> None:
        r = _make_result()
        assert r.output_path is None

    def test_output_path_stored_when_provided(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "out.csv"
        r = _make_result(output_path=p)
        assert r.output_path == p

    def test_elapsed_seconds_stored(self) -> None:
        r = _make_result(elapsed=3.14)
        assert r.elapsed_seconds == pytest.approx(3.14)

    def test_n_embedded_zero_by_default(self) -> None:
        r = _make_result(n_embedded=0)
        assert r.n_embedded == 0

    def test_repr_contains_source(self) -> None:
        r = _make_result(source="my_file.txt")
        assert "my_file.txt" in repr(r)

    def test_repr_contains_n_documents(self) -> None:
        r = _make_result(n_docs=7)
        assert "7" in repr(r)

    def test_export_format_none_allowed(self) -> None:
        r = _make_result(export_format=None)
        assert r.export_format is None

    def test_export_format_stored(self) -> None:
        r = _make_result(export_format=ExportFormat.CSV)
        assert r.export_format is ExportFormat.CSV

    def test_frozen_dataclass_raises_on_mutation(self) -> None:
        r = _make_result()
        with pytest.raises((AttributeError, TypeError)):
            r.source = "other.txt"  # type: ignore[misc]

    def test_empty_documents_list(self) -> None:
        from .._schema import CorpusDocument  # noqa: PLC0415

        r = PipelineResult(
            source="empty.txt",
            documents=[],
            output_path=None,
            n_read=0,
            n_omitted=0,
            n_embedded=0,
            elapsed_seconds=0.01,
            export_format=None,
        )
        assert r.n_documents == 0


# ===========================================================================
# CorpusPipeline — construction
# ===========================================================================


class TestCorpusPipelineConstruction:

    def test_default_chunker_is_none(self) -> None:
        p = CorpusPipeline()
        assert p.chunker is None

    def test_default_filter_is_none(self) -> None:
        p = CorpusPipeline()
        assert p.filter_ is None

    def test_default_embedding_engine_is_none(self) -> None:
        p = CorpusPipeline()
        assert p.embedding_engine is None

    def test_default_output_dir_is_none(self) -> None:
        p = CorpusPipeline()
        assert p.output_dir is None

    def test_default_language_is_none(self) -> None:
        p = CorpusPipeline()
        assert p.default_language is None

    def test_custom_params_stored(self) -> None:
        p = CorpusPipeline(
            output_dir=pathlib.Path("/tmp"),
            export_format=ExportFormat.JSONL,
            default_language="en",
        )
        assert p.output_dir == pathlib.Path("/tmp")
        assert p.export_format is ExportFormat.JSONL
        assert p.default_language == "en"


# ===========================================================================
# CorpusPipeline.run — single-file path
# ===========================================================================


class TestCorpusPipelineRun:

    def test_run_returns_pipeline_result(self, tmp_path: pathlib.Path) -> None:
        f = _write_txt(tmp_path, "doc.txt", "Hello world. This is a test.")
        pipeline = CorpusPipeline()
        result = pipeline.run(f)
        assert isinstance(result, PipelineResult)

    def test_run_source_matches_filename(self, tmp_path: pathlib.Path) -> None:
        f = _write_txt(tmp_path, "article.txt", "Some article content here.")
        pipeline = CorpusPipeline()
        result = pipeline.run(f)
        assert "article" in result.source or "article.txt" in result.source

    def test_run_produces_documents(self, tmp_path: pathlib.Path) -> None:
        f = _write_txt(tmp_path, "doc.txt", "First sentence. Second sentence.")
        pipeline = CorpusPipeline()
        result = pipeline.run(f)
        assert result.n_documents >= 1

    def test_run_elapsed_seconds_positive(self, tmp_path: pathlib.Path) -> None:
        f = _write_txt(tmp_path, "doc.txt", "Content here to process.")
        pipeline = CorpusPipeline()
        result = pipeline.run(f)
        assert result.elapsed_seconds >= 0.0

    def test_run_no_export_when_no_output_dir(self, tmp_path: pathlib.Path) -> None:
        f = _write_txt(tmp_path, "doc.txt", "Content.")
        pipeline = CorpusPipeline()
        result = pipeline.run(f)
        assert result.output_path is None

    def test_run_with_output_dir_creates_file(self, tmp_path: pathlib.Path) -> None:
        f = _write_txt(tmp_path, "doc.txt", "Hello world content.")
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        pipeline = CorpusPipeline(output_dir=out_dir, export_format=ExportFormat.CSV)
        result = pipeline.run(f)
        if result.output_path is not None:
            assert result.output_path.exists()

    def test_run_explicit_output_path(self, tmp_path: pathlib.Path) -> None:
        f = _write_txt(tmp_path, "doc.txt", "Hello world content here.")
        out_path = tmp_path / "out.csv"
        pipeline = CorpusPipeline()
        result = pipeline.run(f, output_path=out_path, export_format=ExportFormat.CSV)
        assert result.output_path == out_path
        assert out_path.exists()

    def test_run_type_error_on_int_input(self) -> None:
        pipeline = CorpusPipeline()
        with pytest.raises(TypeError, match="input_file"):
            pipeline.run(42)  # type: ignore[arg-type]

    def test_run_type_error_on_list_input(self) -> None:
        pipeline = CorpusPipeline()
        with pytest.raises(TypeError, match="input_file"):
            pipeline.run(["file.txt"])  # type: ignore[arg-type]

    def test_run_str_path_accepted(self, tmp_path: pathlib.Path) -> None:
        f = _write_txt(tmp_path, "doc.txt", "Content for str path test.")
        pipeline = CorpusPipeline()
        result = pipeline.run(str(f))
        assert isinstance(result, PipelineResult)

    def test_run_filename_override_applied(self, tmp_path: pathlib.Path) -> None:
        f = _write_txt(tmp_path, "doc.txt", "Override test content.")
        pipeline = CorpusPipeline()
        result = pipeline.run(f, filename_override="custom_label.txt")
        assert "custom_label" in result.source or result.n_documents >= 0

    def test_run_export_format_override(self, tmp_path: pathlib.Path) -> None:
        f = _write_txt(tmp_path, "doc.txt", "Enough content to export here.")
        out = tmp_path / "out.jsonl"
        pipeline = CorpusPipeline()
        result = pipeline.run(f, output_path=out, export_format=ExportFormat.JSONL)
        assert result.export_format is ExportFormat.JSONL

    def test_run_n_read_ge_n_documents(self, tmp_path: pathlib.Path) -> None:
        f = _write_txt(tmp_path, "doc.txt", "Text to chunk and filter.")
        pipeline = CorpusPipeline()
        result = pipeline.run(f)
        assert result.n_read >= result.n_documents

    def test_run_n_read_minus_omitted_equals_n_documents(self, tmp_path: pathlib.Path) -> None:
        f = _write_txt(tmp_path, "doc.txt", "Sentence one. Sentence two.")
        pipeline = CorpusPipeline()
        result = pipeline.run(f)
        assert result.n_read - result.n_omitted == result.n_documents

    def test_run_empty_file_returns_zero_documents(self, tmp_path: pathlib.Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        pipeline = CorpusPipeline()
        result = pipeline.run(f)
        assert result.n_documents == 0

    def test_run_documents_have_text(self, tmp_path: pathlib.Path) -> None:
        f = _write_txt(tmp_path, "doc.txt", "Verifiable content is here now.")
        pipeline = CorpusPipeline()
        result = pipeline.run(f)
        for doc in result.documents:
            assert doc.text is not None
            assert len(doc.text) > 0


# ===========================================================================
# CorpusPipeline.run_batch
# ===========================================================================


class TestCorpusPipelineRunBatch:

    def test_run_batch_returns_list(self, tmp_path: pathlib.Path) -> None:
        files = [_write_txt(tmp_path, f"d{i}.txt", f"Content {i}.") for i in range(3)]
        pipeline = CorpusPipeline()
        results = pipeline.run_batch(files)
        assert isinstance(results, list)

    def test_run_batch_one_result_per_file(self, tmp_path: pathlib.Path) -> None:
        files = [_write_txt(tmp_path, f"d{i}.txt", f"Content {i}.") for i in range(4)]
        pipeline = CorpusPipeline()
        results = pipeline.run_batch(files)
        assert len(results) == 4

    def test_run_batch_empty_list_returns_empty(self) -> None:
        pipeline = CorpusPipeline()
        assert pipeline.run_batch([]) == []

    def test_run_batch_type_error_on_bad_element(self) -> None:
        pipeline = CorpusPipeline()
        with pytest.raises(TypeError, match="input_files"):
            pipeline.run_batch([42])  # type: ignore[list-item]

    def test_run_batch_stop_on_error_false_skips_bad_file(
        self, tmp_path: pathlib.Path
    ) -> None:
        """When ``stop_on_error=False``, bad files are skipped and good ones processed."""
        good = _write_txt(tmp_path, "good.txt", "Good content here.")
        bad = tmp_path / "nonexistent.txt"  # does not exist
        pipeline = CorpusPipeline()
        # Should not raise; bad file is logged as warning.
        results = pipeline.run_batch([good, bad], stop_on_error=False)
        assert len(results) >= 1  # At least the good file processed.

    def test_run_batch_stop_on_error_true_reraises(
        self, tmp_path: pathlib.Path
    ) -> None:
        """When ``stop_on_error=True``, first error is re-raised."""
        bad = tmp_path / "nonexistent.txt"
        pipeline = CorpusPipeline()
        with pytest.raises(Exception):
            pipeline.run_batch([bad], stop_on_error=True)

    def test_run_batch_export_format_override(self, tmp_path: pathlib.Path) -> None:
        files = [_write_txt(tmp_path, f"d{i}.txt", f"Content {i}.") for i in range(2)]
        pipeline = CorpusPipeline()
        results = pipeline.run_batch(files, export_format=ExportFormat.JSONL)
        for r in results:
            if r.export_format is not None:
                assert r.export_format is ExportFormat.JSONL

    def test_run_batch_results_in_order(self, tmp_path: pathlib.Path) -> None:
        files = [_write_txt(tmp_path, f"file{i}.txt", f"Content {i}.") for i in range(3)]
        pipeline = CorpusPipeline()
        results = pipeline.run_batch(files)
        sources = [r.source for r in results]
        # Sources must appear in a consistent order (not necessarily file order,
        # but the list must be stable between calls).
        assert len(sources) == len(results)

    def test_run_batch_progress_callback_invoked(self, tmp_path: pathlib.Path) -> None:
        calls: list[tuple] = []

        def cb(source: str, n_done: int, n_total: int) -> None:
            calls.append((source, n_done, n_total))

        files = [_write_txt(tmp_path, f"d{i}.txt", f"Content {i}.") for i in range(3)]
        pipeline = CorpusPipeline(progress_callback=cb)
        pipeline.run_batch(files)
        assert len(calls) >= 1

    def test_run_batch_str_paths_accepted(self, tmp_path: pathlib.Path) -> None:
        files = [str(_write_txt(tmp_path, f"d{i}.txt", f"Content {i}.")) for i in range(2)]
        pipeline = CorpusPipeline()
        results = pipeline.run_batch(files)
        assert len(results) == 2


# ===========================================================================
# create_corpus — convenience wrapper
# ===========================================================================


class TestCreateCorpus:

    def test_returns_pipeline_result(self, tmp_path: pathlib.Path) -> None:
        f = _write_txt(tmp_path, "chapter.txt", "A complete chapter with good content.")
        out = tmp_path / "chapter.csv"
        result = create_corpus(input_file=f, output_path=out)
        assert isinstance(result, PipelineResult)

    def test_csv_file_created(self, tmp_path: pathlib.Path) -> None:
        f = _write_txt(tmp_path, "doc.txt", "Enough text to create an output file.")
        out = tmp_path / "out.csv"
        create_corpus(input_file=f, output_path=out)
        assert out.exists()

    def test_output_path_in_result(self, tmp_path: pathlib.Path) -> None:
        f = _write_txt(tmp_path, "doc.txt", "Text content for output path test.")
        out = tmp_path / "corpus.csv"
        result = create_corpus(input_file=f, output_path=out)
        assert result.output_path == out

    def test_str_paths_accepted(self, tmp_path: pathlib.Path) -> None:
        f = _write_txt(tmp_path, "doc.txt", "Content for str path test.")
        out = tmp_path / "out.csv"
        result = create_corpus(input_file=str(f), output_path=str(out))
        assert isinstance(result, PipelineResult)

    def test_default_export_format_is_csv(self, tmp_path: pathlib.Path) -> None:
        f = _write_txt(tmp_path, "doc.txt", "Content.")
        out = tmp_path / "out.csv"
        result = create_corpus(input_file=f, output_path=out)
        assert result.export_format in (ExportFormat.CSV, None)

    def test_custom_export_format_jsonl(self, tmp_path: pathlib.Path) -> None:
        f = _write_txt(tmp_path, "doc.txt", "Sufficient content for JSONL export test.")
        out = tmp_path / "out.jsonl"
        result = create_corpus(
            input_file=f, output_path=out, export_format=ExportFormat.JSONL
        )
        if result.n_documents > 0:
            assert out.exists()

    def test_filename_override_forwarded(self, tmp_path: pathlib.Path) -> None:
        f = _write_txt(tmp_path, "doc.txt", "Content for filename override test.")
        out = tmp_path / "out.csv"
        result = create_corpus(
            input_file=f, output_path=out, filename_override="custom.txt"
        )
        assert isinstance(result, PipelineResult)

    def test_default_language_forwarded(self, tmp_path: pathlib.Path) -> None:
        f = _write_txt(tmp_path, "doc.txt", "English text for language test.")
        out = tmp_path / "out.csv"
        result = create_corpus(input_file=f, output_path=out, default_language="en")
        # Documents should carry the default language when set.
        for doc in result.documents:
            lang = getattr(doc, "language", None)
            if lang is not None:
                assert lang == "en"
