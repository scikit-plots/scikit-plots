"""
Stage-outcome regressions for the pipeline executor (slice S-42).

A normalizer or enricher that raised was caught so the run could continue --
one bad stage should not abort a whole ingest -- but the result was then
identical to a clean run. A caller could not distinguish a corpus that was
normalized from one where normalization failed and the text passed through
untouched.

See Also
--------
scikitplot.corpus._pipeline.PipelineResult.status
"""

import tempfile
from pathlib import Path

import pytest

from .._pipeline import CorpusPipeline, PipelineResult


class Boom:
    """Stage that raises, as user-supplied NLP code may."""

    name = "boom"

    def normalize_documents(self, documents):
        raise RuntimeError("injected: normalizer failed")

    def enrich_documents(self, documents):
        raise RuntimeError("injected: enricher failed")


def _run(**kwargs):
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "a.txt"
        source.write_text("hello world. a second sentence here.\n", encoding="utf-8")
        return CorpusPipeline(**kwargs).run(str(source))


def test_a_clean_run_reports_success():
    """The flag states a fact, so it is present and positive when nothing failed."""
    result = _run()
    assert result.status == "success"
    assert result.stage_errors == ()
    assert len(result.documents) == 1


def test_a_failing_normalizer_is_visible():
    """The defect S01 recorded: a lost stage that looked like a clean run."""
    result = _run(normalizer=Boom())
    assert result.status == "degraded"
    assert [stage for stage, _reason in result.stage_errors] == ["normalizer"]


def test_a_failing_enricher_is_visible():
    """The same holds for the other stage that swallows its exceptions."""
    result = _run(enricher=Boom())
    assert result.status == "degraded"
    assert [stage for stage, _reason in result.stage_errors] == ["enricher"]


def test_both_failures_are_reported_not_just_the_first():
    """A caller learns everything that did not happen, not the earliest thing."""
    result = _run(normalizer=Boom(), enricher=Boom())
    assert {stage for stage, _reason in result.stage_errors} == {"normalizer", "enricher"}


def test_the_reason_names_the_exception():
    """A caller must be able to act without reading the build log."""
    reason = dict(_run(normalizer=Boom()).stage_errors)["normalizer"]
    assert "RuntimeError" in reason
    assert "injected" in reason


def test_a_degraded_run_still_returns_its_documents():
    """Continuing past a failed stage stays the behaviour; only the silence goes."""
    result = _run(normalizer=Boom())
    assert len(result.documents) == 1
    assert result.n_read == 1


def test_the_result_stays_immutable():
    """Errors are a tuple, like every other field on this frozen result."""
    result = _run(normalizer=Boom())
    assert isinstance(result.stage_errors, tuple)
    with pytest.raises(Exception):
        result.stage_errors = ()


def test_status_is_derived_not_stored():
    """Two sources of truth for one fact is how they drift apart."""
    assert isinstance(PipelineResult.status, property)
