from __future__ import annotations

import pytest

from .. import _core as core
from .. import (
    backend_info,
    closest,
    distance,
    make_corpus_scorer,
    normalized_similarity,
    rank,
)


def test_classic_distance() -> None:
    assert distance("kitten", "sitting", backend="python") == 3


def test_sequence_distance() -> None:
    assert distance(["spam", "egg"], ["spam", "ham"], backend="python") == 1


def test_normalized_similarity_bounds() -> None:
    assert normalized_similarity("abc", "abc", backend="python") == 1.0
    assert normalized_similarity("", "", backend="python") == 1.0
    assert normalized_similarity("abc", "xyz", backend="python") == 0.0


def test_auto_backend_is_always_available() -> None:
    assert backend_info("auto").available is True


def test_missing_explicit_backend_can_fallback(monkeypatch) -> None:
    real = core._backend_info

    def fake(name):
        if name == "levenshtein":
            return core.BackendInfo(
                name="levenshtein",
                available=False,
                version=None,
                license="GPL-2.0-or-later",
                implementation="Levenshtein",
                reason="missing",
            )
        return real(name)

    monkeypatch.setattr(core, "_backend_info", fake)
    messages = []
    monkeypatch.setattr(core.logger, "warning", lambda msg, *args: messages.append(msg % args))
    assert distance("a", "b", backend="levenshtein", strict=False) == 1
    assert any("falling back" in message for message in messages)


def test_missing_explicit_backend_strict_raises(monkeypatch) -> None:
    real = core._backend_info

    def fake(name):
        if name == "levenshtein":
            return core.BackendInfo(
                name="levenshtein",
                available=False,
                version=None,
                license="GPL-2.0-or-later",
                implementation="Levenshtein",
                reason="missing",
            )
        return real(name)

    monkeypatch.setattr(core, "_backend_info", fake)
    with pytest.raises(ImportError):
        distance("a", "b", backend="levenshtein", strict=True)


def test_rank_is_stable_on_tie() -> None:
    values = ["cat", "bat", "hat"]
    result = rank("mat", values, backend="python")
    assert [m.choice for m in result] == values


def test_closest_empty_returns_none() -> None:
    assert closest("x", [], backend="python") is None


def test_corpus_scorer_is_lazy_and_callable() -> None:
    scorer = make_corpus_scorer(backend="python")
    assert callable(scorer)


def test_corpus_scorer_ranks_documents() -> None:
    from scikitplot.corpus import CorpusDocument, RetrievalConfig

    docs = [
        CorpusDocument.create("a.txt", 0, "kitten"),
        CorpusDocument.create("b.txt", 0, "sitting"),
    ]
    scorer = make_corpus_scorer(backend="python")
    hits = scorer("kitten", docs, RetrievalConfig(top_k=2))
    assert [hit.doc.text for hit in hits] == ["kitten", "sitting"]
    assert hits[0].score == 1.0
    assert hits[0].native_metric == "normalized_levenshtein_similarity"


def test_rapidfuzz_backend_when_available_matches_python() -> None:
    info = backend_info("rapidfuzz")
    if not info.available:
        pytest.skip("RapidFuzz not installed")
    assert distance("kitten", "sitting", backend="rapidfuzz") == 3


def test_rank_reports_actual_runtime_fallback_backend(monkeypatch) -> None:
    monkeypatch.setattr(core, "_resolve_backend_name", lambda name, strict: "rapidfuzz")

    def broken_impl(_name):
        def fail(a, b):
            raise RuntimeError("accelerator broke")
        return fail

    monkeypatch.setattr(core, "_distance_impl", broken_impl)
    monkeypatch.setattr(core.logger, "warning", lambda *args, **kwargs: None)
    match = core.rank("kitten", ["sitting"], backend="auto")[0]
    assert match.distance == 3
    assert match.backend == "python"


def test_corpus_scorer_provenance_uses_actual_match_backend(monkeypatch) -> None:
    from scikitplot.corpus import CorpusDocument, RetrievalConfig

    monkeypatch.setattr(core, "_resolve_backend_name", lambda name, strict: "rapidfuzz")

    def broken_impl(_name):
        def fail(a, b):
            raise RuntimeError("accelerator broke")
        return fail

    monkeypatch.setattr(core, "_distance_impl", broken_impl)
    monkeypatch.setattr(core.logger, "warning", lambda *args, **kwargs: None)
    scorer = core.make_corpus_scorer(backend="auto")
    hits = scorer(
        "kitten",
        [CorpusDocument.create("a.txt", 0, "sitting")],
        RetrievalConfig(top_k=1),
    )
    assert hits[0].backend == "levenshtein:python"


def test_rapidfuzz_matches_python_on_deterministic_unicode_grid() -> None:
    info = backend_info("rapidfuzz")
    if not info.available:
        pytest.skip("RapidFuzz not installed")
    values = ["", "a", "kitten", "sitting", "café", "cafe\u0301", "İstanbul", "東京", "🙂🙃"]
    for left in values:
        for right in values:
            assert distance(left, right, backend="rapidfuzz") == distance(
                left, right, backend="python"
            )


def test_rank_limit_streams_top_k_without_requiring_sequence() -> None:
    seen: list[str] = []

    def choices():
        for value in ("sitting", "kitten", "bitten", "written"):
            seen.append(value)
            yield value

    matches = rank("kitten", choices(), limit=2, backend="python")
    assert [m.choice for m in matches] == ["kitten", "bitten"]
    assert seen == ["sitting", "kitten", "bitten", "written"]


def test_rank_zero_limit_does_not_consume_choices() -> None:
    consumed = False

    def choices():
        nonlocal consumed
        consumed = True
        yield "anything"

    assert rank("query", choices(), limit=0, backend="python") == []
    assert consumed is False


def test_import_facade_does_not_import_optional_accelerators() -> None:
    import subprocess
    import sys
    import textwrap

    source = textwrap.dedent(
        """
        import sys
        import scikitplot
        watched = {"rapidfuzz", "Levenshtein"}
        before = {m for m in watched if m in sys.modules}
        import scikitplot.levenshtein
        after = {m for m in watched if m in sys.modules}
        print(",".join(sorted(after - before)))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    loaded = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    assert loaded == ""


def test_rank_score_cutoff_filters_distant_matches() -> None:
    matches = rank(
        "kitten",
        ["kitten", "bitten", "completely different"],
        backend="python",
        score_cutoff=0.7,
    )
    assert [m.choice for m in matches] == ["kitten", "bitten"]


def test_rank_score_cutoff_validates_bounds() -> None:
    with pytest.raises(ValueError, match="score_cutoff"):
        rank("a", ["a"], score_cutoff=1.1, backend="python")


def test_closest_respects_score_cutoff() -> None:
    assert closest("kitten", ["xxxxxx"], score_cutoff=0.8, backend="python") is None


def test_corpus_scorer_score_cutoff_can_return_fewer_than_top_k() -> None:
    from scikitplot.corpus import CorpusDocument, RetrievalConfig

    docs = [
        CorpusDocument.create("a.txt", 0, "kitten"),
        CorpusDocument.create("b.txt", 0, "xxxxxx"),
    ]
    scorer = make_corpus_scorer(backend="python", score_cutoff=0.8)
    hits = scorer("kitten", docs, RetrievalConfig(top_k=10))
    assert [hit.doc.text for hit in hits] == ["kitten"]


def test_match_to_dict_reports_backend_provenance() -> None:
    match = rank("kitten", ["sitting"], backend="python")[0]
    assert match.to_dict()["backend"] == "python"


def test_top_level_scikitplot_lazy_attribute_exposes_levenshtein() -> None:
    import scikitplot

    assert "levenshtein" in scikitplot._submodules
    assert scikitplot.levenshtein.distance(
        "kitten",
        "sitting",
        backend="python",
    ) == 3
