"""The MCP surface can serve the corpus index's hybrid retrieval (finding CX-04)."""

from __future__ import annotations

import dataclasses
import types

import pytest

from scikitplot.mcp import __main__ as cli
from scikitplot.mcp._corpus_annoy import CorpusAnnoyRetriever, CorpusIndexRetriever
from scikitplot.mcp._outcome import DEGRADED, FAILED, SUCCESS, status_of


@dataclasses.dataclass(frozen=True)
class _Config:
    match_mode: str = "semantic"
    top_k: int = 10
    rrf_k: int = 60


@dataclasses.dataclass
class _Leg:
    leg: str
    status: str
    hit_count: int = 0
    error: object = None


class _Index:
    """A corpus-shaped index: ``search(query, *, config, query_embedding)``."""

    def __init__(self, hits, legs):
        self.config = _Config()
        self.hits, self.legs, self.calls = hits, legs, []

    def search(self, query, *, config, query_embedding=None):
        self.calls.append((query, config, query_embedding))
        return _Response(self.hits, self.legs)


class _Response(list):
    def __init__(self, hits, legs):
        super().__init__(hits)
        self.legs = legs


def _hit(doc_id, text, score):
    doc = types.SimpleNamespace(doc_id=doc_id, text=text, source_uri=f"docs/{doc_id}.md",
                                title=doc_id, anchor="")
    return types.SimpleNamespace(doc=doc, score=score)


class _Embedder:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    def embed(self, text):
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedding service down")
        return [0.1, 0.2]


def test_hybrid_asks_the_corpus_with_its_own_config():
    index = _Index([_hit("a", "roc_auc_score", 0.03)], [_Leg("lexical", "success", 1), _Leg("dense", "success", 1)])
    out = CorpusIndexRetriever(_Embedder(), index).search("roc_auc_score", 4)
    (query, config, vector), = index.calls
    assert config == _Config(match_mode="hybrid", top_k=4, rrf_k=60)
    assert vector == [0.1, 0.2]
    assert [c.doc_id for c in out] == ["a"] and out[0].score == 0.03
    assert [(leg.leg, leg.status) for leg in out.legs] == [("lexical", SUCCESS), ("dense", SUCCESS)]


def test_a_skipped_leg_is_not_reported():
    index = _Index([_hit("a", "x", 1.0)], [_Leg("lexical", "success", 1), _Leg("dense", "skipped")])
    out = CorpusIndexRetriever(None, index, match_mode="keyword").search("x", 3)
    assert [leg.leg for leg in out.legs] == ["lexical"] and status_of(out) == SUCCESS


def test_keyword_mode_never_embeds():
    embedder = _Embedder()
    CorpusIndexRetriever(embedder, _Index([], []), match_mode="keyword").search("x", 3)
    assert embedder.calls == 0


def test_a_failed_embedding_degrades_hybrid_but_fails_semantic():
    legs = [_Leg("lexical", "success", 1), _Leg("dense", "failed", 0, types.SimpleNamespace(message="no query embedding"))]
    hybrid = CorpusIndexRetriever(_Embedder(fail=True), _Index([_hit("a", "x", 1.0)], legs)).search("x", 3)
    assert status_of(hybrid) == DEGRADED and [c.doc_id for c in hybrid] == ["a"]
    assert ("dense", FAILED) in [(leg.leg, leg.status) for leg in hybrid.legs]
    semantic = CorpusIndexRetriever(_Embedder(fail=True), _Index([], []), match_mode="semantic").search("x", 3)
    assert status_of(semantic) == FAILED and list(semantic) == []


def test_strict_reraises():
    with pytest.raises(RuntimeError, match="embedding service down"):
        CorpusIndexRetriever(_Embedder(fail=True), _Index([], []), strict=True).search("x", 3)


def test_hits_without_text_or_repeated_are_dropped():
    index = _Index([_hit("a", "x", 2.0), _hit("a", "x", 1.0), _hit("b", "", 1.0)], [])
    out = CorpusIndexRetriever(_Embedder(), index).search("x", 5)
    assert [c.doc_id for c in out] == ["a"]


@pytest.mark.parametrize("kwargs", [{"match_mode": "strict"}, {"match_mode": "fuzzy"}])
def test_modes_are_validated(kwargs):
    with pytest.raises(ValueError, match="match_mode"):
        CorpusIndexRetriever(_Embedder(), _Index([], []), **kwargs)


def test_an_embedder_is_required_except_for_keyword():
    with pytest.raises(ValueError, match="embedder"):
        CorpusIndexRetriever(None, _Index([], []))


def test_the_factory_rejects_a_bad_mode_before_building(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        CorpusAnnoyRetriever.from_corpus_annoy(str(tmp_path), mode="dense")


def _parse(*argv, environ=None):
    return cli._resolve_config(cli._parser().parse_args(list(argv)), environ=environ or {})


def test_the_cli_default_is_unchanged():
    assert _parse().corpus_mode == "semantic"


def test_the_cli_flag_and_environment():
    assert _parse("--corpus-mode", "hybrid").corpus_mode == "hybrid"
    assert _parse(environ={"SCIKITPLOT_MCP_CORPUS_MODE": "keyword"}).corpus_mode == "keyword"
    assert _parse("--corpus-mode", "semantic", environ={"SCIKITPLOT_MCP_CORPUS_MODE": "hybrid"}).corpus_mode == "semantic"
    with pytest.raises(SystemExit, match="corpus-mode"):
        _parse(environ={"SCIKITPLOT_MCP_CORPUS_MODE": "bogus"})


def test_the_cli_passes_the_mode_to_the_factory(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        CorpusAnnoyRetriever, "from_corpus_annoy",
        classmethod(lambda cls, path, **kwargs: captured.update(kwargs) or "retriever"),
    )
    config = _parse("--corpus-annoy", str(tmp_path), "--corpus-mode", "hybrid")
    assert cli._build_corpus_annoy_retriever(config) == "retriever"
    assert captured["mode"] == "hybrid"


def test_real_corpus_and_annoy_in_hybrid_mode(tmp_path):
    """End to end over a real build; skipped where Annoy cannot load."""
    # importorskip only treats ModuleNotFoundError as "absent" on current
    # pytest; the compiled extension can also raise a plain ImportError while
    # initialising (it imports root-package attributes), which means the same
    # thing here: Annoy cannot load in this environment.
    try:
        import scikitplot.annoy._annoy.annoylib  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"Annoy extension unavailable: {exc}")
    corpus = pytest.importorskip("scikitplot.corpus")
    (tmp_path / "metrics.md").write_text(
        "# Metrics\n\nUse roc_auc_score from sklearn.metrics to compute the area under the ROC curve.\n", encoding="utf-8"
    )
    (tmp_path / "build.md").write_text(
        "# Building\n\nThe C++ extension annoylib is compiled with meson and ninja.\n", encoding="utf-8"
    )
    retriever = CorpusAnnoyRetriever.from_corpus_annoy(
        str(tmp_path), embedder=corpus.HashEmbedder(dimension=64), mode="hybrid", n_trees=5
    )
    assert isinstance(retriever, CorpusIndexRetriever)
    out = retriever.search("roc_auc_score", 2)
    assert out and out[0].source_uri.endswith("metrics.md")
    assert {(leg.leg, leg.status) for leg in out.legs} == {("lexical", SUCCESS), ("dense", SUCCESS)}
