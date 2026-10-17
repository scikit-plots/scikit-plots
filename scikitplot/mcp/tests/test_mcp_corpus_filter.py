"""
MCP retrieval is scoped by the corpus filter contract, never its own (R5-3).
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from scikitplot.mcp import __main__ as cli
from scikitplot.mcp._corpus_annoy import CorpusAnnoyRetriever, CorpusIndexRetriever

corpus = pytest.importorskip("scikitplot.corpus")

KEYWORD = corpus.RetrievalConfig(top_k=5, match_mode="keyword")
EN = {"op": "eq", "field": "language", "value": "en"}


@pytest.fixture(scope="module")
def index():
    docs = [
        corpus.CorpusDocument.create(
            input_path=f"docs/{name}.md", chunk_index=0, text=text, language=lang
        )
        for name, text, lang in (
            ("metrics", "roc auc score for the metrics page", "en"),
            ("metrik", "roc auc score fuer die metrik seite", "de"),
            ("build", "meson builds the extension", "en"),
        )
    ]
    built = corpus.RetrievalIndex(KEYWORD)
    built.build(docs)
    return built


def _uris(outcome):
    return sorted(chunk.source_uri for chunk in outcome)


class TestRetriever:
    def test_without_a_filter_every_document_is_searched(self, index):
        retriever = CorpusIndexRetriever(None, index, match_mode="keyword")
        assert _uris(retriever.search("roc auc score", 5)) == [
            "docs/metrics.md",
            "docs/metrik.md",
        ]
        assert retriever.filter is None

    @pytest.mark.parametrize("scope", [EN, "expression"])
    def test_a_filter_narrows_every_query(self, index, scope):
        expression = corpus.filter_from_dict(EN) if scope == "expression" else scope
        retriever = CorpusIndexRetriever(
            None, index, match_mode="keyword", filter=expression
        )
        assert _uris(retriever.search("roc auc score", 5)) == ["docs/metrics.md"]
        assert corpus.filter_to_dict(retriever.filter) == EN

    def test_the_same_filter_as_the_corpus_gives_the_same_hits(self, index):
        retriever = CorpusIndexRetriever(None, index, match_mode="keyword", filter=EN)
        direct = index.search("roc auc score", config=KEYWORD, filter=EN)
        assert [c.doc_id for c in retriever.search("roc auc score", 5)] == [
            h.doc.doc_id for h in direct
        ]

    @pytest.mark.parametrize(
        ("scope", "error"),
        [
            ({"op": "eq", "field": "no_such_field", "value": 1}, ValueError),
            ({"op": "nope"}, ValueError),
            ("language == en", TypeError),
        ],
    )
    def test_a_malformed_filter_is_refused_at_construction(self, index, scope, error):
        with pytest.raises(error):
            CorpusIndexRetriever(None, index, match_mode="keyword", filter=scope)

    def test_the_factory_checks_the_filter_before_building(self, tmp_path):
        with pytest.raises(ValueError, match="no_such_field"):
            CorpusAnnoyRetriever.from_corpus_annoy(
                str(tmp_path), filter={"op": "eq", "field": "no_such_field", "value": 1}
            )


class TestCli:
    def _parse(self, *argv):
        return cli._resolve_config(cli._parser().parse_args(list(argv)), environ={})

    def test_the_flag_needs_the_corpus_profile(self, tmp_path):
        with pytest.raises(SystemExit, match="--corpus-annoy only"):
            self._parse("--corpus-filter", str(tmp_path / "f.json"))
        assert self._parse().corpus_filter is None

    def test_the_filter_reaches_the_factory(self, monkeypatch, tmp_path):
        path = tmp_path / "scope.json"
        path.write_text(json.dumps(EN), encoding="utf-8")
        captured = {}
        monkeypatch.setattr(
            CorpusAnnoyRetriever,
            "from_corpus_annoy",
            classmethod(
                lambda cls, p, **kwargs: captured.update(kwargs) or "retriever"
            ),
        )
        config = self._parse(
            "--corpus-annoy", str(tmp_path), "--corpus-filter", str(path)
        )
        assert cli._build_corpus_annoy_retriever(config) == "retriever"
        assert corpus.filter_to_dict(captured["filter"]) == EN

    @pytest.mark.parametrize(
        ("content", "match"),
        [
            (None, "does not exist"),
            ("{not json", "not a JSON file"),
            (
                json.dumps({"op": "eq", "field": "no_such_field", "value": 1}),
                "no_such_field",
            ),
        ],
    )
    def test_a_bad_filter_file_stops_the_server(self, tmp_path, content, match):
        path = tmp_path / "scope.json"
        if content is not None:
            path.write_text(content, encoding="utf-8")
        config = self._parse(
            "--corpus-annoy", str(tmp_path), "--corpus-filter", str(path)
        )
        with pytest.raises(SystemExit, match=match):
            cli._build_corpus_annoy_retriever(config)


def test_real_build_in_semantic_mode_with_a_filter(tmp_path):
    """The vector seam takes no filter: with one, semantic runs through search."""
    try:
        import scikitplot.annoy._annoy.annoylib  # noqa: F401, PLC0415
    except ImportError as exc:
        pytest.skip(f"Annoy extension unavailable: {exc}")
    (tmp_path / "a.md").write_text("# A\n\nroc auc score metrics\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\n\nmeson ninja build\n", encoding="utf-8")
    scope = {"op": "eq", "field": "input_path", "value": "a.md"}
    retriever = CorpusAnnoyRetriever.from_corpus_annoy(
        str(tmp_path),
        embedder=corpus.HashEmbedder(dimension=64),
        n_trees=5,
        filter=scope,
    )
    assert isinstance(retriever, CorpusIndexRetriever)
    assert retriever.match_mode == "semantic"
    hits = retriever.search("meson ninja build", 5)
    assert {c.source_uri.rsplit("/", 1)[-1] for c in hits} <= {"a.md"}
    assert [
        c.source_uri.rsplit("/", 1)[-1] for c in retriever.search("roc auc", 5)
    ] == [
        "a.md"
    ]  # the scope matches: the check above is not vacuous
    found = retriever.search("roc auc", 5)[0]
    assert retriever.get(found.doc_id).source_uri.endswith("a.md")


# --------------------------------------------------------------- review 5c


class TestReview:
    def test_get_serves_only_documents_in_scope(self, index):
        documents = list(index._documents)
        by_lang = {d.language: d.doc_id for d in documents if d.language == "de"}
        retriever = CorpusIndexRetriever(
            None, index, match_mode="keyword", filter=EN, documents=documents
        )
        inside = next(d.doc_id for d in documents if d.language == "en")
        assert retriever.get(inside).doc_id == inside
        assert retriever.get(by_lang["de"]) is None  # hidden, as if missing
        assert retriever.get("no-such-id") is None
        assert (
            CorpusIndexRetriever(None, index, match_mode="keyword").get(inside) is None
        )

    def test_get_respects_the_indexs_security_filter(self):
        docs = [
            corpus.CorpusDocument.create(
                input_path=f"{lang}.md",
                chunk_index=0,
                text=f"text {lang}",
                language=lang,
            )
            for lang in ("en", "de")
        ]
        secured = corpus.RetrievalIndex(
            KEYWORD,
            security_filter=corpus.SecurityFilter(
                corpus.filter_from_dict(EN), label="en-only"
            ),
        )
        secured.build(docs)
        retriever = CorpusIndexRetriever(
            None, secured, match_mode="keyword", documents=docs
        )
        assert retriever.get(docs[0].doc_id) is not None
        assert retriever.get(docs[1].doc_id) is None

    @pytest.mark.parametrize(
        "scope",
        [
            {"op": "and", "args": []},
            {"op": "or", "args": []},
            {"op": "not", "arg": {"op": "and", "args": []}},
            {"op": "and", "args": [EN, {"op": "or", "args": []}]},
        ],
    )
    def test_a_scope_that_narrows_nothing_is_refused(self, index, scope):
        with pytest.raises(ValueError, match="empty"):
            CorpusIndexRetriever(None, index, match_mode="keyword", filter=scope)

    @pytest.mark.parametrize("mode", ["hybrid", "keyword"])
    def test_the_server_starts_with_a_corpus_index_retriever(
        self, monkeypatch, tmp_path, mode
    ):
        # The serve path wires ``document_reader=retriever.get``; a
        # CorpusIndexRetriever had no ``get`` (hybrid and keyword modes, and
        # every mode with a filter, crashed at start).
        started = {}

        class Server:
            def run(self, **kwargs):
                started["run"] = kwargs

        def create_server(retriever, **kwargs):
            started["reader"] = kwargs["document_reader"]
            return Server()

        monkeypatch.setattr(cli, "create_server", create_server)
        monkeypatch.setattr(
            cli, "server_runtime_status", lambda: {"server_available": True}
        )
        retriever = CorpusIndexRetriever(None, object(), match_mode="keyword")
        monkeypatch.setattr(cli, "_load_retriever", lambda config: retriever)
        assert cli.main(["--corpus-annoy", str(tmp_path), "--corpus-mode", mode]) == 0
        assert started["reader"] == retriever.get


def test_get_returns_the_first_copy_in_scope():
    """An index may hold two documents with one id; the scope picks between them."""
    de, en = (
        corpus.CorpusDocument.create(
            input_path="same.md", chunk_index=0, text=f"roc auc {lang}", language=lang
        )
        for lang in ("de", "en")
    )
    de = dataclasses.replace(de, doc_id="dup")
    en = dataclasses.replace(en, doc_id="dup")
    index = corpus.RetrievalIndex(KEYWORD)
    index.build([de, en])
    scoped = CorpusIndexRetriever(
        None, index, match_mode="keyword", filter=EN, documents=[de, en]
    )
    assert scoped.get("dup").text == "roc auc en"
    unscoped = CorpusIndexRetriever(
        None, index, match_mode="keyword", documents=[de, en]
    )
    assert unscoped.get("dup").text == "roc auc de"
