# scikitplot/corpus/_metrics/tests/test__scorer.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""scikit-learn compatible corpus scorers."""

from __future__ import annotations

import copy
import pickle
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from ..._diagnostics import ErrorCategory, ErrorRecord
from ..._retrieval import LegKind, LegOutcome, LegStatus, RetrievalResponse
from ..._schema import CorpusDocument
from ..._similarity import RetrievalConfig, RetrievalIndex
from .. import _scorer
from .._ranking import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from .._scorer import (
    CorpusScorerMixin,
    get_corpus_scorer,
    get_corpus_scorer_names,
    make_corpus_scorer,
    scorer_backend_info,
)
from .._text import token_f1_score


class FixedIndex:
    """``search`` returns the same ranking for every query."""

    def __init__(self, ranking):
        self.ranking = ranking

    def search(self, query):
        return [(doc_id, 1.0) for doc_id in self.ranking]


class EchoPredictor:
    """Not a scikit-learn estimator: ``predict`` returns one ranking per query."""

    def predict(self, X):
        return [[query] for query in X]


def _documents():
    texts = {
        "d1": "hemoglobin carries oxygen in red blood cells",
        "d2": "platelets help blood clot",
        "d3": "white blood cells fight infection",
    }
    return [
        CorpusDocument(doc_id=key, input_path=f"memory://{key}", chunk_index=0, text=text)
        for key, text in texts.items()
    ]


def _failed_response():
    error = ErrorRecord(code="NO_QUERY_EMBEDDING", category=ErrorCategory.CAPABILITY, message="no embedding")
    return RetrievalResponse(
        hits=[],
        legs=[
            LegOutcome(LegKind.LEXICAL, LegStatus.EMPTY),
            LegOutcome(LegKind.DENSE, LegStatus.FAILED, error=error),
        ],
    )


class TestPredictions:
    def test_search_scores_a_corpus_index(self):
        scorer = make_corpus_scorer(recall_at_k, response_method="search", k=2)
        assert scorer(FixedIndex(["d1", "d2"]), ["q1", "q2"], [["d2"], ["d3"]]) == 0.5

    def test_search_on_a_real_index(self):
        index = RetrievalIndex(RetrievalConfig(match_mode="keyword", top_k=3))
        index.build(_documents())
        scorer = make_corpus_scorer(recall_at_k, response_method="search", k=1)
        assert scorer(index, ["what carries oxygen"], [["d1"]]) == 1.0

    def test_predict_on_a_plain_object(self):
        scorer = make_corpus_scorer(recall_at_k, k=1)
        assert scorer(EchoPredictor(), ["a", "b"], [["a"], ["x"]]) == 0.5

    def test_id_getter_and_duplicates(self):
        chunks = FixedIndex(["doc1#0", "doc1#1", "doc2#0"])
        scorer = make_corpus_scorer(
            recall_at_k,
            response_method="search",
            id_getter=lambda result: result[0].split("#")[0],
            k=2,
            duplicates="first",
        )
        assert scorer(chunks, ["q"], [["doc2"]]) == 1.0

    def test_greater_is_better_false_negates(self):
        scorer = make_corpus_scorer(recall_at_k, response_method="search", greater_is_better=False, k=2)
        assert scorer(FixedIndex(["d1"]), ["q"], [["d1"]]) == -1.0

    def test_sample_weight(self):
        scorer = make_corpus_scorer(recall_at_k, k=1)
        assert scorer(EchoPredictor(), ["a", "b"], [["a"], ["x"]], sample_weight=[3, 1]) == 0.75

    def test_answer_metric(self):
        class Answerer:
            def predict(self, X):
                return ["the cat sat" for _ in X]

        scorer = make_corpus_scorer(token_f1_score)
        assert scorer(Answerer(), ["q"], ["a cat sat"]) == 1.0

    def test_missing_methods(self):
        with pytest.raises(AttributeError, match="no predict method"):
            make_corpus_scorer(recall_at_k)(FixedIndex([]), ["q"], [["a"]])
        with pytest.raises(AttributeError, match="no search method"):
            make_corpus_scorer(recall_at_k, response_method="search")(EchoPredictor(), ["q"], [["a"]])

    def test_a_string_is_not_a_query_list(self):
        scorer = make_corpus_scorer(recall_at_k, response_method="search")
        with pytest.raises(TypeError, match="sequence of queries"):
            scorer(FixedIndex(["a"]), "q", [["a"]])


class TestDegradedResponses:
    class Degraded:
        def search(self, query):
            return _failed_response()

    def test_raise_by_default(self):
        scorer = make_corpus_scorer(recall_at_k, response_method="search")
        # Lexical ran, dense failed: the response derives DEGRADED, not FAILED.
        with pytest.raises(RuntimeError, match="'degraded' response"):
            scorer(self.Degraded(), ["q"], [["a"]])

    def test_ignore(self):
        scorer = make_corpus_scorer(recall_at_k, response_method="search", on_degraded="ignore")
        assert scorer(self.Degraded(), ["q"], [["a"]]) == 0.0


class TestConstruction:
    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"score_func": 3}, TypeError, "score_func must be callable"),
            ({"response_method": "decision_function"}, ValueError, "response_method"),
            ({"greater_is_better": 1}, TypeError, "greater_is_better"),
            ({"id_getter": 3, "response_method": "search"}, TypeError, "id_getter"),
            ({"id_getter": str}, ValueError, "search' only"),
            ({"on_degraded": "warn"}, ValueError, "on_degraded"),
            ({"kk": 3}, TypeError, "does not accept"),
        ],
    )
    def test_refused(self, kwargs, error, match):
        options = {"score_func": recall_at_k, **kwargs}
        with pytest.raises(error, match=match):
            make_corpus_scorer(**options)

    def test_repr(self):
        assert repr(make_corpus_scorer(ndcg_at_k, response_method="search", k=5)) == (
            "make_corpus_scorer(ndcg_at_k, response_method='search', k=5)"
        )
        assert repr(make_corpus_scorer(recall_at_k, greater_is_better=False)) == (
            "make_corpus_scorer(recall_at_k, greater_is_better=False, response_method='predict')"
        )

    def test_repr_shows_non_default_search_options(self):
        """Review F19: two scorers that behave differently printed the same."""
        def by_source(hit):
            return hit[0]

        assert repr(make_corpus_scorer(recall_at_k, response_method="search", on_degraded="ignore")) == (
            "make_corpus_scorer(recall_at_k, response_method='search', on_degraded='ignore')"
        )
        assert repr(make_corpus_scorer(recall_at_k, response_method="search", id_getter=by_source)) == (
            "make_corpus_scorer(recall_at_k, response_method='search', id_getter=by_source)"
        )

    def test_pickle_and_deepcopy(self):
        scorer = make_corpus_scorer(recall_at_k, response_method="search", k=2)
        for clone in (pickle.loads(pickle.dumps(scorer)), copy.deepcopy(scorer)):  # noqa: S301 - own object
            assert type(clone) is type(scorer)
            assert clone(FixedIndex(["d1", "d2"]), ["q"], [["d2"]]) == 1.0

    def test_is_a_corpus_scorer(self):
        assert isinstance(make_corpus_scorer(recall_at_k), CorpusScorerMixin)


class TestNames:
    def test_known_names(self):
        assert get_corpus_scorer_names() == [
            "anls", "edit_similarity", "exact_match", "hit_rate", "map", "mrr",
            "ndcg", "precision", "r_precision", "recall", "token_f1",
        ]

    def test_cutoff_comes_from_the_name(self):
        scorer = get_corpus_scorer("recall@5", response_method="search", zero_division=0.0)
        assert scorer._kwargs == {"k": 5, "zero_division": 0.0}
        assert get_corpus_scorer("mrr")._kwargs == {"k": None}
        assert get_corpus_scorer("r_precision")._kwargs == {}
        assert get_corpus_scorer("token_f1")._score_func is token_f1_score

    def test_new_instance_each_call(self):
        assert get_corpus_scorer("mrr") is not get_corpus_scorer("mrr")

    def test_refused(self):
        with pytest.raises(ValueError, match="not a corpus scorer name"):
            get_corpus_scorer("accuracy")
        with pytest.raises(ValueError, match="do not pass it again"):
            get_corpus_scorer("recall@5", k=3)
        with pytest.raises(TypeError):
            get_corpus_scorer(None)

    def test_k_messages_are_accurate(self):
        """Review F15: a name without a cutoff did not 'give' k."""
        with pytest.raises(ValueError, match="for example mrr@5"):
            get_corpus_scorer("mrr", k=5)
        with pytest.raises(ValueError, match="r_precision takes no cutoff"):
            get_corpus_scorer("r_precision", k=5)


class TestBackend:
    def test_info(self):
        info = scorer_backend_info()
        try:
            import sklearn  # noqa: F401
        except ImportError:
            assert info.base == "vendored"
            assert info.sklearn_version is None
        else:
            assert info.base == "sklearn"
            assert info.reason is None
        assert set(info.to_dict()) == {"base", "sklearn_version", "reason"}

    def test_contract_probe_detects_a_wrong_result(self):
        from .._sklearn_compat import CorpusScorer as Vendored

        class Wrong(Vendored):
            def _score(self, method_caller, estimator, X, y_true, **kwargs):
                return 0.0

        class Raising(Vendored):
            def _score(self, method_caller, estimator, X, y_true, **kwargs):
                raise KeyError("boom")

        assert "instead of -0.5 and -0.75" in _scorer._contract_problem(Wrong)
        assert "raised KeyError" in _scorer._contract_problem(Raising)
        assert _scorer._contract_problem(Vendored) is None

    @pytest.mark.parametrize(
        ("blocked", "reason"),
        [("sklearn", "scikit-learn is not importable"), ("sklearn.metrics._scorer", "private _BaseScorer")],
    )
    def test_resolution_falls_back_to_the_vendored_base(self, blocked, reason):
        if blocked != "sklearn":
            pytest.importorskip("sklearn")
        source = textwrap.dedent(
            f"""
            import sys
            sys.modules[{blocked!r}] = None
            from scikitplot.corpus._metrics import _scorer, recall_at_k
            info = _scorer.scorer_backend_info()
            scorer = _scorer.make_corpus_scorer(recall_at_k, k=1)
            class P:
                def predict(self, X):
                    return [[q] for q in X]
            print(info.base, "|", info.reason, "|", type(scorer).__module__, "|", scorer(P(), ["a"], [["a"]]))
            """
        )
        proc = subprocess.run(  # noqa: S603 - this interpreter, a fixed program
            [sys.executable, "-c", source], capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, proc.stderr
        base, why, module, value = (part.strip() for part in proc.stdout.strip().splitlines()[-1].split("|"))
        assert base == "vendored"
        assert reason in why
        assert module.endswith("._sklearn_compat")
        assert value == "1.0"


def _run_child(source):
    proc = subprocess.run(  # noqa: S603 - this interpreter, a fixed program
        [sys.executable, "-c", textwrap.dedent(source)], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("error", ["ValueError", "AttributeError"])
def test_a_broken_scikit_learn_falls_back_to_the_vendored_base(tmp_path, error):
    """Review F2: only ImportError fell back; an ABI mismatch raises ValueError."""
    package = tmp_path / "sklearn"
    package.mkdir()
    (package / "__init__.py").write_text(f"raise {error}('broken installation')\n", encoding="utf-8")
    line = _run_child(
        f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        from scikitplot.corpus._metrics import _scorer, recall_at_k
        info = _scorer.scorer_backend_info()
        class P:
            def predict(self, X):
                return [[q] for q in X]
        print(info.base, "|", info.reason, "|", _scorer.make_corpus_scorer(recall_at_k, k=1)(P(), ["a"], [["a"]]))
        """
    )
    base, reason, value = (part.strip() for part in line.split("|"))
    assert base == "vendored"
    assert f"importing scikit-learn raised {error}: broken installation" in reason
    assert value == "1.0"


def test_resolution_runs_once_under_racing_threads():
    """Review F20: lru_cache did not serialise the first call."""
    line = _run_child(
        """
        import threading
        from scikitplot.corpus._metrics import _scorer
        calls = []
        real = _scorer._compute_resolution
        def counted():
            calls.append(1)
            return real()
        _scorer._compute_resolution = counted
        barrier = threading.Barrier(16)
        def worker():
            barrier.wait()
            _scorer._resolve()
        threads = [threading.Thread(target=worker) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        print(len(calls), len({id(_scorer._resolve()[0])}))
        """
    )
    assert line == "1 1"


class TestScikitLearn:
    """The scorers inside scikit-learn's own model-selection machinery."""

    @pytest.fixture
    def sklearn(self):
        return pytest.importorskip("sklearn")

    @staticmethod
    def _retriever_class():
        from sklearn.base import BaseEstimator

        class KeywordRetriever(BaseEstimator):
            calls = 0

            def __init__(self, documents=None, top_k=3):
                self.documents = documents
                self.top_k = top_k

            def fit(self, X, y=None):
                self.index_ = RetrievalIndex(RetrievalConfig(match_mode="keyword", top_k=self.top_k))
                self.index_.build(self.documents)
                return self

            def predict(self, X):
                type(self).calls += 1
                return [[hit.doc.doc_id for hit in self.index_.search(q)] for q in X]

        return KeywordRetriever

    QUERIES = ("what carries oxygen", "blood clotting cells", "fight infection", "red blood")
    QRELS = (("d1",), ("d2",), ("d3",), ("d1",))

    def test_cross_validate_with_a_dict_shares_predict(self, sklearn):
        from sklearn.model_selection import KFold, cross_validate

        retriever = self._retriever_class()
        retriever.calls = 0
        result = cross_validate(
            retriever(documents=_documents()),
            self.QUERIES,
            self.QRELS,
            cv=KFold(2),
            scoring={"recall@1": get_corpus_scorer("recall@1"), "mrr": get_corpus_scorer("mrr")},
        )
        assert set(result) >= {"test_recall@1", "test_mrr"}
        assert np.all((result["test_mrr"] >= 0) & (result["test_mrr"] <= 1))
        version = tuple(int(p) for p in sklearn.__version__.split(".")[:2])
        assert retriever.calls == (2 if version >= (1, 4) else 4)

    def test_grid_search(self, sklearn):
        from sklearn.model_selection import GridSearchCV, KFold

        search = GridSearchCV(
            self._retriever_class()(documents=_documents()),
            {"top_k": [1, 3]},
            scoring={"recall@3": get_corpus_scorer("recall@3"), "mrr": get_corpus_scorer("mrr")},
            refit="recall@3",
            cv=KFold(2),
        ).fit(self.QUERIES, self.QRELS)
        assert search.best_params_ == {"top_k": 3}

    def test_check_scoring_returns_the_scorer(self, sklearn):
        from sklearn.metrics import check_scoring

        scorer = get_corpus_scorer("ndcg@3")
        assert check_scoring(self._retriever_class()(), scoring=scorer) is scorer

    def test_metadata_routing(self, sklearn):
        version = tuple(int(p) for p in sklearn.__version__.split(".")[:2])
        if version < (1, 4):
            pytest.skip("cross_validate takes params= from scikit-learn 1.4")
        from sklearn.model_selection import KFold, cross_validate

        weights = np.array([1.0, 1.0, 3.0, 1.0])
        with sklearn.config_context(enable_metadata_routing=True):
            scorer = get_corpus_scorer("recall@1").set_score_request(sample_weight=True)
            routed = cross_validate(
                self._retriever_class()(documents=_documents()),
                self.QUERIES,
                self.QRELS,
                cv=KFold(2),
                scoring=scorer,
                params={"sample_weight": weights},
            )
        retriever = self._retriever_class()(documents=_documents()).fit(None)
        predictions = retriever.predict(self.QUERIES)
        expected = [
            recall_at_k(self.QRELS[:2], predictions[:2], k=1, sample_weight=weights[:2]),
            recall_at_k(self.QRELS[2:], predictions[2:], k=1, sample_weight=weights[2:]),
        ]
        np.testing.assert_allclose(routed["test_score"], expected)

    def test_overlap_warning(self, sklearn):
        with sklearn.config_context(enable_metadata_routing=True):
            scorer = make_corpus_scorer(mean_reciprocal_rank, sample_weight=None)
            with pytest.warns(UserWarning, match="Overlapping parameters"):
                scorer.set_score_request(sample_weight=True)
