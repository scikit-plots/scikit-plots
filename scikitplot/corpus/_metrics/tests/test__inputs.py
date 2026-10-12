# scikitplot/corpus/_metrics/tests/test__inputs.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Validation and normalisation shared by every evaluation path."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from ..._diagnostics import ErrorCategory, ErrorRecord
from ..._retrieval import (
    LegKind,
    LegOutcome,
    LegStatus,
    RetrievalResponse,
    RetrievalStatus,
)
from ..._schema import CorpusDocument
from ..._similarity import RetrievalHit
from .._inputs import (
    UndefinedMetricWarning,
    aggregate,
    aligned,
    as_id,
    check_average,
    check_duplicates,
    check_id_types,
    check_k,
    check_on_degraded,
    check_relevance_threshold,
    check_sample_weight,
    check_zero_division,
    judgments,
    ranked_ids,
    ranking,
    response_status,
)


class TestCheckers:
    @pytest.mark.parametrize(("value", "expected"), [(1, 1), (np.int64(3), 3), (None, None)])
    def test_k_accepted(self, value, expected):
        assert check_k(value) == expected

    @pytest.mark.parametrize(
        ("value", "error"),
        [(0, ValueError), (-2, ValueError), (True, TypeError), (2.0, TypeError), ("3", TypeError)],
    )
    def test_k_refused(self, value, error):
        with pytest.raises(error):
            check_k(value)

    def test_k_none_where_required(self):
        with pytest.raises(ValueError, match="required"):
            check_k(None, allow_none=False)

    @pytest.mark.parametrize(("value", "expected"), [(1, 1.0), (0.5, 0.5), (np.float32(2), 2.0)])
    def test_threshold_accepted(self, value, expected):
        assert check_relevance_threshold(value) == expected

    @pytest.mark.parametrize(
        ("value", "error"),
        [
            (0, ValueError),
            (-1, ValueError),
            (math.inf, ValueError),
            (math.nan, ValueError),
            (True, TypeError),
            ("1", TypeError),
        ],
    )
    def test_threshold_refused(self, value, error):
        with pytest.raises(error):
            check_relevance_threshold(value)

    @pytest.mark.parametrize("value", ["warn", 0, 0.0, 1, 1.0, np.nan])
    def test_zero_division_accepted(self, value):
        result = check_zero_division(value)
        assert result == "warn" or result in (0.0, 1.0) or math.isnan(result)

    @pytest.mark.parametrize("value", [0.5, "zero", True, None, "nan"])
    def test_zero_division_refused(self, value):
        with pytest.raises(ValueError, match="zero_division"):
            check_zero_division(value)

    def test_average(self):
        assert check_average("mean") == "mean"
        assert check_average(None) is None
        with pytest.raises(ValueError, match="average must be"):
            check_average("macro")

    def test_duplicates_policy(self):
        assert check_duplicates("raise") == "raise"
        assert check_duplicates("first") == "first"
        with pytest.raises(ValueError, match="duplicates must be one of"):
            check_duplicates("last")

    def test_on_degraded_policy(self):
        assert check_on_degraded("warn") == "warn"
        with pytest.raises(ValueError, match="on_degraded must be one of"):
            check_on_degraded("warn", allowed=("raise", "ignore"))

    def test_sample_weight(self):
        assert check_sample_weight(None, 3) is None
        np.testing.assert_array_equal(check_sample_weight([1, 2], 2), [1.0, 2.0])
        with pytest.raises(ValueError, match="shape"):
            check_sample_weight([1, 2], 3)
        for bad in ([1, -1], [1, np.nan], [np.inf, 1]):
            with pytest.raises(ValueError, match="finite and non-negative"):
                check_sample_weight(bad, 2)
        with pytest.raises(TypeError, match="mapping"):
            check_sample_weight({"q1": 1}, 1)


class TestIdentifiers:
    def test_strings_and_integers(self):
        assert as_id("a", label="x") == "a"
        normalised = as_id(np.str_("a"), label="x")
        assert normalised == "a"
        assert type(normalised) is str
        assert as_id(np.int64(3), label="x") == 3
        assert type(as_id(np.int64(3), label="x")) is int

    @pytest.mark.parametrize("value", [True, 1.0, None, b"a", ("a",)])
    def test_other_types_refused(self, value):
        with pytest.raises(TypeError, match="identifiers must be str or int"):
            as_id(value, label="query 'q'")


class TestJudgments:
    def test_graded_mapping(self):
        judged = judgments({"a": 3, "b": 1, "c": 0}, relevance_threshold=1.0, label="q")
        assert judged.grades == {"a": 3.0, "b": 1.0, "c": 0.0}
        assert judged.relevant == frozenset({"a", "b"})

    def test_threshold_decides_relevance(self):
        judged = judgments({"a": 3, "b": 1}, relevance_threshold=2.0, label="q")
        assert judged.relevant == frozenset({"a"})

    def test_collection_is_binary(self):
        judged = judgments({"a", "b"}, relevance_threshold=1.0, label="q")
        assert judged.grades == {"a": 1.0, "b": 1.0}

    def test_numpy_keys_are_one_identifier(self):
        judged = judgments({np.int64(4): 1}, relevance_threshold=1.0, label="q")
        assert list(judged.grades) == [4]

    @pytest.mark.parametrize(
        ("row", "error", "match"),
        [
            ("abc", TypeError, "not read as a collection of characters"),
            (b"abc", TypeError, "not read as a collection"),
            (5, TypeError, "must be a mapping or a collection"),
            ({"a": True}, TypeError, "booleans as 0 and 1"),
            ({"a": "1"}, TypeError, "real number"),
            ({"a": -1}, ValueError, ">= 0"),
            ({"a": math.nan}, ValueError, "finite"),
            (["a", "a"], ValueError, "listed as relevant twice"),
        ],
    )
    def test_malformed_rows(self, row, error, match):
        with pytest.raises(error, match=match):
            judgments(row, relevance_threshold=1.0, label="query 'q7'")

    def test_message_names_the_query(self):
        with pytest.raises(TypeError, match="query 'q7'"):
            judgments("abc", relevance_threshold=1.0, label="query 'q7'")


class TestRanking:
    def test_sequence(self):
        assert ranking(["a", "b"], duplicates="raise", label="q") == ("a", "b")

    def test_duplicates(self):
        with pytest.raises(ValueError, match="duplicates='first'"):
            ranking(["a", "b", "a"], duplicates="raise", label="q")
        assert ranking(["a", "b", "a", "c"], duplicates="first", label="q") == ("a", "b", "c")

    @pytest.mark.parametrize("row", ["ab", b"ab", {"a": 1}])
    def test_refused_shapes(self, row):
        with pytest.raises(TypeError, match="sequence of identifiers"):
            ranking(row, duplicates="raise", label="q")

    def test_not_iterable(self):
        with pytest.raises(TypeError, match="sequence of identifiers"):
            ranking(3, duplicates="raise", label="q")


class TestAligned:
    def test_sequences(self):
        true_rows, pred_rows, keys = aligned([["a"]], [["a"]])
        assert true_rows == [["a"]]
        assert pred_rows == [["a"]]
        assert keys is None

    def test_mappings_follow_y_true_order(self):
        _, pred_rows, keys = aligned({"q2": ["b"], "q1": ["a"]}, {"q1": ["x"], "q2": ["y"]})
        assert keys == ["q2", "q1"]
        assert pred_rows == [["y"], ["x"]]

    def test_mapping_key_mismatch(self):
        with pytest.raises(ValueError, match=r"missing from y_pred: \['q2'\].*unknown in y_pred: \['q3'\]"):
            aligned({"q1": [], "q2": []}, {"q1": [], "q3": []})

    def test_mixed_forms(self):
        with pytest.raises(TypeError, match="both be mappings"):
            aligned({"q1": []}, [[]])

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="aligned by position"):
            aligned([["a"], ["b"]], [["a"]])

    def test_empty(self):
        with pytest.raises(ValueError, match="At least one query"):
            aligned([], [])

    def test_str_refused(self):
        with pytest.raises(TypeError, match="one entry per query"):
            aligned("ab", ["a", "b"])


def test_identifier_types_are_not_mixed():
    judged = [judgments({"1": 1}, relevance_threshold=1.0, label="q")]
    check_id_types(judged, [("1", "2")])
    with pytest.raises(ValueError, match=r"mix str and int \(for example '1' and 2\)"):
        check_id_types(judged, [(2,)])


class TestAggregate:
    def _agg(self, values, **options):
        settings = {
            "zero_division": "warn",
            "sample_weight": None,
            "average": "mean",
            "metric": "recall@1",
            "reason": "no relevant document",
        }
        settings.update(options)
        return aggregate(np.array(values, dtype=np.float64), **settings)

    def test_warn_scores_zero_and_counts(self):
        with pytest.warns(UndefinedMetricWarning, match=r"undefined for 1 of 2 queries \(no relevant document\)"):
            assert self._agg([1.0, np.nan]) == 0.5

    @pytest.mark.parametrize(("policy", "expected"), [(0.0, 0.5), (1.0, 1.0)])
    def test_explicit_value_is_silent(self, policy, expected):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert self._agg([1.0, np.nan], zero_division=policy) == expected

    def test_nan_is_excluded(self):
        assert self._agg([1.0, np.nan, 0.0], zero_division=np.nan) == 0.5
        assert math.isnan(self._agg([np.nan], zero_division=np.nan))

    def test_per_query_values(self):
        out = self._agg([1.0, np.nan], zero_division=np.nan, average=None)
        assert out[0] == 1.0
        assert math.isnan(out[1])

    def test_weights(self):
        assert self._agg([1.0, 0.0], sample_weight=np.array([3.0, 1.0])) == 0.75
        with pytest.raises(ValueError, match="sum to zero"):
            self._agg([1.0, np.nan], zero_division=np.nan, sample_weight=np.array([0.0, 1.0]))

    def test_defined_values_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert self._agg([1.0, 0.0]) == 0.5


class TestRankedIds:
    def test_identifiers_and_pairs(self):
        assert ranked_ids(["d1", 2]) == ["d1", 2]
        assert ranked_ids([("d2", 0.9), ("d1", 0.1)]) == ["d2", "d1"]

    def test_retrieval_hits_and_responses(self):
        docs = [CorpusDocument(doc_id=f"d{i}", input_path="memory://x", chunk_index=i, text="t") for i in range(2)]
        hits = [RetrievalHit(doc=doc, score=1.0, match_mode="keyword") for doc in docs]
        assert ranked_ids(hits) == ["d0", "d1"]
        response = RetrievalResponse(hits=hits, legs=[LegOutcome(LegKind.LEXICAL, LegStatus.SUCCESS)])
        assert ranked_ids(response) == ["d0", "d1"]

    def test_documents_by_doc_id(self):
        doc = CorpusDocument(doc_id="z", input_path="memory://z", chunk_index=0, text="t")
        assert ranked_ids([doc]) == ["z"]

    def test_id_getter_overrides(self):
        assert ranked_ids([{"id": 7}], id_getter=lambda r: r["id"]) == [7]

    def test_refusals(self):
        with pytest.raises(TypeError, match="pass id_getter"):
            ranked_ids([object()])
        with pytest.raises(TypeError, match="iterable of retrieval results"):
            ranked_ids("d1")
        with pytest.raises(TypeError, match="iterable of retrieval results"):
            ranked_ids({"d1": 1})
        with pytest.raises(TypeError, match="id_getter must be callable"):
            ranked_ids([], id_getter=3)
        with pytest.raises(TypeError, match="identifiers must be str or int"):
            ranked_ids([{"id": 1.5}], id_getter=lambda r: r["id"])


def test_response_status():
    ok = RetrievalResponse(hits=[], legs=[LegOutcome(LegKind.LEXICAL, LegStatus.EMPTY)])
    assert response_status(ok) == RetrievalStatus.EMPTY.value
    failed = RetrievalResponse(
        hits=[],
        legs=[
            LegOutcome(
                LegKind.DENSE,
                LegStatus.FAILED,
                error=ErrorRecord(code="X", category=ErrorCategory.CAPABILITY, message="no embedding"),
            )
        ],
    )
    assert response_status(failed) == "failed"
    assert response_status([1, 2]) is None



class SeriesLike:
    """The shape of a pandas Series: iterates values, maps index -> value."""

    def __init__(self, data):
        self._data = dict(data)

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self):
        return len(self._data)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()


class TestMappingLikes:
    """Review F1: a Series of judgments was read as a collection of its grades."""

    def test_judgments_row_is_refused(self):
        with pytest.raises(TypeError, match=r"ambiguous as relevance judgments.*\.to_dict\(\)"):
            judgments(SeriesLike({7: 1}), relevance_threshold=1.0, label="q")

    def test_ranking_row_is_refused(self):
        with pytest.raises(TypeError, match=r"ambiguous as a ranking.*\.tolist\(\)"):
            ranking(SeriesLike({0: "d1"}), duplicates="raise", label="q")

    def test_top_level_indexes_must_agree(self):
        y_true = SeriesLike({"q1": ["a"], "q2": ["b"]})
        aligned(y_true, SeriesLike({"q1": ["a"], "q2": ["b"]}))
        with pytest.raises(ValueError, match="indexed differently"):
            aligned(y_true, SeriesLike({"q2": ["b"], "q1": ["a"]}))

    def test_one_indexed_side_is_positional(self):
        true_rows, _, keys = aligned(SeriesLike({"q1": ["a"]}), [["a"]])
        assert true_rows == [["a"]]
        assert keys is None

    def test_pandas(self):
        pd = pytest.importorskip("pandas")
        qrels = pd.DataFrame({"qid": ["q1", "q2"], "pid": [7, 9], "rel": [1, 1]})
        by_query = {q: g.set_index("pid")["rel"] for q, g in qrels.groupby("qid")}
        from .._ranking import recall_at_k

        with pytest.raises(TypeError, match="ambiguous"):
            recall_at_k(by_query, {"q1": [7], "q2": [9]})
        as_dicts = {q: s.to_dict() for q, s in by_query.items()}
        assert recall_at_k(as_dicts, {"q1": [7], "q2": [9]}) == 1.0
