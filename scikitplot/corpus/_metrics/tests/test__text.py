# scikitplot/corpus/_metrics/tests/test__text.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Reference-based answer metrics and the two edit-distance paths."""

from __future__ import annotations

import collections
import random
import re
import string

import numpy as np
import pytest

from .._text import (
    _python_distance,
    anls_score,
    edit_similarity,
    edit_similarity_score,
    exact_match_score,
    normalize_answer,
    token_f1_score,
)


# The SQuAD v1.1 evaluation normalisation and F1, written as that script does,
# so "normalize='squad'" is held to the published definition.
def _squad_normalize(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _squad_f1(prediction, ground_truth):
    prediction_tokens = _squad_normalize(prediction).split()
    ground_truth_tokens = _squad_normalize(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def _full_matrix_distance(a, b):
    """Levenshtein distance by the full (len(a)+1) x (len(b)+1) table."""
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        table[i][0] = i
    for j in range(len(b) + 1):
        table[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1, table[i - 1][j - 1] + cost)
    return table[-1][-1]


_ALPHABET = "abcAB \u00e9\u0301\U0001F600\u65e5\u672c-., "


def _random_strings(rng, n, max_len=9):
    return ["".join(rng.choice(_ALPHABET) for _ in range(rng.randint(0, max_len))) for _ in range(n)]


class TestNormalize:
    @pytest.mark.parametrize(
        "text",
        ["The  Eiffel Tower!", "an apple, a day", "Theatre", "A.B.C.", "  ", "the", "Paris (France)"],
    )
    def test_squad_mode_is_the_published_normalisation(self, text):
        assert normalize_answer(text) == _squad_normalize(text)

    def test_basic_mode(self):
        assert normalize_answer("Stra\u00dfe\u00a0\u2163", mode="basic") == "strasse iv"

    def test_none_is_identity(self):
        assert normalize_answer(" A ", mode=None) == " A "

    def test_refusals(self):
        with pytest.raises(TypeError):
            normalize_answer(3)
        with pytest.raises(ValueError, match="normalize must be one of"):
            normalize_answer("a", mode="nfkc")


class TestExactMatch:
    def test_multiple_references(self):
        refs = ["The Eiffel Tower", ["Paris", "City of Paris"]]
        assert exact_match_score(refs, ["eiffel tower.", "city of paris"]) == 1.0
        np.testing.assert_array_equal(exact_match_score(refs, ["x", "Paris"], average=None), [0.0, 1.0])

    def test_without_normalisation(self):
        assert exact_match_score(["Paris"], ["paris"], normalize=None) == 0.0

    def test_mapping_inputs(self):
        assert exact_match_score({"q1": "a", "q2": "b"}, {"q2": "b", "q1": "x"}) == 0.5


class TestTokenF1:
    def test_matches_the_squad_definition(self):
        rng = random.Random(5)  # noqa: S311 - seeded test data
        words = ["the", "cat", "sat", "on", "a", "mat", "Mat,", "dog", "an", "!"]
        for _ in range(300):
            ref = " ".join(rng.choice(words) for _ in range(rng.randint(1, 6)))
            pred = " ".join(rng.choice(words) for _ in range(rng.randint(1, 6)))
            expected = _squad_f1(pred, ref)
            if not _squad_normalize(ref).split() or not _squad_normalize(pred).split():
                expected = float(_squad_normalize(ref).split() == _squad_normalize(pred).split())
            assert token_f1_score([ref], [pred]) == pytest.approx(expected, abs=1e-15), (ref, pred)

    def test_empty_token_lists(self):
        assert token_f1_score(["the"], ["a"]) == 1.0  # both empty after normalisation
        assert token_f1_score(["the"], ["cat"]) == 0.0

    def test_custom_tokenizer(self):
        assert token_f1_score(["日本語"], ["日本"], normalize=None, tokenizer=list) == pytest.approx(0.8)
        with pytest.raises(TypeError, match="sequence of str"):
            token_f1_score(["a"], ["a"], tokenizer=lambda s: [1])
        with pytest.raises(TypeError, match="callable"):
            token_f1_score(["a"], ["a"], tokenizer="split")


class TestEditSimilarity:
    def test_classic_values(self):
        assert edit_similarity("kitten", "sitting") == 1 - 3 / 7
        assert edit_similarity("", "") == 1.0
        assert edit_similarity("abc", "") == 0.0
        assert edit_similarity("Paris", "paris", normalize="basic") == 1.0

    def test_python_distance_equals_the_full_table(self):
        rng = random.Random(17)  # noqa: S311 - seeded test data
        for a, b in zip(_random_strings(rng, 400), _random_strings(rng, 400)):
            assert _python_distance(a, b) == _full_matrix_distance(a, b), (a, b)

    def test_both_backends_agree(self):
        pytest.importorskip("scikitplot.levenshtein")
        rng = random.Random(23)  # noqa: S311 - seeded test data
        for a, b in zip(_random_strings(rng, 400), _random_strings(rng, 400)):
            assert edit_similarity(a, b, backend="auto") == edit_similarity(a, b, backend="python"), (a, b)

    def test_backend_is_checked(self):
        with pytest.raises(ValueError, match="backend"):
            edit_similarity("a", "b", backend="rapidfuzz")

    def test_auto_falls_back_without_levenshtein(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "scikitplot.levenshtein", None)
        assert edit_similarity("kitten", "sitting", backend="auto") == 1 - 3 / 7

    def test_score_function(self):
        assert edit_similarity_score(["kitten", "abc"], ["sitting", "abc"]) == pytest.approx((4 / 7 + 1) / 2)


class TestANLS:
    @pytest.mark.parametrize(
        ("ref", "pred", "expected"),
        [
            ("abc", "abc", 1.0),
            ("abc", "abx", 1 - 1 / 3),  # s = 1 - NL, NL = 1/3 < 0.5
            ("ab", "ax", 0.0),  # NL = 0.5 is not below the threshold
            ("kitten", "dog", 0.0),
            ("", "", 1.0),
        ],
    )
    def test_threshold_boundary(self, ref, pred, expected):
        assert anls_score([ref], [pred], normalize=None) == pytest.approx(expected, abs=0)

    def test_threshold_one_keeps_partial_credit(self):
        assert anls_score(["ab"], ["ax"], threshold=1.0, normalize=None) == 0.5

    def test_best_reference_counts(self):
        assert anls_score([["dog", "kitten"]], ["kitten"]) == 1.0

    @pytest.mark.parametrize(("threshold", "error"), [(0, ValueError), (1.5, ValueError), (np.nan, ValueError), (True, TypeError)])
    def test_threshold_is_checked(self, threshold, error):
        with pytest.raises(error):
            anls_score(["a"], ["a"], threshold=threshold)


class TestReferences:
    @pytest.mark.parametrize(
        ("refs", "preds", "error", "match"),
        [
            ([[]], ["a"], ValueError, "at least one reference"),
            ([[1]], ["a"], TypeError, "reference answers must be str"),
            ([{"a": 1}], ["a"], TypeError, "a reference is a str"),
            (["a"], [None], TypeError, "predicted answer must be a str"),
        ],
    )
    def test_malformed(self, refs, preds, error, match):
        with pytest.raises(error, match=match):
            exact_match_score(refs, preds)
