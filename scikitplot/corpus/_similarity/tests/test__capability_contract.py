# scikitplot/corpus/_similarity/tests/test__capability_contract.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for declarative backend capabilities (F-R06-01) and the threshold scale guard (F-R06-02)."""

from __future__ import annotations

import pytest

from .._backends import _BACKENDS, AnnoyBackend, VectorIndexBackend
from .._similarity import RetrievalConfig, RetrievalIndex

__all__: "list[str]" = [
    "TestDeclarativeCapabilities",
    "TestScoreSemantics",
    "TestThresholdScaleGuard",
]

#: The eleven properties guide §18 requires a backend contract to answer.
REQUIRED_CONTRACT = (
    "is_available",
    "build",
    "query",
    "metric",
    "score_semantics",
    "dimension",
    "dtype",
    "supports_persistence",
    "thread_safety",
    "memory_profile",
    "capabilities",
)


class TestDeclarativeCapabilities:
    """F-R06-01 — the contract answered 3 of 11 properties."""

    @pytest.mark.parametrize("member", REQUIRED_CONTRACT)
    def test_base_contract_declares_every_member(self, member: str) -> None:
        assert hasattr(VectorIndexBackend, member)

    @pytest.mark.parametrize("name", sorted(_BACKENDS))
    def test_every_backend_answers(self, name: str) -> None:
        """Base defaults describe current behaviour, so no subclass changed."""
        instance = _BACKENDS[name].__new__(_BACKENDS[name])
        declared = instance.capabilities()
        assert set(declared) >= {
            "metric",
            "score_semantics",
            "dimension",
            "dtype",
            "supports_persistence",
            "thread_safety",
            "memory_profile",
        }

    def test_dimension_is_none_before_build(self) -> None:
        instance = VectorIndexBackend()
        assert instance.dimension is None

    def test_persistence_declaration_matches_reality(self) -> None:
        """The declaration must track what the code can actually do.

        R06 found zero persistence surface in this module, so IMPL-12 declared
        ``False`` -- accurately at the time.  ``ANNIndexArtifact`` then supplied
        one, so the declaration became ``True``.  The point of a *declarative*
        contract is precisely that this flip is a visible, tested change rather
        than an assumption that quietly drifts.
        """
        from ..._artifact import ANNIndexArtifact

        assert hasattr(ANNIndexArtifact, "write")
        for name in _BACKENDS:
            instance = _BACKENDS[name].__new__(_BACKENDS[name])
            assert instance.supports_persistence is True


class TestScoreSemantics:
    """A backend must declare which scale it returns."""

    @pytest.mark.parametrize(
        ("metric", "expected"),
        [
            ("angular", "cosine_similarity"),
            ("cosine", "cosine_similarity"),
            ("euclidean", "bounded_inverse_distance"),
            ("manhattan", "bounded_inverse_distance"),
        ],
    )
    def test_annoy_declares_its_actual_scale(
        self, metric: str, expected: str
    ) -> None:
        backend = AnnoyBackend.__new__(AnnoyBackend)
        backend._metric = metric
        backend.metric = metric
        backend.score_semantics = (
            "cosine_similarity"
            if metric in ("angular", "cosine")
            else "bounded_inverse_distance"
        )
        assert backend.score_semantics == expected

    def test_the_two_scales_genuinely_differ(self) -> None:
        """The measurement behind F-R06-02.

        At the same distance the two branches return values on incomparable
        scales -- one can be negative, the other cannot.
        """
        cosine = AnnoyBackend.__new__(AnnoyBackend)
        cosine._metric = "angular"
        other = AnnoyBackend.__new__(AnnoyBackend)
        other._metric = "euclidean"

        assert cosine._distance_to_score(2.0) == pytest.approx(-1.0)
        assert other._distance_to_score(2.0) == pytest.approx(1 / 3, rel=1e-3)


def _docs():
    from ..._schema import CorpusDocument

    return [
        CorpusDocument.create(
            input_path=f"f{i}.txt", chunk_index=i, text=f"text {i}",
            embedding=[float(i), 1.0, 0.0],
        )
        for i in range(2)
    ]


class TestThresholdScaleGuard:
    """F-R06-02 — a threshold tuned on one metric mis-filtered on another."""

    def test_valid_cosine_threshold_is_accepted(self) -> None:
        index = RetrievalIndex()
        index.build(_docs())
        response = index.search(
            "text",
            config=RetrievalConfig(match_mode="semantic", semantic_threshold=0.5),
            query_embedding=[1.0, 1.0, 0.0],
        )
        assert response is not None

    def test_out_of_range_cosine_threshold_is_rejected(self) -> None:
        index = RetrievalIndex()
        index.build(_docs())
        with pytest.raises(ValueError, match=r"outside the cosine"):
            index.search(
                "text",
                config=RetrievalConfig(match_mode="semantic", semantic_threshold=5.0),
                query_embedding=[1.0, 1.0, 0.0],
            )

    def test_negative_threshold_rejected_on_inverse_distance_scale(self) -> None:
        """A negative threshold cannot match a score in ``(0, 1]``.

        Previously this silently admitted everything, because ``1/(1+d)`` is
        strictly positive -- so the same config behaved differently depending on
        a *different* field.
        """
        index = RetrievalIndex()
        index.build(_docs())
        index._backend.score_semantics = "bounded_inverse_distance"
        index._backend.metric = "euclidean"
        with pytest.raises(ValueError, match="bounded inverse distance"):
            index.search(
                "text",
                config=RetrievalConfig(match_mode="semantic", semantic_threshold=-0.5),
                query_embedding=[1.0, 1.0, 0.0],
            )

    def test_error_names_both_the_backend_and_the_metric(self) -> None:
        index = RetrievalIndex()
        index.build(_docs())
        index._backend.score_semantics = "bounded_inverse_distance"
        index._backend.metric = "euclidean"
        with pytest.raises(ValueError) as excinfo:
            index.search(
                "text",
                config=RetrievalConfig(match_mode="semantic", semantic_threshold=-1.0),
                query_embedding=[1.0, 1.0, 0.0],
            )
        message = str(excinfo.value)
        assert "euclidean" in message and "not portable" in message
