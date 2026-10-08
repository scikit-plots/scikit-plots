# scikitplot/annoy/_annoy/tests/test_sklearn_tags.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for CY-011 (guide 35).

``Index.__sklearn_tags__`` delegated to ``super().__sklearn_tags__()``, but the
class is estimator-like without being a ``BaseEstimator`` subclass, so the parent
chain had no such method and the call raised
``AttributeError: 'super' object has no attribute '__sklearn_tags__'``. It now
delegates to sklearn's root tag builder, which constructs valid default ``Tags``
for the installed sklearn version.
"""
import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn.base import BaseEstimator

from scikitplot.annoy._annoy import annoylib as A

# ``__sklearn_tags__`` and ``sklearn.utils.get_tags`` arrived in scikit-learn
# 1.6.0. The project supports scikit-learn from 1.3 on; an older release has
# no tag builder to delegate to and never asks an estimator for its tags, so
# the tag tests below state a contract that exists from 1.6 on only. The
# condition asks the installed scikit-learn for the feature itself, not for
# its version number.
HAS_SKLEARN_TAGS = hasattr(BaseEstimator, "__sklearn_tags__")
needs_sklearn_tags = pytest.mark.skipif(
    not HAS_SKLEARN_TAGS,
    reason="scikit-learn < 1.6 has no __sklearn_tags__ (the tags API is newer)",
)


def _index():
    return A.Index(4, "euclidean")


@needs_sklearn_tags
def test_sklearn_tags_does_not_raise_and_returns_tags():
    idx = _index()
    tags = idx.__sklearn_tags__()          # previously raised AttributeError
    assert tags is not None
    assert type(tags).__name__ == "Tags"


@needs_sklearn_tags
def test_sklearn_public_get_tags_accessor_works():
    # this is the accessor sklearn machinery uses internally
    from sklearn.utils import get_tags
    tags = get_tags(_index())
    assert type(tags).__name__ == "Tags"


def test_clone_still_works():
    from sklearn.base import clone
    idx = _index()
    idx.set_params(n_neighbors=7)
    cloned = clone(idx)
    assert type(cloned).__name__ == "Index"
    # clone copies params (via __init__), not fitted data
    assert cloned.get_params()["n_neighbors"] == 7


@needs_sklearn_tags
def test_tags_reflect_a_non_classifier_estimator():
    # sensible defaults for a neighbors-style index: not a classifier/regressor
    tags = _index().__sklearn_tags__()
    assert getattr(tags, "estimator_type", None) is None


@pytest.mark.skipif(HAS_SKLEARN_TAGS, reason="scikit-learn >= 1.6 has the tags API")
def test_without_the_tags_api_the_index_still_works_as_an_estimator():
    # On scikit-learn 1.3 to 1.5 nothing calls ``__sklearn_tags__``; what
    # scikit-learn does use there (get_params / set_params / clone) must work.
    from sklearn.base import clone

    idx = _index()
    idx.set_params(n_neighbors=3)
    assert clone(idx).get_params()["n_neighbors"] == 3
    with pytest.raises(AttributeError, match="__sklearn_tags__"):
        idx.__sklearn_tags__()
