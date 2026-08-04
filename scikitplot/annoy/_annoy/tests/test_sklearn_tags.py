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

from scikitplot.annoy._annoy import annoylib as A


def _index():
    return A.Index(4, "euclidean")


def test_sklearn_tags_does_not_raise_and_returns_tags():
    idx = _index()
    tags = idx.__sklearn_tags__()          # previously raised AttributeError
    assert tags is not None
    assert type(tags).__name__ == "Tags"


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


def test_tags_reflect_a_non_classifier_estimator():
    # sensible defaults for a neighbors-style index: not a classifier/regressor
    tags = _index().__sklearn_tags__()
    assert getattr(tags, "estimator_type", None) is None
