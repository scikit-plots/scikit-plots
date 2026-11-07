# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The metadata-routing behaviour that the user guide documents for the pipeline.

``docs/source/user_guide/metadata_routing.rst`` tabulates five calls of the
experimental :class:`~scikitplot.experimental.pipeline.Pipeline`, with routing
off and on; each case here is one row of that table.

Notes
-----
**Developer notes.** The pipeline inherits scikit-learn's
``get_metadata_routing``, so these tests pin scikit-learn's contract as seen
through the subclass. Routing-on cases need scikit-learn 1.4 or later (the
version from which the pipeline routes ``transform`` parameters as well).
"""

from __future__ import annotations

import pytest

sklearn = pytest.importorskip("sklearn")
np = pytest.importorskip("numpy")

from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402
from sklearn.exceptions import UnsetMetadataPassedError  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from .. import Pipeline  # noqa: E402

_ROUTING_ON = pytest.mark.skipif(
    tuple(int(p) for p in sklearn.__version__.split(".")[:2]) < (1, 4),
    reason="the pipeline routes transform parameters from scikit-learn 1.4",
)


class _Recorder(ClassifierMixin, BaseEstimator):
    """A classifier that records the ``sample_weight`` its ``fit`` received."""

    def fit(self, X, y, sample_weight=None):
        self.received_ = sample_weight
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    return rng.rand(12, 3), np.arange(12) % 2, np.arange(1.0, 13.0)


def test_the_pipeline_uses_scikit_learn_routing():
    assert (
        Pipeline.get_metadata_routing is sklearn.pipeline.Pipeline.get_metadata_routing
    )


def test_off_the_step_prefix_reaches_the_step(data):
    X, y, w = data
    pipe = Pipeline([("scale", StandardScaler()), ("clf", _Recorder())]).fit(
        X, y, clf__sample_weight=w
    )
    np.testing.assert_array_equal(pipe.named_steps["clf"].received_, w)


def test_off_an_unprefixed_parameter_is_refused(data):
    X, y, w = data
    pipe = Pipeline([("scale", StandardScaler()), ("clf", _Recorder())])
    with pytest.raises(ValueError, match="sample_weight"):
        pipe.fit(X, y, sample_weight=w)


@_ROUTING_ON
def test_on_only_the_requesting_step_receives_it(data):
    X, y, w = data
    with sklearn.config_context(enable_metadata_routing=True):
        pipe = Pipeline(
            [
                ("scale", StandardScaler().set_fit_request(sample_weight=False)),
                ("clf", _Recorder().set_fit_request(sample_weight=True)),
            ]
        ).fit(X, y, sample_weight=w)
    np.testing.assert_array_equal(pipe.named_steps["clf"].received_, w)


@_ROUTING_ON
def test_on_a_step_without_a_request_is_named(data):
    X, y, w = data
    with sklearn.config_context(enable_metadata_routing=True):
        pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", _Recorder().set_fit_request(sample_weight=True)),
            ]
        )
        with pytest.raises(UnsetMetadataPassedError, match="StandardScaler"):
            pipe.fit(X, y, sample_weight=w)


@_ROUTING_ON
def test_on_the_step_prefix_is_refused(data):
    X, y, w = data
    with sklearn.config_context(enable_metadata_routing=True):
        pipe = Pipeline([("scale", StandardScaler()), ("clf", _Recorder())])
        with pytest.raises(TypeError, match="clf__sample_weight"):
            pipe.fit(X, y, clf__sample_weight=w)
