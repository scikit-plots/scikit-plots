"""
Backend option-discard regressions (slice S-43).

``_construct`` forwards ``dtype`` and ``index_dtype`` and, when the index class
refuses them, rebuilds without them. That fallback is right -- not every Annoy
build accepts those -- but it was recorded at debug level, so a caller who asked
for a precision received a different one with nothing to show for it.

See Also
--------
scikitplot.corpus._similarity._backends.AnnoyBackend.discarded_options
"""

import pytest

from .._similarity import _backends as B


class AcceptsEverything:
    def __init__(self, dim, metric, **kwargs):
        self.kwargs = kwargs


class RefusesKwargs:
    def __init__(self, dim, metric, **kwargs):
        if kwargs:
            raise TypeError("unexpected keyword")


def _shell(**attrs):
    backend = object.__new__(B.AnnoyBackend)
    backend._dim, backend._metric, backend._search_k = 8, "angular", 10
    backend._dtype = attrs.get("dtype")
    backend._index_dtype = attrs.get("index_dtype")
    backend._resolved_impl = "probe"
    return backend


def test_nothing_is_reported_when_everything_is_honoured():
    """The report states a fact, so it is empty when nothing was dropped."""
    backend = _shell(dtype="float32")
    index = backend._construct(AcceptsEverything)
    assert index.kwargs == {"dtype": "float32"}
    assert backend.discarded_options == {}


def test_a_discarded_dtype_is_reported():
    """The caller can see that the index is not built the way they asked."""
    backend = _shell(dtype="float32")
    backend._construct(RefusesKwargs)
    assert backend.discarded_options == {"dtype": "float32"}


def test_both_options_are_reported():
    """Whatever was dropped is named, not just that something was."""
    backend = _shell(dtype="float32", index_dtype="float16")
    backend._construct(RefusesKwargs)
    assert backend.discarded_options == {"dtype": "float32", "index_dtype": "float16"}


def test_the_discard_is_warned_not_debugged(caplog):
    """A downgrade nobody is told about is indistinguishable from none."""
    backend = _shell(dtype="float32")
    with caplog.at_level("WARNING"):
        backend._construct(RefusesKwargs)
    assert any("dtype" in record.getMessage() for record in caplog.records)


def test_a_backend_with_no_options_requested_reports_nothing():
    """No request, no downgrade, no noise."""
    backend = _shell()
    backend._construct(RefusesKwargs)
    assert backend.discarded_options == {}


def test_the_report_is_a_copy():
    """A caller cannot mutate the backend's record of what it dropped."""
    backend = _shell(dtype="float32")
    backend._construct(RefusesKwargs)
    backend.discarded_options["dtype"] = "tampered"
    assert backend.discarded_options == {"dtype": "float32"}
