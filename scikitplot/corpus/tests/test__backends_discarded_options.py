"""
Backend option-discard regressions (slice S-43, reworked by R2-5 / F-40).

A requested ``dtype`` or ``index_dtype`` that the running Annoy binding cannot
honour used to be dropped after a trial construction raised ``TypeError``,
recorded at debug level. Each binding now declares the options it accepts,
``impl="auto"`` picks one that accepts every requested option, and an option
the running binding cannot honour is a reported fallback
(``fallback_reason="option_unsupported"`` plus :attr:`discarded_options` and
one warning), or an error when ``allow_fallback=False``.

See Also
--------
scikitplot.corpus._similarity._backends.AnnoyBackend.discarded_options
"""

import numpy as np
import pytest

from .._similarity import _backends as B

logger = B.logger  # logging.getLogger(B.__name__)

_EMBEDDINGS = np.eye(3, dtype=np.float32)


class _Index:
    """A fake binding that records its constructor options."""

    def __init__(self, dim, metric, **kwargs):
        self.kwargs = kwargs
        self.items = []

    def add_item(self, i, vector):
        self.items.append((i, vector))

    def build(self, n_trees):
        self.n_trees = n_trees


@pytest.fixture
def bindings(monkeypatch):
    """Serve fake classes for both bindings; return the import log."""
    log = []

    def fake_import(impl):
        log.append(impl)
        return type(f"Fake_{impl}", (_Index,), {})

    monkeypatch.setattr(B, "_import_annoy_impl", fake_import)
    return log


def _build(**kwargs):
    backend = B.AnnoyBackend(n_trees=1, **kwargs)
    backend.build(_EMBEDDINGS)
    return backend


def test_nothing_is_reported_when_everything_is_honoured(bindings):
    """The report states a fact, so it is empty when nothing was dropped."""
    backend = _build(impl="cython", dtype="float32")
    assert backend._index.kwargs == {"dtype": "float32"}
    assert backend.discarded_options == {}
    assert backend.fallback_reason is None


def test_a_discarded_dtype_is_reported(bindings):
    """The caller can see that the index is not built the way they asked."""
    backend = _build(impl="highlevel", dtype="float32")
    assert backend._index.kwargs == {}
    assert backend.discarded_options == {"dtype": "float32"}
    assert backend.fallback_reason == "option_unsupported"


def test_both_options_are_reported(bindings):
    """Whatever was dropped is named, not just that something was."""
    backend = _build(impl="highlevel", dtype="float32", index_dtype="int64")
    assert backend.discarded_options == {"dtype": "float32", "index_dtype": "int64"}


def test_the_discard_is_warned_not_debugged(bindings, caplog):
    """A downgrade nobody is told about is indistinguishable from none."""
    with caplog.at_level("WARNING", logger=logger.name):
        logger.addHandler(caplog.handler)
        try:
            _build(impl="highlevel", dtype="float32")
        finally:
            logger.removeHandler(caplog.handler)
    assert any("dtype" in record.getMessage() for record in caplog.records)


def test_a_backend_with_no_options_requested_reports_nothing(bindings):
    """No request, no downgrade, no noise."""
    backend = _build(impl="highlevel")
    assert backend.discarded_options == {}
    assert backend.fallback_reason is None


def test_the_report_is_a_copy(bindings):
    """A caller cannot mutate the backend's record of what it dropped."""
    backend = _build(impl="highlevel", dtype="float32")
    backend.discarded_options["dtype"] = "tampered"
    assert backend.discarded_options == {"dtype": "float32"}


def test_strict_refuses_instead_of_discarding(bindings):
    """``allow_fallback=False`` turns the discard into an error."""
    backend = B.AnnoyBackend(impl="highlevel", dtype="float32", allow_fallback=False)
    with pytest.raises(B.FallbackRefusedError, match="does not accept"):
        backend.build(_EMBEDDINGS)


def test_auto_picks_the_binding_that_honours_the_options(bindings):
    """F-40: ``auto`` no longer picks a binding that would drop the dtype."""
    backend = _build(dtype="float64")
    assert backend.provenance()["impl"] == "cython"
    assert backend._index.kwargs == {"dtype": "float64"}
    assert backend.discarded_options == {}
    assert bindings == ["cython"]
