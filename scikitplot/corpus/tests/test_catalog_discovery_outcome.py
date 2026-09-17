"""
Discovery-outcome regressions for the component catalog (slice S-9).

``_index_specs`` and ``_registry_specs`` returned ``[]`` from a bare
``except Exception``, so a broken registry, an absent optional dependency and a
genuinely empty category were one value. A caller could not tell "there are no
vector backends" from "the module that lists them failed to import", and the
second is a bug someone needs to see.

See Also
--------
scikitplot.corpus._catalog.DiscoveryOutcome
scikitplot.corpus._catalog.component_catalog
"""

import sys
from unittest.mock import patch

import pytest

from .._catalog import DiscoveryStatus, component_catalog


def _broken_import(*args, **kwargs):
    raise ImportError("injected: the registry module is broken")


def test_a_healthy_discovery_reports_ok():
    """The ordinary path reports what it found and that nothing failed."""
    catalog = component_catalog()
    assert catalog.discovery_failures() == []
    assert len(catalog) > 0


def test_a_broken_source_is_not_an_empty_one():
    """A failed import is recorded as a failure, not as "nothing found"."""
    with patch.dict(sys.modules, {"scikitplot.corpus._similarity._backends": None}):
        catalog = component_catalog()
    failures = catalog.discovery_failures()
    assert failures, "a broken import must be visible to the caller"
    assert any(f.status is DiscoveryStatus.BROKEN for f in failures)


def test_the_failure_names_the_source_and_the_reason():
    """A caller must be able to act on it without reading the traceback."""
    with patch.dict(sys.modules, {"scikitplot.corpus._similarity._backends": None}):
        failures = component_catalog().discovery_failures()
    failure = next(f for f in failures if f.status is DiscoveryStatus.BROKEN)
    assert failure.source
    assert failure.reason


def test_every_source_reports_its_own_outcome():
    """
    Each source is accounted for separately, so a partial failure is legible.

    Notes
    -----
    Breaking the backends module leaves the other source working, so the
    catalog reports one ``BROKEN`` and one non-broken outcome. Under the old
    single ``[]`` both collapsed into "nothing found", which is the exact
    confusion this slice removes.
    """
    with patch.dict(sys.modules, {"scikitplot.corpus._similarity._backends": None}):
        catalog = component_catalog()
    broken = catalog.discovery_failures()
    assert len(catalog.outcomes) == 2
    assert len(broken) == 1
    assert "_backends" in broken[0].source
    assert broken[0].reason
    other = next(o for o in catalog.outcomes if o is not broken[0])
    assert other.status is not DiscoveryStatus.BROKEN
    assert other.reason is None


def test_a_catalog_that_is_empty_for_a_reason_says_so():
    """An empty catalog with failures is distinguishable from a genuinely empty one."""
    with patch.dict(sys.modules, {"scikitplot.corpus._similarity._backends": None}):
        broken = component_catalog()
    healthy = component_catalog()
    assert broken.discovery_failures() and not healthy.discovery_failures()
    assert all(o.status is not DiscoveryStatus.BROKEN for o in healthy.outcomes)


def test_the_catalog_still_iterates_and_lists():
    """The existing surface is unchanged; the outcome is additional."""
    catalog = component_catalog()
    # list() sorts and __iter__ preserves collection order, so compare contents
    # rather than sequence. That difference predates this slice and is left
    # alone: it is a separate question from what a source reports.
    assert sorted(list(catalog), key=lambda s: (s.category, s.name)) == sorted(
        catalog.list(), key=lambda s: (s.category, s.name)
    )
    assert sum(len(catalog.list(c)) for c in catalog.categories()) == len(catalog)


@pytest.mark.parametrize("status", list(DiscoveryStatus))
def test_the_vocabulary_is_closed(status):
    """Every status is one of the declared ones."""
    assert status.value == status.value.lower()
    assert status in set(DiscoveryStatus)


def test_ok_and_empty_are_different_answers():
    """A source that ran and found nothing is not a source that failed."""
    assert DiscoveryStatus.OK is not DiscoveryStatus.EMPTY
    assert DiscoveryStatus.EMPTY is not DiscoveryStatus.BROKEN
