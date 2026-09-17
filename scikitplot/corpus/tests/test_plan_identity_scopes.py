"""
Identity-scope regressions for :class:`CorpusPlan` (slice S-7).

One fingerprint covered every configured domain, so changing a retrieval budget
moved the value that names a built index. A query-time change cannot invalidate
an ingest artifact, and the only way to say so is to give the two scopes
separate identities.

See Also
--------
scikitplot.corpus._plan.CorpusPlan.ingest_fingerprint
scikitplot.corpus._plan.CorpusPlan.query_fingerprint
"""

import dataclasses

import pytest

from .._plan import INGEST_DOMAINS, QUERY_DOMAINS, CorpusPlan


@dataclasses.dataclass(frozen=True)
class Cfg:
    """Frozen configuration fragment."""

    value: object


def plan(**fragments):
    """Return a plan built from keyword fragments."""
    return CorpusPlan.of(**{k: Cfg(v) for k, v in fragments.items()})


BASE = dict(reader="files", index="bruteforce", retrieval={"top_k": 5})


def test_the_two_scopes_partition_the_domains():
    """Every configurable domain belongs to exactly one scope."""
    from .._plan import CONFIG_DOMAINS

    assert set(INGEST_DOMAINS) | set(QUERY_DOMAINS) == set(CONFIG_DOMAINS)
    assert not set(INGEST_DOMAINS) & set(QUERY_DOMAINS)


def test_a_query_only_change_leaves_the_ingest_identity_alone():
    """Changing a retrieval budget cannot invalidate a built index."""
    narrow = plan(**BASE)
    wide = plan(**{**BASE, "retrieval": {"top_k": 50}})
    assert narrow.ingest_fingerprint == wide.ingest_fingerprint
    assert narrow.query_fingerprint != wide.query_fingerprint


def test_an_ingest_change_moves_the_ingest_identity():
    """A change that does invalidate a build is visible in the ingest scope."""
    before = plan(**BASE)
    after = plan(**{**BASE, "chunker": "sentences"})
    assert before.ingest_fingerprint != after.ingest_fingerprint
    assert before.query_fingerprint == after.query_fingerprint


def test_the_whole_plan_identity_still_covers_both():
    """The combined fingerprint remains, and moves for either kind of change."""
    base = plan(**BASE)
    assert base.fingerprint != plan(**{**BASE, "retrieval": {"top_k": 50}}).fingerprint
    assert base.fingerprint != plan(**{**BASE, "chunker": "sentences"}).fingerprint


def test_scopes_are_derived_from_the_same_encoder():
    """Both scopes are full digests from the canonical encoder, like the whole."""
    base = plan(**BASE)
    for value in (base.ingest_fingerprint, base.query_fingerprint, base.fingerprint):
        assert len(value) == 64


def test_the_three_identities_are_distinct():
    """A scope digest is not the whole-plan digest under another name."""
    base = plan(**BASE)
    assert len({base.ingest_fingerprint, base.query_fingerprint, base.fingerprint}) == 3


def test_an_unconfigured_scope_still_has_an_identity():
    """A plan with no query configuration has a query identity, not an error."""
    ingest_only = plan(reader="files", index="bruteforce")
    assert len(ingest_only.query_fingerprint) == 64
    assert ingest_only.ingest_fingerprint == plan(
        **{"reader": "files", "index": "bruteforce", "retrieval": {"top_k": 9}}
    ).ingest_fingerprint


def test_stage_order_belongs_to_the_ingest_scope():
    """Pipeline stage order changes what is built, not how it is queried."""
    fragments = {"reader": Cfg("files")}
    default = CorpusPlan(fragments=fragments)
    reordered = CorpusPlan(
        fragments=fragments,
        stages=("read", "chunk", "normalize", "enrich", "embed", "store", "retrieve"),
    )
    assert default.ingest_fingerprint != reordered.ingest_fingerprint
    assert default.query_fingerprint == reordered.query_fingerprint


@pytest.mark.parametrize("scope", ["ingest_fingerprint", "query_fingerprint"])
def test_scope_identities_are_order_independent(scope):
    """Scope identities inherit the encoder's order independence."""
    one = CorpusPlan.of(reader=Cfg("files"), index=Cfg("bruteforce"))
    two = CorpusPlan.of(index=Cfg("bruteforce"), reader=Cfg("files"))
    assert getattr(one, scope) == getattr(two, scope)
