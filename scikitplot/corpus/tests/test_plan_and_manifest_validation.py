"""
Validation regressions for plans and embedding manifests (slice S-8).

Two checks answered a question next to the one that matters. A plan was refused
whenever an index was configured without an embedder, whether or not the index
needs vectors, so a purely lexical configuration could not be expressed. And two
manifests with no resolved revision compared compatible, so "the same model" was
reported as verified when nothing had been verified.

See Also
--------
scikitplot.corpus._plan.CorpusPlan.validate
scikitplot.corpus._embedding_manifest.EmbeddingManifest.is_compatible
"""

import dataclasses

import pytest

from .._embedding_manifest import EmbeddingManifest, IncompatibleEmbeddingsError
from .._plan import CorpusPlan


@dataclasses.dataclass(frozen=True)
class Index:
    """Index fragment that states what it needs."""

    kind: str
    requires_vectors: bool = True


@dataclasses.dataclass(frozen=True)
class Cfg:
    """Plain fragment."""

    value: object


def codes(plan):
    """Return the validation error codes a plan reports."""
    return [record.code for record in plan.validate()]


# -- D09: validation checks the requirement, not the fragment --------------


def test_a_lexical_index_without_an_embedder_is_valid():
    """A lexical index needs no vectors, so it needs no embedder."""
    plan = CorpusPlan.of(reader=Cfg("files"),
                         index=Index("bm25", requires_vectors=False))
    assert "PLAN_INDEX_WITHOUT_EMBEDDER" not in codes(plan)


def test_a_vector_index_without_an_embedder_is_still_refused():
    """The original defect stays closed: vectors with nothing to produce them."""
    plan = CorpusPlan.of(reader=Cfg("files"),
                         index=Index("annoy", requires_vectors=True))
    assert "PLAN_INDEX_WITHOUT_EMBEDDER" in codes(plan)


def test_an_index_that_does_not_state_its_needs_is_treated_as_needing_vectors():
    """Silence keeps the safe answer; an unknown fragment is not assumed lexical."""
    plan = CorpusPlan.of(reader=Cfg("files"), index=Cfg("mystery"))
    assert "PLAN_INDEX_WITHOUT_EMBEDDER" in codes(plan)


def test_an_embedder_satisfies_a_vector_index():
    """The ordinary configuration is unchanged."""
    plan = CorpusPlan.of(reader=Cfg("files"), index=Index("annoy"),
                         embedder=Cfg("st"))
    assert "PLAN_INDEX_WITHOUT_EMBEDDER" not in codes(plan)


# -- D10: unpinned compatibility is explicit ------------------------------


def unpinned(model="m"):
    """Return a manifest with no resolved revision."""
    return EmbeddingManifest(provider="p", model=model, dimension=3)


def pinned(revision="r1", model="m"):
    """Return a manifest with a resolved revision."""
    return EmbeddingManifest(provider="p", model=model, dimension=3,
                             revision=revision)


def test_two_unpinned_manifests_are_not_reported_as_compatible():
    """Matching names are not evidence that the same weights produced both."""
    assert unpinned().is_compatible(unpinned()) is False


def test_unpinned_compatibility_can_be_opted_into():
    """A caller may accept the unverified space, but must say so."""
    assert unpinned().is_compatible(unpinned(), assume_unpinned_match=True) is True


def test_two_pinned_matching_manifests_are_compatible():
    """A resolved revision is what verified compatibility means."""
    assert pinned().is_compatible(pinned()) is True


def test_pinned_manifests_with_different_revisions_are_incompatible():
    """The ordinary negative case is unchanged."""
    assert pinned("r1").is_compatible(pinned("r2")) is False


def test_a_pinned_and_an_unpinned_manifest_are_not_compatible():
    """One side resolved and one side not is not a verified match either."""
    assert pinned().is_compatible(unpinned()) is False
    assert unpinned().is_compatible(pinned()) is False


def test_the_refusal_explains_that_the_revision_is_unresolved():
    """The message distinguishes 'unverified' from 'known different'."""
    with pytest.raises(IncompatibleEmbeddingsError) as excinfo:
        unpinned().require_compatible(unpinned())
    assert "revision" in str(excinfo.value).lower()


def test_require_compatible_accepts_the_same_opt_in():
    """The strict path offers the same explicit escape, not a different one."""
    unpinned().require_compatible(unpinned(), assume_unpinned_match=True)


def test_different_models_are_incompatible_however_they_are_asked():
    """The opt-in covers an unresolved revision, not a different model."""
    assert unpinned("a").is_compatible(unpinned("b"),
                                       assume_unpinned_match=True) is False
