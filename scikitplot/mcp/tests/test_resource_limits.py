"""
Bound-reachability regressions for resource rendering (slice S-22).

The resource guard declared 20,000 characters while ``MAX_CHUNK_CHARS`` capped
chunk text at 4,000 earlier in the same path, so the guard could not fire and
the published limit overstated the effective one roughly fivefold.

See Also
--------
scikitplot.mcp._core.MAX_RESOURCE_CHARS
"""

from .. import _core, _server


def _chunk(text):
    """Return a retrieved chunk carrying ``text``."""
    return _core.RetrievedChunk(
        doc_id="big", title="Big", source_uri="docs://big", anchor="#a",
        text=text, score=1.0,
    )


def test_the_resource_bound_is_derived_from_the_chunk_bound():
    """One declared bound, derived from the tighter one it sits behind."""
    assert _core.MAX_RESOURCE_CHARS == (
        _core.MAX_CHUNK_CHARS + _core.RESOURCE_METADATA_ALLOWANCE
    )
    assert _server._MAX_RESOURCE_CHARS is _core.MAX_RESOURCE_CHARS


def test_the_published_bound_is_the_effective_one():
    """An oversized document renders within the declared bound, not far under it."""
    rendered = _server._read_resource(lambda doc_id: _chunk("x" * 200_000), "big")
    assert len(rendered) <= _core.MAX_RESOURCE_CHARS
    assert len(rendered) > _core.MAX_CHUNK_CHARS


def test_the_allowance_covers_the_metadata_actually_rendered():
    """The allowance is sized from what rendering adds, not guessed."""
    rendered = _server._read_resource(lambda doc_id: _chunk("x" * 200_000), "big")
    overhead = len(rendered) - _core.MAX_CHUNK_CHARS
    assert 0 < overhead <= _core.RESOURCE_METADATA_ALLOWANCE


def test_a_small_document_is_untouched():
    """Rendering below the bound is unchanged."""
    rendered = _server._read_resource(lambda doc_id: _chunk("short body"), "big")
    assert "short body" in rendered
    assert not rendered.endswith("\u2026")
