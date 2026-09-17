"""
Regressions for the shared row-index contract (slice S-1).

Four findings across three submodules were one defect: a position or a count
reaching a public boundary without a domain check. ``doc_id_for`` resolved
``-1`` and ``True`` to real rows, batch scoring accepted booleans into NumPy
fancy indexing, ``get_top_n(n=-1)`` returned the ranking minus its last entry,
and ``build_search_docs_result`` accepted ``max_results=-1``. This module is the
one place that answers the question, so the answer cannot drift between
callers.

See Also
--------
scikitplot.corpus._validation.require_index
scikitplot.corpus._validation.require_count
"""

import pytest

from .._validation import require_count, require_index


class Indexable:
    """Object exposing ``__index__``, as NumPy integers do."""

    def __init__(self, value):
        self._value = value

    def __index__(self):
        return self._value


# -- require_index ---------------------------------------------------------


@pytest.mark.parametrize("value", [0, 1, 2])
def test_in_range_positions_are_returned_as_int(value):
    """A valid position comes back as a plain ``int``."""
    result = require_index(value, 3)
    assert result == value
    assert type(result) is int


@pytest.mark.parametrize("value", [-1, -3, -100])
def test_negative_positions_are_rejected_not_wrapped(value):
    """``-1`` is a caller mistake, not a request for the last row."""
    with pytest.raises(IndexError) as excinfo:
        require_index(value, 3)
    assert str(value) in str(excinfo.value)


@pytest.mark.parametrize("value", [3, 4, 2**70])
def test_positions_at_or_past_the_end_are_rejected(value):
    """The domain is ``[0, count)``; the bound is named in the message."""
    with pytest.raises(IndexError) as excinfo:
        require_index(value, 3)
    assert "3" in str(excinfo.value)


@pytest.mark.parametrize("value", [True, False])
def test_booleans_are_rejected_before_the_integral_check(value):
    """``bool`` is an ``int`` subclass; it is not a row position."""
    with pytest.raises(TypeError) as excinfo:
        require_index(value, 3)
    assert "bool" in str(excinfo.value)


@pytest.mark.parametrize("value", [1.0, "1", None, [1]])
def test_non_integral_positions_are_rejected(value):
    """A value that is not an integer is refused by type, not coerced."""
    with pytest.raises(TypeError):
        require_index(value, 3)


def test_objects_exposing_index_are_converted_then_bounds_checked():
    """A NumPy-style integer is accepted, then held to the same bounds."""
    assert require_index(Indexable(2), 3) == 2
    with pytest.raises(IndexError):
        require_index(Indexable(3), 3)
    with pytest.raises(IndexError):
        require_index(Indexable(-1), 3)


def test_an_empty_collection_accepts_no_position():
    """With no rows there is no valid position, including zero."""
    with pytest.raises(IndexError):
        require_index(0, 0)


def test_the_name_appears_in_the_message():
    """Callers name their own parameter so the message points at the call site."""
    with pytest.raises(IndexError) as excinfo:
        require_index(9, 3, name="doc_position")
    assert "doc_position" in str(excinfo.value)


# -- require_count ---------------------------------------------------------


@pytest.mark.parametrize("value", [0, 1, 20])
def test_non_negative_counts_are_accepted(value):
    """Zero is a legitimate request for nothing."""
    assert require_count(value, maximum=20) == value


@pytest.mark.parametrize("value", [-1, -20])
def test_negative_counts_are_rejected(value):
    """A negative count silently truncated a result list; it is now refused."""
    with pytest.raises(ValueError) as excinfo:
        require_count(value)
    assert str(value) in str(excinfo.value)


def test_counts_above_the_declared_maximum_are_rejected():
    """A declared ceiling is enforced where it is declared."""
    with pytest.raises(ValueError) as excinfo:
        require_count(21, maximum=20)
    assert "20" in str(excinfo.value)


def test_a_count_without_a_maximum_is_unbounded_above():
    """Not every caller has a ceiling; absence of one is not zero."""
    assert require_count(10_000) == 10_000


@pytest.mark.parametrize("value", [True, False])
def test_boolean_counts_are_rejected(value):
    """``True`` is not a number of results."""
    with pytest.raises(TypeError):
        require_count(value)


@pytest.mark.parametrize("value", [1.0, "1", None])
def test_non_integral_counts_are_rejected(value):
    """A count is an integer or it is an error."""
    with pytest.raises(TypeError):
        require_count(value)


def test_the_module_depends_only_on_the_standard_library():
    """``_utils`` helpers must import anywhere without circular imports."""
    from .. import _validation as _indexing

    assert _indexing.__doc__
    assert not hasattr(_indexing, "np")
