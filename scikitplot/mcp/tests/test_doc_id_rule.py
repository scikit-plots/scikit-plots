"""
Single-rule regressions for the document identifier (slice S-21).

``_DOC_ID_RE`` was defined twice. The copy in ``_server`` rejects ``.``, ``..``
and a leading ``:``; the copy in ``__main__`` accepted all three. One concept
validated by two rules means the hardening recorded in one module is not the
rule the package applies.

See Also
--------
scikitplot.mcp._core.is_valid_doc_id
"""

import pytest

from .. import __main__ as entry
from .. import _core, _server

DIVERGENT = [".", "..", ":x"]
REJECTED = ["", "a/b", "../etc", "a" * 201, "a b", "a\nb"]
ACCEPTED = ["a", "a.b", "a:b", "a-b_c", "index.md", "a" * 200]


@pytest.mark.parametrize("value", DIVERGENT + REJECTED + ACCEPTED)
def test_both_modules_agree(value):
    """Every module answers the same question with the same answer."""
    expected = _core.is_valid_doc_id(value)
    assert bool(_server._DOC_ID_RE.fullmatch(value)) is expected
    assert bool(entry._DOC_ID_RE.fullmatch(value)) is expected


@pytest.mark.parametrize("value", DIVERGENT)
def test_traversal_shaped_identifiers_are_refused_everywhere(value):
    """The hardened answer is the shared one, not the permissive one."""
    assert _core.is_valid_doc_id(value) is False


@pytest.mark.parametrize("value", ACCEPTED)
def test_ordinary_identifiers_are_still_accepted(value):
    """Tightening the rule does not narrow what legitimately worked."""
    assert _core.is_valid_doc_id(value) is True


def test_the_rule_has_one_definition():
    """Both modules reference the shared object rather than a copy of it."""
    assert _server._DOC_ID_RE is _core.DOC_ID_RE
    assert entry._DOC_ID_RE is _core.DOC_ID_RE


def test_non_string_input_is_false_not_an_error():
    """A caller passing the wrong type gets a decision, not a traceback."""
    assert _core.is_valid_doc_id(None) is False
    assert _core.is_valid_doc_id(123) is False
