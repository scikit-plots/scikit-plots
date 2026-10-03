"""Tests for :mod:`scikitplot.cleanprompt._formats`."""

from __future__ import annotations

import pytest

from .._formats import FormatSpec, format_from_document
from .._packs import PackError

_DOC = {
    "name": "tickets",
    "version": 1,
    "summary": "Support tickets.",
    "extensions": [".TKT", ".tkt"],
    "splitter": "keyvalue",
    "round_trip": True,
    "options": {"separators": [":"], "comments": ["#"], "prose": True},
    "packs": ["personal", "personal"],
}


def test_builds_a_normalised_hashable_spec():
    spec = format_from_document(dict(_DOC), source="t.yaml")
    assert isinstance(spec, FormatSpec)
    assert spec.extensions == (".tkt",)
    assert spec.packs == ("personal",)
    assert hash(spec) == hash(format_from_document(dict(_DOC)))
    assert spec.option_map() == {"comments": ["#"], "prose": True, "separators": [":"]}


def test_option_map_is_a_copy():
    spec = format_from_document(dict(_DOC))
    spec.option_map()["separators"].append("=")
    assert spec.option_map()["separators"] == [":"]


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"splitter": "yaml"}, "splitter"),
        ({"extensions": ["csv"]}, "extensions[0]"),
        ({"extensions": []}, "non-empty list"),
        ({"round_trip": "yes"}, "round_trip"),
        ({"options": {"delimiter": ";;"}}, "delimiter"),
        ({"options": {"prose": "true"}}, "options.prose"),
        ({"options": {"sheet": 1}}, "unknown key"),
        ({"surprise": 1}, "unknown key"),
    ],
)
def test_invalid_documents_name_the_problem(change, fragment):
    with pytest.raises(PackError) as caught:
        format_from_document(dict(_DOC, **change))
    assert fragment in str(caught.value)
