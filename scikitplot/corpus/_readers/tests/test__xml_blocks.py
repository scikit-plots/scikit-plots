"""XMLReader reads each text node once by default (CX-46)."""

from __future__ import annotations

import sys
from collections import Counter

import pytest

from .._xml import XMLReader, _block_text, _parse_xml

CASES = {
    "nested": ("<doc><sec><p>alpha <b>beta</b> gamma</p></sec></doc>", ["alpha beta gamma"]),
    "inline-in-word": ("<p>Elementar<hi>form</hi> und mehr</p>", ["Elementarform und mehr"]),
    "flat": ("<root><item>a</item><item>b</item></root>", ["a", "b"]),
    "comment-and-pi": ("<doc><p>x<!-- c -->y</p><?pi z?></doc>", ["xy"]),
    # Mixed content is one block, joined exactly as written: no space is
    # invented between "</p>" and "loose" (use block_xpath to split it).
    "mixed-tail": ("<doc><p>one</p> loose <i>two</i></doc>", ["one loose two"]),
    "empty": ("<doc><p/><p>  </p><p>z</p></doc>", ["z"]),
    "cdata": ("<d><p><![CDATA[a < b]]></p></d>", ["a < b"]),
    "tei-namespaced": (
        '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt>'
        "<title>T</title></titleStmt></fileDesc></teiHeader><text><body>"
        "<head>H</head><p>P1 <note>n</note> end</p></body></text></TEI>",
        ["T", "H", "P1 n end"],
    ),
}


@pytest.fixture(params=["lxml", "stdlib"])
def parser(request, monkeypatch):
    if request.param == "lxml":
        pytest.importorskip("lxml")
    else:
        monkeypatch.setitem(sys.modules, "lxml", None)
        monkeypatch.setitem(sys.modules, "lxml.etree", None)
    return request.param


def _chunks(tmp_path, xml, **kwargs):
    path = tmp_path / "doc.xml"
    path.write_text(xml, encoding="utf-8")
    return [c["text"] for c in XMLReader(input_path=path, **kwargs).get_raw_chunks()]


@pytest.mark.parametrize("name", sorted(CASES))
def test_default_gives_each_text_node_once(tmp_path, parser, name):
    xml, expected = CASES[name]
    assert _chunks(tmp_path, xml) == expected


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_character_of_the_text_appears_once(tmp_path, name):
    xml, _ = CASES[name]
    whole = _block_text(_parse_xml(xml.encode()))  # the document read once
    expected = Counter(ch for ch in whole if not ch.isspace())
    got = Counter(ch for ch in "".join(_chunks(tmp_path, xml)) if not ch.isspace())
    assert got == expected


def test_an_explicit_xpath_keeps_its_meaning(tmp_path):
    xml = CASES["nested"][0]
    assert _chunks(tmp_path, xml, block_xpath=".//*") == [
        "alpha beta gamma",
        "alpha beta gamma",
        "beta",
    ]
    assert _chunks(tmp_path, xml, block_xpath=".//b") == ["beta"]


def test_deep_nesting_is_iterative(tmp_path):
    depth = 200  # below lxml's default depth limit (256)
    xml = "<a>" * depth + "deep" + "</a>" * depth
    assert _chunks(tmp_path, xml) == ["deep"]


def test_raw_text_is_the_block_as_written(tmp_path):
    path = tmp_path / "doc.xml"
    path.write_text("<doc><p>a\n   b</p></doc>", encoding="utf-8")
    (chunk,) = XMLReader(input_path=path).get_raw_chunks()
    assert chunk["text"] == "a b" and chunk["raw_text"] == "a\n   b"


def test_an_empty_element_separates_words(tmp_path, parser):
    xml = "<p>und<lb/>einfach, diess<pb n='2'/>buch, Elementar<hi>form</hi></p>"
    assert _chunks(tmp_path, xml) == ["und einfach, diess buch, Elementarform"]
