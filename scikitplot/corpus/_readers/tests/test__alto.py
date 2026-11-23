# corpus/_readers/tests/test__alto.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
ALTOReader treats a malformed page the same with and without lxml (CX-42).

A page that is not valid XML is skipped with one warning, and the other pages
of the archive are read, whichever XML parser is installed.

Run with::

    pytest scikitplot/corpus/_readers/tests/test__alto.py -v
"""

from __future__ import annotations

import logging
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scikitplot.corpus._readers._alto import ALTOReader, _parse_xml_bytes

_PREFIX = "ALTOReader: could not parse XML: "
_LOGGER = "scikitplot.corpus._readers._alto"

#: Pages that are not valid XML.
_UNPARSABLE = {
    "empty": b"",
    "not-xml": b"plain text, no markup",
    "truncated": b"<alto><Page>",
    "mismatched-tag": b"<alto><Page></alto>",
}


def _page(*words: str) -> bytes:
    strings = "".join(f'<String CONTENT="{w}" WC="0.9"/>' for w in words)
    return (
        '<?xml version="1.0"?><alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">'
        "<Layout><Page><PrintSpace><TextBlock ID=\"b1\"><TextLine>"
        f"{strings}</TextLine></TextBlock></PrintSpace></Page></Layout></alto>"
    ).encode()


def _archive(tmp_path: Path, members: list[tuple[str, bytes]]) -> Path:
    path = tmp_path / "pages.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members:
            zf.writestr(name, data)
    return path


@pytest.fixture(params=["lxml", "stdlib"])
def parser(request):
    """Run the test with lxml (skipped if absent) and with lxml hidden."""
    if request.param == "lxml":
        pytest.importorskip("lxml.etree")
        yield "lxml"
        return
    with patch.dict(sys.modules, {"lxml": None, "lxml.etree": None}):
        yield "stdlib"


def _read(path: Path, caplog, captured_logger) -> tuple[list[str], list[str]]:
    caplog.clear()
    # captured_logger, not caplog.at_level: the "scikitplot" logger does not
    # propagate, and which pytest releases see its records differs (Rule 139).
    with captured_logger(logging.WARNING, _LOGGER):
        texts = [c["text"] for c in ALTOReader(input_path=path).get_raw_chunks()]
    warnings = [r.getMessage() for r in caplog.records if "XML parse error" in r.getMessage()]
    return texts, warnings


class TestParseXmlBytes:
    @pytest.mark.parametrize("content", list(_UNPARSABLE.values()), ids=list(_UNPARSABLE))
    def test_one_type_whichever_parser(self, parser, content):
        with pytest.raises(ValueError, match="^ALTOReader: could not parse XML: ") as info:
            _parse_xml_bytes(content)
        assert info.value.__cause__ is not None
        assert str(info.value.__cause__) in str(info.value)

    @pytest.mark.parametrize("content", list(_UNPARSABLE.values()), ids=list(_UNPARSABLE))
    def test_lxml_error_is_the_cause(self, content):
        etree = pytest.importorskip("lxml.etree")
        with pytest.raises(ValueError, match="^ALTOReader") as info:
            _parse_xml_bytes(content)
        assert isinstance(info.value.__cause__, etree.XMLSyntaxError)

    def test_valid_page_still_parses(self, parser):
        root = _parse_xml_bytes(_page("alpha"))
        assert root.tag.endswith("alto")


class TestMalformedPageInArchive:
    @pytest.mark.parametrize("bad", list(_UNPARSABLE.values()), ids=list(_UNPARSABLE))
    def test_bad_page_is_skipped_and_the_rest_read(self, tmp_path, caplog, captured_logger, parser, bad):
        path = _archive(tmp_path, [("p1.xml", _page("alpha")), ("p2.xml", bad), ("p3.xml", _page("gamma"))])
        texts, warnings = _read(path, caplog, captured_logger)
        assert texts == ["alpha", "gamma"]
        assert len(warnings) == 1
        assert "'p2.xml' (page 1)" in warnings[0]
        assert _PREFIX in warnings[0]

    def test_every_page_bad_yields_nothing_and_warns_for_each(self, tmp_path, caplog, captured_logger, parser):
        path = _archive(tmp_path, [("p1.xml", b"<alto>"), ("p2.xml", b"not xml"), ("p3.xml", b"")])
        texts, warnings = _read(path, caplog, captured_logger)
        assert texts == []
        assert len(warnings) == len(("p1", "p2", "p3"))

    def test_good_archive_has_no_warning(self, tmp_path, caplog, captured_logger, parser):
        path = _archive(tmp_path, [("p1.xml", _page("alpha", "beta")), ("p2.xml", _page("gamma"))])
        texts, warnings = _read(path, caplog, captured_logger)
        assert texts == ["alpha beta", "gamma"]
        assert warnings == []

    def test_both_parsers_give_the_same_pages(self, tmp_path, caplog, captured_logger):
        pytest.importorskip("lxml.etree")
        path = _archive(
            tmp_path,
            [("p1.xml", _page("alpha")), ("p2.xml", b"<alto><Page>"), ("p3.xml", b""), ("p4.xml", _page("delta"))],
        )
        with_lxml = _read(path, caplog, captured_logger)
        with patch.dict(sys.modules, {"lxml": None, "lxml.etree": None}):
            without = _read(path, caplog, captured_logger)
        assert with_lxml[0] == without[0] == ["alpha", "delta"]
        assert len(with_lxml[1]) == len(without[1]) == len(("p2", "p3"))


class TestHostilePage:
    def test_entity_expansion_page_does_not_expand(self, tmp_path, caplog, captured_logger, parser):
        # Nested entities: refused by the stdlib guard (page skipped), kept
        # unexpanded by the hardened lxml parser; never expanded either way.
        entities = '<!ENTITY e0 "lol">' + "".join(
            f'<!ENTITY e{i} "{f"&e{i - 1};" * 10}">' for i in range(1, 8)
        )
        hostile = (
            f'<?xml version="1.0"?><!DOCTYPE alto [{entities}]>'
            '<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"><Layout><Page><PrintSpace>'
            '<TextBlock ID="b1"><TextLine><String CONTENT="&e7;"/></TextLine></TextBlock>'
            "</PrintSpace></Page></Layout></alto>"
        ).encode()
        path = _archive(tmp_path, [("p1.xml", hostile), ("p2.xml", _page("gamma"))])
        texts, _ = _read(path, caplog, captured_logger)
        assert texts[-1] == "gamma"
        assert sum(len(t) for t in texts) < len(hostile)
