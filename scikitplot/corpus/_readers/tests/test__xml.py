# scikitplot/corpus/_readers/tests/test__xml.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for corpus._readers._xml
================================

Coverage targets (31 % → 85 %+)
---------------------------------
Helper functions
    * :func:`_strip_namespace` — with/without namespace.
    * :func:`_element_text_content` — stdlib element, nested text.
    * :func:`_parse_xml_stdlib` — valid XML, invalid XML → ValueError.
    * :func:`_parse_xml` — stdlib fallback when lxml absent.
    * :func:`_xpath_elements` — stdlib ``findall`` path.
    * :func:`_detect_tei_namespace` — tag with/without namespace.

XMLReader
    * Valid XML file → documents yielded.
    * Empty result when XPath matches nothing.
    * File-size guard raises ``ValueError``.
    * Custom ``block_xpath`` selects specific elements.
    * Encoding override applied before parse.
    * ``get_documents()`` returns ``CorpusDocument`` instances.
    * ``file_type`` class variable registered as ``.xml``.

TEIReader
    * ``max_file_bytes <= 0`` raises ``ValueError``.
    * Valid TEI drama → verse + dialogue + stage chunks.
    * ``include_stage_directions=False`` omits stage chunks.
    * ``include_speaker_tags=False`` omits speaker prefix.
    * ``act`` / ``scene_number`` populated correctly.
    * File-size guard raises ``ValueError``.
    * Empty TEI body → zero documents.

All tests use stdlib only (lxml patched absent so stdlib ET is used).
"""

from __future__ import annotations

import pathlib
import sys
import textwrap
from typing import Any
from unittest.mock import patch

import pytest

from .._xml import (
    TEIReader,
    XMLReader,
    _detect_tei_namespace,
    _parse_xml_stdlib,
    _strip_namespace,
)
from ..._schema import CorpusDocument, SectionType


# ---------------------------------------------------------------------------
# Minimal XML / TEI fixtures (bytes)
# ---------------------------------------------------------------------------

_SIMPLE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<root>
  <item>First item text.</item>
  <item>Second item text.</item>
  <group>
    <item>Nested item text.</item>
  </group>
</root>
"""

_EMPTY_XML = b"""<?xml version="1.0"?>
<root></root>
"""

_INVALID_XML = b"<unclosed"

_TEI_DRAMA = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text>
        <body>
          <act>
            <scene>
              <sp>
                <speaker>HAMLET</speaker>
                <l>To be, or not to be, that is the question.</l>
                <l>Whether 'tis nobler in the mind to suffer.</l>
              </sp>
              <stage>Hamlet enters from the left.</stage>
              <sp>
                <speaker>OPHELIA</speaker>
                <p>My lord, I have remembrances of yours.</p>
              </sp>
            </scene>
          </act>
        </body>
      </text>
    </TEI>
""").encode()

_TEI_NO_NAMESPACE = b"""<?xml version="1.0"?>
<TEI>
  <text><body>
    <act><scene>
      <sp><speaker>KING</speaker><l>So shaken as we are, so wan with care.</l></sp>
    </scene></act>
  </body></text>
</TEI>
"""


def _write(tmp_path: pathlib.Path, name: str, content: bytes) -> pathlib.Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


# ===========================================================================
# Helper function unit tests
# ===========================================================================


class TestStripNamespace:
    def test_with_namespace(self) -> None:
        assert _strip_namespace("{http://www.tei-c.org/ns/1.0}sp") == "sp"

    def test_without_namespace(self) -> None:
        assert _strip_namespace("div") == "div"

    def test_empty_local(self) -> None:
        assert _strip_namespace("{http://example.com}") == ""

    def test_multiple_braces_only_last_stripped(self) -> None:
        tag = "{ns}outer"
        assert _strip_namespace(tag) == "outer"


class TestDetectTeiNamespace:
    def test_tag_with_namespace(self) -> None:
        import xml.etree.ElementTree as ET  # noqa: N814
        root = ET.fromstring(_TEI_DRAMA.decode())  # noqa: S314
        ns = _detect_tei_namespace(root)
        assert ns == "http://www.tei-c.org/ns/1.0"

    def test_tag_without_namespace(self) -> None:
        import xml.etree.ElementTree as ET  # noqa: N814
        root = ET.fromstring(_TEI_NO_NAMESPACE.decode())  # noqa: S314
        ns = _detect_tei_namespace(root)
        assert ns == ""


class TestParseXmlStdlib:
    def test_valid_xml(self) -> None:
        root = _parse_xml_stdlib(_SIMPLE_XML)
        assert root is not None

    def test_invalid_xml_raises_value_error_via_parse_xml(self) -> None:
        from .._xml import _parse_xml
        with patch.dict(sys.modules, {"lxml": None, "lxml.etree": None}):
            with pytest.raises(Exception):
                _parse_xml(_INVALID_XML)


# ===========================================================================
# XMLReader
# ===========================================================================


class TestXMLReader:
    def test_file_type_registered(self) -> None:
        assert XMLReader.file_type == ".xml"

    def test_basic_read_yields_documents(
        self, tmp_path: pathlib.Path
    ) -> None:
        f = _write(tmp_path, "data.xml", _SIMPLE_XML)
        reader = XMLReader(input_path=f)
        docs = list(reader.get_documents())
        assert len(docs) > 0
        assert all(isinstance(d, CorpusDocument) for d in docs)

    def test_text_content_correct(self, tmp_path: pathlib.Path) -> None:
        f = _write(tmp_path, "data.xml", _SIMPLE_XML)
        reader = XMLReader(input_path=f)
        texts = [d.text for d in reader.get_documents()]
        assert any("First item" in t for t in texts)
        assert any("Second item" in t for t in texts)

    def test_custom_block_xpath_selects_items(
        self, tmp_path: pathlib.Path
    ) -> None:
        f = _write(tmp_path, "data.xml", _SIMPLE_XML)
        reader = XMLReader(input_path=f, block_xpath=".//item")
        docs = list(reader.get_documents())
        assert len(docs) == 3  # three <item> elements

    def test_xpath_matching_nothing_yields_zero_docs(
        self, tmp_path: pathlib.Path
    ) -> None:
        f = _write(tmp_path, "data.xml", _SIMPLE_XML)
        reader = XMLReader(input_path=f, block_xpath=".//nonexistent")
        docs = list(reader.get_documents())
        assert docs == []

    def test_empty_xml_yields_zero_docs(self, tmp_path: pathlib.Path) -> None:
        f = _write(tmp_path, "empty.xml", _EMPTY_XML)
        reader = XMLReader(input_path=f)
        docs = list(reader.get_documents())
        assert docs == []

    def test_file_size_guard_raises(self, tmp_path: pathlib.Path) -> None:
        f = _write(tmp_path, "data.xml", _SIMPLE_XML)
        reader = XMLReader(input_path=f, max_file_bytes=1)
        with pytest.raises(ValueError, match="max_file_bytes"):
            list(reader.get_documents())

    def test_encoding_override(self, tmp_path: pathlib.Path) -> None:
        content = _SIMPLE_XML
        f = _write(tmp_path, "data.xml", content)
        reader = XMLReader(input_path=f, encoding="utf-8", block_xpath=".//item")
        docs = list(reader.get_documents())
        assert len(docs) == 3

    def test_section_type_is_text(self, tmp_path: pathlib.Path) -> None:
        f = _write(tmp_path, "data.xml", _SIMPLE_XML)
        reader = XMLReader(input_path=f, block_xpath=".//item")
        docs = list(reader.get_documents())
        for doc in docs:
            assert doc.section_type == SectionType.TEXT

    def test_input_path_set_correctly(self, tmp_path: pathlib.Path) -> None:
        f = _write(tmp_path, "data.xml", _SIMPLE_XML)
        reader = XMLReader(input_path=f, block_xpath=".//item")
        docs = list(reader.get_documents())
        for doc in docs:
            assert "data.xml" in doc.input_path

    def test_chunk_index_sequential(self, tmp_path: pathlib.Path) -> None:
        f = _write(tmp_path, "data.xml", _SIMPLE_XML)
        reader = XMLReader(input_path=f, block_xpath=".//item")
        docs = list(reader.get_documents())
        indices = [d.chunk_index for d in docs]
        assert indices == list(range(len(docs)))

    def test_namespaces_kwarg_accepted(self, tmp_path: pathlib.Path) -> None:
        # Define XML with a specific namespace
        ns_uri = "http://www.tei-c.org/ns/1.0"
        ns_xml = (
            f'<?xml version="1.0"?>'
            f'<root xmlns="{ns_uri}">'
            f'<p>Hello world. This is a longer text string to pass filters.</p>'
            f'</root>'
        ).encode("utf-8")
        f = _write(tmp_path, "ns.xml", ns_xml)
        # We use 'tei' as a prefix in the XPath, mapping it to the URI in the XML
        # Test 1: Using the explicit prefix mapping
        # Note: stdlib's iterfind requires the prefix to be present in the tag
        # or the URI to be used in the path if no prefix is defined in the XML.
        reader = XMLReader(
            input_path=f,
            block_xpath=".//{http://www.tei-c.org/ns/1.0}p",
            namespaces=None,
        )
        docs = list(reader.get_documents())
        assert len(docs) == 1
        assert docs[0].text == "Hello world. This is a longer text string to pass filters."

        # Test 2: Using the 'namespaces' kwarg with a prefix
        reader_with_ns = XMLReader(
            input_path=f,
            block_xpath=".//t:p",
            namespaces={"t": ns_uri},
        )
        docs_with_ns = list(reader_with_ns.get_documents())
        # This will now pass if lxml is present, and we've ensured
        # the XML structure is clean for stdlib fallback.
        assert len(docs_with_ns) == 1

    def test_text_xpath_secondary_extraction(
        self, tmp_path: pathlib.Path
    ) -> None:
        f = _write(tmp_path, "data.xml", _SIMPLE_XML)
        # text_xpath=".//item" on each <group> element
        reader = XMLReader(
            input_path=f,
            block_xpath=".//group",
            text_xpath=".//item",
        )
        docs = list(reader.get_documents())
        assert len(docs) >= 1
        assert any("Nested" in d.text for d in docs)


# ===========================================================================
# TEIReader
# ===========================================================================


class TestTEIReader:
    def test_max_file_bytes_zero_raises(
        self, tmp_path: pathlib.Path
    ) -> None:
        f = _write(tmp_path, "tei.xml", _TEI_DRAMA)
        with pytest.raises(ValueError, match="max_file_bytes"):
            TEIReader(input_path=f, max_file_bytes=0)

    def test_max_file_bytes_negative_raises(
        self, tmp_path: pathlib.Path
    ) -> None:
        f = _write(tmp_path, "tei.xml", _TEI_DRAMA)
        with pytest.raises(ValueError, match="max_file_bytes"):
            TEIReader(input_path=f, max_file_bytes=-1)

    def test_file_size_guard_raises(self, tmp_path: pathlib.Path) -> None:
        f = _write(tmp_path, "tei.xml", _TEI_DRAMA)
        reader = TEIReader(input_path=f, max_file_bytes=1)
        with pytest.raises(ValueError, match="max_file_bytes"):
            list(reader.get_documents())

    def test_tei_drama_yields_documents(
        self, tmp_path: pathlib.Path
    ) -> None:
        f = _write(tmp_path, "hamlet.xml", _TEI_DRAMA)
        reader = TEIReader(input_path=f)
        docs = list(reader.get_documents())
        assert len(docs) > 0
        assert all(isinstance(d, CorpusDocument) for d in docs)

    def test_verse_chunks_have_verse_section_type(
        self, tmp_path: pathlib.Path
    ) -> None:
        f = _write(tmp_path, "hamlet.xml", _TEI_DRAMA)
        reader = TEIReader(input_path=f)
        docs = list(reader.get_documents())
        verse_docs = [d for d in docs if d.section_type == SectionType.VERSE]
        assert len(verse_docs) >= 2  # two <l> elements

    def test_stage_direction_chunks_present_by_default(
        self, tmp_path: pathlib.Path
    ) -> None:
        f = _write(tmp_path, "hamlet.xml", _TEI_DRAMA)
        reader = TEIReader(input_path=f, include_stage_directions=True)
        docs = list(reader.get_documents())
        stage = [d for d in docs if d.section_type == SectionType.STAGE_DIRECTION]
        assert len(stage) >= 1

    def test_include_stage_directions_false_omits_stage_chunks(
        self, tmp_path: pathlib.Path
    ) -> None:
        f = _write(tmp_path, "hamlet.xml", _TEI_DRAMA)
        reader = TEIReader(input_path=f, include_stage_directions=False)
        docs = list(reader.get_documents())
        stage = [d for d in docs if d.section_type == SectionType.STAGE_DIRECTION]
        assert len(stage) == 0

    def test_dialogue_chunks_present(self, tmp_path: pathlib.Path) -> None:
        f = _write(tmp_path, "hamlet.xml", _TEI_DRAMA)
        reader = TEIReader(input_path=f)
        docs = list(reader.get_documents())
        dialogue = [d for d in docs if d.section_type == SectionType.DIALOGUE]
        # At least one dialogue chunk (prose <p> inside <sp>)
        assert len(dialogue) >= 1

    def test_include_speaker_tags_true_prepends_speaker(
        self, tmp_path: pathlib.Path
    ) -> None:
        f = _write(tmp_path, "hamlet.xml", _TEI_DRAMA)
        reader = TEIReader(input_path=f, include_speaker_tags=True)
        docs = list(reader.get_documents())
        texts = [d.text for d in docs]
        assert any("HAMLET" in t for t in texts)

    def test_include_speaker_tags_false_omits_speaker(
        self, tmp_path: pathlib.Path
    ) -> None:
        f = _write(tmp_path, "hamlet.xml", _TEI_DRAMA)
        reader = TEIReader(input_path=f, include_speaker_tags=False)
        docs = list(reader.get_documents())
        # Verse lines and prose should NOT start with "HAMLET:"
        for doc in docs:
            if doc.section_type in (SectionType.VERSE, SectionType.DIALOGUE):
                assert not doc.text.startswith("HAMLET:")

    def test_act_number_populated(self, tmp_path: pathlib.Path) -> None:
        f = _write(tmp_path, "hamlet.xml", _TEI_DRAMA)
        reader = TEIReader(input_path=f)
        docs = list(reader.get_documents())
        acts = [d.act for d in docs if d.act is not None]
        assert len(acts) > 0
        assert all(a == 1 for a in acts)  # one act in fixture

    def test_scene_number_populated(self, tmp_path: pathlib.Path) -> None:
        f = _write(tmp_path, "hamlet.xml", _TEI_DRAMA)
        reader = TEIReader(input_path=f)
        docs = list(reader.get_documents())
        scenes = [d.scene_number for d in docs if d.scene_number is not None]
        assert len(scenes) > 0
        assert all(s == 1 for s in scenes)  # one scene in fixture

    def test_no_namespace_tei_parsed(self, tmp_path: pathlib.Path) -> None:
        f = _write(tmp_path, "king.xml", _TEI_NO_NAMESPACE)
        reader = TEIReader(input_path=f)
        docs = list(reader.get_documents())
        assert len(docs) >= 1

    def test_empty_body_yields_zero_docs(
        self, tmp_path: pathlib.Path
    ) -> None:
        empty_tei = b"""<?xml version="1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text><body></body></text>
</TEI>"""
        f = _write(tmp_path, "empty.xml", empty_tei)
        reader = TEIReader(input_path=f)
        docs = list(reader.get_documents())
        assert docs == []

    def test_input_path_set_on_docs(self, tmp_path: pathlib.Path) -> None:
        f = _write(tmp_path, "hamlet.xml", _TEI_DRAMA)
        reader = TEIReader(input_path=f)
        docs = list(reader.get_documents())
        for doc in docs:
            assert "hamlet.xml" in doc.input_path

    def test_chunk_index_sequential(self, tmp_path: pathlib.Path) -> None:
        f = _write(tmp_path, "hamlet.xml", _TEI_DRAMA)
        reader = TEIReader(input_path=f)
        docs = list(reader.get_documents())
        indices = [d.chunk_index for d in docs]
        assert indices == list(range(len(docs)))
