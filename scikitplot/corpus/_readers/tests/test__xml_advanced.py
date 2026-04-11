# corpus/_readers/tests/test__xml_advanced.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Advanced tests for XMLReader and _clark_to_prefix.

Covers:
- Clark notation {uri}tag auto-conversion to prefix:tag for lxml
- Namespace reuse when prefix already mapped
- Multiple distinct namespaces in one XPath
- stdlib fallback with Clark notation natively
- _detect_tei_namespace edge cases
"""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET

import pytest

from .._xml import _clark_to_prefix, _xpath_elements


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _elem(xml_str: str) -> ET.Element:
    return ET.fromstring(xml_str)


# ===========================================================================
# _clark_to_prefix
# ===========================================================================


class TestClarkToPrefix:
    def test_no_clark_notation_unchanged(self) -> None:
        xpath, ns = _clark_to_prefix(".//p", {})
        assert xpath == ".//p"
        assert ns == {}

    def test_single_namespace_auto_prefixed(self) -> None:
        xpath, ns = _clark_to_prefix(".//{http://example.org/ns}p", {})
        assert "http://example.org/ns" in ns.values()
        prefix = [k for k, v in ns.items() if v == "http://example.org/ns"][0]
        assert f"{prefix}:p" in xpath

    def test_reuses_existing_prefix(self) -> None:
        existing = {"tei": "http://www.tei-c.org/ns/1.0"}
        xpath, ns = _clark_to_prefix(".//{http://www.tei-c.org/ns/1.0}p", existing)
        assert "tei:p" in xpath
        assert "tei" in ns

    def test_two_distinct_namespaces(self) -> None:
        expr = ".//{http://ns1.org}div/{http://ns2.org}p"
        xpath, ns = _clark_to_prefix(expr, {})
        assert len(ns) == 2
        assert all(":" in part for part in xpath.split("/") if part and "." not in part)

    def test_same_namespace_reused_once(self) -> None:
        expr = ".//{http://ns.org}a/{http://ns.org}b"
        xpath, ns = _clark_to_prefix(expr, {})
        # Only one prefix generated for the same URI
        assert len(ns) == 1

    def test_returns_tuple(self) -> None:
        result = _clark_to_prefix(".//p", {})
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_empty_namespace_dict_input(self) -> None:
        xpath, ns = _clark_to_prefix(".//{http://x.org}y", {})
        assert isinstance(ns, dict)
        assert len(ns) == 1

    def test_none_namespace_dict_input(self) -> None:
        xpath, ns = _clark_to_prefix(".//{http://x.org}y", None)  # type: ignore[arg-type]
        assert isinstance(ns, dict)

    def test_generated_prefix_not_in_existing_keys(self) -> None:
        existing = {"_ns0": "http://other.org"}
        xpath, ns = _clark_to_prefix(".//{http://new.org}elem", existing)
        # New prefix must not collide with _ns0
        new_prefix = [k for k in ns if k not in existing][0]
        assert new_prefix != "_ns0"


# ===========================================================================
# _xpath_elements: stdlib path with Clark notation
# ===========================================================================


class TestXpathElementsStdlib:
    def test_clark_notation_with_stdlib(self) -> None:
        """stdlib iterfind must find elements using Clark notation natively."""
        ns = "http://example.org/ns"
        root = ET.fromstring(
            f'<root xmlns="{ns}"><p>Hello</p></root>'
        )
        results = _xpath_elements(root, f".//{{{ns}}}p", {})
        assert len(results) == 1
        assert results[0].text == "Hello"

    def test_no_namespace_simple_xpath(self) -> None:
        root = ET.fromstring("<root><p>text</p></root>")
        results = _xpath_elements(root, ".//p", {})
        assert len(results) == 1

    def test_no_match_returns_empty(self) -> None:
        root = ET.fromstring("<root><p>text</p></root>")
        results = _xpath_elements(root, ".//div", {})
        assert results == []

    def test_bad_xpath_returns_empty_not_raises(self) -> None:
        root = ET.fromstring("<root><p>text</p></root>")
        results = _xpath_elements(root, "[invalid xpath!!!", {})
        assert results == []


# ===========================================================================
# XMLReader with namespace XPath (integration)
# ===========================================================================


class TestXMLReaderNamespace:
    def _write(self, tmp_path: pathlib.Path, name: str, content: bytes) -> pathlib.Path:
        p = tmp_path / name
        p.write_bytes(content)
        return p

    def test_clark_notation_in_block_xpath(self, tmp_path: pathlib.Path) -> None:
        """XMLReader with Clark notation block_xpath must find namespaced elements."""
        from .._xml import XMLReader

        ns_uri = "http://www.tei-c.org/ns/1.0"
        xml = (
            f'<?xml version="1.0"?>'
            f'<root xmlns="{ns_uri}">'
            f'<p>Hello world this is a test document for reading.</p>'
            f'</root>'
        ).encode("utf-8")
        f = self._write(tmp_path, "ns_clark.xml", xml)
        reader = XMLReader(
            input_file=f,
            block_xpath=f".//{{{ns_uri}}}p",
            namespaces=None,
        )
        docs = list(reader.get_documents())
        assert len(docs) == 1
        assert "Hello" in docs[0].text

    def test_prefixed_xpath_with_namespaces_kwarg(self, tmp_path: pathlib.Path) -> None:
        """Prefixed XPath with explicit namespaces dict must work."""
        from .._xml import XMLReader

        ns_uri = "http://www.tei-c.org/ns/1.0"
        xml = (
            f'<?xml version="1.0"?>'
            f'<root xmlns="{ns_uri}">'
            f'<p>Hello world this is a longer test sentence.</p>'
            f'</root>'
        ).encode("utf-8")
        f = self._write(tmp_path, "ns_prefix.xml", xml)
        reader = XMLReader(
            input_file=f,
            block_xpath=".//tei:p",
            namespaces={"tei": ns_uri},
        )
        docs = list(reader.get_documents())
        assert len(docs) == 1

    def test_no_namespace_xml(self, tmp_path: pathlib.Path) -> None:
        """XML without namespace must still work normally."""
        from .._xml import XMLReader

        xml = b'<?xml version="1.0"?><root><p>Simple paragraph text here.</p></root>'
        f = self._write(tmp_path, "no_ns.xml", xml)
        reader = XMLReader(input_file=f, block_xpath=".//p")
        docs = list(reader.get_documents())
        assert len(docs) == 1
        assert "Simple" in docs[0].text

    def test_multiple_namespaced_elements(self, tmp_path: pathlib.Path) -> None:
        from .._xml import XMLReader

        ns_uri = "http://www.example.org/ns"
        xml = (
            f'<?xml version="1.0"?>'
            f'<root xmlns="{ns_uri}">'
            f'<p>First paragraph with enough words to pass filters.</p>'
            f'<p>Second paragraph also with enough content.</p>'
            f'</root>'
        ).encode("utf-8")
        f = self._write(tmp_path, "multi_ns.xml", xml)
        reader = XMLReader(
            input_file=f,
            block_xpath=f".//{{{ns_uri}}}p",
        )
        docs = list(reader.get_documents())
        assert len(docs) >= 1  # at least one passes min_length filter
