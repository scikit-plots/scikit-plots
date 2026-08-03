# scikitplot/corpus/_readers/tests/test__xml_safety.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Hardened XML parsing gate (CORPUS-XML-001)
==========================================

The tests verify security behavior, namespace preservation, DTD-default
handling, and semantic parity between the stdlib Expat and optional lxml
backends.

Run with::

    pytest scikitplot/corpus/_readers/tests/test__xml_safety.py -v
"""

from __future__ import annotations

from xml.parsers import expat

import pytest

from scikitplot.corpus._readers._xml_safety import (
    XmlSecurityError,
    hardened_lxml_parser,
    parse_stdlib_secure,
)

BILLION_LAUGHS = b"""<?xml version="1.0"?>
<!DOCTYPE lolz [
  <!ENTITY a "AAAAAAAAAA">
  <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">
  <!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">
]>
<lolz>&c;</lolz>"""

XXE_FILE = b"""<?xml version="1.0"?>
<!DOCTYPE r [<!ENTITY x SYSTEM "file:///etc/hostname">]>
<r>&x;</r>"""

XXE_META = b"""<?xml version="1.0"?>
<!DOCTYPE r [
  <!ENTITY x SYSTEM "http://169.254.169.254/latest/">
]>
<r>&x;</r>"""

BENIGN = b"<doc><title>Hello</title><p>World of <b>XML</b></p></doc>"

DEFAULT_NAMESPACE = b"""\
<root xmlns="urn:default">
  <item plain="value">Default namespace</item>
</root>
"""

PREFIXED_NAMESPACE = b"""\
<root xmlns:t="urn:test" plain="root-value" t:identifier="123">
  <t:item t:status="active">Prefixed namespace</t:item>
</root>
"""

BENIGN_DOCTYPE = b'<!DOCTYPE root><root><item value="safe"/></root>'

DTD_DEFAULT_ATTRIBUTE = b"""\
<!DOCTYPE root [
  <!ATTLIST root role CDATA "admin">
]>
<root/>
"""

EXTERNAL_DTD = b"""\
<!DOCTYPE root SYSTEM "http://169.254.169.254/external.dtd">
<root/>
"""


class TestStdlibHardened:
    @pytest.mark.parametrize(
        "payload",
        [BILLION_LAUGHS, XXE_FILE, XXE_META],
        ids=["billion-laughs", "xxe-file", "xxe-network"],
    )
    def test_entity_declarations_raise_security_error(
        self,
        payload: bytes,
    ) -> None:
        with pytest.raises(XmlSecurityError, match="entity declarations"):
            parse_stdlib_secure(payload)

    def test_benign_parses(self) -> None:
        root = parse_stdlib_secure(BENIGN)

        assert root.tag == "doc"
        assert "".join(root.itertext()) == "HelloWorld of XML"

    def test_string_input_is_supported(self) -> None:
        root = parse_stdlib_secure("<root><item>text</item></root>")

        assert root.tag == "root"
        assert list(root)[0].text == "text"

    def test_default_namespace_is_preserved_as_clark_name(self) -> None:
        root = parse_stdlib_secure(DEFAULT_NAMESPACE)
        item = list(root)[0]

        assert root.tag == "{urn:default}root"
        assert item.tag == "{urn:default}item"
        # Default namespaces never apply to unprefixed attributes.
        assert item.attrib == {"plain": "value"}

    def test_prefixed_namespace_and_attributes_are_preserved(self) -> None:
        root = parse_stdlib_secure(PREFIXED_NAMESPACE)
        item = list(root)[0]

        assert root.tag == "root"
        assert root.attrib == {
            "plain": "root-value",
            "{urn:test}identifier": "123",
        }
        assert item.tag == "{urn:test}item"
        assert item.attrib == {"{urn:test}status": "active"}

    def test_benign_doctype_is_accepted(self) -> None:
        root = parse_stdlib_secure(BENIGN_DOCTYPE)

        assert root.tag == "root"
        assert list(root)[0].attrib == {"value": "safe"}

    def test_dtd_default_attributes_are_not_injected(self) -> None:
        root = parse_stdlib_secure(DTD_DEFAULT_ATTRIBUTE)

        assert root.attrib == {}

    def test_external_dtd_is_not_loaded(self) -> None:
        root = parse_stdlib_secure(EXTERNAL_DTD)

        assert root.tag == "root"
        assert root.attrib == {}

    def test_malformed_xml_raises_expat_error(self) -> None:
        with pytest.raises(expat.ExpatError):
            parse_stdlib_secure(b"<root>")


class TestLxmlHardened:
    def setup_method(self) -> None:
        pytest.importorskip("lxml")

    @staticmethod
    def _parse(data: bytes):
        from lxml import etree  # noqa: PLC0415

        return etree.fromstring(data, parser=hardened_lxml_parser())

    @staticmethod
    def _parse_or_syntax_error(data: bytes):
        from lxml import etree  # noqa: PLC0415

        try:
            return etree.fromstring(data, parser=hardened_lxml_parser())
        except etree.XMLSyntaxError:
            return None

    def test_billion_laughs_is_not_expanded(self) -> None:
        root = self._parse_or_syntax_error(BILLION_LAUGHS)
        if root is None:
            return

        text = "".join(root.itertext())
        assert "AAAAAAAAAA" not in text
        assert len(text) < 100

    def test_local_external_entity_is_not_resolved(
        self,
        tmp_path,
    ) -> None:
        marker = "CORPUS_XML_LOCAL_SECRET_7f4d7d6b"
        secret = tmp_path / "secret.txt"
        secret.write_text(marker, encoding="utf-8")

        payload = (
            '<?xml version="1.0"?>'
            f'<!DOCTYPE r [<!ENTITY x SYSTEM "{secret.as_uri()}">]>'
            "<r>&x;</r>"
        ).encode("utf-8")

        root = self._parse_or_syntax_error(payload)
        if root is None:
            return

        assert marker not in "".join(root.itertext())

    def test_network_external_entity_is_not_resolved(self) -> None:
        root = self._parse_or_syntax_error(XXE_META)
        if root is None:
            return

        assert "".join(root.itertext()).strip() in {"", "&x;"}

    def test_external_dtd_is_not_loaded(
        self,
        tmp_path,
    ) -> None:
        dtd = tmp_path / "external.dtd"
        dtd.write_text(
            '<!ATTLIST root injected CDATA "from-external-dtd">',
            encoding="utf-8",
        )
        payload = (
            f'<!DOCTYPE root SYSTEM "{dtd.as_uri()}"><root/>'
        ).encode("utf-8")

        root = self._parse(payload)

        assert "injected" not in root.attrib

    def test_internal_dtd_default_attributes_are_not_injected(self) -> None:
        root = self._parse(DTD_DEFAULT_ATTRIBUTE)

        assert root.attrib == {}

    def test_namespace_names_are_preserved(self) -> None:
        root = self._parse(DEFAULT_NAMESPACE)

        assert root.tag == "{urn:default}root"
        assert list(root)[0].tag == "{urn:default}item"

    def test_benign_parses(self) -> None:
        root = self._parse(BENIGN)

        assert root.tag == "doc"
        assert "".join(root.itertext()) == "HelloWorld of XML"
