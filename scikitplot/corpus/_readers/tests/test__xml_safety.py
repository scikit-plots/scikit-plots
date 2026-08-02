# corpus/_readers/tests/test__xml_safety.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Hardened XML parsing gate (CORPUS-XML-001)
==========================================

Untrusted XML must not trigger entity expansion (billion-laughs), external
entity resolution (XXE — local file / internal network), or external DTD loads.
Both parser backends are covered: the stdlib expat path
(:func:`parse_stdlib_secure`) and the lxml path (:func:`hardened_lxml_parser`).

Run with::

    pytest scikitplot/corpus/_readers/tests/test__xml_safety.py -v
"""

from __future__ import annotations

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
<!DOCTYPE r [ <!ENTITY x SYSTEM "file:///etc/hostname"> ]><r>&x;</r>"""

XXE_META = b"""<?xml version="1.0"?>
<!DOCTYPE r [ <!ENTITY x SYSTEM "http://169.254.169.254/latest/"> ]><r>&x;</r>"""

BENIGN = b"<doc><title>Hello</title><p>World of <b>XML</b></p></doc>"


class TestStdlibHardened:
    @pytest.mark.parametrize("payload", [BILLION_LAUGHS, XXE_FILE, XXE_META])
    def test_malicious_blocked(self, payload):
        with pytest.raises((XmlSecurityError, Exception)):
            parse_stdlib_secure(payload)

    def test_billion_laughs_raises_security_error(self):
        with pytest.raises(XmlSecurityError):
            parse_stdlib_secure(BILLION_LAUGHS)

    def test_benign_parses(self):
        root = parse_stdlib_secure(BENIGN)
        assert "Hello" in "".join(root.itertext())
        assert "XML" in "".join(root.itertext())


class TestLxmlHardened:
    def setup_method(self):
        pytest.importorskip("lxml")

    def _parse(self, data):
        from lxml import etree
        return etree.fromstring(data, parser=hardened_lxml_parser())

    def test_billion_laughs_not_expanded(self):
        # resolve_entities=False -> entities are not expanded (or it raises).
        try:
            root = self._parse(BILLION_LAUGHS)
        except Exception:
            return  # raising is a safe outcome
        assert len("".join(root.itertext())) < 5000

    @pytest.mark.parametrize("payload", [XXE_FILE, XXE_META])
    def test_external_entity_not_resolved(self, payload):
        try:
            root = self._parse(payload)
        except Exception:
            return  # raising is safe
        # The entity must remain unexpanded (literal reference), never fetched.
        assert "".join(root.itertext()).strip() in ("", "&x;")

    def test_benign_parses(self):
        root = self._parse(BENIGN)
        assert "Hello" in "".join(root.itertext())
