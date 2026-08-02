# scikitplot/corpus/_readers/_xml_safety.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Hardened XML parsing for untrusted documents (CORPUS-XML-001).

XML readers parse bytes that may come from arbitrary sources. A naive
``etree.fromstring`` / ``ET.fromstring`` is exposed to:

* **XXE** — external entities that read local files or reach internal network
  endpoints (``<!ENTITY x SYSTEM "file:///etc/passwd">`` /
  ``"http://169.254.169.254/">``).
* **Entity-expansion DoS** ("billion laughs") — nested internal entities that
  expand to gigabytes of text.
* **External DTD** loading over the network.

This module provides a single hardened parser factory used by every XML/ALTO
parse site, so the policy is defined once and cannot drift:

* **lxml** — ``resolve_entities=False`` (no expansion), ``no_network=True`` (no
  external fetch), ``load_dtd=False`` / ``dtd_validation=False`` (no external
  DTD), ``huge_tree=False`` (keep libxml2's built-in size limits).
* **stdlib expat** (:func:`parse_stdlib_secure`) — entity declarations and
  external-entity references are rejected (mirrors ``defusedxml``'s
  ``forbid_entities`` / ``forbid_external`` defaults) without adding a
  dependency. A benign ``DOCTYPE`` carrying no entities is still accepted.
"""

from __future__ import annotations

from typing import Any

__all__ = ["XmlSecurityError", "hardened_lxml_parser", "parse_stdlib_secure"]


class XmlSecurityError(ValueError):
    """Raised when untrusted XML uses a security-disabled feature (entity/DTD)."""


def hardened_lxml_parser() -> Any:
    """Construct a hardened ``lxml.etree.XMLParser``.

    Returns
    -------
    lxml.etree.XMLParser
        A parser with entity resolution, network access, and DTD loading
        disabled, and size limits enabled.

    Raises
    ------
    ImportError
        If lxml is not installed.
    """
    from lxml import etree  # type: ignore[import]  # noqa: PLC0415

    return etree.XMLParser(
        resolve_entities=False,  # no entity expansion (billion laughs)
        no_network=True,  # no external fetch (XXE / SSRF)
        load_dtd=False,  # do not load external DTDs
        dtd_validation=False,
        huge_tree=False,  # keep libxml2's built-in size limits
        recover=False,
    )


def parse_stdlib_secure(content: bytes) -> Any:
    """Parse XML bytes with a hardened stdlib (expat) parser.

    Uses ``xml.parsers.expat`` directly with a :class:`~xml.etree.ElementTree.
    TreeBuilder`, because the C-accelerated ``ElementTree.XMLParser`` does not
    expose the expat entity handlers on all Python versions. Entity declarations
    (internal / unparsed) and external-entity references are rejected, blocking
    both billion-laughs expansion and XXE. A benign ``DOCTYPE`` without entities
    is still accepted.

    Parameters
    ----------
    content : bytes or str
        Raw XML.

    Returns
    -------
    xml.etree.ElementTree.Element
        The parsed document root.

    Raises
    ------
    XmlSecurityError
        If the document declares entities or references external entities.
    xml.parsers.expat.ExpatError
        If the XML is malformed.
    """
    import xml.etree.ElementTree as ET  # noqa: N817, PLC0415
    from xml.parsers import expat  # noqa: PLC0415

    if isinstance(content, str):
        content = content.encode("utf-8")

    def _forbid_entities(*_args: Any, **_kwargs: Any) -> None:
        raise XmlSecurityError(
            "XML entity declarations are disabled for security "
            "(CORPUS-XML-001: billion-laughs / XXE prevention)."
        )

    def _forbid_external(*_args: Any, **_kwargs: Any) -> None:
        raise XmlSecurityError(
            "XML external entities are disabled for security "
            "(CORPUS-XML-001: XXE prevention)."
        )

    builder = ET.TreeBuilder()
    parser = expat.ParserCreate()
    parser.buffer_text = True
    parser.EntityDeclHandler = _forbid_entities
    parser.UnparsedEntityDeclHandler = _forbid_entities
    parser.ExternalEntityRefHandler = _forbid_external
    parser.StartElementHandler = builder.start
    parser.EndElementHandler = builder.end
    parser.CharacterDataHandler = builder.data
    parser.Parse(content, True)
    return builder.close()
