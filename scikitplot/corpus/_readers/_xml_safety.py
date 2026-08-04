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

Namespace representation::

    XML bytes
        |
        v
    Expat reports URI}local
        |
        v
    Normalize to {URI}local
        |
        v
    ElementTree stores Clark names

Clark notation is the canonical internal identity of an expanded XML element
or attribute name. It is not a replacement for XPath. Full lxml XPath queries
still use prefixes plus a prefix-to-URI namespace mapping.
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
        Parser with entity expansion, network access, DTD loading, DTD-default
        attributes, recovery, and oversized-tree mode disabled.

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
        attribute_defaults=False,
        huge_tree=False,  # keep libxml2's built-in size limits
        recover=False,
    )


def parse_stdlib_secure(content: bytes) -> Any:
    """Parse XML bytes with a hardened, namespace-aware stdlib (expat) parser.

    Uses ``xml.parsers.expat`` directly with a :class:`~xml.etree.ElementTree.
    TreeBuilder`, because the C-accelerated ``ElementTree.XMLParser`` does not
    expose the expat entity handlers on all Python versions. Entity declarations
    (internal / unparsed) and external-entity references are rejected, blocking
    both billion-laughs expansion and XXE. A benign ``DOCTYPE`` without entities
    is still accepted.

    Namespace-expanded Expat names are converted to ElementTree's Clark
    notation, for example::

        http://www.tei-c.org/ns/1.0}p  # Expat reports expanded names like

    becomes::

        {http://www.tei-c.org/ns/1.0}p  # TreeBuilder and ElementTree expect

    Parameters
    ----------
    content : bytes or str
        Raw XML content. String input is encoded as UTF-8.

    Returns
    -------
    xml.etree.ElementTree.Element
        Parsed document root with namespace-qualified element and attribute
        names preserved in Clark notation.

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

    def _to_clark_name(name: str) -> str:
        """Convert an Expat expanded name to ElementTree Clark notation."""
        if "}" not in name:
            return name

        # With namespace_separator="}", Expat emits:
        #     namespace-uri}local-name  # URI}local
        #
        # ElementTree expects:
        #     {namespace-uri}local-name  # {URI}local
        return f"{{{name}"

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

    def _start_element(name: str, attrs: dict[str, str]) -> None:
        builder.start(
            _to_clark_name(name),
            {_to_clark_name(attr_name): value for attr_name, value in attrs.items()},
        )

    def _end_element(name: str) -> None:
        builder.end(_to_clark_name(name))

    # A separator enables namespace processing. Expat reports expanded names
    # as ``URI}local``; the handlers convert them to ``{URI}local``.
    parser = expat.ParserCreate(namespace_separator="}")
    parser.buffer_text = True

    # Report only attributes explicitly present in the XML source. Without
    # this, Expat can inject default attributes declared in an internal DTD.
    parser.specified_attributes = True

    parser.EntityDeclHandler = _forbid_entities
    parser.UnparsedEntityDeclHandler = _forbid_entities
    parser.ExternalEntityRefHandler = _forbid_external

    # Never parse external parameter entities or external DTD subsets.
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)

    parser.StartElementHandler = _start_element  # builder.start
    parser.EndElementHandler = _end_element  # builder.end
    parser.CharacterDataHandler = builder.data

    parser.Parse(content, True)
    return builder.close()
