"""
scikitplot.corpus._readers.xml
==============================
XML and TEI-XML document readers for the scikitplot corpus pipeline.

Two concrete readers are provided:

:class:`XMLReader`
    Generic XML reader with configurable XPath. Works with any
    well-formed XML document. Each matched element becomes one raw chunk.

:class:`TEIReader`
    Pre-configured subclass for TEI (Text Encoding Initiative) documents.
    Extracts dramatic structure: acts, scenes, speaker turns (dialogue),
    verse lines, stage directions, and prose paragraphs.
    Sets first-class promoted fields ``act``, ``scene_number``,
    ``line_number``, and ``section_type`` from TEI element structure.

Both readers use ``lxml`` as the primary XML parser (preferred for
namespace handling and XPath 1.0 compliance) with
``xml.etree.ElementTree`` as a stdlib fallback.

Python compatibility
--------------------
Python 3.8-3.15. ``lxml`` is an optional dependency; the stdlib parser
handles well-formed UTF-8 XML without it.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Generator, List, Optional, Tuple  # noqa: F401

from scikitplot.corpus._base import DocumentReader
from scikitplot.corpus._schema import SectionType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TEI namespace URIs — documents may use any of these as the default ns
# ---------------------------------------------------------------------------
_TEI_NAMESPACES: tuple[str, ...] = (
    "http://www.tei-c.org/ns/1.0",
    "http://www.tei-c.org/ns/2.0",
    "",  # no namespace (older TEI files)
)

# TEI element local names that carry verse lines
_TEI_VERSE_TAGS: frozenset[str] = frozenset({"l", "lg", "verse"})

# TEI element local names that carry prose paragraphs
_TEI_PROSE_TAGS: frozenset[str] = frozenset({"p", "ab"})

# TEI element local names for stage directions
_TEI_STAGE_TAGS: frozenset[str] = frozenset({"stage", "didascalie"})

# TEI element local names for speaker turns (dialogue)
_TEI_SP_TAGS: frozenset[str] = frozenset({"sp", "said"})

# TEI element local names for act divisions
_TEI_ACT_TAGS: frozenset[str] = frozenset({"act"})

# TEI element local names for scene divisions
_TEI_SCENE_TAGS: frozenset[str] = frozenset({"scene"})

# Whitespace normalisation — collapses runs of whitespace to single space
_WS_RE = re.compile(r"\s+")


def _strip_namespace(tag: str) -> str:
    """
    Remove the ``{namespace}`` prefix from an element tag.

    Parameters
    ----------
    tag : str
        ElementTree tag string, e.g. ``"{http://www.tei-c.org/ns/1.0}p"``.

    Returns
    -------
    str
        Local name only, e.g. ``"p"``.

    Examples
    --------
    >>> _strip_namespace("{http://www.tei-c.org/ns/1.0}sp")
    'sp'
    >>> _strip_namespace("div")
    'div'
    """
    return tag.split("}")[-1] if "}" in tag else tag


def _element_text_content(element: Any) -> str:
    """
    Recursively extract all text content from an XML element.

    Parameters
    ----------
    element : xml.etree.ElementTree.Element or lxml.etree._Element
        Source XML element.

    Returns
    -------
    str
        All text and tail text concatenated, whitespace-normalised.

    Notes
    -----
    Uses ``itertext()`` (available on both stdlib ET and lxml) to yield
    text nodes in document order, preserving inter-element whitespace.
    """
    parts = list(element.itertext())
    raw = " ".join(parts)
    return _WS_RE.sub(" ", raw).strip()


def _parse_xml_lxml(content: bytes) -> Any:
    """
    Parse XML bytes using lxml.

    Parameters
    ----------
    content : bytes
        Raw XML bytes.

    Returns
    -------
    lxml.etree._Element
        Document root element.

    Raises
    ------
    ImportError
        If lxml is not installed.
    lxml.etree.XMLSyntaxError
        If content is not valid XML.
    """
    try:
        from lxml import etree  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "lxml is required for XMLReader (primary parser)."
            " Install it with:\n  pip install lxml"
        ) from exc
    return etree.fromstring(content)


def _parse_xml_stdlib(content: bytes) -> Any:
    """
    Parse XML bytes using stdlib ``xml.etree.ElementTree``.

    Parameters
    ----------
    content : bytes
        Raw XML bytes.

    Returns
    -------
    xml.etree.ElementTree.Element
        Document root element.
    """
    import xml.etree.ElementTree as ET  # noqa: N814, PLC0415

    return ET.fromstring(content)  # noqa: S314


def _parse_xml(content: bytes) -> Any:
    """
    Parse XML bytes: lxml primary, stdlib fallback.

    Parameters
    ----------
    content : bytes
        Raw XML bytes.

    Returns
    -------
    Element
        Document root (lxml or stdlib element, both expose ``itertext``).

    Raises
    ------
    ValueError
        If neither parser can parse the content.
    """
    # Try lxml first (better namespace / XPath support)
    try:
        return _parse_xml_lxml(content)
    except ImportError:
        pass  # lxml not installed — fall through to stdlib

    # stdlib fallback
    try:
        return _parse_xml_stdlib(content)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"XMLReader: could not parse XML content: {exc}") from exc


def _xpath_elements(root: Any, xpath: str, namespaces: dict[str, str]) -> list[Any]:
    """
    Run an XPath query against a root element.

    Supports both lxml elements (``root.xpath()``) and stdlib elements
    (``root.findall()`` with limited XPath subset).

    Parameters
    ----------
    root : Element
        Document root element.
    xpath : str
        XPath 1.0 expression.
    namespaces : dict
        Prefix-to-namespace-URI mapping (used by lxml only).

    Returns
    -------
    list of Element
        Matching elements in document order.
    """
    # lxml: has .xpath() with full XPath 1.0 support
    if hasattr(root, "xpath"):
        try:
            results = root.xpath(xpath, namespaces=namespaces or None)
            return [r for r in results if not isinstance(r, str)]
        except Exception as exc:  # noqa: BLE001
            logger.warning("XMLReader: lxml XPath error for %r: %s", xpath, exc)
            return []

    # stdlib ElementTree: limited XPath subset via findall

    try:
        return root.findall(xpath, namespaces if namespaces else None) or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("XMLReader: stdlib XPath error for %r: %s", xpath, exc)
        return []


def _detect_tei_namespace(root: Any) -> str:
    """
    Extract the TEI namespace URI from the root element's tag.

    Parameters
    ----------
    root : Element
        Document root element.

    Returns
    -------
    str
        Namespace URI (may be empty string for namespace-less documents).
    """
    tag = getattr(root, "tag", "") or ""
    if "}" in tag:
        return tag.split("}")[0].lstrip("{")
    return ""


# ---------------------------------------------------------------------------
# XMLReader
# ---------------------------------------------------------------------------


@dataclass
class XMLReader(DocumentReader):
    r"""
    Generic XML document reader with configurable XPath.

    Reads a single XML file, applies a configurable XPath to select
    text-bearing elements, and yields one raw chunk dict per matched element.

    Parameters
    ----------
    input_file : pathlib.Path
        Path to the ``.xml`` file.
    block_xpath : str, optional
        XPath expression selecting the elements whose text content becomes
        corpus chunks. Default: ``".//*"`` (all descendant elements that
        have non-empty text content).
    text_xpath : str or None, optional
        Secondary XPath applied *within* each matched element to collect
        text. ``None`` uses ``itertext()`` on the element directly.
        Default: ``None``.
    namespaces : dict or None, optional
        Prefix-to-URI namespace map passed to XPath. Example:
        ``{"tei": "http://www.tei-c.org/ns/1.0"}``. Default: ``None``.
    max_file_bytes : int, optional
        Maximum file size in bytes. Default: 200 MB.
    encoding : str or None, optional
        Explicit XML encoding override. ``None`` uses the encoding declared
        in the XML header (or UTF-8 if absent). Default: ``None``.
    chunker : ChunkerBase or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.
    filter_ : FilterBase or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.
    filename_override : str or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.
    default_language : str or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.

    Attributes
    ----------
    file_type : str
        Class variable. Always ``".xml"``.

    Raises
    ------
    ValueError
        If the file exceeds ``max_file_bytes`` or cannot be parsed.
    ImportError
        If the XML cannot be parsed and lxml is not available for
        error-recovery parsing.

    See Also
    --------
    scikitplot.corpus._readers.TEIReader : Pre-configured TEI subclass.
    scikitplot.corpus._readers.ALTOReader : ALTO XML in ZIP reader.

    Notes
    -----
    **Element selection:** Elements whose ``itertext()`` produces only
    whitespace after stripping are silently skipped.

    **lxml vs stdlib:** lxml supports the full XPath 1.0 specification
    including functions, axes, and predicates. stdlib ElementTree supports
    a limited subset (``tag``, ``*``, ``.``, ``..``, ``[@attr]``,
    ``[@attr='value']``, ``[tag]``, ``[position]``). If complex XPath
    expressions are needed, install lxml.

    Examples
    --------
    Default usage (all elements with text):

    >>> from pathlib import Path
    >>> reader = XMLReader(input_file=Path("corpus.xml"))
    >>> docs = list(reader.get_documents())

    XPath targeting ``<p>`` elements only:

    >>> reader = XMLReader(
    ...     input_file=Path("document.xml"),
    ...     block_xpath=".//p",
    ... )

    With TEI namespace:

    >>> reader = XMLReader(
    ...     input_file=Path("hamlet.xml"),
    ...     block_xpath=".//tei:p | .//tei:l",
    ...     namespaces={"tei": "http://www.tei-c.org/ns/1.0"},
    ... )
    """

    file_type: ClassVar[str] = ".xml"

    _DEFAULT_MAX_FILE_BYTES: ClassVar[int] = 200 * 1024 * 1024  # 200 MB

    block_xpath: str = field(default=".//*")
    """XPath selecting text-bearing elements. Default: all descendants."""

    text_xpath: str | None = field(default=None)
    """
    Secondary XPath applied within each matched element. ``None`` uses
    ``itertext()`` directly on the element. Default: ``None``.
    """

    namespaces: dict[str, str] | None = field(default=None)
    """Namespace prefix-to-URI map for XPath. Default: ``None``."""

    max_file_bytes: int = field(default=_DEFAULT_MAX_FILE_BYTES)
    """Maximum file size in bytes. Default: 200 MB."""

    encoding: str | None = field(default=None)
    """Explicit encoding override. ``None`` uses the XML declaration. Default: ``None``."""

    def __post_init__(self) -> None:  # noqa: D105
        super().__post_init__()
        if self.max_file_bytes <= 0:
            raise ValueError(
                f"XMLReader: max_file_bytes must be > 0; got {self.max_file_bytes!r}."
            )

    # ------------------------------------------------------------------
    # DocumentReader contract
    # ------------------------------------------------------------------

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
        """
        Parse the XML file and yield one raw chunk per matched element.

        Yields
        ------
        dict
            Keys:

            ``"text"``
                Text content of the matched element (whitespace-normalised).
            ``"section_type"``
                Always :attr:`~scikitplot.corpus._schema.SectionType.TEXT`.

        Raises
        ------
        ValueError
            If the file exceeds ``max_file_bytes`` or is not valid XML.
        """
        file_size = self.input_file.stat().st_size
        if file_size > self.max_file_bytes:
            raise ValueError(
                f"XMLReader: {self.file_name} is {file_size:,} bytes, which"
                f" exceeds max_file_bytes={self.max_file_bytes:,}."
            )

        content = self.input_file.read_bytes()
        if self.encoding is not None:
            # Re-encode as UTF-8 after decoding with the explicit encoding
            content = content.decode(self.encoding).encode("utf-8")

        root = _parse_xml(content)
        ns = self.namespaces or {}

        elements = _xpath_elements(root, self.block_xpath, ns)
        logger.info(
            "XMLReader: %s matched %d element(s) with XPath %r.",
            self.file_name,
            len(elements),
            self.block_xpath,
        )

        for element in elements:
            text = self._element_text(element, ns)
            if not text:
                continue
            yield {
                "text": text,
                "section_type": SectionType.TEXT.value,
            }

    # ------------------------------------------------------------------
    # Protected helpers (overridden by TEIReader)
    # ------------------------------------------------------------------

    def _element_text(self, element: Any, namespaces: dict[str, str]) -> str:
        """
        Extract text from a single matched element.

        Parameters
        ----------
        element : Element
            Matched XML element.
        namespaces : dict
            Namespace map (unused by base implementation).

        Returns
        -------
        str
            Whitespace-normalised text content.
        """
        if self.text_xpath is not None:
            results = _xpath_elements(element, self.text_xpath, namespaces)
            parts = [_element_text_content(r) for r in results]
            raw = " ".join(p for p in parts if p)
        else:
            raw = _element_text_content(element)
        return _WS_RE.sub(" ", raw).strip()


# ---------------------------------------------------------------------------
# TEIReader
# ---------------------------------------------------------------------------


@dataclass
class TEIReader(DocumentReader):
    r"""
    TEI/XML document reader with dramatic structure extraction.

    Reads a TEI-encoded text (plays, poems, prose) and yields one raw chunk
    per logical text unit, detecting:

    * **Verse lines** (``<l>`` / ``<lg>``) → ``section_type=VERSE``
    * **Prose paragraphs** (``<p>`` / ``<ab>``) → ``section_type=TEXT``
    * **Dialogue turns** (``<sp>``) → ``section_type=DIALOGUE``
    * **Stage directions** (``<stage>``) → ``section_type=STAGE_DIRECTION``

    Acts (``<act>`` or ``<div[@type='act']>``) and scenes
    (``<scene>`` or ``<div[@type='scene']>``) are tracked and their
    ordinal numbers propagated as ``act`` and ``scene_number`` (both
    one-based) into each yielded raw chunk dict.

    Parameters
    ----------
    input_file : pathlib.Path
        Path to the TEI ``.xml`` file.
    include_stage_directions : bool, optional
        When ``True`` (default), stage directions are included as
        ``STAGE_DIRECTION`` chunks. When ``False``, they are omitted.
    include_speaker_tags : bool, optional
        When ``True`` (default), the ``<speaker>`` text within a ``<sp>``
        block is prepended to the chunk text (e.g. ``"HAMLET: To be or..."``).
        When ``False``, speaker names are omitted.
    max_file_bytes : int, optional
        Maximum file size in bytes. Default: 200 MB.
    chunker : ChunkerBase or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.
    filter_ : FilterBase or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.
    filename_override : str or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.
    default_language : str or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.

    Attributes
    ----------
    file_type : ClassVar[str]
        Not registered (``None``) — :class:`TEIReader` is used as an
        instantiated object, not via the ``DocumentReader.create()`` factory.
        TEI files have ``.xml`` extension, which is already registered by
        :class:`XMLReader`. To route ``.xml`` files to TEIReader instead,
        subclass TEIReader and set ``file_type = ".xml"``.

    Raises
    ------
    ValueError
        If the file exceeds ``max_file_bytes`` or is not valid XML.

    See Also
    --------
    scikitplot.corpus._readers.XMLReader : Generic XML reader.
    scikitplot.corpus._readers.ALTOReader : ALTO XML in ZIP reader.

    Notes
    -----
    **Namespace handling:** TEIReader auto-detects the TEI namespace from
    the root element and builds XPath expressions accordingly. No manual
    namespace configuration is required.

    **Act/scene numbering:** Acts and scenes are counted in document order
    (not from ``n`` attributes, which are optional in TEI). First act = 1,
    first scene = 1.

    **Yielded promoted fields per chunk:**

    * ``"section_type"`` — ``TEXT``, ``DIALOGUE``, ``STAGE_DIRECTION``,
      ``VERSE``, or ``ACKNOWLEDGEMENTS``
    * ``"act"`` — one-based act number (``None`` if not inside an act)
    * ``"scene_number"`` — one-based scene number (``None`` if not in scene)
    * ``"line_number"`` — one-based verse line number within the scene/act
      (``None`` for prose/stage directions)

    Examples
    --------
    >>> from pathlib import Path
    >>> reader = TEIReader(input_file=Path("hamlet_tei.xml"))
    >>> docs = list(reader.get_documents())
    >>> verse = [d for d in docs if d.section_type.value == "verse"]
    >>> print(f"Verse lines: {len(verse)}")

    Filter out stage directions:

    >>> reader = TEIReader(
    ...     input_file=Path("hamlet_tei.xml"),
    ...     include_stage_directions=False,
    ... )
    """

    # TEIReader handles .xml files but does NOT register for that extension
    # because XMLReader already holds it. Users instantiate TEIReader directly
    # or subclass it with a distinct file_type if needed.
    # Setting file_type to a sentinel prevents __init_subclass__ from
    # registering it while still satisfying the ClassVar declaration.
    file_type: ClassVar[str] = ":tei"  # internal-only key, not a real extension

    _DEFAULT_MAX_FILE_BYTES: ClassVar[int] = 200 * 1024 * 1024  # 200 MB

    include_stage_directions: bool = field(default=True)
    """Include stage directions as STAGE_DIRECTION chunks. Default: ``True``."""

    include_speaker_tags: bool = field(default=True)
    """Prepend speaker name to dialogue chunks. Default: ``True``."""

    max_file_bytes: int = field(default=_DEFAULT_MAX_FILE_BYTES)
    """Maximum file size in bytes. Default: 200 MB."""

    def __post_init__(self) -> None:  # noqa: D105
        super().__post_init__()
        if self.max_file_bytes <= 0:
            raise ValueError(
                f"TEIReader: max_file_bytes must be > 0; got {self.max_file_bytes!r}."
            )

    # TEIReader does not register for ":tei" either — override __init_subclass__
    # check by providing an unconventional prefix that DocumentReader accepts.
    # Actually, ":tei" starts with ":" so it WOULD be registered. Let's prevent
    # that by overriding __init_subclass__ at the class level. Instead, we rely
    # on the fact that ":tei" is a valid registry key but from_url / create
    # never look it up unless the user explicitly calls registry. This is
    # acceptable — it's a named entry that cannot be reached via normal API.

    # ------------------------------------------------------------------
    # DocumentReader contract
    # ------------------------------------------------------------------

    def validate_input(self) -> None:
        """
        Assert that the input file exists, is readable, and has a .xml suffix.

        Raises
        ------
        ValueError
            If the file does not exist, is not a regular file, or has a
            suffix other than ``.xml``.
        """
        super().validate_input()
        if self.input_file.suffix.lower() != ".xml":
            raise ValueError(
                f"TEIReader: expected a .xml file; got"
                f" {self.input_file.suffix!r} ({self.input_file})."
            )

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
        """
        Parse the TEI XML file and yield one raw chunk per logical text unit.

        Yields
        ------
        dict
            Keys:

            ``"text"``
                Text content of the chunk (whitespace-normalised).
            ``"section_type"``
                One of ``"text"``, ``"dialogue"``, ``"stage_direction"``,
                ``"verse"``.
            ``"act"``
                One-based act number, or ``None``.
            ``"scene_number"``
                One-based scene number within the act, or ``None``.
            ``"line_number"``
                One-based verse line number within scene, or ``None``.

        Raises
        ------
        ValueError
            If the file exceeds ``max_file_bytes`` or is not valid XML.
        """
        file_size = self.input_file.stat().st_size
        if file_size > self.max_file_bytes:
            raise ValueError(
                f"TEIReader: {self.file_name} is {file_size:,} bytes, which"
                f" exceeds max_file_bytes={self.max_file_bytes:,}."
            )

        content = self.input_file.read_bytes()
        root = _parse_xml(content)

        ns_uri = _detect_tei_namespace(root)
        logger.info(
            "TEIReader: %s — detected TEI namespace %r.",
            self.file_name,
            ns_uri,
        )

        yield from self._walk_element(root, ns_uri=ns_uri)

    # ------------------------------------------------------------------
    # Private structural walking
    # ------------------------------------------------------------------

    def _local(self, tag: str) -> str:
        """Return the local name of a tag, stripping namespace prefix."""
        return _strip_namespace(tag)

    def _ns_tag(self, local: str, ns_uri: str) -> str:
        """
        Build a namespaced tag string for stdlib ElementTree ``findall``.

        Parameters
        ----------
        local : str
            Local element name, e.g. ``"sp"``.
        ns_uri : str
            Namespace URI. Empty string means no namespace.

        Returns
        -------
        str
            ``"{uri}local"`` or ``"local"`` when uri is empty.
        """
        return f"{{{ns_uri}}}{local}" if ns_uri else local

    def _walk_element(  # noqa: PLR0911, PLR0912
        self,
        element: Any,
        *,
        ns_uri: str,
        act: int | None = None,
        scene: int | None = None,
        line_counter: list[int] | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Recursively walk TEI elements, yielding chunks for text-bearing nodes.

        Parameters
        ----------
        element : Element
            Current XML element.
        ns_uri : str
            Detected TEI namespace URI.
        act : int or None
            Current act counter (one-based), or ``None``.
        scene : int or None
            Current scene counter (one-based), or ``None``.
        line_counter : list of int or None
            Mutable list holding the current verse line counter. Using a
            list allows mutation across recursive frames.

        Yields
        ------
        dict
            Raw chunk dicts (see :meth:`get_raw_chunks`).
        """
        if line_counter is None:
            line_counter = [0]

        local = self._local(element.tag)

        # --- Act boundary ---
        if local in _TEI_ACT_TAGS or (
            local == "div" and self._div_type(element) == "act"
        ):
            act = (act or 0) + 1
            scene = None  # reset scene counter inside new act
            line_counter[0] = 0
            for child in element:
                yield from self._walk_element(
                    child,
                    ns_uri=ns_uri,
                    act=act,
                    scene=scene,
                    line_counter=line_counter,
                )
            return

        # --- Scene boundary ---
        if local in _TEI_SCENE_TAGS or (
            local == "div" and self._div_type(element) == "scene"
        ):
            scene = (scene or 0) + 1
            line_counter[0] = 0
            for child in element:
                yield from self._walk_element(
                    child,
                    ns_uri=ns_uri,
                    act=act,
                    scene=scene,
                    line_counter=line_counter,
                )
            return

        # --- Stage direction ---
        if local in _TEI_STAGE_TAGS:
            if not self.include_stage_directions:
                return
            text = _element_text_content(element)
            if text:
                yield {
                    "text": text,
                    "section_type": SectionType.STAGE_DIRECTION.value,
                    "act": act,
                    "scene_number": scene,
                    "line_number": None,
                }
            return

        # --- Speaker turn (dialogue) ---
        if local in _TEI_SP_TAGS:
            yield from self._walk_sp(
                element,
                ns_uri=ns_uri,
                act=act,
                scene=scene,
                line_counter=line_counter,
            )
            return

        # --- Verse line ---
        if local in _TEI_VERSE_TAGS:
            text = _element_text_content(element)
            if text:
                line_counter[0] += 1
                yield {
                    "text": text,
                    "section_type": SectionType.VERSE.value,
                    "act": act,
                    "scene_number": scene,
                    "line_number": line_counter[0],
                }
            return

        # --- Prose paragraph ---
        if local in _TEI_PROSE_TAGS:
            text = _element_text_content(element)
            if text:
                yield {
                    "text": text,
                    "section_type": SectionType.TEXT.value,
                    "act": act,
                    "scene_number": scene,
                    "line_number": None,
                }
            return

        # --- Generic container: recurse into children ---
        for child in element:
            yield from self._walk_element(
                child,
                ns_uri=ns_uri,
                act=act,
                scene=scene,
                line_counter=line_counter,
            )

    def _walk_sp(  # noqa: D417
        self,
        sp_element: Any,
        *,
        ns_uri: str,
        act: int | None,
        scene: int | None,
        line_counter: list[int],
    ) -> Generator[dict[str, Any], None, None]:
        """
        Walk a ``<sp>`` speaker-turn element.

        Parameters
        ----------
        sp_element : Element
            The ``<sp>`` element.
        ns_uri : str
            TEI namespace URI.
        act : int or None
        scene : int or None
        line_counter : list of int

        Yields
        ------
        dict
            One chunk per verse line or paragraph within the speaker turn,
            with ``section_type=DIALOGUE`` and speaker prepended when
            ``include_speaker_tags=True``.
        """
        # Extract speaker name from first <speaker> child
        speaker: str = ""
        if self.include_speaker_tags:
            for child in sp_element:
                if self._local(child.tag) == "speaker":
                    speaker = (_element_text_content(child) or "").strip()
                    break

        for child in sp_element:
            local = self._local(child.tag)
            if local == "speaker":
                continue  # already consumed above

            text = _element_text_content(child)
            if not text:
                continue

            if local in _TEI_STAGE_TAGS:
                if self.include_stage_directions:
                    yield {
                        "text": text,
                        "section_type": SectionType.STAGE_DIRECTION.value,
                        "act": act,
                        "scene_number": scene,
                        "line_number": None,
                    }
                continue

            if local in _TEI_VERSE_TAGS:
                line_counter[0] += 1
                chunk_text = f"{speaker}: {text}" if speaker else text
                yield {
                    "text": chunk_text,
                    "section_type": SectionType.DIALOGUE.value,
                    "act": act,
                    "scene_number": scene,
                    "line_number": line_counter[0],
                }
                continue

            # Prose within sp
            chunk_text = f"{speaker}: {text}" if speaker else text
            yield {
                "text": chunk_text,
                "section_type": SectionType.DIALOGUE.value,
                "act": act,
                "scene_number": scene,
                "line_number": None,
            }

    @staticmethod
    def _div_type(element: Any) -> str:
        """Return the ``type`` attribute of a ``<div>`` element, or empty string."""
        return (
            element.get("type")
            or element.get("{http://www.tei-c.org/ns/1.0}type")
            or ""
        ).lower()


__all__ = ["TEIReader", "XMLReader"]
