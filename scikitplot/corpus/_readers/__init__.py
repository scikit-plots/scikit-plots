"""
scikitplot.corpus._readers
==========================
Format-specific document readers for the scikitplot corpus pipeline.

Importing this package registers all built-in readers in the
:class:`~scikitplot.corpus._base.DocumentReader` registry, making them
available via :meth:`~scikitplot.corpus._base.DocumentReader.create` (for
file-based sources) and
:meth:`~scikitplot.corpus._base.DocumentReader.from_url` (for URLs).

Available readers
-----------------

**File-based (via** ``DocumentReader.create(path)`` **)**

:class:`TextReader`
    Plain text files (``.txt``). Encoding-aware: BOM detection,
    chardet fallback, Latin-1 last resort.

:class:`MarkdownReader`
    Markdown files (``.md``). Inherits all :class:`TextReader` behaviour.

:class:`ReSTReader`
    reStructuredText files (``.rst``). Inherits all :class:`TextReader`
    behaviour.

:class:`XMLReader`
    Generic XML files (``.xml``) via configurable lxml XPath.

:class:`TEIReader`
    MEGA TEI/XML files. Pre-configured :class:`XMLReader` subclass with
    page and line-number metadata.

:class:`ALTOReader`
    ALTO XML inside a ZIP archive (``.zip``). Supports ALTO namespace
    v2/v3/v4. Natural-sort ordering; ZipSlip guard.

:class:`PDFReader`
    PDF files (``.pdf``) via ``pdfminer.six`` -> ``pypdf`` cascade.
    Per-page error recovery; encrypted PDF support.

:class:`ImageReader`
    Raster images (``.png``, ``.jpg``, ``.jpeg``, ``.gif``, ``.webp``,
    ``.tiff``, ``.tif``, ``.bmp``) via OCR.
    Backend: ``pytesseract`` (primary) -> ``easyocr`` (fallback).
    Multi-frame GIF/TIFF yields one chunk per frame.

:class:`VideoReader`
    Video files (``.mp4``, ``.avi``, ``.mkv``, ``.mov``, ``.webm``,
    ``.m4v``, ``.wmv``, ``.flv``).
    Companion subtitle detection (SRT/VTT/SBV/SUB) with zero dependencies.
    Optional Whisper transcription fallback (``faster-whisper`` or
    ``openai-whisper``).

:class:`AudioReader`
    Audio files (``.mp3``, ``.wav``, ``.flac``, ``.ogg``, ``.m4a``,
    ``.wma``, ``.aac``, ``.aiff``, ``.opus``, ``.wv``).
    Companion lyrics/transcript detection (LRC/SRT/VTT/TXT).
    Optional Whisper ASR and audio classification (animal sounds,
    instruments, environmental sounds).

**URL-based (via** ``DocumentReader.from_url(url)`` **)**

:class:`WebReader`
    Any ``http://`` / ``https://`` URL. Fetches HTML with ``requests``
    and extracts text sections via ``beautifulsoup4``.

:class:`YouTubeReader`
    YouTube video URLs. Retrieves transcript via ``youtube-transcript-api``.
    No audio download or model required.

Quick usage
-----------
File-based:

>>> from pathlib import Path
>>> from scikitplot.corpus._base import DocumentReader
>>> import scikitplot.corpus._readers  # registers all readers
>>> reader = DocumentReader.create(Path("corpus.txt"))
>>> docs = list(reader.get_documents())

URL-based:

>>> reader = DocumentReader.from_url("https://en.wikipedia.org/wiki/Python")
>>> docs = list(reader.get_documents())

YouTube:

>>> reader = DocumentReader.from_url("https://youtu.be/dQw4w9WgXcQ")
>>> docs = list(reader.get_documents())

Image (OCR):

>>> reader = DocumentReader.create(Path("scan.png"))
>>> docs = list(reader.get_documents())

Video (subtitle-first, Whisper fallback):

>>> reader = DocumentReader.create(
...     Path("lecture.mp4"),
...     transcribe=True,
...     whisper_model="small",
... )
>>> docs = list(reader.get_documents())
"""  # noqa: D205, D400

from __future__ import annotations

from scikitplot.corpus._readers._alto import ALTOReader
from scikitplot.corpus._readers._audio import AudioReader
from scikitplot.corpus._readers._image import ImageReader
from scikitplot.corpus._readers._pdf import PDFReader

# Import order is intentional: each import triggers __init_subclass__
# which registers the class in DocumentReader._registry.
# All readers are imported unconditionally so the registry is complete
# whenever this package is imported.
from scikitplot.corpus._readers._text import MarkdownReader, ReSTReader, TextReader
from scikitplot.corpus._readers._video import VideoReader
from scikitplot.corpus._readers._web import WebReader, YouTubeReader
from scikitplot.corpus._readers._xml import TEIReader, XMLReader

__all__ = [  # noqa: RUF022
    # Text family
    "TextReader",
    "MarkdownReader",
    "ReSTReader",
    # XML family
    "XMLReader",
    "TEIReader",
    # Archive / structured document
    "ALTOReader",
    "PDFReader",
    # Media (OCR / transcription)
    "ImageReader",
    "VideoReader",
    "AudioReader",
    # URL-based
    "WebReader",
    "YouTubeReader",
]
