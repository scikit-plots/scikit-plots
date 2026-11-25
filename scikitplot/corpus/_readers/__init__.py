"""
scikitplot.corpus._readers
==========================
Format-specific document readers for the scikitplot corpus pipeline.

Importing this package registers all built-in readers in the
:class:`~scikitplot.corpus._base.DocumentReader` registry, making them
available via :meth:`~scikitplot.corpus._base.DocumentReader.create` (for
file-based sources) and
:meth:`~scikitplot.corpus._base.DocumentReader.from_url` (for URLs).

Available readers:


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

:class:`OfficeReader`
    Office documents: Office Open XML (``.docx``, ``.pptx``, ``.xlsx`` and
    their macro and template variants) and OpenDocument (``.odt``,
    ``.ods``, ``.odp``, ``.odg``, ``.odf``, templates and flat XML forms). Standard library only; nothing in the
    package is run.

:class:`ImageReader`
    Raster images (``.png``, ``.jpg``, ``.jpeg``, ``.gif``, ``.webp``,
    ``.tiff``, ``.tif``, ``.bmp``) via OCR.
    Backend selection is explicit: ``pytesseract`` by default,
    ``easyocr`` only when requested because it may download model weights.
    Multi-frame GIF/TIFF yields one chunk per frame.

:class:`VideoReader`
    Video files (``.mp4``, ``.avi``, ``.mkv``, ``.mov``, ``.webm``,
    ``.m4v``, ``.wmv``, ``.flv``).
    Companion subtitle detection (SRT/VTT/SBV/SUB) with zero dependencies.
    Optional Whisper transcription fallback (``faster-whisper`` or
    ``openai-whisper``).

:class:`SubtitleReader`
    Subtitle and caption files (``.srt``, ``.vtt``, ``.sbv``, ``.sub``,
    ``.ass``, ``.ssa``): timed cues, alone or grouped, with the speaker
    when named. The same parser reads audio and video companions.

:class:`AudioReader`
    Audio files (``.mp3``, ``.wav``, ``.flac``, ``.ogg``, ``.m4a``,
    ``.wma``, ``.aac``, ``.aiff``, ``.opus``, ``.wv``).
    Companion lyrics/transcript detection (LRC/SRT/VTT/TXT).
    Optional Whisper ASR and audio classification (animal sounds,
    instruments, environmental sounds).

:class:`CHMReader`
    Compiled HTML Help (``.chm``): offline help files and e-books. Pages
    are decompressed in pure Python and parsed as :class:`HTMLReader`
    parses a file, in table-of-contents order.

:class:`CodeReader`
    Source code and plain configuration (``.py``, ``.sh``, ``.js``,
    ``.c``, ``.sql``, ``.ini`` and about 70 more, see ``LANGUAGES``):
    blocks of whole lines with their language and line range.

:class:`CSVReader`
    Delimited tables (``.csv``, ``.tsv``): rows as tab-separated lines,
    ``rows_per_chunk`` a chunk, each chunk opening with the header row.

:class:`JSONReader`, :class:`YAMLReader`, :class:`TOMLReader`
    Structured data (``.json``, ``.jsonl``, ``.ndjson``, ``.yaml``,
    ``.yml``, ``.toml``): every value as a ``path: value`` line, grouped
    by top-level item. YAML needs PyYAML; TOML before Python 3.11 needs
    ``tomli``.

:class:`HTMLReader`
    Local HTML files (``.html``, ``.htm``, ``.xhtml``). Parsed offline
    exactly as :class:`WebReader` parses a fetched page; a ``<meta>``
    charset is honoured when the bytes are not UTF-8.

:class:`NotebookReader`
    Jupyter notebooks (``.ipynb``): one chunk per cell, markdown as text
    and code in the kernel's language; printed output on request.

:class:`MHTMLReader`
    Saved web pages (``.mhtml``, ``.mht``): every HTML part of the MIME
    message, page first, parsed as :class:`HTMLReader` parses a file.

**URL-based (via** ``DocumentReader.from_url(url)`` **)**

:class:`WebReader`
    Any ``http://`` / ``https://`` URL. Fetches HTML with ``requests``
    and extracts text sections via ``beautifulsoup4``.

:class:`YouTubeReader`
    YouTube video URLs. Retrieves transcript via ``youtube-transcript-api``.
    No audio download or model required.

:class:`ZipReader`
    Generic ZIP archives (``.zip``). Extracts members to a temporary
    directory and dispatches each by extension to the correct reader.
    Supports ``reader_kwargs`` for per-extension kwargs (e.g.
    ``{".mp3": {"transcribe": True}}``).  Overrides :class:`ALTOReader`
    for ``".zip"`` — use :class:`ALTOReader` directly for ALTO XML archives.

:class:`DummyReader`
    No-op reader registered as ``":dummy"``.  Validates source accessibility
    without producing documents.  Use :meth:`DummyReader.check` for batch
    pre-flight validation of file or URL sources.

Examples
--------
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

>>> reader = DocumentReader.from_url("https://youtu.be/rwPISgZcYIk")
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

# Import order is intentional: each import triggers __init_subclass__
# which registers the class in DocumentReader._registry.
# All readers are imported unconditionally so the registry is complete
# whenever this package is imported.
#
# IMPORTANT: _zip must be imported AFTER _alto so that ZipReader's ".zip"
# registration overwrites ALTOReader's ".zip" registration.  This makes
# DocumentReader.create("corpus.zip") dispatch to the generic ZipReader
# which handles any content mix.  To use ALTOReader directly, instantiate
# it explicitly: ALTOReader(input_path=Path("alto.zip")).
from .._base import DummyReader  # noqa: F401 — registers :dummy
from . import (
    _alto,
    _audio,
    _chm,
    _code,
    _custom,
    _image,
    _mhtml,
    _notebook,
    _office,
    _pdf,
    _structured,
    _subtitle,
    _table,
    _tar,
    _text,
    _video,
    _web,
    _whisper,
    _xml,
    _zip,
)
from ._alto import *  # noqa: F403
from ._audio import *  # noqa: F403
from ._chm import *  # noqa: F403
from ._code import *  # noqa: F403
from ._custom import *  # noqa: F403  # CustomReader + normalize_extractor_output
from ._image import *  # noqa: F403
from ._mhtml import *  # noqa: F403
from ._notebook import *  # noqa: F403
from ._office import *  # noqa: F403
from ._pdf import *  # noqa: F403
from ._structured import *  # noqa: F403
from ._subtitle import *  # noqa: F403
from ._table import *  # noqa: F403
from ._tar import *  # noqa: F403  # a ZipReader: after _zip
from ._text import *  # noqa: F403
from ._video import *  # noqa: F403
from ._web import *  # noqa: F403
from ._whisper import *  # noqa: F403  # public ASR customization contracts
from ._xml import *  # noqa: F403
from ._zip import *  # noqa: F403  # must come after ALTOReader

__all__ = ["DummyReader"]
__all__ += _alto.__all__
__all__ += _audio.__all__
__all__ += _chm.__all__
__all__ += _code.__all__
__all__ += _custom.__all__
__all__ += _image.__all__
__all__ += _mhtml.__all__
__all__ += _notebook.__all__
__all__ += _office.__all__
__all__ += _pdf.__all__
__all__ += _structured.__all__
__all__ += _subtitle.__all__
__all__ += _table.__all__
__all__ += _text.__all__
__all__ += _video.__all__
__all__ += _web.__all__
__all__ += _whisper.__all__
__all__ += _xml.__all__
__all__ += _zip.__all__
__all__ += _tar.__all__
