# scikitplot/corpus/_readers/_video.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.corpus._readers._video
=================================
Text extraction from video files via subtitle detection and/or
automatic speech recognition (transcription).

Supported formats: MP4, AVI, MKV, MOV, WEBM, M4V, WMV, FLV.

Extraction strategy
-------------------
Two strategies are attempted in order, stopping at the first success:

1. **Companion subtitle file** (zero dependencies, instant).
   The reader looks for a file with the same stem as the video but with
   a subtitle extension — in priority order:
   ``.srt`` > ``.vtt`` > ``.sbv`` > ``.sub``.
   If found, the subtitle is parsed and each cue is yielded as one chunk.

2. **Whisper transcription** (opt-in, requires ``openai-whisper`` or
   ``faster-whisper``).  Set ``transcribe=True`` to enable.
   ``faster-whisper`` is tried first (faster, lower VRAM); falls back to
   ``openai-whisper`` (reference implementation).

   .. warning::
      Whisper downloads model weights (~75 MB - 6 GB depending on size)
      on first use. In CI/Docker, pre-cache the weights or use subtitle
      files instead.

Format support details
-----------------------
``SRT``
    Sub Rip Text. Index line + HH:MM:SS,mmm timecodes + text block.
``WebVTT`` (``.vtt``)
    Web Video Text Tracks. ``WEBVTT`` header + dot-separated timecodes.
    Supports ``<b>``, ``<i>``, ``<ruby>`` HTML tags (stripped).
``SBV``
    YouTube's subtitle format. Comma-separated timecodes, no index lines.
``SUB``
    MicroDVD frame-based format — timecodes converted to seconds using
    the ``frame_rate`` parameter (default 25 fps).

Python compatibility
--------------------
Python 3.8-3.15. All transcription dependencies are optional lazy
imports. The subtitle parser is pure Python (stdlib only).
"""  # noqa: D205, D400

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Generator, List, Optional, Tuple  # noqa: F401

from scikitplot.corpus._base import DocumentReader
from scikitplot.corpus._schema import SectionType, SourceType

logger = logging.getLogger(__name__)


def _load_faster():
    """Lazily import faster_whisper and return the WhisperModel class.

    Returns
    -------
    type
        The ``faster_whisper.WhisperModel`` class.

    Raises
    ------
    ImportError
        If ``faster_whisper`` is not installed.
    """
    from faster_whisper import WhisperModel  # noqa: PLC0415

    return WhisperModel


def _load_openai():
    """Lazily import openai-whisper and return the module.

    Returns
    -------
    module
        The ``whisper`` module (openai-whisper).

    Raises
    ------
    ImportError
        If ``openai-whisper`` is not installed.
    """
    import whisper  # noqa: PLC0415

    return whisper


# ---------------------------------------------------------------------------
# Video extensions registered by this reader
# ---------------------------------------------------------------------------
_VIDEO_EXTENSIONS: list[str] = [
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".webm",
    ".m4v",
    ".wmv",
    ".flv",
]

# Priority order for companion subtitle files
_SUBTITLE_EXTENSIONS: tuple[str, ...] = (".srt", ".vtt", ".sbv", ".sub")

# Timecode regex: handles SRT (HH:MM:SS,mmm) and VTT (HH:MM:SS.mmm)
_TC_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
    r"\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
)

# SBV timecode: H:MM:SS.mmm,H:MM:SS.mmm
_SBV_TC_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})\.(\d{3}),(\d{1,2}):(\d{2}):(\d{2})\.(\d{3})"
)

# MicroDVD SUB: {start_frame}{end_frame}text
_SUB_RE = re.compile(r"\{(\d+)\}\{(\d+)\}(.*)")

# HTML tag stripper
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Whisper backend identifiers
_WHISPER_BACKEND_FASTER = "faster-whisper"
_WHISPER_BACKEND_OPENAI = "openai-whisper"


def _tc_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    """Convert timecode components to total seconds."""
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _parse_srt(content: str) -> list[dict[str, Any]]:
    """
    Parse SRT subtitle content into a list of cue dicts.

    Parameters
    ----------
    content : str
        Full SRT file contents.

    Returns
    -------
    list of dict
        Each dict has ``"text"``, ``"timecode_start"``, ``"timecode_end"``.
    """
    results = []
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        tc_match = None
        text_start = 0
        for i, line in enumerate(lines):
            m = _TC_RE.match(line.strip())
            if m:
                tc_match = m
                text_start = i + 1
                break
        if tc_match is None:
            continue
        start = _tc_to_seconds(*tc_match.groups()[:4])
        end = _tc_to_seconds(*tc_match.groups()[4:])
        text = " ".join(lines[text_start:]).strip()
        text = _HTML_TAG_RE.sub("", text).strip()
        if text:
            results.append({"text": text, "timecode_start": start, "timecode_end": end})
    return results


def _parse_vtt(content: str) -> list[dict[str, Any]]:
    """
    Parse WebVTT subtitle content. Strips the ``WEBVTT`` header line then
    delegates to the SRT parser (same timecode format, dot ms separator
    is already handled by ``_TC_RE``).

    Parameters
    ----------
    content : str
        Full VTT file contents.

    Returns
    -------
    list of dict
        See :func:`_parse_srt`.
    """  # noqa: D205
    lines = content.splitlines(keepends=True)
    # Drop WEBVTT header (first line, possibly with metadata)
    stripped = lines[1:] if lines and lines[0].startswith("WEBVTT") else lines
    return _parse_srt("".join(stripped))


def _parse_sbv(content: str) -> list[dict[str, Any]]:
    """
    Parse SBV (YouTube) subtitle content.

    Parameters
    ----------
    content : str
        Full SBV file contents.

    Returns
    -------
    list of dict
        Each dict has ``"text"``, ``"timecode_start"``, ``"timecode_end"``.
        See :func:`_parse_srt`.
    """
    results = []
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        m = _SBV_TC_RE.match(lines[0].strip())
        if not m:
            continue
        start = _tc_to_seconds(*m.groups()[:4])
        end = _tc_to_seconds(*m.groups()[4:])
        text = " ".join(lines[1:]).strip()
        text = _HTML_TAG_RE.sub("", text).strip()
        if text:
            results.append({"text": text, "timecode_start": start, "timecode_end": end})
    return results


def _parse_sub(content: str, frame_rate: float = 25.0) -> list[dict[str, Any]]:
    """
    Parse MicroDVD SUB subtitle content.

    Parameters
    ----------
    content : str
        Full SUB file contents.
    frame_rate : float
        Frames per second used to convert frame numbers to seconds.
        Default: 25.0.

    Returns
    -------
    list of dict
        Each dict has ``"text"``, ``"timecode_start"``, ``"timecode_end"``.
        See :func:`_parse_srt`.
    """
    results = []
    for line in content.splitlines():
        m = _SUB_RE.match(line.strip())
        if not m:
            continue
        start_f, end_f, text = m.group(1), m.group(2), m.group(3)
        start = int(start_f) / frame_rate
        end = int(end_f) / frame_rate
        # MicroDVD uses | as line separator
        text = text.replace("|", " ").strip()
        text = _HTML_TAG_RE.sub("", text).strip()
        if text:
            results.append({"text": text, "timecode_start": start, "timecode_end": end})
    return results


def _find_subtitle(video_path: Path) -> tuple[Path, str] | None:
    """
    Search for a companion subtitle file next to the video.

    Parameters
    ----------
    video_path : Path
        Path to the video file.

    Returns
    -------
    tuple of (Path, str) or None
        ``(subtitle_path, format_name)`` where ``format_name`` is one of
        ``"srt"``, ``"vtt"``, ``"sbv"``, ``"sub"``.
        ``None`` if no companion subtitle file found.
    """
    for ext in _SUBTITLE_EXTENSIONS:
        candidate = video_path.with_suffix(ext)
        if candidate.is_file():
            logger.info(
                "VideoReader: found companion subtitle %s for %s.",
                candidate.name,
                video_path.name,
            )
            return candidate, ext.lstrip(".")
    return None


def _parse_subtitle(path: Path, fmt: str, frame_rate: float) -> list[dict[str, Any]]:
    """
    Parse a subtitle file and return a list of cue dicts.

    Parameters
    ----------
    path : Path
        Path to the subtitle file.
    fmt : str
        Format identifier: ``"srt"``, ``"vtt"``, ``"sbv"``, or ``"sub"``.
    frame_rate : float
        Frames per second (only used for SUB format).

    Returns
    -------
    list of dict
    """
    # Try UTF-8 first; fall back to latin-1
    try:
        content = path.read_text("utf-8")
    except UnicodeDecodeError:
        content = path.read_text("latin-1")

    dispatch = {
        "srt": _parse_srt,
        "vtt": _parse_vtt,
        "sbv": _parse_sbv,
        "sub": lambda c: _parse_sub(c, frame_rate),
    }
    parser = dispatch.get(fmt, _parse_srt)
    return parser(content)


def _transcribe_whisper(
    video_path: Path,
    model_size: str,
    language: str | None,
) -> list[dict[str, Any]]:
    """
    Transcribe a video file using Whisper (faster-whisper → openai-whisper).

    Parameters
    ----------
    video_path : Path
        Path to the video/audio file. Whisper handles both directly.
    model_size : str
        Whisper model size: ``"tiny"``, ``"base"``, ``"small"``,
        ``"medium"``, ``"large"``, ``"large-v2"``, ``"large-v3"``.
        See ``pip install openai-whisper``.
    language : str or None
        ISO 639-1 language code hint (e.g. ``"en"``, ``"de"``). ``None``
        triggers automatic language detection.

    Returns
    -------
    list of dict
        Each dict has ``"text"``, ``"timecode_start"``, ``"timecode_end"``.
        ``source_type`` is intentionally absent here — it is set at the
        ``get_raw_chunks`` yield boundary so that the promoted key name and
        enum value are controlled in one place only.

    Raises
    ------
    ImportError
        If neither ``faster-whisper`` nor ``openai-whisper`` is installed.
    """
    # Try faster-whisper first
    try:
        # pip install faster-whisper
        from faster_whisper import WhisperModel  # noqa: PLC0415

        logger.info(
            "VideoReader: transcribing with faster-whisper (model=%s).", model_size
        )
        model = WhisperModel(model_size)
        segments, _info = model.transcribe(
            str(video_path),
            language=language,
        )
        results = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                results.append(
                    {
                        "text": text,
                        "timecode_start": round(seg.start, 3),
                        "timecode_end": round(seg.end, 3),
                    }
                )
        return results

    except ImportError:
        pass  # faster-whisper not installed; try openai-whisper

    # Try openai-whisper
    try:
        # pip install openai-whisper
        import whisper  # noqa: PLC0415

        logger.info(
            "VideoReader: transcribing with openai-whisper (model=%s).", model_size
        )
        model = whisper.load_model(model_size)
        result = model.transcribe(str(video_path), language=language)
        segments = result.get("segments", [])
        results = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if text:
                results.append(
                    {
                        "text": text,
                        "timecode_start": round(seg.get("start", 0.0), 3),
                        "timecode_end": round(seg.get("end", 0.0), 3),
                    }
                )
        return results

    except ImportError:
        pass

    raise ImportError(
        "VideoReader: transcribe=True requires either faster-whisper or"
        " openai-whisper.\n"
        "Install one of:\n"
        "  pip install faster-whisper   # recommended (faster, lower VRAM)\n"
        "  pip install openai-whisper   # reference implementation\n"
        "Or provide a companion subtitle file (.srt/.vtt/.sbv next to the video)"
        " to avoid transcription entirely."
    )


@dataclass
class VideoReader(DocumentReader):
    """
    Text extraction from video files via subtitle parsing and/or
    automatic speech recognition.

    Two extraction paths are available and attempted in order:

    1. **Companion subtitle** — zero-dependency, instant. The reader looks
       for ``.srt``, ``.vtt``, ``.sbv``, or ``.sub`` files with the same
       stem as the video. If found, the subtitle is parsed; Whisper is
       never invoked.

    2. **Whisper transcription** — opt-in. Enable with
       ``transcribe=True``. Requires ``faster-whisper`` or
       ``openai-whisper``.

    Parameters
    ----------
    input_file : pathlib.Path
        Path to the video file.
    transcribe : bool, optional
        When ``True``, fall back to Whisper transcription if no companion
        subtitle is found. When ``False`` (default), a missing subtitle
        file causes the reader to yield no chunks rather than error.
    whisper_model : str, optional
        Whisper model size. One of ``"tiny"``, ``"base"``, ``"small"``,
        ``"medium"``, ``"large"``, ``"large-v2"``, ``"large-v3"``.
        Smaller models are faster but less accurate. Default: ``"base"``.
    subtitle_frame_rate : float, optional
        Frames per second used to convert MicroDVD ``.sub`` frame numbers
        to seconds. Ignored for all other subtitle formats. Default: 25.0.
    max_file_bytes : int, optional
        Maximum file size in bytes. Files larger than this raise
        ``ValueError`` before processing. Default: 10 GB.
    chunker : ChunkerBase or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.
    filter_ : FilterBase or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.
    filename_override : str or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.
    default_language : str or None, optional
        ISO 639-1 language code. Used as language hint for Whisper when
        ``transcribe=True``. Default: ``None`` (auto-detect).

    Attributes
    ----------
    file_types : list of str
        Class variable. Registered extensions:
        ``[".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v", ".wmv", ".flv"]``.

    Raises
    ------
    ValueError
        If ``whisper_model`` is not a valid Whisper model size.
    ImportError
        If ``transcribe=True`` and neither Whisper variant is installed.

    See Also
    --------
    scikitplot.corpus._readers.ImageReader : Image OCR reader.
    scikitplot.corpus._readers.WebReader : Web page reader.

    Notes
    -----
    **Recommended workflow for offline/CI environments:**

    Pre-generate subtitle files and place them next to each video::

        $ whisper video.mp4 --output_format srt --output_dir .

    This keeps model weights out of the runtime path and makes the corpus
    pipeline fully reproducible without network access.

    **Chunk metadata keys:**

    Each yielded chunk dict contains:

    - ``"text"`` — subtitle cue or transcription segment text
    - ``"section_type"`` — always :attr:`SectionType.TEXT`
    - ``"timecode_start"`` — start time in seconds (float); promoted to
      :attr:`CorpusDocument.timecode_start`
    - ``"timecode_end"`` — end time in seconds (float); promoted to
      :attr:`CorpusDocument.timecode_end`
    - ``"source_type"`` — :attr:`SourceType.SUBTITLE` for subtitle cues,
      :attr:`SourceType.VIDEO` for Whisper transcription; promoted to
      :attr:`CorpusDocument.source_type`
    - ``"subtitle_format"`` — ``"srt"``, ``"vtt"``, ``"sbv"``, ``"sub"``,
      or ``None`` for transcription (goes to ``metadata``)
    - ``"transcript_type"`` — ``"whisper"`` for Whisper transcription,
      absent for subtitle cues (goes to ``metadata``)

    Examples
    --------
    Subtitle-only (no transcription):

    >>> from pathlib import Path
    >>> reader = VideoReader(input_file=Path("lecture.mp4"))
    >>> docs = list(reader.get_documents())
    >>> print(f"Subtitle cues: {len(docs)}")

    With Whisper fallback:

    >>> reader = VideoReader(
    ...     input_file=Path("interview.mp4"),
    ...     transcribe=True,
    ...     whisper_model="small",
    ...     default_language="en",
    ... )
    >>> docs = list(reader.get_documents())
    """  # noqa: D205

    file_types: ClassVar[list[str]] = _VIDEO_EXTENSIONS

    _VALID_WHISPER_MODELS: ClassVar[tuple[str, ...]] = (
        "tiny",
        "base",
        "small",
        "medium",
        "large",
        "large-v2",
        "large-v3",
    )

    transcribe: bool = field(default=False)
    """Enable Whisper fallback when no subtitle file is found."""

    whisper_model: str = field(default="base")
    """Whisper model size for transcription. Default: ``"base"``."""

    subtitle_frame_rate: float = field(default=25.0)
    """Frames per second for MicroDVD SUB format. Default: 25.0."""

    max_file_bytes: int = field(default=10 * 1024 * 1024 * 1024)
    """Maximum video file size. Default: 10 GB."""

    yield_frames: bool = field(default=False)
    """Include decoded video frames as raw tensors in output chunks.

    When ``True``, the reader extracts raw pixel data from video frames
    (requires ``opencv-python`` or equivalent) and sets ``modality`` to
    ``"video"`` (or ``"multimodal"`` when subtitle/transcript text is also
    present).  When ``False`` (default), only text is yielded and
    ``modality`` is ``"text"``.

    .. note::
        Frame extraction is compute-intensive.  Use ``yield_frames=True``
        only when the downstream model requires visual input alongside the
        transcript.  For text-only pipelines, leave this at ``False``.
    """

    def __post_init__(self) -> None:
        """Validate VideoReader fields and resolve subtitle/transcription strategy.

        Raises
        ------
        ValueError
            If ``whisper_model`` is not a recognised Whisper model size.
        """
        super().__post_init__()
        if self.whisper_model not in self._VALID_WHISPER_MODELS:
            raise ValueError(
                f"VideoReader: whisper_model must be one of"
                f" {self._VALID_WHISPER_MODELS}; got {self.whisper_model!r}."
            )
        if self.subtitle_frame_rate <= 0:
            raise ValueError(
                f"VideoReader: subtitle_frame_rate must be > 0;"
                f" got {self.subtitle_frame_rate!r}."
            )
        if self.max_file_bytes <= 0:
            raise ValueError(
                f"VideoReader: max_file_bytes must be > 0; got {self.max_file_bytes!r}."
            )

    # ------------------------------------------------------------------
    # DocumentReader contract
    # ------------------------------------------------------------------

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
        """
        Extract text from the video via subtitle or transcription.

        Attempts subtitle detection first. Falls back to Whisper only
        when ``transcribe=True`` and no subtitle was found.

        Yields
        ------
        dict
            Keys: ``"text"``, ``"section_type"``, ``"timecode_start"``,
            ``"timecode_end"``, ``"source_type"``, ``"subtitle_format"``.
            Transcription chunks also carry ``"transcript_type": "whisper"``.

        Raises
        ------
        ValueError
            If the file exceeds ``max_file_bytes``.
        ImportError
            If ``transcribe=True`` and Whisper is not installed.
        """
        file_size = self.input_file.stat().st_size
        if file_size > self.max_file_bytes:
            raise ValueError(
                f"VideoReader: {self.file_name} is {file_size:,} bytes,"
                f" which exceeds max_file_bytes={self.max_file_bytes:,}."
            )

        # --- Strategy 1: companion subtitle file ---
        subtitle_result = _find_subtitle(self.input_file)
        if subtitle_result is not None:
            sub_path, sub_fmt = subtitle_result
            cues = _parse_subtitle(sub_path, sub_fmt, self.subtitle_frame_rate)
            logger.info(
                "VideoReader: parsed %d cues from %s (%s).",
                len(cues),
                sub_path.name,
                sub_fmt.upper(),
            )
            for cue in cues:
                yield {
                    "text": cue["text"],
                    "section_type": SectionType.TEXT.value,
                    "timecode_start": cue[
                        "timecode_start"
                    ],  # promoted → CorpusDocument.timecode_start
                    "timecode_end": cue[
                        "timecode_end"
                    ],  # promoted → CorpusDocument.timecode_end
                    # promoted → CorpusDocument.source_type
                    "source_type": SourceType.SUBTITLE.value,
                    "subtitle_format": sub_fmt,  # non-promoted → metadata
                }
            return  # Done — do not fall through to transcription

        # --- Strategy 2: Whisper transcription (opt-in) ---
        if self.transcribe:
            logger.info(
                "VideoReader: no subtitle found for %s; transcribing with"
                " Whisper model=%r.",
                self.file_name,
                self.whisper_model,
            )
            segments = _transcribe_whisper(
                self.input_file,
                self.whisper_model,
                self.default_language,
            )
            logger.info(
                "VideoReader: transcription produced %d segments for %s.",
                len(segments),
                self.file_name,
            )
            for seg in segments:
                yield {
                    "text": seg["text"],
                    "section_type": SectionType.TEXT.value,
                    "timecode_start": seg[
                        "timecode_start"
                    ],  # promoted → CorpusDocument.timecode_start
                    "timecode_end": seg[
                        "timecode_end"
                    ],  # promoted → CorpusDocument.timecode_end
                    # promoted → CorpusDocument.source_type
                    "source_type": SourceType.VIDEO.value,
                    "subtitle_format": None,  # non-promoted → metadata
                    # non-promoted → metadata (sub-type detail)
                    "transcript_type": "whisper",
                }
            return

        # --- No subtitle, transcription disabled ---
        logger.warning(
            "VideoReader: no subtitle file found for %s and transcribe=False."
            " Yielding no chunks. To enable transcription, set transcribe=True.",
            self.file_name,
        )


__all__ = ["VideoReader"]
