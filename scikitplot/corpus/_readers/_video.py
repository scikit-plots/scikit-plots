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

Extraction strategy:

Two strategies are attempted in order, stopping at the first success:

1. **Companion subtitle file** (zero dependencies, instant).
   The reader looks for a file with the same stem as the video but with
   a subtitle extension — in priority order:
   ``.srt`` > ``.vtt`` > ``.sbv`` > ``.sub``.
   If found, the subtitle is parsed and each cue is yielded as one chunk.

2. **Whisper transcription** (opt-in, requires ``openai-whisper`` or
   ``faster-whisper``).  Set ``transcribe=True`` to enable.
   ``faster-whisper`` is tried first (faster, lower VRAM); import *and runtime*
   failures fall back to ``openai-whisper`` (reference implementation).
   With the default ``strict=False``, failure of both optional backends is
   logged and yields no transcription chunks instead of aborting ingestion.

   .. warning::
      Whisper downloads model weights (~75 MB - 6 GB depending on size)
      on first use. In CI/Docker, pre-cache the weights or use subtitle
      files instead.

Format support details:

``SRT``
    Sub Rip Text. Index line + HH:MM:SS,mmm timecodes + text block.
``WebVTT`` (``.vtt``)
    Web Video Text Tracks. ``WEBVTT`` header + dot-separated timecodes.
    Supports ``<b>``, ``<i>``, ``<ruby>`` HTML tags (stripped).
``SBV``
    YouTube's subtitle format. Comma-separated timecodes, no index lines.
``SUB``
    MicroDVD frame-based format — timecodes converted to seconds using
    ``subtitle_frame_rate``, else the file's own header, else 25 fps.
``ASS`` / ``SSA``
    Advanced SubStation Alpha ``Dialogue`` lines; ``Name`` is the speaker.

All formats are parsed by :mod:`scikitplot.corpus._readers._subtitle`,
which :class:`SubtitleReader` also uses for standalone files. A tagged
companion (``talk.en.srt``) is found too, in any letter case.

Python compatibility:

Python 3.8-3.15. All transcription dependencies are optional lazy
imports. The subtitle parser is pure Python (stdlib only).
"""  # noqa: D205, D400

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import (  # noqa: F401
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    List,
    Mapping,
    Optional,
    Tuple,
)

from .._backends import BackendPolicy
from .._base import DocumentReader
from .._capabilities import CapabilityRegistry
from .._schema import SectionType, SourceType
from . import _subtitle
from ._whisper import WHISPER_MODELS, ASRBackend, asr_segment_metadata
from ._whisper import plan_whisper_backends as _shared_plan_whisper_backends
from ._whisper import transcribe_whisper as _shared_transcribe_whisper

logger = logging.getLogger(__name__)

__all__ = ["VideoReader"]


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
_SUBTITLE_EXTENSIONS: tuple[str, ...] = (".srt", ".vtt", ".ass", ".ssa", ".sbv", ".sub")

# One parser for companions and standalone files (scikitplot.corpus._subtitle).
_parse_srt = _subtitle.parse_srt
_parse_vtt = _subtitle.parse_vtt
_parse_sbv = _subtitle.parse_sbv
_parse_sub = _subtitle.parse_sub
_tc_to_seconds = _subtitle._seconds


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
        ``(subtitle_path, format_name)``: ``talk.srt`` or a tagged
        ``talk.en.srt``, any case, in ``_SUBTITLE_EXTENSIONS`` order (see
        :func:`scikitplot.corpus._readers._subtitle.find_companion`);
        ``None`` if there is none.
    """
    found = _subtitle.find_companion(video_path, _SUBTITLE_EXTENSIONS)
    if found is None:
        return None
    logger.info(
        "VideoReader: found companion subtitle %s for %s.", found.name, video_path.name
    )
    return found, _subtitle.FORMATS[found.suffix.lower()]


def _parse_subtitle(
    path: Path, format: str, frame_rate: float | None = None
) -> list[dict[str, Any]]:
    """
    Parse a subtitle file with the shared subtitle reader.

    Decoding follows :class:`~scikitplot.corpus._readers.TextReader`. A
    file that cannot be read (binary, or UTF-16 without a byte-order mark)
    logs a warning and gives no cues, so the video falls back to
    transcription instead of failing.

    Returns
    -------
    list of dict
        ``text``, ``raw_text``, ``timecode_start``, ``timecode_end`` and,
        when named, ``speaker``.
    """
    try:
        return _subtitle.read_cues(path, format, frame_rate=frame_rate)[0]
    except ValueError as exc:
        logger.warning(
            "VideoReader: companion subtitle %s is unreadable: %s", path.name, exc
        )
        return []


def _transcribe_whisper(
    video_path: Path,
    model_size: str,
    language: str | None,
    *,
    strict: bool = False,
    report: Callable[[Any], None] | None = None,
    policy: BackendPolicy | str | dict[str, Any] | None = None,
    custom_backends: tuple[ASRBackend, ...] = (),
    capability_registry: CapabilityRegistry | None = None,
) -> list[dict[str, Any]]:
    """Transcribe video through the shared Whisper backend cascade."""
    return _shared_transcribe_whisper(
        video_path,
        model_size,
        language,
        component="VideoReader",
        logger=logger,
        strict=strict,
        include_confidence=False,
        report=report,
        policy=policy,
        custom_backends=custom_backends,
        capability_registry=capability_registry,
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
    input_path : pathlib.Path
        Path to the video file.
    transcribe : bool, optional
        When ``True``, fall back to Whisper transcription if no companion
        subtitle is found. When ``False`` (default), a missing subtitle
        file causes the reader to yield no chunks rather than error.
    whisper_model : str, optional
        Whisper model size. One of ``"tiny"``, ``"base"``, ``"small"``,
        ``"medium"``, ``"large"``, ``"large-v2"``, ``"large-v3"``.
        Smaller models are faster but less accurate. Default: ``"base"``.
    strict : bool, optional
        Whisper failure policy. ``False`` (default) logs backend failures,
        tries the next backend, and yields no ASR chunks if all backends fail.
        ``True`` raises after the fallback cascade is exhausted.
    subtitle_frame_rate : float or None, optional
        Frames per second used to convert MicroDVD ``.sub`` frame numbers
        to seconds; ``None`` (default) uses the file's own ``{1}{1}fps``
        header, else 25. Ignored for all other subtitle formats.
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
        If ``transcribe=True``, ``strict=True``, and neither Whisper backend
        can be imported.
    RuntimeError
        If ``transcribe=True``, ``strict=True``, and all available Whisper
        backends fail at runtime.

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
    >>> reader = VideoReader(input_path=Path("lecture.mp4"))
    >>> docs = list(reader.get_documents())
    >>> print(f"Subtitle cues: {len(docs)}")

    With Whisper fallback:

    >>> reader = VideoReader(
    ...     input_path=Path("interview.mp4"),
    ...     transcribe=True,
    ...     whisper_model="small",
    ...     default_language="en",
    ... )
    >>> docs = list(reader.get_documents())
    """  # noqa: D205

    file_type: ClassVar[str | None] = None
    file_types: ClassVar[list[str] | None] = _VIDEO_EXTENSIONS

    _VALID_WHISPER_MODELS: ClassVar[tuple[str, ...]] = WHISPER_MODELS

    transcribe: bool = field(default=False)
    """Enable Whisper fallback when no subtitle file is found."""

    whisper_model: str = field(default="base")
    """Whisper model size for transcription. Default: ``"base"``."""

    subtitle_frame_rate: float | None = field(default=None)
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

    strict: bool = field(default=False)
    """Raise when all Whisper backends fail instead of yielding no ASR chunks."""

    backend_policy: BackendPolicy | str | Mapping[str, Any] | None = field(
        default=None,
        repr=False,
    )
    """Optional backend policy preset/object/config mapping."""

    asr_backends: tuple[ASRBackend, ...] = field(default_factory=tuple, repr=False)
    """User-provided ASR backends appended to the built-in backend registry."""

    capability_registry: CapabilityRegistry | None = field(default=None, repr=False)
    """Optional private readiness registry for custom ASR capabilities."""

    # BUG-08/09 fix: custom_extractor and custom_extractor_kwargs are
    # inherited from DocumentReader. Redeclaring them here changed the
    # dataclass __init__ field order (subclass fields come after base fields
    # in MRO but re-declaration moved them to the subclass position). Removed.

    def __post_init__(self) -> None:
        """Validate VideoReader fields.

        Raises
        ------
        ValueError
            If ``whisper_model`` is not a recognised Whisper model size.
        TypeError
            If ``custom_extractor`` is not callable (and not ``None``).
        """
        super().__post_init__()
        if self.whisper_model not in self._VALID_WHISPER_MODELS:
            raise ValueError(
                f"VideoReader: whisper_model must be one of"
                f" {self._VALID_WHISPER_MODELS}; got {self.whisper_model!r}."
            )
        if self.custom_extractor is not None and not callable(self.custom_extractor):
            raise TypeError(
                f"VideoReader: custom_extractor must be callable or None; "
                f"got {type(self.custom_extractor).__name__!r}."
            )
        if self.subtitle_frame_rate is not None and self.subtitle_frame_rate <= 0:
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

    def _custom_extractor_source_type(self) -> SourceType:
        """Return :attr:`~scikitplot.corpus._schema.SourceType.VIDEO` for custom-extractor chunks."""
        return SourceType.VIDEO

    def plan_asr_backends(self):
        """Return the exact side-effect-free ASR backend preflight plan."""
        return _shared_plan_whisper_backends(
            self.input_path,
            self.whisper_model,
            self.default_language,
            component="VideoReader",
            strict=self.strict,
            include_confidence=False,
            policy=self.backend_policy,
            custom_backends=self.asr_backends,
            capability_registry=self.capability_registry,
        )

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
            If ``transcribe=True``, ``strict=True``, and neither Whisper backend
            can be imported.
        RuntimeError
            If ``transcribe=True``, ``strict=True``, and all available Whisper
            backends fail at runtime.
        """
        file_size = self.input_path.stat().st_size
        if file_size > self.max_file_bytes:
            raise ValueError(
                f"VideoReader: {self.file_name} is {file_size:,} bytes,"
                f" which exceeds max_file_bytes={self.max_file_bytes:,}."
            )

        # ── Strategy 0: custom extractor (highest priority) ───────────
        if self.custom_extractor is not None:
            yield from self._iter_custom_extractor_chunks()
            return

        # --- Strategy 1: companion subtitle file ---
        subtitle_result = _find_subtitle(self.input_path)
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
                chunk = {
                    "text": cue["text"],
                    # raw_text: the cue's lines as written, markup kept.
                    "raw_text": cue.get("raw_text", cue["text"]),
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
                if cue.get("speaker"):
                    chunk["speaker"] = cue["speaker"]
                yield chunk
            if cues:
                return  # Done — do not fall through to transcription
            logger.warning(
                "VideoReader: companion subtitle %s gave no cues; trying"
                " transcription if enabled.",
                sub_path.name,
            )

        # --- Strategy 2: Whisper transcription (opt-in) ---
        if self.transcribe:
            logger.info(
                "VideoReader: no subtitle found for %s; transcribing with"
                " Whisper model=%r.",
                self.file_name,
                self.whisper_model,
            )
            segments = _transcribe_whisper(
                self.input_path,
                self.whisper_model,
                self.default_language,
                strict=self.strict,
                report=self._record_backend_outcome,
                policy=self.backend_policy,
                custom_backends=self.asr_backends,
                capability_registry=self.capability_registry,
            )
            logger.info(
                "VideoReader: transcription produced %d segments for %s.",
                len(segments),
                self.file_name,
            )
            for seg in segments:
                chunk = {
                    "text": seg["text"],
                    # raw_text: ASR transcription — verbatim Whisper output.
                    "raw_text": seg.get("raw_text", seg["text"]),
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
                if "confidence" in seg:
                    chunk["confidence"] = seg["confidence"]
                if seg.get("asr_backend") is not None:
                    chunk["asr_backend"] = seg["asr_backend"]
                extra_asr_metadata = asr_segment_metadata(seg)
                if extra_asr_metadata:
                    chunk["asr_metadata"] = extra_asr_metadata
                yield chunk
            return

        # --- No subtitle, transcription disabled ---
        logger.warning(
            "VideoReader: no subtitle file found for %s and transcribe=False."
            " Yielding no chunks. To enable transcription, set transcribe=True.",
            self.file_name,
        )
