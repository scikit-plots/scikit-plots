# scikitplot/corpus/_readers/_audio.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.corpus._readers.audio
=================================
Text extraction from audio files via companion transcript/lyrics detection,
automatic speech recognition (Whisper), and optional audio classification.

Supported formats: MP3, WAV, FLAC, OGG, M4A, WMA, AAC, AIFF, OPUS, WV.

Extraction strategy
-------------------
Three strategies are attempted in order, stopping at the first success:

1. **Companion transcript / lyrics file** (zero dependencies, instant).
   The reader looks for a file with the same stem as the audio but with
   a transcript or lyrics extension — in priority order:
   ``.lrc`` > ``.srt`` > ``.vtt`` > ``.txt``.
   If found, the file is parsed and each cue/line is yielded as one chunk.

2. **Whisper transcription** (opt-in, requires ``openai-whisper`` or
   ``faster-whisper``). Set ``transcribe=True`` to enable.
   ``faster-whisper`` is tried first (faster, lower VRAM); falls back to
   ``openai-whisper`` (reference implementation).

3. **Audio classification** (opt-in, requires a user-supplied callable
   or a ``librosa``-backed feature extractor). Set ``classify=True`` and
   provide ``classifier`` to enable.
   Produces labelled chunks with ``audio_label`` and ``confidence`` for
   matching non-speech audio (animal sounds, instruments, environmental
   sounds) against text corpora.

   .. warning::
      Whisper downloads model weights (~75 MB - 6 GB depending on size)
      on first use. In CI/Docker, pre-cache the weights or use companion
      transcript files instead.

Companion file formats
----------------------
``LRC``
    Timestamped lyrics format. ``[MM:SS.xx]`` or ``[HH:MM:SS.xx]``
    timecodes per line. Supports enhanced LRC with word-level timestamps
    ``<MM:SS.xx>`` (inline tags stripped, line-level timecodes preserved).
``SRT`` / ``VTT``
    Subtitle formats — same parsers as :class:`VideoReader`.
    Useful when audio is extracted from video.
``TXT``
    Plain-text transcript. No timecodes; entire file yielded as one
    chunk per line (or as a single chunk if ``txt_as_single_chunk=True``).

Scenario support
----------------
**Scenario 11 — Beethoven MP3 + Music Notes Book**
    Reads Beethoven MP3 files via Whisper ASR or companion ``.lrc`` files.
    Each segment carries ``timecode_start`` / ``timecode_end`` for temporal
    alignment against book text. ``source_type = SourceType.AUDIO``.
    ``SimilarityIndex`` performs STRICT / KEYWORD / SEMANTIC matching
    between audio-derived text and music-notes book text.

**Scenario 12 — Animal Sounds + Children's Book (Bremen)**
    Reads animal sound files via audio classification (bird, donkey, cat,
    dog, rooster). Each chunk carries ``audio_label`` in ``metadata`` and
    a text description (e.g. ``"bird call"``). ``SimilarityIndex`` matches
    ``audio_label`` or generated text against children's book text via
    KEYWORD or SEMANTIC match.

Python compatibility
--------------------
Python 3.8-3.15. All audio dependencies are optional lazy imports.
The LRC parser is pure Python (stdlib only).
"""  # noqa: D205, D400

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,  # noqa: F401
    Generator,
    List,  # noqa: F401
    Optional,  # noqa: F401
    Tuple,  # noqa: F401
)

from scikitplot.corpus._base import DocumentReader
from scikitplot.corpus._schema import SectionType, SourceType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audio extensions registered by this reader
# ---------------------------------------------------------------------------
_AUDIO_EXTENSIONS: list[str] = [
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".wma",
    ".aac",
    ".aiff",
    ".opus",
    ".wv",
]

# Priority order for companion transcript / lyrics files
_COMPANION_EXTENSIONS: tuple[str, ...] = (".lrc", ".srt", ".vtt", ".txt")

# ---------------------------------------------------------------------------
# LRC timestamp patterns
# ---------------------------------------------------------------------------
# Standard LRC: [MM:SS.xx] or [MM:SS.xxx]
_LRC_LINE_RE = re.compile(r"\[(\d{1,2}):(\d{2})(?:[.:](\d{2,3}))?\]\s*(.*)")

# Extended LRC: [HH:MM:SS.xx]
_LRC_LINE_EXT_RE = re.compile(r"\[(\d{1,2}):(\d{2}):(\d{2})(?:[.:](\d{2,3}))?\]\s*(.*)")

# LRC metadata tags: [ti:Title], [ar:Artist], [al:Album], etc.
_LRC_META_RE = re.compile(r"\[([a-zA-Z]{2,}):\s*(.*?)\s*\]")

# Enhanced LRC word-level inline tags: <MM:SS.xx>word
_LRC_INLINE_TAG_RE = re.compile(r"<\d{1,2}:\d{2}(?:[.:]\d{2,3})?>")

# SRT timecode: HH:MM:SS,mmm --> HH:MM:SS,mmm
_SRT_TC_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
    r"\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})"
)

# HTML tag stripper (for VTT formatting tags)
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# VTT header
_VTT_HEADER = "WEBVTT"

# Whisper backend identifiers
_WHISPER_BACKEND_FASTER = "faster-whisper"
_WHISPER_BACKEND_OPENAI = "openai-whisper"


# =========================================================================
# Companion file detection
# =========================================================================


def _find_companion(audio_path: Path) -> tuple[Path, str] | None:
    """
    Find a companion transcript or lyrics file for the given audio file.

    Parameters
    ----------
    audio_path : pathlib.Path
        Path to the audio file.

    Returns
    -------
    tuple of (pathlib.Path, str) or None
        ``(companion_path, format_name)`` if found; ``None`` otherwise.
        ``format_name`` is one of ``"lrc"``, ``"srt"``, ``"vtt"``, ``"txt"``.
    """
    stem = audio_path.stem
    parent = audio_path.parent
    for ext in _COMPANION_EXTENSIONS:
        candidate = parent / (stem + ext)
        if candidate.is_file():
            fmt = ext.lstrip(".")
            logger.debug(
                "AudioReader: found companion file %s (format=%s).",
                candidate.name,
                fmt,
            )
            return candidate, fmt
    return None


# =========================================================================
# LRC parser
# =========================================================================


def _lrc_ts_to_seconds(
    minutes: str,
    seconds: str,
    centis: str | None,
    hours: str | None = None,
) -> float:
    """
    Convert LRC timestamp components to seconds.

    Parameters
    ----------
    minutes : str
        Minutes component (or hours if ``hours`` is provided).
    seconds : str
        Seconds component.
    centis : str or None
        Centisecond (2-digit) or millisecond (3-digit) component.
    hours : str or None, optional
        Hours component for extended LRC format.

    Returns
    -------
    float
        Timestamp in seconds.
    """
    h = int(hours) if hours else 0
    m = int(minutes)
    s = int(seconds)
    if centis is not None:
        if len(centis) == 2:  # noqa: PLR2004, SIM108
            frac = int(centis) / 100.0
        else:
            frac = int(centis) / 1000.0
    else:
        frac = 0.0
    return h * 3600.0 + m * 60.0 + s + frac


def _parse_lrc(  # noqa: PLR0912
    lrc_path: Path,
    encoding: str = "utf-8",
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """
    Parse an LRC (timestamped lyrics) file.

    Parameters
    ----------
    lrc_path : pathlib.Path
        Path to the ``.lrc`` file.
    encoding : str, optional
        File encoding. Default: ``"utf-8"``.

    Returns
    -------
    cues : list of dict
        Each dict has keys: ``"text"``, ``"timecode_start"``,
        ``"timecode_end"`` (estimated from next cue), ``"line_index"``.
    lrc_meta : dict of str to str
        LRC metadata tags (``ti`` → title, ``ar`` → artist, ``al`` →
        album, ``by`` → LRC author, ``offset`` → timing offset ms).

    Notes
    -----
    ``timecode_end`` for each cue is estimated as the ``timecode_start``
    of the next cue. The last cue has ``timecode_end = timecode_start + 5.0``
    as a reasonable default. This is the standard LRC convention since the
    format does not include explicit end times.

    Enhanced LRC word-level inline tags ``<MM:SS.xx>`` are stripped from
    the text since word-level timing is below the chunk granularity of this
    reader. The line-level timestamp is preserved.
    """
    try:
        raw = lrc_path.read_text(encoding)
    except UnicodeDecodeError:
        raw = lrc_path.read_text("latin-1")

    lrc_meta: dict[str, str] = {}
    raw_cues: list[tuple[float, str]] = []

    for line in raw.splitlines():
        line = line.strip()  # noqa: PLW2901
        if not line:
            continue

        # Check for metadata tags first
        meta_match = _LRC_META_RE.match(line)
        if meta_match and not _LRC_LINE_RE.match(line):
            tag, value = meta_match.group(1).lower(), meta_match.group(2)
            lrc_meta[tag] = value
            continue

        # Try extended format [HH:MM:SS.xx] first
        ext_match = _LRC_LINE_EXT_RE.match(line)
        if ext_match:
            hours_str = ext_match.group(1)
            mins_str = ext_match.group(2)
            secs_str = ext_match.group(3)
            centis_str = ext_match.group(4)
            text = ext_match.group(5)
            ts = _lrc_ts_to_seconds(mins_str, secs_str, centis_str, hours_str)
            # Strip enhanced LRC inline tags
            text = _LRC_INLINE_TAG_RE.sub("", text).strip()
            if text:
                raw_cues.append((ts, text))
            continue

        # Try standard format [MM:SS.xx]
        std_match = _LRC_LINE_RE.match(line)
        if std_match:
            mins_str = std_match.group(1)
            secs_str = std_match.group(2)
            centis_str = std_match.group(3)
            text = std_match.group(4)
            ts = _lrc_ts_to_seconds(mins_str, secs_str, centis_str)
            # Strip enhanced LRC inline tags
            text = _LRC_INLINE_TAG_RE.sub("", text).strip()
            if text:
                raw_cues.append((ts, text))

    # Apply offset if present
    offset_ms = 0.0
    if "offset" in lrc_meta:
        try:
            offset_ms = float(lrc_meta["offset"])
        except ValueError:
            logger.warning(
                "AudioReader: invalid LRC offset value %r; ignoring.",
                lrc_meta["offset"],
            )

    offset_s = offset_ms / 1000.0

    # Sort by timestamp and build cue dicts with estimated end times
    raw_cues.sort(key=lambda c: c[0])
    cues: list[dict[str, Any]] = []
    for i, (ts, text) in enumerate(raw_cues):
        adjusted_ts = ts + offset_s
        adjusted_ts = max(adjusted_ts, 0.0)
        # Estimate end time from next cue's start
        if i + 1 < len(raw_cues):
            end_ts = raw_cues[i + 1][0] + offset_s
            end_ts = max(end_ts, 0.0)
        else:
            # Last cue: assume 5 seconds duration
            end_ts = adjusted_ts + 5.0
        cues.append(
            {
                "text": text,
                "timecode_start": round(adjusted_ts, 3),
                "timecode_end": round(end_ts, 3),
                "line_index": i,
            }
        )

    return cues, lrc_meta


# =========================================================================
# SRT / VTT parser (reused from VideoReader conventions)
# =========================================================================


def _tc_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    """
    Convert ``HH:MM:SS,mmm`` components to seconds.

    Parameters
    ----------
    h, m, s, ms : str
        Hours, minutes, seconds, milliseconds as strings.

    Returns
    -------
    float
        Timestamp in seconds.
    """
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _parse_srt(srt_path: Path) -> list[dict[str, Any]]:
    """
    Parse an SRT subtitle file.

    Parameters
    ----------
    srt_path : pathlib.Path
        Path to the ``.srt`` file.

    Returns
    -------
    list of dict
        Each dict has keys: ``"text"``, ``"timecode_start"``,
        ``"timecode_end"``.
    """
    try:
        raw = srt_path.read_text("utf-8-sig")
    except UnicodeDecodeError:
        raw = srt_path.read_text("latin-1")

    cues: list[dict[str, Any]] = []
    blocks = re.split(r"\n\s*\n", raw.strip())

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:  # noqa: PLR2004
            continue

        # Find the timecode line
        tc_match = None
        text_start = 0
        for i, line in enumerate(lines):
            tc_match = _SRT_TC_RE.search(line)
            if tc_match:
                text_start = i + 1
                break
        if tc_match is None:
            continue

        text = " ".join(lines[text_start:]).strip()
        if not text:
            continue

        cues.append(
            {
                "text": text,
                "timecode_start": _tc_to_seconds(
                    tc_match.group(1),
                    tc_match.group(2),
                    tc_match.group(3),
                    tc_match.group(4),
                ),
                "timecode_end": _tc_to_seconds(
                    tc_match.group(5),
                    tc_match.group(6),
                    tc_match.group(7),
                    tc_match.group(8),
                ),
            }
        )

    return cues


def _parse_vtt(vtt_path: Path) -> list[dict[str, Any]]:
    """
    Parse a WebVTT file.

    Parameters
    ----------
    vtt_path : pathlib.Path
        Path to the ``.vtt`` file.

    Returns
    -------
    list of dict
        Each dict has keys: ``"text"``, ``"timecode_start"``,
        ``"timecode_end"``.
    """
    try:
        raw = vtt_path.read_text("utf-8-sig")
    except UnicodeDecodeError:
        raw = vtt_path.read_text("latin-1")

    # Skip header
    lines = raw.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith(_VTT_HEADER):
            start_idx = i + 1
            break
    raw = "\n".join(lines[start_idx:])

    cues: list[dict[str, Any]] = []
    blocks = re.split(r"\n\s*\n", raw.strip())

    for block in blocks:
        blines = block.strip().splitlines()
        if not blines:
            continue

        tc_match = None
        text_start = 0
        for i, line in enumerate(blines):
            tc_match = _SRT_TC_RE.search(line)
            if tc_match:
                text_start = i + 1
                break
        if tc_match is None:
            continue

        text = " ".join(blines[text_start:]).strip()
        # Strip HTML formatting tags
        text = _HTML_TAG_RE.sub("", text).strip()
        if not text:
            continue

        cues.append(
            {
                "text": text,
                "timecode_start": _tc_to_seconds(
                    tc_match.group(1),
                    tc_match.group(2),
                    tc_match.group(3),
                    tc_match.group(4),
                ),
                "timecode_end": _tc_to_seconds(
                    tc_match.group(5),
                    tc_match.group(6),
                    tc_match.group(7),
                    tc_match.group(8),
                ),
            }
        )

    return cues


def _parse_txt_companion(
    txt_path: Path,
    as_single_chunk: bool = False,
) -> list[dict[str, Any]]:
    """
    Parse a plain-text companion transcript.

    Parameters
    ----------
    txt_path : pathlib.Path
        Path to the ``.txt`` file.
    as_single_chunk : bool, optional
        When ``True``, the entire file is yielded as one chunk.
        When ``False`` (default), each non-empty line is a separate chunk.

    Returns
    -------
    list of dict
        Each dict has keys: ``"text"``, ``"line_index"``.
        Plain-text companions have no timecodes.
    """
    try:
        raw = txt_path.read_text("utf-8")
    except UnicodeDecodeError:
        raw = txt_path.read_text("latin-1")

    if as_single_chunk:
        text = raw.strip()
        if text:
            return [{"text": text, "line_index": 0}]
        return []

    cues: list[dict[str, Any]] = []
    for i, line in enumerate(raw.splitlines()):
        text = line.strip()
        if text:
            cues.append({"text": text, "line_index": i})
    return cues


def _parse_companion(
    companion_path: Path,
    fmt: str,
    txt_as_single_chunk: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """
    Dispatch companion file parsing to the appropriate format parser.

    Parameters
    ----------
    companion_path : pathlib.Path
        Path to the companion file.
    fmt : str
        Format identifier: ``"lrc"``, ``"srt"``, ``"vtt"``, or ``"txt"``.
    txt_as_single_chunk : bool, optional
        Passed through to ``_parse_txt_companion``. Default: ``False``.

    Returns
    -------
    cues : list of dict
        Parsed cues/lines.
    companion_meta : dict of str to str
        Metadata extracted from companion file (LRC tags only; empty
        for SRT/VTT/TXT).

    Raises
    ------
    ValueError
        If ``fmt`` is not a supported format.
    """
    if fmt == "lrc":
        return _parse_lrc(companion_path)
    if fmt == "srt":
        return _parse_srt(companion_path), {}
    if fmt == "vtt":
        return _parse_vtt(companion_path), {}
    if fmt == "txt":
        return (
            _parse_txt_companion(companion_path, as_single_chunk=txt_as_single_chunk),
            {},
        )
    raise ValueError(
        f"AudioReader: unsupported companion format {fmt!r}."
        f" Supported: lrc, srt, vtt, txt."
    )


# =========================================================================
# Whisper ASR transcription
# =========================================================================


def _transcribe_whisper(
    audio_path: Path,
    model_size: str,
    language: str | None,
) -> list[dict[str, Any]]:
    """
    Transcribe an audio file using Whisper ASR.

    Parameters
    ----------
    audio_path : pathlib.Path
        Path to the audio file.
    model_size : str
        Whisper model size (``"tiny"``, ``"base"``, ``"small"``, etc.).
    language : str or None
        ISO 639-1 language hint. ``None`` = auto-detect.

    Returns
    -------
    list of dict
        Each dict has keys: ``"text"``, ``"timecode_start"``,
        ``"timecode_end"``, ``"confidence"`` (when available).

    Raises
    ------
    ImportError
        If neither ``faster-whisper`` nor ``openai-whisper`` is installed.

    Notes
    -----
    ``faster-whisper`` is tried first (CTranslate2-based, lower VRAM,
    faster inference). Falls back to ``openai-whisper`` (reference
    implementation, PyTorch-based).

    **User-facing note:** For batch processing, pre-generate transcript
    files offline and place them next to the audio files::

        $ whisper audio.mp3 --output_format srt --output_dir .

    This avoids downloading model weights at corpus-build time.
    """
    # Try faster-whisper first
    try:
        from faster_whisper import WhisperModel  # noqa: PLC0415

        logger.info(
            "AudioReader: transcribing with faster-whisper (model=%s).",
            model_size,
        )
        model = WhisperModel(model_size, device="auto")
        segments, _info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
        )
        result: list[dict[str, Any]] = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                chunk: dict[str, Any] = {
                    "text": text,
                    "timecode_start": round(seg.start, 3),
                    "timecode_end": round(seg.end, 3),
                }
                if hasattr(seg, "avg_logprob"):
                    # Convert log probability to [0, 1] confidence
                    import math  # noqa: PLC0415

                    chunk["confidence"] = round(math.exp(seg.avg_logprob), 4)
                result.append(chunk)
        return result
    except ImportError:
        pass  # faster-whisper not installed; try openai-whisper

    # Try openai-whisper
    try:
        import whisper  # noqa: PLC0415

        logger.info(
            "AudioReader: transcribing with openai-whisper (model=%s).",
            model_size,
        )
        model = whisper.load_model(model_size)
        wresult = model.transcribe(str(audio_path), language=language)
        result = []
        for seg in wresult.get("segments", []):
            text = seg.get("text", "").strip()
            if text:
                chunk = {
                    "text": text,
                    "timecode_start": round(seg["start"], 3),
                    "timecode_end": round(seg["end"], 3),
                }
                if "avg_logprob" in seg:
                    import math  # noqa: PLC0415

                    chunk["confidence"] = round(math.exp(seg["avg_logprob"]), 4)
                result.append(chunk)
        return result
    except ImportError:
        pass  # openai-whisper not installed

    raise ImportError(
        "AudioReader: transcribe=True requires either faster-whisper or"
        " openai-whisper.\n"
        "Install one of:\n"
        "  pip install faster-whisper   # recommended (faster, lower VRAM)\n"
        "  pip install openai-whisper   # reference implementation\n"
    )


# =========================================================================
# Audio classification (for non-speech: animal sounds, instruments, etc.)
# =========================================================================


def _classify_audio(
    audio_path: Path,
    classifier: Callable[..., list[dict[str, Any]]],
    segment_duration: float,
    overlap_duration: float,
) -> list[dict[str, Any]]:
    """
    Classify audio segments using a user-provided classifier callable.

    Parameters
    ----------
    audio_path : pathlib.Path
        Path to the audio file.
    classifier : callable
        A callable with signature::

            classifier(audio_path: Path, offset: float, duration: float)
                -> list[dict[str, Any]]

        Must return a list of dicts, each with at least:
        ``"label"`` (str), ``"confidence"`` (float in [0, 1]).
        May optionally include ``"text"`` (str) for a text description.
    segment_duration : float
        Duration in seconds of each classification window.
    overlap_duration : float
        Overlap in seconds between consecutive windows.

    Returns
    -------
    list of dict
        Each dict has keys: ``"text"``, ``"timecode_start"``,
        ``"timecode_end"``, ``"confidence"``, ``"audio_label"``.

    Raises
    ------
    ImportError
        If ``librosa`` or ``soundfile`` is not installed (needed for
        duration detection).

    Notes
    -----
    **Developer note:** The ``classifier`` callable is the user's
    responsibility. This reader provides the windowing, timecoding, and
    integration with the corpus pipeline. Example classifiers:

    - A ``transformers`` ``pipeline("audio-classification")`` wrapper
    - A ``librosa``-based MFCC + scikit-learn model
    - A pre-trained YAMNet / VGGish / PANNs model wrapper

    The callable receives the full file path plus offset/duration so it
    can load only the required audio segment.

    **Scenario 12 example (animal sounds):**

    >>> def animal_classifier(path, offset, duration):
    ...     # Your classification logic here
    ...     return [
    ...         {"label": "bird", "confidence": 0.92, "text": "bird call (sparrow)"}
    ...     ]
    """
    # Get audio duration
    total_duration = _get_audio_duration(audio_path)
    if total_duration is None or total_duration <= 0:
        logger.warning(
            "AudioReader: could not determine duration of %s; skipping classification.",
            audio_path.name,
        )
        return []

    step = segment_duration - overlap_duration
    if step <= 0:
        raise ValueError(
            f"AudioReader: segment_duration ({segment_duration}) must be"
            f" greater than overlap_duration ({overlap_duration})."
        )

    result: list[dict[str, Any]] = []
    offset = 0.0
    while offset < total_duration:
        dur = min(segment_duration, total_duration - offset)
        if dur < 0.1:  # noqa: PLR2004
            break  # Skip negligible trailing segments

        try:
            labels = classifier(audio_path, offset, dur)
        except Exception:
            logger.warning(
                "AudioReader: classifier raised an exception for"
                " %s at offset=%.2f; skipping segment.",
                audio_path.name,
                offset,
                exc_info=True,
            )
            offset += step
            continue

        if not labels:
            offset += step
            continue

        # Take top prediction
        top = max(labels, key=lambda x: x.get("confidence", 0.0))
        label_str = str(top.get("label", "unknown"))
        conf = float(top.get("confidence", 0.0))
        text = str(top.get("text", label_str))

        result.append(
            {
                "text": text,
                "timecode_start": round(offset, 3),
                "timecode_end": round(offset + dur, 3),
                "confidence": round(conf, 4),
                "audio_label": label_str,
            }
        )

        offset += step

    return result


def _get_audio_duration(audio_path: Path) -> float | None:
    """
    Get the duration of an audio file in seconds.

    Parameters
    ----------
    audio_path : pathlib.Path
        Path to the audio file.

    Returns
    -------
    float or None
        Duration in seconds, or ``None`` if detection fails.

    Notes
    -----
    Tries backends in order: ``mutagen`` (metadata-only, fast) →
    ``librosa`` (loads audio, slower) → ``soundfile`` (loads header).
    Returns ``None`` without error if no backend is available.
    """
    # Try mutagen (fast, metadata-only)
    try:
        import mutagen  # type: ignore[] # noqa: PLC0415

        audio = mutagen.File(str(audio_path))
        if audio is not None and audio.info is not None:
            return float(audio.info.length)
    except Exception:  # noqa: BLE001
        logger.debug(
            "AudioReader: mutagen failed for %s; trying next backend.",
            audio_path.name,
        )

    # Try librosa
    try:
        import librosa  # type: ignore[] # noqa: PLC0415

        return float(librosa.get_duration(path=str(audio_path)))
    except Exception:  # noqa: BLE001
        logger.debug(
            "AudioReader: librosa failed for %s; trying next backend.",
            audio_path.name,
        )

    # Try soundfile
    try:
        import soundfile as sf  # type: ignore[] # noqa: PLC0415

        info = sf.info(str(audio_path))
        return float(info.duration)
    except Exception:  # noqa: BLE001
        logger.debug(
            "AudioReader: soundfile failed for %s.",
            audio_path.name,
        )

    logger.debug(
        "AudioReader: could not detect duration of %s. "
        "Install mutagen, librosa, or soundfile for duration detection.",
        audio_path.name,
    )
    return None


# =========================================================================
# Audio feature extraction (for embedding-based matching)
# =========================================================================


def _extract_audio_features(
    audio_path: Path,
    offset: float,
    duration: float | None,
    sr: int = 22050,
    n_mfcc: int = 13,
) -> dict[str, Any]:
    """
    Extract audio features using librosa for embedding-based matching.

    Parameters
    ----------
    audio_path : pathlib.Path
        Path to the audio file.
    offset : float
        Start time in seconds.
    duration : float or None
        Duration to load in seconds. ``None`` = load to end.
    sr : int, optional
        Target sample rate. Default: 22050.
    n_mfcc : int, optional
        Number of MFCC coefficients. Default: 13.

    Returns
    -------
    dict
        Keys: ``"mfcc_mean"`` (list of float), ``"chroma_mean"``
        (list of float), ``"spectral_centroid"`` (float),
        ``"sample_rate"`` (int), ``"rms_energy"`` (float).
        Empty dict if ``librosa`` is not installed.

    Notes
    -----
    These features are stored in ``CorpusDocument.metadata`` and can be
    used for audio similarity matching. For Scenario 11 (Beethoven),
    chroma features capture harmonic content that correlates with musical
    notation in the book text.
    """
    try:
        import librosa  # type: ignore[] # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError:
        logger.debug(
            "AudioReader: librosa not installed; skipping audio feature extraction."
        )
        return {}

    try:
        y, sr_actual = librosa.load(
            str(audio_path),
            sr=sr,
            offset=offset,
            duration=duration,
        )
    except Exception:
        logger.warning(
            "AudioReader: failed to load audio from %s at offset=%.2f;"
            " skipping feature extraction.",
            audio_path.name,
            offset,
            exc_info=True,
        )
        return {}

    if len(y) == 0:
        return {}

    features: dict[str, Any] = {"sample_rate": sr_actual}

    # MFCCs — timbral fingerprint
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr_actual, n_mfcc=n_mfcc)
        features["mfcc_mean"] = [round(float(x), 6) for x in np.mean(mfcc, axis=1)]
    except Exception:  # noqa: BLE001
        logger.debug(
            "AudioReader: MFCC extraction failed for %s.",
            audio_path.name,
        )

    # Chroma — harmonic / pitch class content (key for music matching)
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr_actual)
        features["chroma_mean"] = [round(float(x), 6) for x in np.mean(chroma, axis=1)]
    except Exception:  # noqa: BLE001
        logger.debug(
            "AudioReader: chroma extraction failed for %s.",
            audio_path.name,
        )

    # Spectral centroid — brightness indicator
    try:
        sc = librosa.feature.spectral_centroid(y=y, sr=sr_actual)
        features["spectral_centroid"] = round(float(np.mean(sc)), 2)
    except Exception:  # noqa: BLE001
        logger.debug(
            "AudioReader: spectral centroid extraction failed for %s.",
            audio_path.name,
        )

    # RMS energy — loudness
    try:
        rms = librosa.feature.rms(y=y)
        features["rms_energy"] = round(float(np.mean(rms)), 6)
    except Exception:  # noqa: BLE001
        logger.debug(
            "AudioReader: RMS energy extraction failed for %s.",
            audio_path.name,
        )

    return features


# =========================================================================
# AudioReader
# =========================================================================


@dataclass
class AudioReader(DocumentReader):
    """
    Text extraction from audio files via companion transcript/lyrics parsing,
    Whisper ASR, and optional audio classification.

    Three extraction paths are attempted in order:

    1. **Companion file** — zero-dependency, instant. The reader looks
       for ``.lrc``, ``.srt``, ``.vtt``, or ``.txt`` files with the same
       stem as the audio. If found, the companion is parsed; transcription
       is never invoked.

    2. **Whisper transcription** — opt-in. Enable with
       ``transcribe=True``. Requires ``faster-whisper`` or
       ``openai-whisper``.

    3. **Audio classification** — opt-in. Enable with
       ``classify=True`` and provide a ``classifier`` callable.
       For non-speech audio (animal sounds, instruments, environmental
       sounds).

    Parameters
    ----------
    input_file : pathlib.Path
        Path to the audio file.
    transcribe : bool, optional
        When ``True``, fall back to Whisper ASR if no companion file is
        found. When ``False`` (default), a missing companion file causes
        the reader to yield no chunks (unless ``classify=True``).
    whisper_model : str, optional
        Whisper model size. One of ``"tiny"``, ``"base"``, ``"small"``,
        ``"medium"``, ``"large"``, ``"large-v2"``, ``"large-v3"``.
        Default: ``"base"``.
    classify : bool, optional
        When ``True``, apply audio classification using the
        ``classifier`` callable. Can be combined with ``transcribe``:
        transcription produces speech text, classification produces
        non-speech labels. Default: ``False``.
    classifier : callable or None, optional
        A callable for audio classification. Signature::

            classifier(audio_path: Path, offset: float, duration: float)
                -> list[dict[str, Any]]

        Must return dicts with ``"label"`` (str) and ``"confidence"``
        (float). May include ``"text"`` (str). Required when
        ``classify=True``.
    segment_duration : float, optional
        Duration in seconds of each classification window when
        ``classify=True``. Default: 5.0.
    segment_overlap : float, optional
        Overlap in seconds between consecutive classification windows.
        Default: 1.0.
    extract_features : bool, optional
        When ``True``, extract audio features (MFCCs, chroma, spectral)
        for each segment and store them in ``metadata``. Requires
        ``librosa``. Default: ``False``.
    txt_as_single_chunk : bool, optional
        When a ``.txt`` companion is found, yield the entire file as one
        chunk (``True``) or one chunk per non-empty line (``False``).
        Default: ``False``.
    max_file_bytes : int, optional
        Maximum file size in bytes. Default: 5 GB.
    chunker : ChunkerBase or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.
    filter_ : FilterBase or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.
    filename_override : str or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.
    default_language : str or None, optional
        ISO 639-1 language code. Used as language hint for Whisper.
        Default: ``None`` (auto-detect).

    Attributes
    ----------
    file_types : list of str
        Class variable. Registered extensions:
        ``[".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aac",
        ".aiff", ".opus", ".wv"]``.

    Raises
    ------
    ValueError
        If ``whisper_model`` is not a valid Whisper model size.
    ValueError
        If ``classify=True`` but ``classifier`` is ``None``.
    ValueError
        If ``segment_duration <= segment_overlap``.
    ImportError
        If ``transcribe=True`` and no Whisper backend is installed.

    See Also
    --------
    scikitplot.corpus._readers.VideoReader : Video/subtitle reader.
    scikitplot.corpus._readers.TextReader : Plain-text file reader.

    Notes
    -----
    **Scenario 11 — Beethoven MP3 + Music Notes Book:**

    Build a corpus of Beethoven recordings and a book of music notes.
    Use Whisper ASR to transcribe audio segments (or provide companion
    ``.lrc`` files with lyrics). Each audio segment carries ``timecode_start``
    and ``timecode_end`` for temporal alignment. Use ``SimilarityIndex``
    with ``MatchMode.SEMANTIC`` to find which book passages match which
    audio segments — like Shazam for text-to-audio alignment.

    With ``extract_features=True``, chroma features capture harmonic
    content that correlates with musical notation in the book.

    **Scenario 12 — Animal Sounds + Children's Book (Bremen):**

    Build a corpus of animal sound recordings using ``classify=True``
    with a classifier that labels sounds (``"bird"``, ``"donkey"``,
    ``"cat"``, ``"dog"``, ``"rooster"``). Each chunk carries
    ``metadata["audio_label"]`` and a text description. Use
    ``SimilarityIndex`` with ``MatchMode.KEYWORD`` to match labels
    against sentences in *The Town Musicians of Bremen*.

    **Chunk metadata keys (companion):**

    - ``"text"`` — lyrics line or transcript text
    - ``"section_type"`` — :attr:`SectionType.LYRICS` (LRC) or
      :attr:`SectionType.TEXT` (SRT/VTT/TXT)
    - ``"timecode_start"`` — start time in seconds (float), if available
    - ``"timecode_end"`` — end time in seconds (float), if available
    - ``"source_type"`` — ``SourceType.AUDIO``
    - ``"companion_format"`` — ``"lrc"``/``"srt"``/``"vtt"``/``"txt"``

    **Chunk metadata keys (transcription):**

    - ``"text"`` — Whisper-generated transcription
    - ``"section_type"`` — :attr:`SectionType.TRANSCRIPT`
    - ``"timecode_start"`` / ``"timecode_end"`` — segment timecodes
    - ``"confidence"`` — ASR confidence (when available)
    - ``"source_type"`` — ``SourceType.AUDIO``

    **Chunk metadata keys (classification):**

    - ``"text"`` — label text or description
    - ``"section_type"`` — :attr:`SectionType.TEXT`
    - ``"timecode_start"`` / ``"timecode_end"`` — window timecodes
    - ``"confidence"`` — classification confidence
    - ``"audio_label"`` — classification label string (in metadata)
    - ``"source_type"`` — ``SourceType.AUDIO``

    Examples
    --------
    Companion LRC lyrics:

    >>> from pathlib import Path
    >>> reader = AudioReader(input_file=Path("beethoven_moonlight.mp3"))
    >>> docs = list(reader.get_documents())
    >>> for d in docs[:3]:
    ...     print(f"{d.timecode_start:.1f}s: {d.text[:50]}")

    Whisper transcription:

    >>> reader = AudioReader(
    ...     input_file=Path("lecture.mp3"),
    ...     transcribe=True,
    ...     whisper_model="small",
    ...     default_language="en",
    ... )
    >>> docs = list(reader.get_documents())

    Audio classification (animal sounds):

    >>> def my_classifier(path, offset, duration):
    ...     # Your classification model here
    ...     return [{"label": "bird", "confidence": 0.95, "text": "bird chirping"}]
    >>> reader = AudioReader(
    ...     input_file=Path("forest_sounds.wav"),
    ...     classify=True,
    ...     classifier=my_classifier,
    ...     segment_duration=3.0,
    ... )
    >>> docs = list(reader.get_documents())
    """  # noqa: D205

    file_types: ClassVar[list[str]] = _AUDIO_EXTENSIONS

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
    """Enable Whisper ASR fallback when no companion file is found."""

    whisper_model: str = field(default="base")
    """Whisper model size for transcription. Default: ``"base"``."""

    classify: bool = field(default=False)
    """Enable audio classification via ``classifier`` callable."""

    classifier: Callable[..., list[dict[str, Any]]] | None = field(
        default=None, repr=False
    )
    """
    Audio classification callable. Signature::

        classifier(audio_path: Path, offset: float, duration: float)
            -> list[dict[str, Any]]

    Required when ``classify=True``.
    """

    segment_duration: float = field(default=5.0)
    """Classification window duration in seconds. Default: 5.0."""

    segment_overlap: float = field(default=1.0)
    """Classification window overlap in seconds. Default: 1.0."""

    extract_features: bool = field(default=False)
    """Extract audio features (MFCCs, chroma) via librosa."""

    txt_as_single_chunk: bool = field(default=False)
    """Yield entire ``.txt`` companion as one chunk if ``True``."""

    max_file_bytes: int = field(default=5 * 1024 * 1024 * 1024)
    """Maximum audio file size. Default: 5 GB."""

    def __post_init__(self) -> None:
        """Validate constructor fields and resolve companion file strategy.

        Raises
        ------
        ValueError
            If ``whisper_model`` is not a recognised Whisper size.
        ValueError
            If ``classify=True`` but ``classifier`` callable is ``None``.
        """
        super().__post_init__()
        if self.whisper_model not in self._VALID_WHISPER_MODELS:
            raise ValueError(
                f"AudioReader: whisper_model must be one of"
                f" {self._VALID_WHISPER_MODELS}; got {self.whisper_model!r}."
            )
        if self.classify and self.classifier is None:
            raise ValueError(
                "AudioReader: classify=True requires a 'classifier' callable."
                " Provide a function with signature:"
                " classifier(path: Path, offset: float, duration: float)"
                " -> list[dict[str, Any]]."
            )
        if self.segment_duration <= 0:
            raise ValueError(
                f"AudioReader: segment_duration must be > 0;"
                f" got {self.segment_duration!r}."
            )
        if self.segment_overlap < 0:
            raise ValueError(
                f"AudioReader: segment_overlap must be >= 0;"
                f" got {self.segment_overlap!r}."
            )
        if self.segment_duration <= self.segment_overlap:
            raise ValueError(
                f"AudioReader: segment_duration ({self.segment_duration})"
                f" must be > segment_overlap ({self.segment_overlap})."
            )
        if self.max_file_bytes <= 0:
            raise ValueError(
                f"AudioReader: max_file_bytes must be > 0; got {self.max_file_bytes!r}."
            )

    # ------------------------------------------------------------------
    # DocumentReader contract
    # ------------------------------------------------------------------

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:  # noqa: PLR0912
        """
        Extract text from the audio via companion, transcription, or
        classification.

        Attempts companion detection first. Falls back to Whisper only
        when ``transcribe=True`` and no companion was found. Classification
        via ``classify=True`` runs independently (can combine with
        transcription).

        Yields
        ------
        dict
            Keys always include ``"text"`` and ``"section_type"``.
            May include ``"timecode_start"``, ``"timecode_end"``,
            ``"confidence"``, ``"source_type"``, and format-specific keys.

        Raises
        ------
        ValueError
            If the file exceeds ``max_file_bytes``.
        ImportError
            If ``transcribe=True`` and Whisper is not installed.
        """  # noqa: D205
        file_size = self.input_file.stat().st_size
        if file_size > self.max_file_bytes:
            raise ValueError(
                f"AudioReader: {self.file_name} is {file_size:,} bytes,"
                f" which exceeds max_file_bytes={self.max_file_bytes:,}."
            )

        yielded_any = False

        # --- Strategy 1: companion transcript / lyrics file ---
        companion_result = _find_companion(self.input_file)
        if companion_result is not None:
            comp_path, comp_fmt = companion_result
            cues, comp_meta = _parse_companion(
                comp_path,
                comp_fmt,
                txt_as_single_chunk=self.txt_as_single_chunk,
            )
            logger.info(
                "AudioReader: parsed %d cues from %s (format=%s).",
                len(cues),
                comp_path.name,
                comp_fmt.upper(),
            )

            # Determine section type based on companion format
            if comp_fmt == "lrc":
                section = SectionType.LYRICS.value
            else:
                section = SectionType.TEXT.value

            for cue in cues:
                chunk: dict[str, Any] = {
                    "text": cue["text"],
                    "section_type": section,
                    "source_type": SourceType.AUDIO.value,
                    "companion_format": comp_fmt,
                }
                # Add timecodes if available
                if "timecode_start" in cue:
                    chunk["timecode_start"] = cue["timecode_start"]
                if "timecode_end" in cue:
                    chunk["timecode_end"] = cue["timecode_end"]
                # Add LRC metadata to first chunk
                if comp_meta and cue is cues[0]:
                    for meta_key, meta_val in comp_meta.items():
                        if meta_key == "ti":
                            chunk["source_title"] = meta_val
                        elif meta_key == "ar":
                            chunk["source_author"] = meta_val
                        else:
                            chunk[meta_key] = meta_val
                # Add audio features if requested
                if self.extract_features and "timecode_start" in cue:
                    features = _extract_audio_features(
                        self.input_file,
                        offset=cue["timecode_start"],
                        duration=(
                            cue.get("timecode_end", cue["timecode_start"] + 5.0)
                            - cue["timecode_start"]
                        ),
                    )
                    if features:
                        chunk["audio_features"] = features
                yield chunk
                yielded_any = True

            if not self.classify:
                return  # Done — companion found and classification not requested

        # --- Strategy 2: Whisper transcription (opt-in) ---
        if self.transcribe and not yielded_any:
            logger.info(
                "AudioReader: no companion found for %s; transcribing with"
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
                "AudioReader: transcription produced %d segments for %s.",
                len(segments),
                self.file_name,
            )
            for seg in segments:
                chunk = {
                    "text": seg["text"],
                    "section_type": SectionType.TRANSCRIPT.value,
                    "source_type": SourceType.AUDIO.value,
                    "timecode_start": seg["timecode_start"],
                    "timecode_end": seg["timecode_end"],
                }
                if "confidence" in seg:
                    chunk["confidence"] = seg["confidence"]
                # Add audio features if requested
                if self.extract_features:
                    features = _extract_audio_features(
                        self.input_file,
                        offset=seg["timecode_start"],
                        duration=seg["timecode_end"] - seg["timecode_start"],
                    )
                    if features:
                        chunk["audio_features"] = features
                yield chunk
                yielded_any = True

        # --- Strategy 3: Audio classification (opt-in) ---
        if self.classify and self.classifier is not None:
            logger.info(
                "AudioReader: classifying %s (segment=%.1fs, overlap=%.1fs).",
                self.file_name,
                self.segment_duration,
                self.segment_overlap,
            )
            labels = _classify_audio(
                self.input_file,
                self.classifier,
                self.segment_duration,
                self.segment_overlap,
            )
            logger.info(
                "AudioReader: classification produced %d labelled segments for %s.",
                len(labels),
                self.file_name,
            )
            for lbl in labels:
                chunk = {
                    "text": lbl["text"],
                    "section_type": SectionType.TEXT.value,
                    "source_type": SourceType.AUDIO.value,
                    "timecode_start": lbl["timecode_start"],
                    "timecode_end": lbl["timecode_end"],
                    "confidence": lbl.get("confidence", 0.0),
                    "audio_label": lbl.get("audio_label", "unknown"),
                }
                # Add audio features if requested
                if self.extract_features:
                    features = _extract_audio_features(
                        self.input_file,
                        offset=lbl["timecode_start"],
                        duration=lbl["timecode_end"] - lbl["timecode_start"],
                    )
                    if features:
                        chunk["audio_features"] = features
                yield chunk
                yielded_any = True

        # --- No strategy produced output ---
        if not yielded_any:
            logger.warning(
                "AudioReader: no companion file found for %s,"
                " transcribe=%s, classify=%s."
                " Yielding no chunks."
                " To enable transcription, set transcribe=True."
                " To enable classification, set classify=True with a"
                " classifier callable.",
                self.file_name,
                self.transcribe,
                self.classify,
            )


__all__ = ["AudioReader"]
