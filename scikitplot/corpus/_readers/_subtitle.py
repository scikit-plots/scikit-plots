# scikitplot/corpus/_readers/_subtitle.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Subtitle and caption files: one parser for every reader.

Parses SubRip (``.srt``), WebVTT (``.vtt``), YouTube SBV (``.sbv``),
MicroDVD (``.sub``), Advanced SubStation Alpha (``.ass``, ``.ssa``) and
LRC lyrics (``.lrc``)
into *cues* ``{"text", "raw_text", "timecode_start", "timecode_end"}``
(plus ``"speaker"`` when the file names one). :class:`SubtitleReader`
reads a subtitle file on its own; :class:`~scikitplot.corpus._readers.VideoReader`
and :class:`~scikitplot.corpus._readers.AudioReader` use the same parser
for companion files, so a file reads the same either way.

Every parser is pure (text in, cues out), standard library only, and
linear in the file: one pass over its lines, no backtracking regex.
"""

from __future__ import annotations

import bisect
import html
import logging
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (  # noqa: F401
    Any,
    Callable,
    ClassVar,
    Generator,
    Iterable,
    Iterator,
    Optional,
)

from .._base import ReadCoverage
from .._schema import SectionType, SourceType
from ._text import TextReader

logger = logging.getLogger(__name__)

__all__ = ["SubtitleReader"]

#: Formats by extension. ``.sub`` is MicroDVD text; a binary VobSub
#: ``.sub`` (an image track) is refused as not a text file.
FORMATS = {
    ".srt": "srt",
    ".vtt": "vtt",
    ".sbv": "sbv",
    ".sub": "sub",
    ".ass": "ass",
    ".ssa": "ass",
    ".lrc": "lrc",
}

# ---------------------------------------------------------------------------
# Timecodes
# ---------------------------------------------------------------------------

#: ``[H:]MM:SS[,.fraction]`` -- hours optional (WebVTT allows it), any
#: number of fraction digits (``,5`` is half a second, not 5 ms).
_TIME = r"(?:(\d+):)?(\d{1,2}):(\d{2})(?:[,.](\d{1,9}))?"
_CUE_TIME_RE = re.compile(rf"^\s*{_TIME}\s*-->\s*{_TIME}(?:\s|$)")
_SBV_TIME_RE = re.compile(rf"^\s*{_TIME}\s*,\s*{_TIME}\s*$")


def _seconds(
    hours: str | None, minutes: str, seconds: str, fraction: str | None
) -> float:
    """
    Return the seconds a timecode's parts name.

    Examples
    --------
    >>> _seconds(None, "01", "02", "5"), _seconds("1", "00", "00", "250")
    (62.5, 3600.25)
    """
    value: float = int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)
    if fraction:
        value += int(fraction) / 10 ** len(fraction)
    return value


def _span(match: re.Match[str]) -> tuple[float, float]:
    groups = match.groups()
    return _seconds(*groups[:4]), _seconds(*groups[4:8])


# ---------------------------------------------------------------------------
# Cue text
# ---------------------------------------------------------------------------

# Every markup pattern stops at the next opener ([^<>], [^{}]): a scan
# from an opener that never closes ends there instead of at the end of the
# line, so text full of "<" or "{" costs linear time, not quadratic.
_VOICE_RE = re.compile(r"<v(?:\.[^\s<>]*)?\s+([^<>\n]+)>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^<>\n]*>")
_ASS_OVERRIDE_RE = re.compile(r"\{\\[^{}]*\}")
_MICRODVD_STYLE_RE = re.compile(r"\{[a-zA-Z]:[^{}]*\}")
_SPACES_RE = re.compile(r"[ \t\u00a0]+")


def _clean(lines: list[str]) -> tuple[str, str, str | None]:
    r"""
    Return ``(text, raw_text, speaker)`` for a cue's lines.

    Markup is removed: HTML and WebVTT tags (``<i>``, ``<c.x>``, inline
    ``<00:00:01.000>`` times), ASS override blocks (``{\an8}``) and
    MicroDVD style codes (``{y:i}``); entities are unescaped; lines join
    with a space. The first WebVTT voice tag (``<v Alice>``) names the
    speaker. ``raw_text`` is the lines as written, joined by line breaks.
    """
    raw = "\n".join(lines).strip()
    voice = _VOICE_RE.search(raw)
    speaker = voice.group(1).strip() if voice else None
    text = _MICRODVD_STYLE_RE.sub("", _ASS_OVERRIDE_RE.sub("", raw))
    text = html.unescape(_TAG_RE.sub("", text))
    text = _SPACES_RE.sub(" ", " ".join(part.strip() for part in text.splitlines()))
    return text.strip(), raw, speaker


def _cue(
    lines: list[str], start: float, end: float, speaker: str | None = None
) -> dict[str, Any] | None:
    text, raw, voice = _clean(lines)
    if not text:
        return None
    cue = {"text": text, "raw_text": raw, "timecode_start": start, "timecode_end": end}
    if speaker or voice:
        cue["speaker"] = speaker or voice
    return cue


# ---------------------------------------------------------------------------
# Parsers: text in, cues out
# ---------------------------------------------------------------------------

_VTT_BLOCKS = ("NOTE", "STYLE", "REGION")


def _blocks(content: str) -> Iterator[list[str]]:
    """Yield the blank-line separated blocks of *content*, one pass."""
    block: list[str] = []
    for line in content.lstrip("\ufeff").splitlines():
        if line.strip():
            block.append(line)
        elif block:
            yield block
            block = []
    if block:
        yield block


def _malformed(stats: dict[str, int] | None) -> None:
    if stats is not None:
        stats["malformed"] = stats.get("malformed", 0) + 1


def parse_timed_blocks(  # noqa: PLR0912
    content: str, *, vtt: bool = False, stats: dict[str, int] | None = None
) -> list[dict[str, Any]]:
    r"""
    Parse SubRip or WebVTT text into cues, block by block.

    As both specifications say, a blank line ends a cue. A block's timing
    line ``start --> end`` is its first line, or its second after a cue
    number or identifier. A block whose timing line is broken, or that
    has none, is dropped and counted in *stats* (``"malformed"``), so a
    bad cue never spills into its neighbour. A valid timing line inside a
    block (a missing blank line) starts a new cue there, and a bare
    number just before it is taken as that cue's number. WebVTT
    ``WEBVTT``, ``NOTE``, ``STYLE`` and ``REGION`` blocks are skipped.

    Parameters
    ----------
    content : str
        File text.
    vtt : bool, optional
        Skip WebVTT header and blocks. Default: ``False``.
    stats : dict or None, optional
        Counts of dropped blocks, updated in place.

    Returns
    -------
    list of dict
        Cues in file order.

    Examples
    --------
    >>> [
    ...     c["text"]
    ...     for c in parse_timed_blocks("1\n00:00:01,000 --> 00:00:02,000\nHi\n")
    ... ]
    ['Hi']
    """
    cues: list[dict[str, Any]] = []
    for block in _blocks(content):
        head = block[0].strip()
        if vtt and (head.startswith(_VTT_BLOCKS) or head.startswith("WEBVTT")):
            continue
        timing = next(
            (i for i in range(min(2, len(block))) if _CUE_TIME_RE.match(block[i])), None
        )
        if timing is None:
            _malformed(stats)
            continue
        span = _span(_CUE_TIME_RE.match(block[timing]))  # type: ignore[arg-type]
        body: list[str] = []
        for line in block[timing + 1 :]:
            match = _CUE_TIME_RE.match(line)
            if match:  # a missing blank line: a new cue starts here
                if len(body) > 1 and body[-1].strip().isdigit():
                    body.pop()
                made = _cue(body, *span)
                if made:
                    cues.append(made)
                span, body = _span(match), []
            elif "-->" in line:
                # A broken timing line: keep the cue before it, drop the
                # one it would start (and its number, if any).
                if body and body[-1].strip().isdigit():
                    body.pop()
                made = _cue(body, *span)
                if made:
                    cues.append(made)
                _malformed(stats)
                break
            else:
                body.append(line)
        else:
            made = _cue(body, *span)
            if made:
                cues.append(made)
    return cues


def parse_srt(
    content: str, stats: dict[str, int] | None = None
) -> list[dict[str, Any]]:
    """Parse SubRip text (see :func:`parse_timed_blocks`)."""
    return parse_timed_blocks(content, stats=stats)


def parse_vtt(
    content: str, stats: dict[str, int] | None = None
) -> list[dict[str, Any]]:
    """Parse WebVTT text (see :func:`parse_timed_blocks`)."""
    return parse_timed_blocks(content, vtt=True, stats=stats)


def parse_sbv(
    content: str, stats: dict[str, int] | None = None
) -> list[dict[str, Any]]:
    r"""
    Parse YouTube SBV text: blocks of a ``start,end`` line, then the lines.

    A block whose first line is not a valid timing is dropped and counted
    in *stats* (``"malformed"``).

    Examples
    --------
    >>> parse_sbv("0:00:01.000,0:00:02.500\nHello\n")[0]["timecode_end"]
    2.5
    """
    cues: list[dict[str, Any]] = []
    for block in _blocks(content):
        match = _SBV_TIME_RE.match(block[0])
        if not match:
            _malformed(stats)
            continue
        made = _cue(block[1:], *_span(match))
        if made:
            cues.append(made)
    return cues


_MICRODVD_RE = re.compile(r"^\s*\{(\d+)\}\{(\d*)\}(.*)$")


def parse_sub(content: str, frame_rate: float | None = None) -> list[dict[str, Any]]:
    r"""
    Parse MicroDVD text: ``{start frame}{end frame}text``, ``|`` breaks lines.

    The frame rate is *frame_rate* when given, else the file's own
    ``{1}{1}23.976`` header, else 25. An empty end frame means the cue
    lasts until the next one starts.

    Examples
    --------
    >>> parse_sub("{1}{1}25\n{25}{50}Hello|world\n")[0]["text"]
    'Hello world'
    """
    rows: list[tuple[int, int | None, str]] = []
    header_rate: float | None = None
    for line in content.lstrip("\ufeff").splitlines():
        match = _MICRODVD_RE.match(line)
        if not match:
            continue
        start, end, text = int(match.group(1)), match.group(2), match.group(3)
        if not rows and header_rate is None and start == 1 and end == "1":
            # The frame-rate header is never dialogue, whatever rate wins.
            try:
                header_rate = float(text.strip())
            except ValueError:
                pass
            else:
                continue
        rows.append((start, int(end) if end else None, text))
    rate = frame_rate or header_rate
    rate = rate if rate and rate > 0 else 25.0
    cues: list[dict[str, Any]] = []
    for i, (start, stop, text) in enumerate(rows):
        end = stop
        if end is None:
            end = rows[i + 1][0] if i + 1 < len(rows) else start
        made = _cue(text.split("|"), start / rate, end / rate)
        if made:
            cues.append(made)
    return cues


def parse_ass(content: str) -> list[dict[str, Any]]:
    r"""
    Parse Advanced SubStation Alpha (``.ass``) or SSA ``Dialogue`` lines.

    The ``[Events]`` section's ``Format:`` line gives the field order (the
    standard order when absent); ``Name`` is the speaker, ``Text`` the
    last field (it may hold commas). ``\N`` and ``\n`` break lines,
    ``\h`` is a space, override blocks ``{...}`` are removed.
    ``Comment:`` lines are skipped.

    Examples
    --------
    >>> ass = (
    ...     "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, "
    ...     "MarginR, MarginV, Effect, Text\n"
    ...     "Dialogue: 0,0:00:01.00,0:00:02.50,Default,Ann,0,0,0,,Hi,\\Nthere\n"
    ... )
    >>> c = parse_ass(ass)[0]
    >>> c["text"], c["speaker"], c["timecode_end"]
    ('Hi, there', 'Ann', 2.5)
    """
    fields = [
        "layer",
        "start",
        "end",
        "style",
        "name",
        "marginl",
        "marginr",
        "marginv",
        "effect",
        "text",
    ]
    in_events = False
    cues: list[dict[str, Any]] = []
    time_re = re.compile(rf"^\s*{_TIME}\s*$")
    for line in content.lstrip("\ufeff").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_events = stripped.lower() == "[events]"
            continue
        if not in_events:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip().lower()
        if key == "format":
            fields = [f.strip().lower() for f in value.split(",")]
            continue
        if key != "dialogue" or "text" not in fields:
            continue
        parts = value.split(",", len(fields) - 1)
        if len(parts) < len(fields):
            continue
        row = dict(zip(fields, parts))
        start = time_re.match(row.get("start", ""))
        end = time_re.match(row.get("end", ""))
        if not (start and end):
            continue
        text = row["text"].replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ")
        name = row.get("name", "").strip() or None
        made = _cue(
            text.split("\n"), _seconds(*start.groups()), _seconds(*end.groups()), name
        )
        if made:
            cues.append(made)
    return cues


_LRC_TAG_RE = re.compile(r"\[(\d{1,3}):(\d{2})(?::(\d{2}))?(?:[.:](\d{1,3}))?\]")
_LRC_META_RE = re.compile(r"^\[([A-Za-z#]{1,10}):(.*)\]$")
_LRC_WORD_RE = re.compile(r"<\d{1,3}:\d{2}(?:[.:]\d{1,3})?>")
#: Seconds the last lyric line lasts: the format gives no end time.
LRC_LAST_LINE_SECONDS = 5.0


def lrc_parts(content: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    r"""
    Parse LRC lyrics into cues and their ``[tag:value]`` metadata.

    A line may carry several timestamps (``[00:12.00][00:45.00]Chorus``):
    it becomes a cue at each. ``[mm:ss]``, ``[mm:ss.xx]`` and
    ``[hh:mm:ss.xx]`` are read (three colon-separated numbers are always
    hours, minutes, seconds, as before); enhanced word tags
    ``<mm:ss.xx>`` are removed. A cue ends where the next timestamp starts, an empty one
    (an instrumental break) included; the last lasts
    :data:`LRC_LAST_LINE_SECONDS`. ``[offset:ms]`` follows the format: a
    positive offset makes lyrics appear sooner, so it is subtracted.

    Returns
    -------
    tuple of (list of dict, dict)
        Cues sorted by time, with ``line_index``; metadata by lower-case tag
        (``ti`` title, ``ar`` artist, ``al`` album, ``by``, ``offset``).

    Examples
    --------
    >>> cues, meta = lrc_parts("[ti:Song]\n[00:01.00][00:03.00]La\n[00:02.00]Da\n")
    >>> [(c["text"], c["timecode_start"], c["timecode_end"]) for c in cues]
    [('La', 1.0, 2.0), ('Da', 2.0, 3.0), ('La', 3.0, 8.0)]
    >>> meta
    {'ti': 'Song'}
    """
    meta: dict[str, str] = {}
    marks: list[tuple[float, int, str, str]] = []  # (time, order, text, raw)
    for line in content.lstrip("\ufeff").splitlines():
        stripped = line.strip()
        times: list[float] = []
        at = 0
        while True:
            match = _LRC_TAG_RE.match(stripped, at)
            if not match:
                break
            first, second, third, fraction = match.groups()
            if third is not None:  # [hh:mm:ss(.xx)]
                times.append(_seconds(first, second, third, fraction))
            else:
                times.append(_seconds(None, first, second, fraction))
            at = match.end()
        if not times:
            tag = _LRC_META_RE.match(stripped)
            if tag:
                meta[tag.group(1).lower()] = tag.group(2).strip()
            continue
        raw = stripped[at:].strip()
        text = " ".join(_LRC_WORD_RE.sub("", raw).split())
        for time in times:
            marks.append((time, len(marks), text, raw))
    offset = 0.0
    if "offset" in meta:
        try:
            offset = float(meta["offset"]) / 1000.0
        except ValueError:
            logger.warning("LRC: ignoring an invalid offset %r.", meta["offset"])
    marks.sort()
    cues: list[dict[str, Any]] = []
    for i, (time, _, text, raw) in enumerate(marks):
        if not text:
            continue
        start = max(time - offset, 0.0)
        end = (
            max(marks[i + 1][0] - offset, 0.0)
            if i + 1 < len(marks)
            else start + LRC_LAST_LINE_SECONDS
        )
        cues.append(
            {
                "text": text,
                "raw_text": raw,
                "timecode_start": start,
                "timecode_end": end,
                "line_index": len(cues),
            }
        )
    return cues, meta


def parse_lrc(content: str) -> list[dict[str, Any]]:
    """Parse LRC lyrics into cues (see :func:`lrc_parts`)."""
    return lrc_parts(content)[0]


PARSERS: dict[str, Callable[[str], list[dict[str, Any]]]] = {
    "srt": parse_srt,
    "vtt": parse_vtt,
    "sbv": parse_sbv,
    "sub": parse_sub,
    "ass": parse_ass,
    "lrc": parse_lrc,
}


def merge_rolling(cues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    r"""
    Drop lines a cue repeats from the end of the cue before (rolling captions).

    Automatic captions (YouTube's among them) show each line twice: as
    the new line of one cue and the old line of the next. A cue's leading
    lines that equal the previous cue's trailing lines are removed; a cue
    left empty is dropped. Only exact repeats of adjacent cues change.

    Examples
    --------
    >>> cues = [
    ...     {"text": "a", "raw_text": "a", "timecode_start": 0, "timecode_end": 1},
    ...     {
    ...         "text": "a b",
    ...         "raw_text": "a\nb",
    ...         "timecode_start": 1,
    ...         "timecode_end": 2,
    ...     },
    ... ]
    >>> [c["text"] for c in merge_rolling(cues)]
    ['a', 'b']
    """
    out: list[dict[str, Any]] = []
    previous: list[str] = []
    for cue in cues:
        lines = [line.strip() for line in cue["raw_text"].split("\n") if line.strip()]
        keep = lines
        for size in range(min(len(previous), len(lines)), 0, -1):
            if previous[-size:] == lines[:size]:
                keep = lines[size:]
                break
        previous = lines
        if not keep:
            continue
        if keep is lines:
            out.append(cue)
            continue
        made = _cue(
            keep, cue["timecode_start"], cue["timecode_end"], cue.get("speaker")
        )
        if made:
            out.append(made)
    return out


# ---------------------------------------------------------------------------
# Companion files of audio and video
# ---------------------------------------------------------------------------


#: Folders whose listing is kept: one entry per folder, replaced when the
#: folder changes, so memory holds at most this many listings.
_LISTING_FOLDERS = 8
_LISTINGS: OrderedDict[str, tuple[int, tuple[tuple[str, str], ...]]] = OrderedDict()
_LISTINGS_LOCK = threading.Lock()


def _listing(folder: str) -> tuple[tuple[str, str], ...]:
    """
    Return ``(lower-case name, name)`` of a folder's files, sorted.

    Kept per folder and replaced when the folder's modification time
    changes, so a folder is listed once per change, not once per media
    file in it; at most :data:`_LISTING_FOLDERS` folders are kept. ``scandir``
    entries carry their type, so no file is stat-ed. Thread-safe.

    Raises
    ------
    OSError
        If the folder cannot be read.
    """
    stamp = os.stat(folder).st_mtime_ns
    with _LISTINGS_LOCK:
        kept = _LISTINGS.get(folder)
        if kept is not None and kept[0] == stamp:
            _LISTINGS.move_to_end(folder)
            return kept[1]
    with os.scandir(folder) as entries:
        listing = tuple(
            sorted((e.name.lower(), e.name) for e in entries if e.is_file())
        )
    with _LISTINGS_LOCK:
        _LISTINGS[folder] = (stamp, listing)
        _LISTINGS.move_to_end(folder)
        while len(_LISTINGS) > _LISTING_FOLDERS:
            _LISTINGS.popitem(last=False)
    return listing


def find_companion(
    media: Path, extensions: Iterable[str], *, language: str | None = None
) -> Path | None:
    """
    Return the subtitle file that belongs to *media*, or ``None``.

    Candidates sit in the same folder and share the media file's stem:
    ``talk.srt``, or a tagged one such as ``talk.en.srt`` or
    ``talk.en-US.forced.srt``. Names match in any letter case. The first
    extension in *extensions* that has a candidate wins; among its
    candidates the order is: with *language*, a tag whose first part is
    that language (``en`` matches ``en`` and ``en-US``); then the exact
    stem; then tagged names in sorted order. The choice is the same on
    every file system.

    Parameters
    ----------
    media : pathlib.Path
        The audio or video file.
    extensions : iterable of str
        Companion extensions in priority order, lower case with the dot.
    language : str or None, optional
        Preferred language tag (``"en"``, ``"pt-BR"``). Default: none.

    Returns
    -------
    pathlib.Path or None
        The companion, or ``None``.

    Notes
    -----
    The folder listing is kept per folder (eight at most) and replaced
    when the folder's modification time changes: a lookup costs a binary
    search, so a folder of thousands of media files is not listed once
    per file, and memory stays bounded. A file system whose time stamps are
    coarse may hide a companion created within the same tick until the
    folder changes again.

    Examples
    --------
    >>> find_companion(Path("/nonexistent/talk.mp4"), [".srt"]) is None
    True
    """
    folder = media.parent
    try:
        listing = _listing(str(folder))
    except OSError:
        return None
    prefix = media.stem.lower() + "."
    start = bisect.bisect_left(listing, (prefix,))
    rests: list[tuple[str, str]] = []  # (lower-case rest after "stem.", name)
    for lower, name in listing[start:]:
        if not lower.startswith(prefix):
            break
        rests.append((lower[len(prefix) :], name))
    wanted = language.lower() if language else None
    for ext in extensions:
        bare = ext.lstrip(".")
        exact = [name for rest, name in rests if rest == bare]
        tagged = sorted(
            (rest[: -len(ext)], name)
            for rest, name in rests
            if rest.endswith(ext) and rest != bare
        )
        ordered: list[str] = []
        if wanted:
            ordered += [
                name
                for tag, name in tagged
                if tag.split(".")[0] == wanted
                or tag.split(".")[0].startswith(wanted + "-")
            ]
        ordered += exact + [name for _, name in tagged]
        if ordered:
            return folder / ordered[0]
    return None


#: Keys ``companion_options`` accepts on the audio and video readers.
COMPANION_OPTIONS = frozenset({"encoding", "dedupe_rolling", "language"})


def companion_options(value: Any, owner: str) -> dict[str, Any]:
    """
    Validate a reader's ``companion_options`` and return them as a dict.

    Keys: ``encoding`` (the :class:`TextReader` policy for the companion),
    ``dedupe_rolling`` (see :func:`merge_rolling`) and ``language`` (the
    tag :func:`find_companion` prefers). Values are checked by building a
    :class:`SubtitleReader` with them, so a bad one fails when the owning
    reader is made, not in the middle of a corpus build.

    Raises
    ------
    TypeError
        If *value* is not a mapping or ``language`` is not a str.
    ValueError
        If a key is unknown or a value is invalid.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(
            f"{owner}: companion_options must be a dict or None; got {value!r}."
        )
    unknown = sorted(set(value) - COMPANION_OPTIONS)
    if unknown:
        raise ValueError(
            f"{owner}: companion_options has unknown key(s) {unknown};"
            f" known: {sorted(COMPANION_OPTIONS)}."
        )
    language = value.get("language")
    if language is not None and (not isinstance(language, str) or not language.strip()):
        raise TypeError(
            f"{owner}: companion_options['language'] must be a non-empty str."
        )
    try:
        SubtitleReader(
            input_path=Path("companion.srt"),
            encoding=value.get("encoding"),
            dedupe_rolling=value.get("dedupe_rolling", False),
        )
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{owner}: companion_options: {exc}") from exc
    return dict(value)


def read_text(path: Path, *, encoding: Any = None) -> str:
    """
    Decode a subtitle file as :class:`SubtitleReader` does.

    Raises
    ------
    ValueError
        If the file is too large or binary.
    """
    reader = SubtitleReader(input_path=Path(path), encoding=encoding)
    whole = next(iter(TextReader.get_raw_chunks(reader)), None)
    return "" if whole is None else whole["text"]


def read_cues(
    path: Path,
    fmt: str | None = None,
    *,
    encoding: Any = None,
    frame_rate: float | None = None,
    dedupe_rolling: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    """
    Decode a subtitle file as :class:`SubtitleReader` does and parse it.

    Returns
    -------
    tuple of (list of dict, str)
        The cues and the codec used.

    Raises
    ------
    ValueError
        If the file is too large or binary (a VobSub ``.sub``, or UTF-16
        without a byte-order mark; pass ``encoding=``).
    """
    reader = SubtitleReader(
        input_path=Path(path),
        encoding=encoding,
        frame_rate=frame_rate,
        dedupe_rolling=dedupe_rolling,
        subtitle_format=fmt,
    )
    return reader.cues(), reader.last_encoding


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


@dataclass
class SubtitleReader(TextReader):
    """
    Read a subtitle or caption file into timed chunks.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to a ``.srt``, ``.vtt``, ``.sbv``, ``.sub`` (MicroDVD),
        ``.ass``, ``.ssa`` or ``.lrc`` (lyrics) file.
    subtitle_format : str or None, optional
        ``"srt"``, ``"vtt"``, ``"sbv"``, ``"sub"``, ``"ass"`` or ``"lrc"``; ``None``
        (default) takes it from the extension.
    encoding : str, sequence of str, callable or None, optional
        As for :class:`~scikitplot.corpus._readers.TextReader`. Legacy
        subtitles are often in a national code page: ``encoding="detect"``
        (with ``charset-normalizer``) or a codec reads them right.
    cues_per_chunk : int, optional
        Most cues in one chunk. Default: ``1`` (one chunk per cue).
    max_chunk_seconds : float or None, optional
        With several cues a chunk, also end a chunk once it spans this
        many seconds. Default: ``None``.
    dedupe_rolling : bool, optional
        Remove lines a cue repeats from the cue before, as automatic
        "rolling" captions do (see :func:`merge_rolling`). Default:
        ``False``.
    frame_rate : float or None, optional
        MicroDVD frames per second; ``None`` uses the file's header, else
        25. Default: ``None``.
    max_file_bytes : int, optional
        Larger files are refused. Default: 50 MB.

    Attributes
    ----------
    file_types : list of str
        Class variable. ``[".ass", ".lrc", ".sbv", ".srt", ".ssa", ".sub", ".vtt"]``.
    coverage : ReadCoverage
        After a read: cues read, and blocks dropped as ``"malformed"`` (a
        broken or missing timing line).

    Raises
    ------
    ValueError
        If the file is too large or binary, or an option is invalid.

    See Also
    --------
    scikitplot.corpus._readers.VideoReader : Uses the same parser for a
        video's companion subtitles.

    Notes
    -----
    **User note:** Each chunk carries ``timecode_start`` and
    ``timecode_end`` (seconds, promoted to the document), ``cue_start``
    and ``cue_end`` (0-based cue numbers), ``subtitle_format``,
    ``encoding`` and, when the file names one, ``speaker`` (a WebVTT
    ``<v Name>`` tag or the ASS ``Name`` field). LRC lyrics add
    ``source_title``, ``source_author`` and ``album`` from their
    ``[ti:]``, ``[ar:]`` and ``[al:]`` tags. ``source_type`` is
    :attr:`SourceType.SUBTITLE`. Markup is removed and entities
    unescaped; ``raw_text`` keeps the lines as written.

    **What changed for companions:** audio and video companion files
    now parse with this reader: short WebVTT timecodes (``01:02.000``),
    any number of fraction digits, UTF-16 with a byte-order mark and the
    encoding policy all apply there too.

    Examples
    --------
    >>> from pathlib import Path
    >>> reader = SubtitleReader(input_path=Path("talk.en.vtt"), cues_per_chunk=5)
    >>> docs = list(reader.get_documents())
    """

    file_type: ClassVar[str] = ".srt"
    file_types: ClassVar[list[str] | None] = sorted(FORMATS)

    subtitle_format: str | None = field(default=None)
    """Format name; ``None`` takes it from the extension."""

    cues_per_chunk: int = field(default=1)
    """Most cues in one chunk."""

    max_chunk_seconds: float | None = field(default=None)
    """With several cues a chunk, the longest span a chunk may cover."""

    dedupe_rolling: bool = field(default=False)
    """Remove lines repeated from the previous cue (rolling captions)."""

    frame_rate: float | None = field(default=None)
    """MicroDVD frames per second; ``None`` uses the file's, else 25."""

    max_file_bytes: int = field(default=50 * 1024 * 1024)
    """Larger files are refused. Default: 50 MB."""

    def __post_init__(self) -> None:
        """Validate the options."""
        super().__post_init__()
        fmt = self.subtitle_format
        if fmt is not None and fmt not in PARSERS:
            raise ValueError(
                f"SubtitleReader: subtitle_format must be one of {sorted(PARSERS)}"
                f" or None; got {fmt!r}."
            )
        n = self.cues_per_chunk
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError(
                f"SubtitleReader: cues_per_chunk must be an int > 0; got {n!r}."
            )
        for name in ("max_chunk_seconds", "frame_rate"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not value > 0
            ):
                raise ValueError(
                    f"SubtitleReader: {name} must be a number > 0 or None; got {value!r}."
                )
        if not isinstance(self.dedupe_rolling, bool):
            raise TypeError("SubtitleReader: dedupe_rolling must be a bool.")
        self.coverage = ReadCoverage(unit="cue")
        self.last_encoding = ""
        self.lrc_meta: dict[str, str] = {}
        self.skipped: dict[str, int] = {}

    def _format(self) -> str:
        if self.subtitle_format is not None:
            return self.subtitle_format
        return FORMATS.get(self.input_path.suffix.lower(), "srt")

    def cues(self) -> list[dict[str, Any]]:
        """
        Return the file's cues in file order.

        Returns
        -------
        list of dict
            ``text``, ``raw_text``, ``timecode_start``, ``timecode_end``
            and, when named, ``speaker``.

        Raises
        ------
        ValueError
            If the file is too large or binary.
        """
        whole = next(iter(super().get_raw_chunks()), None)
        if whole is None:
            self.last_encoding = ""
            return []
        self.last_encoding = whole["encoding"]
        fmt = self._format()
        self.lrc_meta = {}
        self.skipped = {}
        text = whole["text"]
        if fmt == "sub":
            cues = parse_sub(text, self.frame_rate)
        elif fmt == "lrc":
            cues, self.lrc_meta = lrc_parts(text)
        elif fmt in ("srt", "vtt", "sbv"):
            cues = PARSERS[fmt](text, self.skipped)  # type: ignore[call-arg]
        else:
            cues = PARSERS[fmt](text)
        return merge_rolling(cues) if self.dedupe_rolling else cues

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
        """
        Yield chunks of ``cues_per_chunk`` cues.

        Yields
        ------
        dict
            ``text``, ``raw_text``, ``section_type``, ``source_type``,
            ``timecode_start``, ``timecode_end``, ``cue_start``,
            ``cue_end``, ``subtitle_format``, ``encoding`` and, when named,
            ``speaker``.
        """
        coverage = ReadCoverage(unit="cue")
        self.coverage = coverage
        cues = self.cues()
        for reason, count in self.skipped.items():
            coverage.skip(reason, count)
        if self.skipped:
            logger.warning(
                "SubtitleReader: %s: dropped %d malformed block(s).",
                self.file_name,
                sum(self.skipped.values()),
            )
        if not cues:
            if self.last_encoding:
                logger.warning("SubtitleReader: %s holds no timed cue.", self.file_name)
            return
        meta: dict[str, Any] = {
            "subtitle_format": self._format(),
            "encoding": self.last_encoding,
        }
        # LRC tags name the song; every chunk carries them.
        for tag, key in (
            ("ti", "source_title"),
            ("ar", "source_author"),
            ("al", "album"),
        ):
            if self.lrc_meta.get(tag):
                meta[key] = self.lrc_meta[tag]
        group: list[tuple[int, dict[str, Any]]] = []
        for index, cue in enumerate(cues):
            coverage.read += 1
            if group and (
                len(group) >= self.cues_per_chunk
                or (
                    self.max_chunk_seconds is not None
                    and cue["timecode_end"] - group[0][1]["timecode_start"]
                    > self.max_chunk_seconds
                )
            ):
                yield _group_chunk(group, meta)
                group = []
            group.append((index, cue))
        if group:
            yield _group_chunk(group, meta)


def _group_chunk(
    group: list[tuple[int, dict[str, Any]]], meta: dict[str, Any]
) -> dict[str, Any]:
    cues = [cue for _, cue in group]
    chunk = {
        "text": " ".join(c["text"] for c in cues),
        "raw_text": "\n\n".join(c["raw_text"] for c in cues),
        "section_type": SectionType.TEXT.value,
        "source_type": SourceType.SUBTITLE.value,
        "timecode_start": min(c["timecode_start"] for c in cues),
        "timecode_end": max(c["timecode_end"] for c in cues),
        "cue_start": group[0][0],
        "cue_end": group[-1][0],
        **meta,
    }
    speakers = {c["speaker"] for c in cues if c.get("speaker")}
    if len(speakers) == 1:
        chunk["speaker"] = speakers.pop()
    elif speakers:
        chunk["speakers"] = sorted(speakers)
    return chunk
