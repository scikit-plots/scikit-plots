"""
The cleaner: a validated plan applied to text, files, folders and archives.

:class:`Cleaner` is what :meth:`FluentCleanPrompt.materialize
<scikitplot.cleanprompt._plan.FluentCleanPrompt.materialize>` returns. The plan
says *what* to hide; the cleaner reads each input with the format that claims
it, hides what the plan's packs and the core patterns find, and remembers every
value it removed so a model's reply can be put back.

Notes
-----
**User notes.** One cleaner is one conversation or one batch. Every file it
encodes shares a single vault, so the same patient number is ``[MRN-1]`` in the
CSV, in the notebook and in the note that mentions it::

    from scikitplot.cleanprompt import FluentCleanPrompt

    with FluentCleanPrompt().packs("patient").materialize() as cleaner:
        for item in cleaner.encode_tree("study/", "study_safe/"):
            print(item.status, item.relative, item.reason)
        reply = ask_the_model(open("study_safe/cohort.csv").read())
        print(cleaner.decode(reply))
    # leaving the block drops every value the cleaner held

What each kind of file becomes:

========================  ==========================  =====================
format                    output                      decode restores
========================  ==========================  =====================
text, markdown, csv,      the same file, values       the original bytes
json, env, shell, …       replaced in place
python, notebook          the same file, columns      the original bytes
                          renamed, outputs dropped
docx, xlsx, pptx, pdf     ``name.ext.txt`` holding    the extracted text
                          the redacted text
zip                       a new zip of the above      each member as above
========================  ==========================  =====================

A file no selected format reads is **skipped** and a file that cannot be read
safely is **refused**; neither is ever written to the output, so nothing
unexamined leaves the machine. Both are listed with a reason.

**Developer notes — the decisions that make this safe.**

*One vault, seeded forward* (invariant ``I4`` kept). The engine stays
stateless: each call receives the entries issued so far as a ``seed``, which
is how the session layer already links turns. The cleaner owns the history;
the engine never does.

*JSON stays JSON.* A placeholder written over a JSON number or ``true`` would
leave a file no parser opens, and a model given invalid JSON "repairs" it —
usually by quoting the placeholder, which then restores as a string. So in a
JSON document every span is first clipped to the scalar token it falls in
(never across structure, never through an escape sequence), a number or
literal token is always replaced whole, and its stand-in is a reserved
negative number from :func:`sentinel` rather than a bracket label. The output
is parsed before it is returned; a document that no longer parses is an error,
never a result.

*Refuse rather than guess.* A file that is not UTF-8, a JSON file that does
not parse, an archive member with ``..`` in its name, a nested archive: each
is reported and left out. Nothing here falls back to "read it as plain text
and hope".

*No symlink is followed and no version-control folder is entered*, so a tree
walk cannot be steered outside the folder it was given, and a repository's
history is not mistaken for its content.

See Also
--------
scikitplot.cleanprompt._plan : Builds the plan a cleaner runs.
scikitplot.cleanprompt._structured : Finds field values in each format.
scikitplot.cleanprompt._corpus : Reads PDFs and redacts corpus documents.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import re
import threading
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Annotations are strings under ``from __future__ import annotations``, so
    # ``Self`` is needed only by a type checker. Importing it at runtime made
    # the base tier depend on typing_extensions, which Python 3.11+ does not
    # ship and nothing here declares (CP-054).
    from typing_extensions import Self

from ._api import Handle, _synchronized
from ._artifacts import PATH_PATTERN, encode_artifact, path_seed
from ._canonical import canonical, value_pattern
from ._catalog import AUTO, NONE
from ._detectors import (
    DetectorRegistry,
    LiteralDetector,
    RegexDetector,
    default_registry,
)
from ._documents import regions_for
from ._engine import Redactor, restore
from ._exceptions import CleanPromptError
from ._logging import VaultScrubber, audit, get_logger
from ._office import OFFICE_EXTENSIONS, OfficeLimits, _Budget, extract_office_text
from ._packs import pack_detectors
from ._pattern_risk import enforce, pack_findings
from ._plan import CleanPlan
from ._policy import DEFAULT_POLICY, RedactionPolicy
from ._policy import profile as _profile
from ._structured import (
    FieldDetector,
    FieldRegion,
    field_regions,
    json_tokens,
    record_starts,
)
from ._types import Entry, RestorationResult, Span
from ._vault import Vault

__all__ = [
    "SKIPPED_DIRECTORIES",
    "Cleaner",
    "Encoded",
    "Item",
    "kind_totals",
    "restore_archive",
    "restore_tree",
    "sentinel",
]

logger = get_logger(__name__)

#: Directories a tree walk never enters. Their content is either history
#: (version control) or a copy of what is already being encoded (checkpoints),
#: and neither is written to the output.
SKIPPED_DIRECTORIES = frozenset(
    {".git", ".hg", ".svn", "__pycache__", ".ipynb_checkpoints"}
)

#: Office kinds whose text is read as ``label: value`` or ``label<TAB>value``.
_OFFICE_SPLITTER = {
    "docx": ("keyvalue", {"separators": [":", "\t"], "comments": [], "prose": True}),
    "pptx": ("keyvalue", {"separators": [":", "\t"], "comments": [], "prose": True}),
    "xlsx": ("sheets", {}),
}

_SENTINEL = re.compile(r"^-99\d{8}$")
_JSON_ESCAPE = re.compile(r"\\(?:u[0-9a-fA-F]{4}|.)", re.DOTALL)
#: Splitters whose documents are sequences of records no value can span.
_RECORDS = ("delimited", "jsonl")
_NATIVE = ("text", "script", "json", "jsonl", "delimited", "keyvalue", "sheets")
_ARTIFACT = ("python", "notebook")


def sentinel(ordinal: int) -> str:
    """
    Return the JSON-number stand-in for the ``ordinal``-th number hidden.

    Parameters
    ----------
    ordinal : int
        Counting from one.

    Returns
    -------
    str
        ``-99`` followed by the ordinal in eight digits.

    Raises
    ------
    ValueError
        If ``ordinal`` is not between 1 and 99,999,999.

    Notes
    -----
    **Developer notes.** Fixed width is what makes these safe to restore by
    literal match: no sentinel is a prefix of another, so ``-9900000001``
    cannot claim the front of ``-9900000011``. The cleaner also checks each
    one against the document before issuing it, exactly as a surrogate is
    checked, so a sentinel never matches something the author wrote.

    Examples
    --------
    >>> sentinel(1)
    '-9900000001'
    """
    if not 1 <= ordinal <= 99_999_999:  # ruff: ignore[magic-value-comparison]
        msg = f"sentinel ordinal must be between 1 and 99999999, got {ordinal!r}"
        raise ValueError(msg)
    return f"-99{ordinal:08d}"


@dataclass(frozen=True)
class Encoded:
    r"""
    One encoded input: safe text and a report. **Holds no removed value**.

    Parameters
    ----------
    text : str
        Safe to send.
    source : str
        The name the input was given as.
    format : str
        The format that read it.
    output_name : str
        The name to write it under: the source name for formats that round
        trip, ``name.ext.txt`` for formats whose text was extracted.
    round_trip : bool
        Whether decoding restores the original bytes rather than extracted
        text.
    report : dict
        Counts only — packs, kinds, fields matched — never a value.
    """

    text: str = field(repr=False)
    source: str
    format: str
    output_name: str
    round_trip: bool
    report: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def count(self) -> int:
        """int: Distinct values replaced in this input."""
        return int(self.report.get("values", 0))


@dataclass(frozen=True)
class Item:
    """
    What happened to one file of a folder or archive.

    Parameters
    ----------
    status : {'encoded', 'decoded', 'skipped', 'refused'}
        ``skipped``: no selected format reads it, or it is a link or a
        version-control folder. ``refused``: it could not be read safely.
        Neither is written to the output.
    relative : str
        Its path relative to the folder or archive root, with ``/``.
    format : str or None
        The format that read it.
    output : str or None
        Where it was written, relative to the output root.
    reason : str
        Why it was skipped or refused; empty otherwise.
    count : int
        Distinct values replaced.
    kinds : dict
        Distinct values replaced, by kind.
    """

    status: str
    relative: str
    format: str | None = None
    output: str | None = None
    reason: str = ""
    count: int = 0
    kinds: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _Setup:
    """The detectors and vocabulary one format's packs contribute."""

    packs: tuple[str, ...]
    index: dict
    detectors: tuple
    vocabulary: Any


#: Stand-in kinds ``remember`` does not repeat into prose; see
#: :meth:`Cleaner._known_detectors`.
_NOT_REMEMBERED = frozenset({"COLUMN", "FIGURE", "OUTPUT"})


class _KnownValues:
    r"""
    Find values already hidden, wherever they recur, as whole tokens.

    Notes
    -----
    **Developer notes.** Bounded by "no word character on either side"
    rather than by ``\b``: ``\b`` needs a word character *inside* the edge,
    so it never bounds ``+1 555 010 4477`` or a value ending in ``.``. Longest
    first, so a value containing another is found whole (``CP-001``).

    **Developer notes — a recurrence is the value however it is written
    (``CP-070``).** A name learned from a CSV cell recurs in prose in
    capitals, broken across a line, with a non-breaking space, in full-width
    letters or with a typographic apostrophe; each of those was sent in the
    clear. Matching runs on :func:`canonical` text, case-insensitively, with
    any run of whitespace standing for any other. These are equivalences, not
    guesses: reordering (``Holt, Marion``), initials and a surname alone are
    not matched. The span is the original surface, so the engine gives a
    differently written recurrence its own label and restoration stays exact.
    """

    __slots__ = ("_pattern", "confidence", "kind", "name", "priority")

    def __init__(self, kind: str, values: Iterable[str], ignore_case: bool) -> None:
        del ignore_case  # a hidden value is hidden in any case (CP-070)
        self.kind = kind
        self.name = f"known:{kind}"
        self.priority = 90
        self.confidence = 1.0
        self._pattern = value_pattern(values)

    def detect(self, text: str, policy: RedactionPolicy) -> Iterator[Span]:
        """Yield one span per recurrence, over the original text."""
        del policy
        for match in self._pattern.finditer(canonical(text)):
            if match.end() > match.start():
                yield Span(
                    start=match.start(),
                    end=match.end(),
                    kind=self.kind,
                    text=text[match.start() : match.end()],
                    detector=self.name,
                    priority=self.priority,
                    confidence=1.0,
                )


class _TokenBound:
    """
    Keep a detector's spans inside JSON scalar tokens.

    Notes
    -----
    **Developer notes.** A span is cut at every token boundary: the part in a
    string is kept to the string's inside, moved outward so it never splits an
    escape sequence; a part touching a number or a literal becomes that whole
    token; a part over structure or whitespace is dropped, because JSON puts
    nothing there but punctuation. Clipping never hides *less* inside a token
    than the detector found, so it cannot open a leak.

    A number token that is itself a sentinel (``-99`` and eight digits) is
    never reported: it is a stand-in already, reserved exactly as a bracket
    placeholder is, which is what keeps re-encoding idempotent (``I7``).
    """

    __slots__ = (
        "_inner",
        "_starts",
        "_tokens",
        "confidence",
        "kind",
        "name",
        "priority",
    )

    def __init__(self, inner: Any, tokens: list[tuple[int, int, str]]) -> None:
        self._inner = inner
        self._tokens = tokens
        self._starts = [token[0] for token in tokens]
        self.name = inner.name
        self.kind = inner.kind
        self.priority = inner.priority
        self.confidence = inner.confidence

    def detect(self, text: str, policy: RedactionPolicy) -> Iterator[Span]:
        """Yield the detector's spans, clipped to tokens."""
        for span in self._inner.detect(text, policy):
            seen = set()
            first = max(0, bisect.bisect_right(self._starts, span.start) - 1)
            for start, end, token in self._tokens[first:]:
                if start >= span.end:
                    break
                if end <= span.start:
                    continue
                if token == "string":  # ruff: ignore[hardcoded-password-string]
                    lo, hi = max(span.start, start + 1), min(span.end, end - 1)
                    if hi <= lo:
                        continue
                    lo, hi = _outside_escapes(text, start + 1, end - 1, lo, hi)
                elif _SENTINEL.match(text[start:end]):
                    # A stand-in this module issued: reserved, like a
                    # placeholder, so encoding an encoded file changes nothing.
                    continue
                else:
                    lo, hi = start, end
                if (lo, hi) in seen:
                    continue
                seen.add((lo, hi))
                yield replace(span, start=lo, end=hi, text=text[lo:hi])


def _outside_escapes(
    text: str, inner_start: int, inner_end: int, lo: int, hi: int
) -> tuple[int, int]:
    """Widen ``[lo, hi)`` so neither end falls inside a JSON escape sequence."""
    for match in _JSON_ESCAPE.finditer(text, inner_start, inner_end):
        if match.start() >= hi:
            break
        if match.start() < lo < match.end():
            lo = match.start()
        if match.start() < hi < match.end():
            hi = match.end()
    return lo, hi


def _parses(text: str, splitter: str) -> bool:
    """Return whether a JSON or JSON Lines document still parses."""
    try:
        if splitter == "json":
            json.loads(text)
        else:
            for line in text.split("\n"):
                if line.strip():
                    json.loads(line)
    except ValueError:
        return False
    return True


def _safe_member(name: str) -> str | None:
    """Return why an archive member name is unsafe, or ``None``."""
    if "\\" in name or "\x00" in name:
        return "its name contains a backslash or NUL"
    pure = PurePosixPath(name)
    if pure.is_absolute() or re.match(r"^[A-Za-z]:", name):
        return "its name is an absolute path"
    if ".." in pure.parts:
        return "its name climbs out of the archive with '..'"
    return None


class Cleaner:
    """
    Apply one validated plan to text, files, folders and archives.

    Parameters
    ----------
    plan : CleanPlan, optional
        Defaults to ``CleanPlan()``: packs chosen by each file's format, every
        format, placeholders.
    limits : OfficeLimits, optional
        Size bounds for a file read from disk (``max_part_bytes``), and for
        the members and total size of an Office file or archive.
    prior : iterable of Entry, optional
        Entries issued earlier — a vault being appended to. A value among them
        keeps its stand-in, and new stand-ins never repeat one of theirs.
    chunk_chars : int, optional
        Largest piece of a CSV, TSV or JSON Lines file encoded at once.
        Defaults to the policy's ``max_input_chars``. Larger record files are
        encoded piece by piece, cut between records, with the same result as
        one pass; any other format above the limit is refused.

    Raises
    ------
    CleanPromptError
        If the plan does not validate, or an entity engine it asks for is not
        installed.

    Notes
    -----
    **User notes.** Use it as a context manager, or call :meth:`clear` when
    done: the cleaner holds every value it removed until then. Its
    representation shows counts, never values.

    **Developer notes.** Not thread-safe: the shared vault is the point of a
    cleaner, and two threads appending to it would number values in an order
    that depends on scheduling. Use one cleaner per thread when determinism
    across runs matters, which it does for anything tested.

    Examples
    --------
    >>> from scikitplot.cleanprompt._plan import FluentCleanPrompt
    >>> cleaner = FluentCleanPrompt().packs("patient").materialize()
    >>> out = cleaner.encode_text('{"mrn": 20417, "email": "ann@example.com"}', "json")
    >>> out.text
    '{"mrn": -9900000001, "email": "[EMAIL-1]"}'
    >>> cleaner.decode(out.text)
    '{"mrn": 20417, "email": "ann@example.com"}'
    """

    __slots__ = (
        "__weakref__",
        "_chunk_chars",
        "_closed",
        "_entries",
        "_fingerprint",
        "_formats",
        "_known",
        "_learning",
        "_limits",
        "_lock",
        "_ner",
        "_plan",
        "_policy",
        "_scrubber",
        "_sentinels",
        "_setups",
        "catalog",
    )

    def __init__(
        self,
        plan: CleanPlan | None = None,
        limits: OfficeLimits | None = None,
        prior: Iterable[Entry] = (),
        chunk_chars: int | None = None,
    ) -> None:
        plan = plan if plan is not None else CleanPlan()
        problems = plan.validate()
        if problems:
            msg = "the plan is not valid:\n" + "\n".join(f"  - {p}" for p in problems)
            raise CleanPromptError(msg)
        self._plan = plan
        self._limits = limits if limits is not None else OfficeLimits()
        self.catalog = plan.catalog()
        if plan.custom:
            # The plan's own mode: validate() already refused under 'refuse';
            # here 'warn' speaks once per open finding, at the point of use.
            enforce(
                pack_findings(self.catalog.packs.values()),
                plan.pattern_risk_mode(),
                ", ".join(plan.custom),
            )
        self._formats = self.catalog.resolve_formats(plan.formats_selection())
        policy = _profile(plan.profile) if plan.profile else DEFAULT_POLICY
        tag_style = replace(
            policy.tag_style, style=plan.style, surrogate_set=plan.surrogate_set()
        )
        self._policy = policy.evolve(allow=plan.allow, tag_style=tag_style)
        if plan.hide:
            Redactor._check_terms(  # noqa: SLF001 - same validation as the engine
                plan.hide,
                self._policy.limits,
            )
        self._ner = self._build_ner()
        self._setups: dict[str, _Setup] = {}
        self._lock = threading.RLock()
        self._entries: dict[str, Entry] = {}
        self._known: tuple[int, tuple] = (-1, ())
        self._learning = False
        if chunk_chars is not None and (
            isinstance(chunk_chars, bool)
            or not isinstance(chunk_chars, int)
            or chunk_chars < 1
        ):
            msg = f"chunk_chars must be a positive integer, got {chunk_chars!r}"
            raise CleanPromptError(msg)
        self._chunk_chars = chunk_chars
        self._scrubber = VaultScrubber(owner=self)
        self._fingerprint = plan.fingerprint()
        self._absorb(prior)
        self._sentinels = max(
            (int(label[3:]) for label in self._entries if _SENTINEL.match(label)),
            default=0,
        )
        self._closed = False

    # -- construction helpers -------------------------------------------

    def _build_ner(self) -> tuple:
        """Return the entity detectors the plan asks for, or none."""
        if self._plan.ner == NONE:
            return ()
        from ._engines import build_detectors  # ruff: ignore[import-outside-top-level]

        return tuple(
            build_detectors(
                mode=self._plan.ner, language=self._plan.language, required=True
            )
        )

    def _setup(self, fmt: Any) -> _Setup:
        """Return, and cache, the pack contribution for one format."""
        cached = self._setups.get(fmt.name)
        if cached is not None:
            return cached
        selection = self._plan.packs_selection()
        packs = self.catalog.resolve_packs(
            selection, formats=(fmt,) if selection == AUTO else None
        )
        setup = _Setup(
            packs=tuple(p.name for p in packs),
            index=self.catalog.field_index(packs),
            detectors=tuple(pack_detectors(packs)),
            vocabulary=self.catalog.code_vocabulary(packs),
        )
        self._setups[fmt.name] = setup
        return setup

    def _check_open(self) -> None:
        """Raise if :meth:`clear` has been called."""
        if self._closed:
            msg = "this cleaner was cleared; its values are gone, so it cannot encode or decode"
            raise CleanPromptError(msg)

    # -- identity ---------------------------------------------------------

    @property
    def plan(self) -> CleanPlan:
        """CleanPlan: The plan this cleaner runs."""
        return self._plan

    @property
    def policy(self) -> RedactionPolicy:
        """RedactionPolicy: The policy labels are issued under."""
        return self._policy

    @property
    def formats(self) -> tuple[str, ...]:
        """Tuple of str: Names of the formats this cleaner reads."""
        return tuple(spec.name for spec in self._formats)

    def fingerprint(self) -> str:
        """
        Return the plan's content fingerprint.

        Returns
        -------
        str
            As :meth:`CleanPlan.fingerprint`.
        """
        return self._plan.fingerprint()

    def __repr__(self) -> str:
        state = "cleared" if self._closed else f"{len(self._entries)} value(s)"
        return f"Cleaner({state}, formats={len(self._formats)}, packs={list(self._plan.packs)})"

    __str__ = __repr__

    def __len__(self) -> int:
        return len(self._entries)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.clear()

    @_synchronized
    def clear(self) -> None:
        """Drop every value this cleaner holds. It cannot be used afterwards."""
        self._entries.clear()
        self._known = (-1, ())
        self._scrubber.close()
        self._closed = True

    # -- the vault ------------------------------------------------------

    @_synchronized
    def vault(self) -> Vault:
        """
        Return a vault of every value removed so far.

        Returns
        -------
        Vault
            A fresh copy; clearing it does not clear the cleaner.
        """
        self._check_open()
        mapping = {label: entry.original for label, entry in self._entries.items()}
        return Vault(mapping, grammar_fingerprint=self._policy.tag_style.fingerprint)

    @_synchronized
    def handle(self) -> Handle:
        """
        Return a :class:`~scikitplot.cleanprompt.Handle` over every value removed.

        Returns
        -------
        Handle
            Usable with :func:`~scikitplot.cleanprompt.decode`, and exportable
            with :meth:`Handle.export` for a vault file.
        """
        return Handle(
            vault=self.vault(),
            policy=self._policy,
            entries=tuple(self._entries.values()),
        )

    @_synchronized
    def leaks(self, text: str) -> tuple[str, ...]:
        """
        Return the kinds of held values that still occur in ``text``.

        Parameters
        ----------
        text : str
            Text about to leave, usually an encoded output.

        Returns
        -------
        tuple of str
            One kind per distinct value found, sorted; empty when clean.

        Notes
        -----
        **Developer notes.** An independent second check, not a repeat of the
        first: encoding decides spans with detectors and arbitration; this
        searches the finished text for the removed values themselves, as whole
        tokens. It checks exactly the kinds ``remember`` repeats, so with
        ``remember`` on it can only fire on a defect, and with it off it fires
        on every recurrence the rules did not catch. Nothing it returns is a
        value.
        """
        self._check_open()
        by_kind: dict[str, set] = {}
        labels = set(self._entries)
        for entry in self._entries.values():
            if (
                entry.kind in _NOT_REMEMBERED
                or not entry.original.strip()
                or entry.original in labels
            ):
                continue
            by_kind.setdefault(entry.kind, set()).add(entry.original)
        found = []
        for kind, values in sorted(by_kind.items()):
            seen = {
                span.text
                for span in _KnownValues(
                    kind, values, self._policy.case_insensitive
                ).detect(text, self._policy)
            }
            found.extend([kind] * len(seen))
        return tuple(found)

    def _absorb(self, entries: Iterable[Entry]) -> None:
        """
        Record new entries; positions are per document and are not kept.

        Notes
        -----
        **Developer notes.** Every value recorded is also handed to this
        cleaner's :class:`~scikitplot.cleanprompt._logging.VaultScrubber`, so
        from this moment no log record from this submodule can carry it.
        """
        fresh = []
        for entry in entries:
            if entry.label not in self._entries:
                self._entries[entry.label] = replace(entry, occurrences=())
                fresh.append(entry.original)
        self._scrubber.add(fresh)

    # -- format resolution ---------------------------------------------

    def _format(self, name_or_path: str, explicit: str | None) -> Any:
        """Return the selected format for an explicit name or a file name."""
        if explicit is not None:
            chosen = self.catalog.resolve_formats(explicit)[0]
            if chosen not in self._formats:
                msg = f"format {chosen.name!r} is not selected by this plan (selected: {', '.join(self.formats)})"
                raise CleanPromptError(msg)
            return chosen
        found = self.catalog.format_for(name_or_path, self._formats)
        if found is None:
            msg = f"no selected format reads {name_or_path!r}"
            raise CleanPromptError(msg)
        return found

    def text_format(self, format: str) -> str:  # noqa: A002 - public keyword
        """
        Return the name of the text format ``format`` resolves to here.

        Parameters
        ----------
        format : str
            A format name or extension.

        Returns
        -------
        str
            The format's name.

        Raises
        ------
        CleanPromptError
            If the plan does not select it, or it is read from bytes.
        """
        return self._text_format(format).name

    def _text_format(self, format: str) -> Any:  # noqa: A002 - public keyword
        """Resolve a format that :meth:`encode_text` accepts."""
        fmt = self._format(format, format)
        if fmt.splitter not in (*_NATIVE, *_ARTIFACT):
            msg = f"format {fmt.name!r} is read from bytes; use encode_file or encode_bytes"
            raise CleanPromptError(msg)
        return fmt

    # -- encoding: text -------------------------------------------------

    def encode_text(
        self, text: str, format: str = "text", name: str | None = None
    ) -> Encoded:  # noqa: A002 - public keyword
        """
        Encode a string read as one format.

        Parameters
        ----------
        text : str
            The content.
        format : str, default='text'
            A format name or extension. Formats whose text must first be
            extracted from bytes (Office, PDF, zip) are refused here.
        name : str, optional
            A name for reports; defaults to ``<text>``.

        Returns
        -------
        Encoded
            The safe text and a report.

        Raises
        ------
        TypeError
            If ``text`` is not a string.
        CleanPromptError
            If the format is not selected, needs bytes, or the text does not
            parse as it.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__!r}")
        self._check_open()
        fmt = self._text_format(format)
        return self._encode_string(
            text, fmt, name or "<text>", fmt.splitter, fmt.option_map()
        )

    @_synchronized
    def _encode_string(
        self, text: str, fmt: Any, name: str, splitter: str, options: dict
    ) -> Encoded:
        """Dispatch a decoded document to the path its splitter needs."""
        setup = self._setup(fmt)
        prior = tuple(self._entries.values())
        if splitter in _ARTIFACT:
            safe, entries, report = self._encode_artifact(
                text, splitter, name, setup, prior
            )
        elif splitter in _RECORDS and len(text) > self._chunk_limit():
            safe, entries, report = self._encode_chunked(text, splitter, options, setup)
        else:
            safe, entries, report = self._encode_native(
                text, splitter, options, setup, prior
            )
        self._absorb(entries)
        report.update(
            {
                "format": fmt.name,
                "packs": list(setup.packs),
                "values": len(entries),
                "kinds": _count_kinds(entries),
                "preexisting_labels": _preexisting(text, prior, self._policy),
            }
        )
        if report["preexisting_labels"] and not self._learning:
            # Counts only: a file name can itself be personal data.
            logger.warning(
                "an input already contains %d stand-in(s) issued earlier; decoding it will "
                "replace those too",
                len(report["preexisting_labels"]),
            )
        extracted = splitter != fmt.splitter
        audit(
            "learned" if self._learning else "encoded",
            format=fmt.name,
            packs=list(setup.packs),
            values=len(entries),
            kinds=report["kinds"],
            plan=self._fingerprint[:16],
            output_sha256=hashlib.sha256(safe.encode("utf-8")).hexdigest()[:16],
        )
        return Encoded(
            text=safe,
            source=name,
            format=fmt.name,
            output_name=(
                f"{PurePosixPath(name).name}.txt"
                if extracted
                else PurePosixPath(name).name
            ),
            round_trip=fmt.round_trip and not extracted,
            report=report,
        )

    def _registry(
        self,
        setup: _Setup,
        regions: tuple[FieldRegion, ...],
        known: tuple | None = None,
    ) -> DetectorRegistry:
        """Build the detector set for a text-native document."""
        plan = self._plan
        registry = default_registry(kinds=self._policy.kinds if plan.core else ())
        for detector in setup.detectors:
            registry.add(detector)
        if setup.index and regions:
            registry.add(FieldDetector(regions, setup.index))
        if "paths" not in plan.keep:
            registry.add(RegexDetector(PATH_PATTERN))
        if plan.hide:
            registry.add(
                LiteralDetector(
                    plan.hide, kind="CUSTOM", ignore_case=self._policy.case_insensitive
                )
            )
        for detector in self._ner:
            registry.add(detector)
        for detector in self._known_detectors() if known is None else known:
            registry.add(detector)
        return registry

    def _known_detectors(self) -> tuple:
        """
        Return one detector per kind for every value hidden so far.

        Notes
        -----
        **Developer notes.** This is ``remember``: a value found once — by a
        field rule in a CSV, say — is found again wherever it recurs, in prose
        no rule could read. One detector per *kind*, so a recurrence carries
        the kind it was first hidden under and the seed hands it the same
        label. Column, figure and output stand-ins are excluded: they rename
        code identifiers and rendered blobs, which the artefact layer places
        with identifier boundaries; repeating a column name such as ``age``
        across prose would hide an English word. Cached on the number of
        entries, since entries are only ever appended.
        """
        if not self._plan.remember:
            return ()
        count, detectors = self._known
        if count == len(self._entries):
            return detectors
        by_kind: dict[str, set] = {}
        for entry in self._entries.values():
            if entry.kind in _NOT_REMEMBERED or not entry.original.strip():
                continue
            by_kind.setdefault(entry.kind, set()).add(entry.original)
        detectors = tuple(
            _KnownValues(kind, values, self._policy.case_insensitive)
            for kind, values in sorted(by_kind.items())
        )
        self._known = (len(self._entries), detectors)
        return detectors

    def _encode_native(  # ruff: ignore[too-many-positional-arguments]
        self,
        text: str,
        splitter: str,
        options: dict,
        setup: _Setup,
        prior: tuple[Entry, ...],
        regions: tuple[FieldRegion, ...] | None = None,
        known: tuple | None = None,
    ) -> tuple[str, tuple[Entry, ...], dict]:
        """Encode a document whose fields are found by a splitter."""
        if regions is None:
            regions = (
                field_regions(text, splitter, options)
                if setup.index or splitter in ("json", "jsonl")
                else ()
            )
        registry = self._registry(setup, regions, known)
        tokens = json_tokens(text) if splitter in ("json", "jsonl") else None
        if tokens is not None:
            registry = DetectorRegistry(
                _TokenBound(detector, tokens) for detector in registry
            )
        policy = self._policy.evolve(kinds=None)
        seed = list(prior)
        if "paths" not in self._plan.keep:
            seed.extend(path_seed(text, seed))
        redactor = Redactor(policy=policy, registry=registry)
        result = redactor.redact(text, seed=seed)
        if tokens is not None:
            result = self._sentinel_pass(text, tokens, result, redactor, seed, prior)
            if not _parses(result.text, splitter):
                msg = "internal error: the redacted document is no longer valid JSON; nothing was returned"
                raise CleanPromptError(msg)
        matched = (
            sum(1 for _ in FieldDetector(regions, setup.index).matches())
            if setup.index
            else 0
        )
        return result.text, result.entries, {"fields": matched}

    def _chunk_limit(self) -> int:
        """Return the largest piece of a record file encoded in one call."""
        return self._chunk_chars or self._policy.limits.max_input_chars

    def _encode_chunked(
        self, text: str, splitter: str, options: dict, setup: _Setup
    ) -> tuple[str, tuple[Entry, ...], dict]:
        """
        Encode a large CSV, TSV or JSON Lines file in record-aligned pieces.

        Notes
        -----
        **Developer notes — why this equals encoding the file at once.**

        *Cut only between records.* A CSV record (quoted newlines included)
        and a JSON Lines line hold whole values, so no field value and no
        value a rule could sensibly find is cut. Boundaries come from the same
        scan that locates the cells (:func:`record_starts`).

        *Fields keep their names.* Field regions are located once over the
        whole file — the CSV header is only in the first piece — and each
        piece gets its own, shifted.

        *One vault, in order.* Each piece is seeded with every entry issued so
        far, so labels continue exactly as a single pass would issue them.

        *Remembered values are frozen.* The known-value detectors are taken
        once, before the first piece, so a value first hidden in piece one is
        not found in piece two by memory alone — a single pass could not have
        done that either.

        A single record longer than the limit is refused with the limit's own
        error, as an unchunked file would be.
        """
        limit = self._chunk_limit()
        # The loop below walks from offset 0; the scan puts a record there for
        # any non-empty text, and the set makes that an invariant, not a hope.
        starts = [*sorted({0, *record_starts(text, splitter, options)}), len(text)]
        regions = (
            field_regions(text, splitter, options)
            if setup.index or splitter == "jsonl"
            else ()
        )
        known = self._known_detectors()
        pieces: list[str] = []
        issued: dict[str, Entry] = {}
        fields_matched = 0
        low = 0
        cursor = 0
        chunks = 0
        while low < len(text):
            high = low
            while cursor + 1 < len(starts) and starts[cursor + 1] - low <= limit:
                cursor += 1
                high = starts[cursor]
            if high == low:  # one record longer than the limit
                cursor += 1
                high = starts[cursor]
            piece = text[low:high]
            local = tuple(
                FieldRegion(r.start - low, r.end - low, r.field, r.token)
                for r in regions
                if low <= r.start and r.end <= high
            )
            safe, entries, report = self._encode_native(
                piece,
                splitter,
                options,
                setup,
                tuple(self._entries.values()),
                regions=local,
                known=known,
            )
            self._absorb(entries)
            for entry in entries:
                issued.setdefault(entry.label, entry)
            fields_matched += report["fields"]
            pieces.append(safe)
            chunks += 1
            low = high
        return (
            "".join(pieces),
            tuple(issued.values()),
            {"fields": fields_matched, "chunks": chunks},
        )

    def _sentinel_pass(  # ruff: ignore[too-many-positional-arguments]
        self,
        text: str,
        tokens: list[tuple[int, int, str]],
        result: Any,
        redactor: Redactor,
        seed: list[Entry],
        prior: tuple[Entry, ...],
    ) -> Any:
        """
        Re-issue every value that sits on a JSON number or literal as a sentinel.

        Notes
        -----
        **Developer notes.** The first pass decides *which* values are hidden;
        this one only changes how some of them are spelled. Every new entry of
        the first pass is seeded back with its label, except those on a number
        or literal token, which are seeded with a sentinel. The detectors and
        the text are unchanged, so the second pass resolves the same spans and
        the only difference is the stand-in.
        """
        scalars = sorted(
            (start, end)
            for start, end, token in tokens  # lint
            if token != "string"  # ruff: ignore[hardcoded-password-string]
        )
        starts = [start for start, _ in scalars]
        known = {entry.label for entry in prior}
        needs = set()
        for entry in result.entries:
            for start, end in entry.occurrences:
                first = max(0, bisect.bisect_right(starts, start) - 1)
                for low, high in scalars[first : bisect.bisect_left(starts, end)]:
                    if not (low < end and start < high):
                        continue
                    if (low, high) != (start, end):
                        msg = "internal error: a span covers part of a JSON number; nothing was returned"
                        raise CleanPromptError(msg)
                    needs.add(entry.label)
        pending = [
            entry
            for entry in result.entries
            if entry.label in needs
            and entry.label not in known
            and not _SENTINEL.match(entry.label)
        ]
        if not pending:
            return result
        taken = set(self._entries) | {entry.label for entry in result.entries}
        relabel = {}
        for entry in pending:
            while True:
                self._sentinels += 1
                label = sentinel(self._sentinels)
                if label not in taken and label not in text:
                    break
            taken.add(label)
            relabel[entry.label] = label
        fresh = [
            replace(entry, label=relabel.get(entry.label, entry.label), occurrences=())
            for entry in result.entries
            if entry.label not in known
        ]
        return redactor.redact(text, seed=[*seed, *fresh])

    def _encode_artifact(
        self,
        text: str,
        splitter: str,
        name: str,
        setup: _Setup,
        prior: tuple[Entry, ...],
    ) -> tuple[str, tuple[Entry, ...], dict]:
        """Encode a Python module or a notebook through the artefact layer."""
        plan = self._plan
        regions = field_regions(text, splitter) if setup.index else ()
        if splitter == "notebook" and regions:
            readable = [
                r
                for r in regions_for(text, "notebook", name)
                if r.role not in ("metadata", "binary")
            ]
            regions = tuple(
                region
                for region in regions
                if any(
                    r.start <= region.start and region.end <= r.end for r in readable
                )
            )
        extra = list(setup.detectors)
        extra.extend(self._known_detectors())
        if setup.index and regions:
            extra.append(FieldDetector(regions, setup.index))
        extra.extend(self._ner)
        policy = self._policy if plan.core else self._policy.evolve(kinds=None)
        result, artifact = encode_artifact(
            text,
            fmt=splitter,
            path=name,
            policy=policy,
            declared_roles=dict(plan.roles) or None,
            infer_roles=plan.infer_roles,
            hide_paths="paths" not in plan.keep,
            drop_binary="figures" not in plan.keep,
            drop_outputs="outputs" not in plan.keep,
            extra_terms=plan.hide or None,
            extra_detectors=extra,
            prior=prior,
            vocabulary=setup.vocabulary,
            core=plan.core,
        )
        matched = (
            sum(1 for _ in FieldDetector(regions, setup.index).matches())
            if setup.index
            else 0
        )
        report = {
            "fields": matched,
            "columns": len(artifact.columns),
            "refused_columns": len(artifact.refused),
            "unparsed": len(artifact.unparsed),
        }
        return result.text, result.entries, report

    # -- encoding: bytes and files -------------------------------------

    def encode_bytes(
        self, data: bytes, name: str, format: str | None = None
    ) -> Encoded:  # noqa: A002 - public keyword
        """
        Encode a file's bytes, choosing the format by name unless given.

        Parameters
        ----------
        data : bytes
            The file's content.
        name : str
            The file name; its extension selects the format.
        format : str, optional
            Force a format by name or extension.

        Returns
        -------
        Encoded
            The safe text and a report.

        Raises
        ------
        CleanPromptError
            If no selected format reads it, it is not UTF-8, it does not parse
            as its format, or it is a PDF or zip (which need a file).
        """
        self._check_open()
        fmt = self._format(name, format)
        return self._encode_data(data, name, fmt)

    def _encode_data(
        self, data: bytes, name: str, fmt: Any, budget: _Budget | None = None
    ) -> Encoded:
        """
        Encode bytes already matched to a format.

        Notes
        -----
        **Developer notes.** ``budget`` is an enclosing archive's. An Office
        member is itself a zip; decompressing it against its own fresh budget
        would bound each member but not the archive, so ten thousand small
        ``.docx`` members could each expand to the per-file ceiling.
        """
        if fmt.splitter == "office":
            kind = OFFICE_EXTENSIONS.get(PurePosixPath(name).suffix.lower())
            if kind is None:
                msg = f"{name!r} is not an Office Open XML file"
                raise CleanPromptError(msg)
            text = extract_office_text(data, kind, self._limits, budget=budget)
            splitter, options = _OFFICE_SPLITTER[kind]
            return self._encode_string(text, fmt, name, splitter, dict(options))
        if fmt.splitter in ("corpus", "archive"):
            msg = f"format {fmt.name!r} is read from a file on disk; use encode_file or encode_archive"
            raise CleanPromptError(msg)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            msg = f"{name!r} is not UTF-8 (byte {exc.start}); convert it first, nothing was read"
            raise CleanPromptError(msg) from exc
        return self._encode_string(text, fmt, name, fmt.splitter, fmt.option_map())

    def encode_file(
        self, path: str | os.PathLike, format: str | None = None
    ) -> Encoded:  # noqa: A002 - public keyword
        """
        Encode one file from disk.

        Parameters
        ----------
        path : path-like
            The file.
        format : str, optional
            Force a format by name or extension.

        Returns
        -------
        Encoded
            The safe text and a report. Nothing is written.

        Raises
        ------
        CleanPromptError
            If the file is missing, too large, or cannot be read safely.
        """
        self._check_open()
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            msg = f"{str(source)!r} is not a regular file"
            raise CleanPromptError(msg)
        fmt = self._format(source.name, format)
        if fmt.splitter == "archive":
            msg = f"{source.name!r} is an archive; use encode_archive, which writes a new one"
            raise CleanPromptError(msg)
        size = source.stat().st_size
        if size > self._limits.max_part_bytes:
            msg = f"{source.name!r} is {size} bytes, above the limit of {self._limits.max_part_bytes}; nothing was read"
            raise CleanPromptError(msg)
        if fmt.splitter == "corpus":
            from ._corpus import read_text  # ruff: ignore[import-outside-top-level]

            return self._encode_string(read_text(source), fmt, source.name, "text", {})
        return self._encode_data(source.read_bytes(), source.name, fmt)

    def encode_files(self, paths: Iterable[str | os.PathLike]) -> list[Encoded]:
        """
        Encode several files into the one shared vault, in the order given.

        Parameters
        ----------
        paths : iterable of path-like
            The files.

        Returns
        -------
        list of Encoded
            One per path.

        Raises
        ------
        CleanPromptError
            On the first file that cannot be encoded.
        """
        paths = list(paths)
        self._learn([(lambda one=path: self.encode_file(one)) for path in paths])
        return [self.encode_file(path) for path in paths]

    # -- encoding: folders and archives -------------------------------

    def encode_tree(
        self, source: str | os.PathLike, target: str | os.PathLike
    ) -> Iterator[Item]:
        """
        Encode every readable file under a folder into a mirror folder.

        Parameters
        ----------
        source : path-like
            The folder to read.
        target : path-like
            Where encoded files are written, mirroring ``source``. Created if
            missing; must not be inside ``source``.

        Yields
        ------
        Item
            One per file or skipped folder, in sorted path order.

        Raises
        ------
        CleanPromptError
            If ``source`` is not a folder or ``target`` is inside it.

        Notes
        -----
        **User notes.** It is a generator: files are encoded as you iterate,
        so wrap it in :func:`list` to run it all. A file that is skipped or
        refused is not written, and says why.
        """
        self._check_open()
        root, out = _roots(source, target)
        return self._walk_tree(root, out)

    def survey_tree(self, source: str | os.PathLike) -> list[Item]:
        """
        Report what encoding a folder would do, writing nothing.

        Parameters
        ----------
        source : path-like
            The folder.

        Returns
        -------
        list of Item
            As :meth:`encode_tree` would yield, with ``output`` left empty.

        Notes
        -----
        **User notes.** A dry run to read before sending a folder anywhere:
        which files would be encoded, skipped or refused, and how many values
        of each kind each holds. No file is written and this cleaner's vault
        is unchanged — the survey runs on a scratch copy of the plan.
        """
        self._check_open()
        root = Path(source).resolve()
        if not root.is_dir():
            msg = f"{str(source)!r} is not a folder"
            raise CleanPromptError(msg)
        probe = Cleaner(self._plan, self._limits, chunk_chars=self._chunk_chars)
        try:
            return list(probe._walk_tree(root, None))  # noqa: SLF001 - same class
        finally:
            probe.clear()

    def _walk_tree(self, root: Path, out: Path | None) -> Iterator[Item]:
        """Learn, then encode every file; write under ``out`` unless it is ``None``."""
        self._learn([self._tree_learn(here / name) for here, name in _tree_files(root)])
        for folder, dirnames, filenames in os.walk(root, followlinks=False):
            here = Path(folder)
            kept = []
            for name in sorted(dirnames):
                relative = (here / name).relative_to(root).as_posix()
                if name in SKIPPED_DIRECTORIES:
                    yield Item("skipped", relative + "/", reason="folder is never read")
                elif (here / name).is_symlink():
                    yield Item(
                        "skipped", relative + "/", reason="symbolic link not followed"
                    )
                else:
                    kept.append(name)
            dirnames[:] = kept
            for name in sorted(filenames):
                yield self._tree_file(here / name, root, out)

    @_synchronized
    def _learn(self, steps: Iterable[Any]) -> None:
        """
        Run an encoding pass whose only effect is to fill the vault.

        Notes
        -----
        **Developer notes — why a folder is read twice when ``remember`` is
        on.** Remembering in one pass makes the output depend on file order: a
        note sorted before the CSV that names the patient would be written
        before the name was known, and sent (measured, round fourteen). So the
        first pass encodes everything and discards the text, learning every
        value; the second pass writes, with every value already known. Labels
        are issued in the first pass, in sorted order, so the result is
        deterministic. A file that fails here is not reported here: the
        writing pass meets the same failure and reports it.
        """
        if not self._plan.remember:
            return
        self._learning = True
        try:
            for step in steps:
                if step is None:
                    continue
                try:
                    step()
                except CleanPromptError:
                    continue  # reported by the writing pass, which fails the same way
        finally:
            self._learning = False

    def _tree_learn(self, path: Path) -> Any:
        """Return a learning step for one file of a tree, or ``None``."""
        if path.is_symlink():
            return None
        fmt = self.catalog.format_for(path.name, self._formats)
        if fmt is None:
            return None
        if fmt.splitter == "archive":
            return lambda: self._archive_pass(path, None)
        return lambda: self.encode_file(path, fmt.name)

    def _tree_file(self, path: Path, root: Path, out: Path | None) -> Item:
        """Encode one file of a tree walk; write it unless ``out`` is ``None``."""
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            return Item("skipped", relative, reason="symbolic link not followed")
        fmt = self.catalog.format_for(path.name, self._formats)
        if fmt is None:
            return Item(
                "skipped", relative, reason="no selected format reads this file"
            )
        try:
            if fmt.splitter == "archive":
                if out is None:
                    items = self._archive_pass(path, None)
                else:
                    target = out / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    items = self.encode_archive(path, target)
                kinds: dict[str, int] = {}
                for item in items:
                    for kind, number in item.kinds.items():
                        kinds[kind] = kinds.get(kind, 0) + number
                return Item(
                    "encoded",
                    relative,
                    fmt.name,
                    relative if out is not None else None,
                    count=sum(i.count for i in items),
                    kinds=dict(sorted(kinds.items())),
                )
            encoded = self.encode_file(path, fmt.name)
        except CleanPromptError as exc:
            return Item("refused", relative, fmt.name, reason=str(exc))
        kinds = dict(encoded.report.get("kinds", {}))
        if out is None:
            return Item(
                "encoded", relative, fmt.name, None, count=encoded.count, kinds=kinds
            )
        output = PurePosixPath(relative).with_name(encoded.output_name).as_posix()
        target = out / output
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded.text.encode("utf-8"))
        return Item(
            "encoded", relative, fmt.name, output, count=encoded.count, kinds=kinds
        )

    def encode_archive(
        self, source: str | os.PathLike, target: str | os.PathLike
    ) -> list[Item]:
        """
        Encode every readable member of a zip into a new zip.

        Parameters
        ----------
        source : path-like
            The zip to read.
        target : path-like
            The zip to write. Must differ from ``source``.

        Returns
        -------
        list of Item
            One per member, in name order.

        Raises
        ------
        CleanPromptError
            If the archive is unreadable, has too many members, or
            decompresses past the total size limit.

        Notes
        -----
        **Developer notes.** The output is deterministic — members sorted,
        timestamps fixed — so encoding the same archive twice with fresh
        cleaners gives the same bytes. It is written to a temporary name and
        moved into place, so a failure never leaves half an archive.
        """
        self._check_open()
        src, dst = Path(source), Path(target)
        if src.resolve() == dst.resolve():
            msg = "encode_archive would overwrite its own input; choose another target"
            raise CleanPromptError(msg)
        self._learn([lambda: self._archive_pass(src, None)])
        return self._archive_pass(src, dst)

    def _archive_pass(self, src: Path, dst: Path | None) -> list[Item]:
        """Encode every member; write the new archive unless ``dst`` is ``None``."""
        items: list[Item] = []
        outputs: dict[str, bytes] = {}
        budget = _Budget(self._limits)
        try:
            archive = zipfile.ZipFile(src)
        except (zipfile.BadZipFile, OSError) as exc:
            msg = f"{src.name!r} is not a readable zip: {exc}"
            raise CleanPromptError(msg) from exc
        with archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            if len(infos) > self._limits.max_members:
                msg = f"{src.name!r} has {len(infos)} members, above the limit of {self._limits.max_members}"
                raise CleanPromptError(msg)
            for info in sorted(infos, key=lambda one: one.filename):
                items.append(self._archive_member(archive, info, budget, outputs))
                if budget.used > self._limits.max_total_bytes:
                    msg = f"{src.name!r} decompresses past {self._limits.max_total_bytes} bytes; nothing was written"
                    raise CleanPromptError(msg)
        if dst is not None:
            _write_zip(dst, outputs)
        return items

    def _archive_member(  # ruff: ignore[too-many-return-statements]
        self,
        archive,
        info,
        budget,
        outputs,
    ) -> Item:
        """Encode one archive member into ``outputs``."""
        name = info.filename
        unsafe = _safe_member(name)
        if unsafe is not None:
            return Item("refused", name, reason=unsafe)
        if info.flag_bits & 0x1:
            return Item("refused", name, reason="the member is encrypted")
        fmt = self.catalog.format_for(name, self._formats)
        if fmt is None:
            return Item("skipped", name, reason="no selected format reads this member")
        if fmt.splitter == "archive":
            return Item(
                "skipped", name, fmt.name, reason="nested archives are not opened"
            )
        if fmt.splitter == "corpus":
            return Item(
                "skipped",
                name,
                fmt.name,
                reason="PDF members are read only from files on disk",
            )
        try:
            data = budget.read(archive, info, xml=False)
            encoded = self._encode_data(data, name, fmt, budget)
        except CleanPromptError as exc:
            return Item("refused", name, fmt.name, reason=str(exc))
        output = PurePosixPath(name).with_name(encoded.output_name).as_posix()
        if output in outputs:
            return Item(
                "refused",
                name,
                fmt.name,
                reason=f"its output name {output!r} is already taken",
            )
        outputs[output] = encoded.text.encode("utf-8")
        return Item(
            "encoded",
            name,
            fmt.name,
            output,
            count=encoded.count,
            kinds=dict(encoded.report.get("kinds", {})),
        )

    # -- decoding -------------------------------------------------------

    @_synchronized
    def decode_report(
        self,
        text: str,
        strict: bool = False,
        kinds: Iterable[str] | None = None,
    ) -> RestorationResult:
        """
        Put values back into ``text`` and say what was and was not restored.

        Parameters
        ----------
        text : str
            A model's reply, or an encoded file edited by one.
        strict : bool, default=False
            Raise on a placeholder this cleaner never issued.
        kinds : iterable of str, optional
            Restore only values of these kinds; placeholders of other kinds
            are left as they are. ``None`` restores every kind.

        Returns
        -------
        RestorationResult
            With ``unknown`` and ``repaired`` filled in.
        """
        outcome = self._restore_report(text, strict=strict, kinds=kinds)
        self._audit_decoded(
            len(outcome.restored), len(outcome.unknown), len(outcome.repaired)
        )
        return outcome

    @_synchronized
    def _restore_report(
        self,
        text: str,
        strict: bool = False,
        kinds: Iterable[str] | None = None,
    ) -> RestorationResult:
        """
        Restore ``text`` without recording an audit event.

        Parameters
        ----------
        text : str
            A reply, or one piece of a reply.
        strict : bool, default=False
            Raise on a placeholder this cleaner never issued.
        kinds : iterable of str, optional
            Restore only values of these kinds.

        Returns
        -------
        RestorationResult
            The same result :meth:`decode_report` returns.

        Notes
        -----
        **Developer notes.** A caller that decodes one reply in many pieces
        uses this and records the reply once with :meth:`_audit_decoded`
        (``CP-090``). Every other caller wants :meth:`decode_report`.
        """
        self._check_open()
        return restore(text, self._vault_of(kinds), policy=self._policy, strict=strict)

    def _audit_decoded(
        self, restored: int, unknown: int, repaired: int, **fields: Any
    ) -> None:
        """
        Record one ``decoded`` audit event for one reply.

        Parameters
        ----------
        restored, unknown, repaired : int
            Counts over the whole reply.
        **fields
            Further counts, for example ``chunks`` for a streamed reply.
            Never a value.
        """
        audit(
            "decoded",
            restored=restored,
            unknown=unknown,
            repaired=repaired,
            plan=self._fingerprint[:16],
            **fields,
        )

    def _vault_of(self, kinds: Iterable[str] | None) -> Vault:
        """Return a vault of every value, or of the values of ``kinds`` only."""
        if kinds is None:
            return self.vault()
        wanted = frozenset(kinds)
        mapping = {
            label: entry.original
            for label, entry in self._entries.items()
            if entry.kind in wanted
        }
        return Vault(mapping, grammar_fingerprint=self._policy.tag_style.fingerprint)

    @_synchronized
    def kinds_in(self, text: str) -> tuple[str, ...]:
        """
        Return the kinds of held values ``text`` refers to, sorted.

        Parameters
        ----------
        text : str
            Text a model wrote: a reply, or a tool call's arguments.

        Returns
        -------
        tuple of str
            One kind per distinct held value named in ``text`` — by its
            placeholder or stand-in, exactly or as the model rewrote it.
            Nothing it returns is a value.

        Notes
        -----
        **Developer notes.** Decided by the same restoration that decoding
        uses, so "would this text receive a value of kind K" and "does
        decoding put one there" cannot disagree (``CP-073``).
        """
        self._check_open()
        outcome = restore(text, self.vault(), policy=self._policy)
        named = dict.fromkeys(
            [*outcome.restored, *(label for _, label in outcome.repaired)]
        )
        return tuple(
            sorted(
                self._entries[label].kind for label in named if label in self._entries
            )
        )

    def decode(self, text: str, strict: bool = False) -> str:
        """
        Put values back into ``text``.

        Parameters
        ----------
        text : str
            A model's reply, or an encoded file edited by one.
        strict : bool, default=False
            Raise on a placeholder this cleaner never issued.

        Returns
        -------
        str
            The restored text.
        """
        return self.decode_report(text, strict=strict).text

    def decode_tree(
        self, source: str | os.PathLike, target: str | os.PathLike
    ) -> Iterator[Item]:
        """
        Restore every UTF-8 file under a folder into a mirror folder.

        Parameters
        ----------
        source : path-like
            Encoded files, possibly edited by a model.
        target : path-like
            Where restored files are written; must not be inside ``source``.

        Yields
        ------
        Item
            One per file, in sorted path order.

        See Also
        --------
        restore_tree : The same, given a vault instead of a cleaner.
        """
        self._check_open()
        return restore_tree(source, target, self.vault(), self._policy)

    def decode_archive(
        self, source: str | os.PathLike, target: str | os.PathLike
    ) -> list[Item]:
        """
        Restore every member of an encoded zip into a new zip.

        Parameters
        ----------
        source : path-like
            An archive written by :meth:`encode_archive`.
        target : path-like
            The zip to write.

        Returns
        -------
        list of Item
            One per member.

        See Also
        --------
        restore_archive : The same, given a vault instead of a cleaner.
        """
        self._check_open()
        return restore_archive(source, target, self.vault(), self._policy, self._limits)


def kind_totals(items: Iterable[Item]) -> dict[str, int]:
    """
    Add up the values of each kind across a folder's items.

    Parameters
    ----------
    items : iterable of Item
        From :meth:`Cleaner.encode_tree` or :meth:`Cleaner.survey_tree`.

    Returns
    -------
    dict of str to int
        Kind to count, sorted by kind. Holds no value.

    Examples
    --------
    >>> kind_totals(
    ...     [
    ...         Item("encoded", "a", kinds={"EMAIL": 2}),
    ...         Item("encoded", "b", kinds={"EMAIL": 1, "MRN": 1}),
    ...     ]
    ... )
    {'EMAIL': 3, 'MRN': 1}
    """
    totals: dict[str, int] = {}
    for item in items:
        for kind, number in item.kinds.items():
            totals[kind] = totals.get(kind, 0) + number
    return dict(sorted(totals.items()))


def restore_tree(
    source: str | os.PathLike,
    target: str | os.PathLike,
    vault: Vault,
    policy: RedactionPolicy | None = None,
) -> Iterator[Item]:
    """
    Restore every UTF-8 file under a folder into a mirror folder.

    Parameters
    ----------
    source : path-like
        Encoded files, possibly edited by a model.
    target : path-like
        Where restored files are written; must not be inside ``source``.
    vault : Vault
        The values to put back, for example read from a vault file.
    policy : RedactionPolicy, optional
        Supplies the placeholder grammar the vault was issued under.

    Yields
    ------
    Item
        One per file, in sorted path order. A file that is not UTF-8 holds
        no stand-in and is skipped, not copied.

    Raises
    ------
    CleanPromptError
        If ``source`` is not a folder or ``target`` is inside it.
    """
    root, out = _roots(source, target)
    for folder, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(folder)
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in SKIPPED_DIRECTORIES and not (here / d).is_symlink()
        )
        for name in sorted(filenames):
            path = here / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                yield Item("skipped", relative, reason="symbolic link not followed")
                continue
            try:
                text = path.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                yield Item(
                    "skipped", relative, reason="not UTF-8, so it holds no stand-in"
                )
                continue
            outcome = restore(text, vault, policy=policy)
            destination = out / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(outcome.text.encode("utf-8"))
            yield Item(
                "decoded", relative, output=relative, count=len(outcome.restored)
            )


def restore_archive(
    source: str | os.PathLike,
    target: str | os.PathLike,
    vault: Vault,
    policy: RedactionPolicy | None = None,
    limits: OfficeLimits | None = None,
) -> list[Item]:
    """
    Restore every member of an encoded zip into a new zip.

    Parameters
    ----------
    source : path-like
        An archive written by :meth:`Cleaner.encode_archive`.
    target : path-like
        The zip to write; must differ from ``source``.
    vault : Vault
        The values to put back.
    policy : RedactionPolicy, optional
        Supplies the placeholder grammar.
    limits : OfficeLimits, optional
        Member count and size bounds.

    Returns
    -------
    list of Item
        One per member.

    Raises
    ------
    CleanPromptError
        If a member is unsafe, oversized, or not UTF-8 — an encoded archive
        holds only text, so anything else did not come from
        :meth:`Cleaner.encode_archive`.
    """
    src, dst = Path(source), Path(target)
    if src.resolve() == dst.resolve():
        msg = "decoding an archive would overwrite its own input; choose another target"
        raise CleanPromptError(msg)
    bounds = limits if limits is not None else OfficeLimits()
    budget = _Budget(bounds)
    outputs: dict[str, bytes] = {}
    items = []
    try:
        archive = zipfile.ZipFile(src)
    except (zipfile.BadZipFile, OSError) as exc:
        msg = f"{src.name!r} is not a readable zip: {exc}"
        raise CleanPromptError(msg) from exc
    with archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if len(infos) > bounds.max_members:
            msg = f"{src.name!r} has {len(infos)} members, above the limit of {bounds.max_members}"
            raise CleanPromptError(msg)
        for info in sorted(infos, key=lambda one: one.filename):
            unsafe = _safe_member(info.filename)
            if unsafe is not None:
                msg = f"member {info.filename!r} is unsafe: {unsafe}"
                raise CleanPromptError(msg)
            data = budget.read(archive, info, xml=False)
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                msg = f"member {info.filename!r} is not UTF-8, so it was not written by encode_archive"
                raise CleanPromptError(msg) from exc
            outcome = restore(text, vault, policy=policy)
            outputs[info.filename] = outcome.text.encode("utf-8")
            items.append(
                Item(
                    "decoded",
                    info.filename,
                    output=info.filename,
                    count=len(outcome.restored),
                )
            )
    _write_zip(dst, outputs)
    return items


def _tree_files(root: Path) -> Iterator[tuple[Path, str]]:
    """
    Yield ``(folder, name)`` for every file a tree walk would consider.

    Notes
    -----
    **Developer notes.** The same rules as :meth:`Cleaner.encode_tree` — no
    symbolic-link folders, no version-control or checkpoint folders, sorted —
    so the learning pass reads exactly the files the writing pass writes.
    """
    for folder, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(folder)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in SKIPPED_DIRECTORIES and not (here / name).is_symlink()
        )
        for name in sorted(filenames):
            yield here, name


def _roots(source: str | os.PathLike, target: str | os.PathLike) -> tuple[Path, Path]:
    """Resolve a source folder and a target folder that is not inside it."""
    root = Path(source).resolve()
    if not root.is_dir():
        msg = f"{str(source)!r} is not a folder"
        raise CleanPromptError(msg)
    out = Path(target).resolve()
    if out == root or root in out.parents:
        msg = "the target folder must not be the source folder or inside it"
        raise CleanPromptError(msg)
    out.mkdir(parents=True, exist_ok=True)
    return root, out


def _write_zip(target: Path, outputs: dict[str, bytes]) -> None:
    """Write members deterministically, via a temporary file."""
    temporary = target.with_name(target.name + ".partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(outputs):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, outputs[name])
    os.replace(temporary, target)


def _count_kinds(entries: Iterable[Entry]) -> dict[str, int]:
    """Return the number of distinct values per kind, sorted by kind."""
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.kind] = counts.get(entry.kind, 0) + 1
    return dict(sorted(counts.items()))


def _preexisting(
    text: str, prior: tuple[Entry, ...], policy: RedactionPolicy
) -> list[str]:
    """
    Return earlier stand-ins that already occur in a new document.

    Notes
    -----
    **Developer notes.** A bracket label is protected by the engine's reserved
    ranges. A *literal* stand-in — a surrogate name, a column stand-in, a
    sentinel — is not a grammar label, and if a later file already contains
    one, decoding that file would replace the author's text too. That is
    reported, not silently accepted, and not refused either: the encoded text
    is still safe to send.
    """
    literal = [
        entry.label
        for entry in prior
        if policy.tag_style.normalize(entry.label) is None
    ]
    if not literal:
        return []
    # Bounded by "no word character on either side" rather than by \b, which
    # needs a word character *inside* the edge and so never bounds a sentinel
    # that starts with '-'.
    alternation = "|".join(
        re.escape(label)
        for label in sorted(set(literal), key=lambda one: (-len(one), one))
    )
    return sorted(set(re.findall(rf"(?<!\w)(?:{alternation})(?!\w)", text)))
