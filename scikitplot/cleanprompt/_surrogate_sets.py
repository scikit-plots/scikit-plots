r"""
Custom surrogate sets: your own invented names, under the core's safety rules.

Notes
-----
**User notes.** ``--style surrogate`` replaces names with invented ones
(``Marion Holt``, ``Northwind Logistics``). A *surrogate set* swaps in your
own lists — names in your language, a fictional cast your team recognises —
without changing anything else::

    # nordic.yaml
    name: nordic
    version: 1
    summary: Nordic-sounding invented names.
    kinds:
      PERSON: {first: [Aino, Eero, Liv], last: [Halvorsen, Lindgren, Virtanen]}
      ORG:    {first: [Fjord, Norrsken], second: [Data, Logistik]}
      GPE:    [Granvik, Solberga]
      LOC:    [the Tunturi Fells]
      FAC:    [Granvik Station]

.. code-block:: bash

    cleanprompt encode --style surrogate --surrogates nordic.yaml --ner

Every key under ``kinds`` is optional; a kind the set does not list keeps
the built-in names. ``decode`` needs nothing extra: the vault records which
set wrote it, and restoring reads the stand-ins from the vault.

**What a set cannot change.** Credentials and identifiers never get an
invented value, and e-mail addresses, telephone numbers and links are always
built by cleanprompt in reserved forms (``example.invalid``, ``+1 555
0100``–``0199``) that cannot reach anyone. A set that tries to define them is
refused, with the reason. An entry must read as a name: letters, combining
marks, spaces and ``' - . ’`` only, 1–64 characters, starting and ending with
a letter. No entry may look like a value cleanprompt detects (a pattern from
the core or a built-in pack); otherwise the next encode would hide the
stand-in itself.

**Developer notes.** The design, the invariants (G1–G6) and the growth plan
are in ``maintenances/cleanprompt/_maintenance/GENERATOR_DESIGN.md``. In
short:

* validation is total, like a pack's: every problem in one
  :class:`~scikitplot.cleanprompt._packs.PackError`;
* a set is identified by ``name@version#digest16``, where the digest covers
  the *validated* content, so the same set in YAML or JSON has one identity
  and editing any entry gives a new one;
* :class:`SurrogateSet` only *proposes* candidates
  (:meth:`SurrogateSet.candidate`). Uniqueness, absence from the source,
  absence of held values and the bounded search stay in
  :func:`~scikitplot.cleanprompt._surrogates.surrogate_for`, which is what
  makes a later provider protocol (slice B) safe to add.

See Also
--------
scikitplot.cleanprompt._surrogates : The core loop and the built-in names.
scikitplot.cleanprompt._policy.TagStyle : Where a set is attached and recorded.
"""  # ruff: ignore[ambiguous-unicode-character-docstring]

from __future__ import annotations

import functools
import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "CORE_FORMS",
    "SETTABLE_KINDS",
    "SurrogateSet",
    "detected_by",
    "entry_problem",
    "load_surrogate_set",
    "surrogate_set_from_document",
]

#: Kinds a set may define, and the parts each takes. ``None``: one list.
SETTABLE_KINDS: dict[str, tuple[str, ...] | None] = {
    "PERSON": ("first", "last"),
    "ORG": ("first", "second"),
    "GPE": None,
    "LOC": None,
    "FAC": None,
}

#: Surrogated kinds whose form the core owns: reserved domains and numbers.
CORE_FORMS = ("EMAIL", "PHONE", "URL")

#: The most entries one list may hold. A name list is data, not a database.
MAX_ENTRIES = 256

#: The longest entry, in characters.
MAX_ENTRY_CHARS = 64

_SET_NAME = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
_TOP_KEYS = frozenset({"name", "version", "summary", "kinds"})
#: Punctuation a name may contain, besides letters, marks and spaces.
_NAME_PUNCTUATION = frozenset("'-.\u2019")


#: Code points Unicode declares default-ignorable (DerivedCoreProperties,
#: ``Default_Ignorable_Code_Point``): rendered as nothing, so two entries that
#: differ only by one look the same. Python's :mod:`unicodedata` does not
#: expose the property; the ranges are stable by Unicode policy. The ``Cf``
#: ones are refused by category already; listed for completeness.
_DEFAULT_IGNORABLE = (
    (0x00AD, 0x00AD),
    (0x034F, 0x034F),
    (0x061C, 0x061C),
    (0x115F, 0x1160),
    (0x17B4, 0x17B5),
    (0x180B, 0x180F),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x206F),
    (0x3164, 0x3164),
    (0xFE00, 0xFE0F),
    (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0),
    (0xFFF0, 0xFFF8),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0000, 0xE0FFF),
)

#: Compatibility decompositions whose character is another character drawn
#: differently (full-width ``\uff21`` for ``A``, a circled or superscript
#: letter, an Arabic presentation form): an entry using one looks like an
#: entry that does not. ``<compat>`` itself is allowed, because ordinary
#: letters carry it — the Thai SARA AM in ``\u0e19\u0e49\u0e33``, for one —
#: and requiring NFKC refused such names (round 26, checked across scripts).
_LOOK_ALIKE_FORMS = frozenset(
    {
        "<wide>",
        "<narrow>",
        "<font>",
        "<circle>",
        "<super>",
        "<sub>",
        "<square>",
        "<small>",
        "<vertical>",
        "<fraction>",
        "<initial>",
        "<medial>",
        "<final>",
        "<isolated>",
    }
)

#: Combining marks allowed in a row after a letter (``e`` + acute + dot below).
_MAX_MARKS = 2

#: Scripts that are written together in one name (Japanese mixes all three).
_SCRIPT_GROUPS = {"CJK": "HAN", "HIRAGANA": "HAN", "KATAKANA": "HAN"}


def _ignorable(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _DEFAULT_IGNORABLE)


def _script(char: str) -> str:
    """Return the script a letter belongs to, read from its Unicode name."""
    word = unicodedata.name(char, "UNKNOWN").split(" ")[0]
    return _SCRIPT_GROUPS.get(word, word)


def entry_problem(  # ruff: ignore[too-many-return-statements, too-many-branches]
    entry: Any,
) -> str | None:
    """
    Return why ``entry`` cannot be a name stand-in, or ``None`` if it can.

    Parameters
    ----------
    entry : object
        A candidate, as written in a set or proposed by a provider.

    Returns
    -------
    str or None
        A short reason, or ``None`` when the entry passes the safety floor.

    Notes
    -----
    **Developer notes.** This is floor rule 3 of ``GENERATOR_DESIGN.md``, and
    it is checked twice: when a set loads (so a bad file is refused with a
    clear message) and on every candidate at run time (so nothing a future
    provider proposes bypasses it). It is a whitelist, so a digit, ``@``,
    ``/``, a bracket or a control character can never appear in a name
    stand-in. The round-26 review added what "looks like a name" also needs,
    so that two different entries can never *look* the same: no
    default-ignorable code point (the combining grapheme joiner, variation
    selectors, Hangul fillers), combining marks only after a letter and at
    most two in a row, no full-width or other drawn-differently compatibility
    form, and one script per
    entry (``A`` in Latin and ``\u0410`` in Cyrillic are different
    characters that look alike).

    Examples
    --------
    >>> entry_problem("Aino") is None
    True
    >>> entry_problem("Ag3nt")
    "'3' is not a letter, mark, space or one of ' - . \u2019"
    """
    if not isinstance(entry, str):
        return "must be a string"
    if not 1 <= len(entry) <= MAX_ENTRY_CHARS:
        return f"must be 1 to {MAX_ENTRY_CHARS} characters"
    for char in entry:
        if _ignorable(char):
            return f"U+{ord(char):04X} is an invisible character"
    for char in entry:
        tag = unicodedata.decomposition(char).split(" ")[0]
        if tag in _LOOK_ALIKE_FORMS:
            return (
                f"U+{ord(char):04X} is a {tag[1:-1]} compatibility form of another "
                "character; write the ordinary character"
            )
    if not _is_letter(entry[0]):
        return "must start and end with a letter"
    if "  " in entry:
        return "must not contain two spaces in a row"
    marks, scripts = 0, set()
    for char in entry:
        category = unicodedata.category(char)
        if category.startswith("M"):
            marks += 1
            if marks > _MAX_MARKS:
                return f"has more than {_MAX_MARKS} combining marks in a row"
            continue
        marks = 0
        if category.startswith("L"):
            scripts.add(_script(char))
        elif char != " " and char not in _NAME_PUNCTUATION:
            return f"{char!r} is not a letter, mark, space or one of ' - . \u2019"
    base = next(
        char
        for char in reversed(entry)
        if not unicodedata.category(char).startswith("M")
    )
    if not _is_letter(base):
        return "must start and end with a letter"
    if len(scripts) > 1:
        return f"mixes scripts ({', '.join(sorted(scripts))})"
    return None


def _is_letter(char: str) -> bool:
    return unicodedata.category(char).startswith("L")


@dataclass(frozen=True)
class SurrogateSet:
    """
    A validated set of invented names.

    Parameters
    ----------
    name : str
        Set name: lower case, starts with a letter, at most 32 characters.
    version : int
        The set's own version; bump it when entries change meaning.
    summary : str
        One line saying what the set is for.
    pools : tuple of (str, tuple of tuple of str)
        Each kind with its lists, in :data:`SETTABLE_KINDS` order.
    identity : str
        ``name@version#digest16`` — what a vault and a plan record.
    source : str, default='<memory>'
        Where it was loaded from. Not part of equality.

    Notes
    -----
    **Developer notes.** Build one with :func:`surrogate_set_from_document` or
    :func:`load_surrogate_set`; the constructor does not validate, and the
    identity is only meaningful when it was computed from the content.
    """

    name: str
    version: int
    summary: str
    pools: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...]
    identity: str
    source: str = field(default="<memory>", compare=False)

    def kinds(self) -> tuple[str, ...]:
        """Return the kinds this set defines."""
        return tuple(kind for kind, _ in self.pools)

    def candidate(self, kind: str, index: int) -> str | None:
        """
        Propose the ``index``-th stand-in for ``kind``.

        Parameters
        ----------
        kind : str
            The placeholder category.
        index : int
            Zero-based position in this kind's sequence.

        Returns
        -------
        str or None
            A candidate, or ``None`` when the set does not define ``kind``
            (the core then uses its built-in names).

        Notes
        -----
        **Developer notes.** Only a proposal: the core decides whether it is
        used (``GENERATOR_DESIGN.md`` section 6). Two lists are combined with
        the same offset rule as the built-in names, so consecutive people do
        not share a surname.
        """
        from ._surrogates import _pair  # ruff: ignore[import-outside-top-level]

        for name, lists in self.pools:
            if name != kind:
                continue
            if len(lists) == 2:  # ruff: ignore[magic-value-comparison]
                return _pair(lists[0], lists[1], index)
            return lists[0][index % len(lists[0])]
        return None

    def capacity(self, kind: str) -> int:
        """Return how many distinct stand-ins this set can propose for ``kind``."""
        for name, lists in self.pools:
            if name == kind:
                total = 1
                for one in lists:
                    total *= len(one)
                return total
        return 0


@functools.lru_cache(maxsize=1)
def _detection_rules() -> tuple:
    """
    Return the patterns a stand-in must not match, compiled once per process.

    Notes
    -----
    **Developer notes.** Core patterns that run by default, and every
    built-in pack pattern (a pack can be selected on any run). Opt-in core
    patterns are left out: ``TITLE_CASE`` matches every two-word name, the
    built-in names included, and including it refused the design's own
    example set (round 26 review).
    """
    from ._catalog import builtin_catalog  # ruff: ignore[import-outside-top-level]
    from ._patterns import PATTERNS  # ruff: ignore[import-outside-top-level]

    specs = [spec for spec in PATTERNS.values() if spec.enabled_by_default]
    for pack in builtin_catalog().packs.values():
        specs.extend(pack.patterns)
    return tuple(
        (spec.kind, re.compile(spec.pattern, spec.flags), spec.validate)
        for spec in specs
    )


def detected_by(entry: str) -> str | None:
    """
    Return the kind of the first detection rule ``entry`` matches, or ``None``.

    Parameters
    ----------
    entry : str
        A name, or a combined two-part stand-in.

    Returns
    -------
    str or None
        The kind that would detect it — so the next encode would hide the
        stand-in itself — or ``None``.
    """
    for kind, compiled, validate in _detection_rules():
        for match in compiled.finditer(entry):
            if match.group() and (validate is None or validate(match)):
                return kind
    return None


def _shown(entry: Any) -> str:
    """
    Quote an entry for a message, with invisible characters spelled out.

    Notes
    -----
    **Developer notes.** A message that quotes ``'Ai\u034fno'`` literally
    looks like ``'Aino'`` on screen, which hides the very problem it reports.
    """
    if not isinstance(entry, str):
        return repr(entry)
    inner = "".join(
        (
            f"\\u{ord(char):04x}"
            if _ignorable(char)
            or (unicodedata.category(char)[0] in "CZ" and char != " ")
            else char
        )
        for char in entry
    )
    return f"'{inner}'"


def _list(value: Any, path: str, problems: list[str]) -> tuple[str, ...]:
    """Validate one list of entries, collecting every problem."""
    if not isinstance(value, list) or not value:
        problems.append(f"{path}: must be a non-empty list of names")
        return ()
    if len(value) > MAX_ENTRIES:
        problems.append(f"{path}: holds {len(value)} entries, above {MAX_ENTRIES}")
        return ()
    from ._canonical import canonical  # ruff: ignore[import-outside-top-level]

    seen: dict[str, str] = {}
    out = []
    for index, written in enumerate(value):
        where = f"{path}[{index}]"
        # Canonically equivalent spellings (a precomposed letter, or the base
        # letter and its mark in any order) are one name: the set stores the
        # composed (NFC) form, so the identity and every stand-in are the same
        # however the file was typed.
        entry = (
            unicodedata.normalize("NFC", written)
            if isinstance(written, str)
            else written
        )
        problem = entry_problem(entry)
        if problem is not None:
            problems.append(f"{where}: {_shown(written)} {problem}")
            continue
        key = canonical(entry).casefold()
        if key in seen:
            problems.append(f"{where}: {_shown(entry)} repeats {_shown(seen[key])}")
            continue
        kind = detected_by(entry)
        if kind is not None:
            problems.append(
                f"{where}: {_shown(entry)} is detected as {kind}; the next encode would "
                "hide the stand-in itself"
            )
            continue
        seen[key] = entry
        out.append(entry)
    return tuple(out)


def surrogate_set_from_document(  # ruff: ignore[too-many-branches]
    document: Any,
    source: str = "<memory>",
) -> SurrogateSet:
    """
    Validate a parsed set document and build the set.

    Parameters
    ----------
    document : mapping
        The set, as parsed from YAML or JSON.
    source : str, default='<memory>'
        Where it came from, for messages.

    Returns
    -------
    SurrogateSet
        Validated, with its identity computed from the content.

    Raises
    ------
    PackError
        Listing every problem found; nothing is returned for a partly valid
        document.

    Examples
    --------
    >>> names = surrogate_set_from_document(
    ...     {
    ...         "name": "demo",
    ...         "version": 1,
    ...         "summary": "Demo names.",
    ...         "kinds": {"GPE": ["Granvik", "Solberga"]},
    ...     }
    ... )
    >>> names.candidate("GPE", 1), names.candidate("PERSON", 0)
    ('Solberga', None)
    >>> names.identity.startswith("demo@1#")
    True
    """
    from ._packs import PackError  # ruff: ignore[import-outside-top-level]

    if not isinstance(document, Mapping):
        raise PackError(source, ["the document must be a mapping"])
    problems: list[str] = []
    problems.extend(
        f"set: unknown key {key!r} (allowed: {', '.join(sorted(_TOP_KEYS))})"
        for key in sorted(set(document) - _TOP_KEYS)
    )
    name = document.get("name")
    if not isinstance(name, str) or not _SET_NAME.match(name):
        problems.append(f"name: {name!r} must match {_SET_NAME.pattern}")
    version = document.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        problems.append(f"version: {version!r} must be a positive integer")
    summary = document.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        problems.append("summary: must be a non-empty string")
    kinds = document.get("kinds")
    pools: list[tuple[str, tuple[tuple[str, ...], ...]]] = []
    if not isinstance(kinds, Mapping) or not kinds:
        problems.append("kinds: must be a non-empty mapping of kind to names")
        kinds = {}
    for kind in kinds:
        if kind in CORE_FORMS:
            problems.append(
                f"kinds.{kind}: e-mail addresses, telephone numbers and links are "
                "always built by cleanprompt in reserved forms that cannot reach "
                "anyone; remove this kind"
            )
        elif kind not in SETTABLE_KINDS:
            problems.append(
                f"kinds.{kind}: only {', '.join(SETTABLE_KINDS)} take invented "
                "names; credentials, identifiers and every other kind keep "
                "placeholders"
            )
    for kind, parts in SETTABLE_KINDS.items():
        if kind not in kinds:
            continue
        value = kinds[kind]
        if parts is None:
            lists = (_list(value, f"kinds.{kind}", problems),)
        elif not isinstance(value, Mapping):
            problems.append(
                f"kinds.{kind}: must be a mapping with {' and '.join(parts)}"
            )
            continue
        else:
            problems.extend(
                f"kinds.{kind}: unknown key {key!r} (allowed: {', '.join(parts)})"
                for key in sorted(set(value) - set(parts))
            )
            lists = tuple(
                _list(value.get(part), f"kinds.{kind}.{part}", problems)
                for part in parts
            )
        if all(lists):
            pools.append((kind, lists))
    if problems:
        raise PackError(source, problems)
    content = {
        "name": name,
        "version": version,
        "summary": summary.strip(),
        "kinds": {kind: [list(one) for one in lists] for kind, lists in pools},
    }
    digest = hashlib.sha256(
        json.dumps(
            content, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return SurrogateSet(
        name=name,
        version=version,
        summary=summary.strip(),
        pools=tuple(pools),
        identity=f"{name}@{version}#{digest[:16]}",
        source=source,
    )


def load_surrogate_set(path: str | Path) -> SurrogateSet:
    """
    Read and validate a surrogate set from a ``.yaml``, ``.yml`` or ``.json`` file.

    Parameters
    ----------
    path : path-like
        The set file. YAML needs PyYAML; JSON needs nothing.

    Returns
    -------
    SurrogateSet
        Validated.

    Raises
    ------
    PackError
        If the document is invalid, listing every problem.
    CleanPromptError
        If the file is missing, too large, or of an unknown type.
    """
    from ._custom import _read  # ruff: ignore[import-outside-top-level]

    file = Path(path)
    return surrogate_set_from_document(_read(file), file.name)
