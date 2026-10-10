r"""
Find regular-expression shapes that can backtrack catastrophically (base tier).

Notes
-----
**User notes.** A pack's patterns run on every document. Python's :mod:`re`
backtracks, so a few shapes take time that grows exponentially — or with a
high power — on a near-match, and one document can stall a run. This module
reads a pattern's *source* and reports those shapes, with concrete rewrites::

    (a+)+          nested repetition          -> a+
    (\w+\s?)*      nested repetition          -> (\w+\s)*\w*  or bound it
    (a|ab)+        overlapping alternatives   -> a(?:b)?  repeated
    \d+\d+         adjacent repetitions       -> \d{2,}

What happens next is policy, and the user's (:data:`PATTERN_RISK_MODES`):

``warn`` (default)
    load the pattern and say what was found, and how to fix or accept it;
``ignore``
    load it silently (``packs --check`` still lists the findings);
``refuse``
    refuse the pack, naming every finding.

A single pattern can be accepted deliberately in its pack, with the reason
recorded beside it::

    patterns:
      - kind: TICKET
        pattern: '(?:[A-Z]+-)+\d+'
        risk: accepted
        risk_reason: inputs are single ticket ids, never prose

**Developer notes — why a parser of our own, and what it does not claim.**

``sre_parse`` / ``re._parser`` would give a parse tree for free, but they are
private, they moved in Python 3.11, and importing ``sre_parse`` there emits a
``DeprecationWarning`` the suite treats as an error. So the source is read by
the small parser below, which understands the syntax that matters for
repetition: escapes, character classes, groups of every kind, alternation and
every quantifier form (greedy, lazy, possessive).

Whether two pieces can match the same character is decided by Python's public
:mod:`re` itself: each piece is compiled alone and tried against a *probe
alphabet* — every character the pattern mentions, the endpoints of every range
and their neighbours, and a fixed set of representatives (letters, digits,
underscore, whitespace kinds, punctuation, a non-ASCII letter and digit). Two
pieces overlap when some probe matches both.

Two runs separated only by an *optional* element (``\s*:?\s*``) are not
reported: the cost is at most quadratic, the idiom is everywhere in label
patterns (the built-in ``MRN`` rule uses it), and every span is already
bounded by the policy's length limits. Directly adjacent runs (``\d+\d+``)
are reported, because merging them is always possible and always clearer.

The analysis is a *warning*, deliberately: it reports shapes known to
backtrack badly, it can report a shape that is harmless on the inputs a
pattern will really see, and it cannot prove a pattern is fast (nothing short
of a regex engine with a timeout can). Nothing it reports is silent: a pattern
it cannot parse is reported as ``not-analysed``, never passed as safe.

See Also
--------
scikitplot.cleanprompt._custom : Where custom packs are loaded and this runs.
"""

from __future__ import annotations

import os
import re
import unicodedata
import warnings
from dataclasses import dataclass, field

__all__ = [
    "DEFAULT_PATTERN_RISK",
    "PATTERN_RISK_ENV",
    "PATTERN_RISK_MODES",
    "PackFinding",
    "PatternRisk",
    "PatternRiskWarning",
    "analyse_pattern",
    "enforce",
    "pack_findings",
    "resolve_pattern_risk",
]

#: What may be done with a finding, in order of strictness.
PATTERN_RISK_MODES: tuple[str, ...] = ("ignore", "warn", "refuse")

#: The default: load, and say so.
DEFAULT_PATTERN_RISK = "warn"

#: Environment variable that sets the default for a machine or a CI job.
PATTERN_RISK_ENV = "CLEANPROMPT_PATTERN_RISK"

#: Advice every finding carries, so the way out is always on screen.
_QUICK_OPTIONS = (
    (
        "accept this pattern in its pack: add `risk: accepted` and a "
        "`risk_reason:` saying why its inputs are safe"
    ),
    (
        "silence the check for this run: --pattern-risk ignore "
        "(or pattern_risk='ignore', or CLEANPROMPT_PATTERN_RISK=ignore)"
    ),
    (
        "make findings fatal: --pattern-risk refuse (or pin pattern_risk: refuse "
        "in a team plan file)"
    ),
)


class PatternRiskWarning(UserWarning):
    """A pack pattern has a shape that can backtrack catastrophically."""


@dataclass(frozen=True)
class PatternRisk:
    """
    One risky shape found in a pattern.

    Parameters
    ----------
    rule : str
        ``nested-quantifier``, ``overlapping-alternation``,
        ``adjacent-quantifiers`` or ``not-analysed``.
    severity : str
        ``high`` (exponential), ``medium`` (polynomial or possible), or
        ``info`` (not analysed).
    fragment : str
        The part of the source the finding is about.
    message : str
        One sentence saying what the shape does.
    suggestions : tuple of str
        Concrete rewrites, then the policy options.
    """

    rule: str
    severity: str
    fragment: str
    message: str
    suggestions: tuple[str, ...] = field(default=())

    def describe(self, where: str = "") -> str:
        """Return the finding as a self-contained, multi-line message."""
        head = f"{where}: " if where else ""
        lines = [
            f"{head}{self.rule} ({self.severity}) in `{self.fragment}`: {self.message}"
        ]
        lines += [f"  - {item}" for item in self.suggestions]
        return "\n".join(lines)


def resolve_pattern_risk(explicit: str | None = None) -> str:
    """
    Return the mode in force: explicit, else the environment, else ``warn``.

    Parameters
    ----------
    explicit : str, optional
        A mode given by the caller (a flag, a plan file, an argument).

    Returns
    -------
    str
        One of :data:`PATTERN_RISK_MODES`.

    Raises
    ------
    ValueError
        If a mode, explicit or from the environment, is not a known one.

    Examples
    --------
    >>> resolve_pattern_risk("refuse")
    'refuse'
    """
    chosen = explicit if explicit is not None else os.environ.get(PATTERN_RISK_ENV)
    if chosen is None or chosen == "":
        return DEFAULT_PATTERN_RISK
    if chosen not in PATTERN_RISK_MODES:
        origin = "pattern_risk" if explicit is not None else PATTERN_RISK_ENV
        msg = (
            f"{origin}={chosen!r} is not a pattern-risk mode; choose from "
            f"{', '.join(PATTERN_RISK_MODES)}"
        )
        raise ValueError(msg)
    return chosen


# ---------------------------------------------------------------------------
# preparing the source: verbose mode and inline flags
# ---------------------------------------------------------------------------


class _UnparsedError(ValueError):
    """The source uses a construct this parser does not model."""


#: A repetition with more than this many *choices* counts as repeating:
#: ``{1,40}`` (39 choices) splits a run in as many ways as ``+`` does on any
#: text a pack will meet, while ``{4}`` (none: a fixed count cannot give a
#: character back) or ``{1,3}`` costs at most a small constant factor.
_REPEAT_BOUND = 3

_INLINE_FLAGS = re.compile(r"\(\?([aiLmsux]*)(?:-([imsx]+))?([:)])")
_FLAG_BITS = {"i": re.IGNORECASE, "s": re.DOTALL}


def _walk(source: str):
    """
    Yield ``(index, in_class)`` for each position that is not inside an escape.

    Notes
    -----
    **Developer notes.** The one place that knows where a character class
    starts and ends (``[]]`` and ``[^]]`` hold a literal ``]``) and that an
    escape covers two characters. Verbose stripping and the inline-flag scan
    both read the source through it, so they cannot disagree with each other.
    """
    index, in_class, class_start = 0, False, -1
    while index < len(source):
        char = source[index]
        if char == "\\":
            yield index, in_class
            index += 2
            continue
        yield index, in_class
        if in_class:
            first = class_start + 1 + (source[class_start + 1 : class_start + 2] == "^")
            if char == "]" and index > first:
                in_class = False
        elif char == "[":
            in_class, class_start = True, index
        index += 1


def _prepare(source: str, flags: int) -> tuple[str, int]:
    r"""
    Return ``(source to parse, flags to decide overlap with)``.

    Notes
    -----
    **Developer notes — why (round 26 review).** In verbose mode whitespace
    and ``#`` comments mean nothing, so ``(?: \w+ \s? )+`` *is*
    ``(?:\w+\s?)+``; read literally, the spaces looked like separators and
    an exponential pattern passed under ``refuse``. Verbose patterns are
    stripped first. Inline ``i`` and ``s`` flags (``(?i)``, ``(?s:...)``)
    widen what characters match, so they are added to the overlap flags
    wherever they appear — adding them can only find more overlap, never
    less, so a scoped flag applied to the whole pattern errs towards a
    report. ``a`` and ``L`` narrow matching and are ignored for the same
    reason. A *scoped* verbose group is not modelled and is reported.
    """
    verbose = bool(flags & re.VERBOSE)
    extra = 0
    for index, in_class in _walk(source):
        if in_class or source[index] != "(":
            continue
        match = _INLINE_FLAGS.match(source, index)
        if not match:
            continue
        added, _removed, end = match.groups()
        for letter, bit in _FLAG_BITS.items():
            if letter in added:
                extra |= bit
        if "x" in added:
            if end == ":":
                msg = "a scoped verbose group (?x:...)"
                raise _UnparsedError(msg)
            verbose = True
    if verbose:
        source = _strip_verbose(source)
    return source, (flags | extra) & ~re.VERBOSE


def _strip_verbose(source: str) -> str:
    """Remove what verbose mode ignores: unescaped whitespace and comments."""
    out = []
    positions = list(_walk(source))
    skip_to = -1
    for index, in_class in positions:
        if index < skip_to:
            continue
        char = source[index]
        if char == "\\":
            out.append(source[index : index + 2])
            continue
        if not in_class and char in " \t\n\r\f\v":
            continue
        if not in_class and char == "#":
            newline = source.find("\n", index)
            skip_to = len(source) if newline == -1 else newline + 1
            continue
        out.append(char)
    return "".join(out)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


@dataclass
class _Item:
    """One element of a sequence: an atom or a group, with its repetition."""

    source: str  # the element without its quantifier, compilable alone
    low: int = 1
    high: int | None = 1  # None means unbounded
    group: list | None = None  # alternatives (lists of _Item) for a group
    empty_ok: bool = False  # zero-width (assertion, anchor, lookaround)
    atomic: bool = False  # possessive quantifier or atomic group: never gives back

    @property
    def repeats(self) -> bool:
        """Whether this element has enough choices to split a run many ways."""
        return self.high is None or self.high - self.low > _REPEAT_BOUND


_ZERO_WIDTH_ESCAPES = set("bBAZz")
_QUANT = re.compile(r"\{(\d*)(,?)(\d*)\}")
_NUMERIC_ESCAPE = re.compile(
    r"\\(x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|N\{[^}]*\}|0[0-7]{0,2}|[1-9][0-9]?)"
)


def _parse(source: str) -> list:
    """Return the top-level alternatives: a list of sequences of :class:`_Item`."""
    alternatives, end = _parse_alternatives(source, 0, top=True)
    if end != len(source):
        raise _UnparsedError(f"unbalanced ')' at {end}")
    return alternatives


def _parse_alternatives(source: str, index: int, top: bool = False):
    alternatives = [[]]
    while index < len(source):
        char = source[index]
        if char == ")":
            if top:
                raise _UnparsedError(f"unbalanced ')' at {index}")
            return alternatives, index
        if char == "|":
            alternatives.append([])
            index += 1
            continue
        item, index = _parse_atom(source, index)
        index = _parse_quantifier(source, index, item)
        alternatives[-1].append(item)
    if not top:
        raise _UnparsedError("unclosed '('")
    return alternatives, index


def _parse_atom(  # ruff: ignore[too-many-return-statements]
    source: str,
    index: int,
):
    char = source[index]
    if char == "\\":
        if index + 1 >= len(source):
            raise _UnparsedError("trailing backslash")
        nxt = source[index + 1]
        if nxt in "xuUN0123456789":
            match = _NUMERIC_ESCAPE.match(source, index)
            if not match:
                raise _UnparsedError(f"escape at {index}")
            text = match.group(0)
            if text[1] in "123456789":
                # A backreference: its width depends on a group, so it is
                # treated as an opaque element that consumes nothing here.
                return _Item(source="(?:)", empty_ok=True), index + len(text)
            return _Item(source=text), index + len(text)
        if nxt in _ZERO_WIDTH_ESCAPES:
            return _Item(source="(?:)", empty_ok=True), index + 2
        return _Item(source=source[index : index + 2]), index + 2
    if char == "[":
        end = _class_end(source, index)
        return _Item(source=source[index : end + 1]), end + 1
    if char == "(":
        return _parse_group(source, index)
    if char in "^$":
        return _Item(source="(?:)", empty_ok=True), index + 1
    if char in "*+?{":
        if char == "{" and not _QUANT.match(source, index):
            return _Item(source=re.escape(char)), index + 1
        raise _UnparsedError(f"quantifier with nothing to repeat at {index}")
    return _Item(source=re.escape(char) if char != "." else "."), index + 1


def _class_end(source: str, start: int) -> int:
    index = start + 1
    if index < len(source) and source[index] == "^":
        index += 1
    if index < len(source) and source[index] == "]":
        index += 1
    while index < len(source):
        if source[index] == "\\":
            index += 2
            continue
        if source[index] == "]":
            return index
        index += 1
    raise _UnparsedError("unclosed '['")


def _parse_group(  # ruff: ignore[too-many-branches]
    source: str,
    index: int,
):
    start = index
    index += 1
    zero_width = atomic = False
    if source.startswith("?", index):
        rest = source[index:]
        if rest.startswith("?#"):
            end = source.find(")", index)
            if end == -1:
                raise _UnparsedError("unclosed comment")
            return _Item(source="(?:)", empty_ok=True), end + 1
        if rest.startswith(("?=", "?!")):
            zero_width, index = True, index + 2
        elif rest.startswith(("?<=", "?<!")):
            zero_width, index = True, index + 3
        elif rest.startswith(("?P<", "?<")):
            close = source.find(">", index)
            if close == -1:
                raise _UnparsedError("unclosed group name")
            index = close + 1
        elif rest.startswith("?P="):
            close = source.find(")", index)
            if close == -1:
                raise _UnparsedError("unclosed backreference")
            return _Item(source="(?:)", empty_ok=True), close + 1
        elif rest.startswith("?:"):
            index += 2
        elif rest.startswith("?>"):
            atomic, index = True, index + 2
        elif rest.startswith("?("):
            raise _UnparsedError("conditional group")
        else:
            match = _INLINE_FLAGS.match(source, start)
            if not match:
                raise _UnparsedError(f"group syntax at {start}")
            if match.group(3) == ")":
                return _Item(source="(?:)", empty_ok=True), match.end()
            index = match.end()
    alternatives, end = _parse_alternatives(source, index)
    item = _Item(
        source=source[start : end + 1],
        group=alternatives,
        empty_ok=zero_width,
        atomic=atomic,
    )
    return item, end + 1


def _parse_quantifier(source: str, index: int, item: _Item) -> int:
    if index >= len(source):
        return index
    char = source[index]
    if char in "*+?":
        item.low, item.high = {"*": (0, None), "+": (1, None), "?": (0, 1)}[char]
        index += 1
    elif char == "{":
        match = _QUANT.match(source, index)
        if not match:
            return index
        low, comma, high = match.groups()
        item.low = int(low) if low else 0
        item.high = (None if not high else int(high)) if comma else item.low
        index = match.end()
    else:
        return index
    if index < len(source) and source[index] == "+":  # possessive: never gives back
        item.atomic = True
        index += 1
    elif index < len(source) and source[index] == "?":  # lazy: still backtracks
        index += 1
    return index


# ---------------------------------------------------------------------------
# overlap, decided by re itself
# ---------------------------------------------------------------------------

#: Printable code points span from the space to the top of Unicode.
_FIRST_PRINTABLE, _CODE_POINTS = 0x20, 0x110000

_REPRESENTATIVES = "aZm5_ \t\n.-@,:/;'\"\u00e9\u0663\u00df\uff10"


def _decoded(source: str) -> set:
    r"""Return every character an escape in ``source`` names (``\x41`` -> ``A``)."""
    found = set()
    for match in _NUMERIC_ESCAPE.finditer(source):
        body = match.group(1)
        try:
            if body[0] in "xuU":
                found.add(chr(int(body[1:], 16)))
            elif body[0] == "N":
                found.add(unicodedata.lookup(body[2:-1]))
            elif body[0] == "0":
                found.add(chr(int(body, 8)))
        except (KeyError, ValueError):
            continue
    return found


def _probe_alphabet(source: str) -> str:
    r"""
    Return the characters overlap is tested on.

    Notes
    -----
    **Developer notes.** Every character the source mentions, every character
    an escape in it names (``[\x41-\x5a]`` mentions only ASCII digits and
    letters as written), the upper- and lower-case forms of each, their
    neighbours (so a range endpoint's inside is covered), and fixed
    representatives. Overlap is then decided by compiling each piece with
    :mod:`re` and asking it — so the answer is the engine's, not a model of
    it.
    """
    chars = set(_REPRESENTATIVES) | set(source) | _decoded(source)
    for char in list(chars):
        for variant in (char.lower(), char.upper()):
            if len(variant) == 1:
                chars.add(variant)
    for char in list(chars):
        if char.isprintable():
            chars.update(
                chr(code)
                for code in (ord(char) - 1, ord(char) + 1)
                if _FIRST_PRINTABLE <= code < _CODE_POINTS
            )
    return "".join(sorted(chars))


def _first_matchers(item: _Item, flags: int, cache: dict) -> list:
    """Return compiled single-character matchers for what ``item`` can start with."""
    if item.empty_ok and item.group is None:
        return []
    if item.group is None:
        return [_compile(item.source, flags, cache)]
    found = []
    for alternative in item.group:
        found.extend(_sequence_first(alternative, flags, cache))
    return found


def _sequence_first(sequence: list, flags: int, cache: dict) -> list:
    found = []
    for element in sequence:
        found.extend(_first_matchers(element, flags, cache))
        if not _nullable(element):
            break
    return found


def _compile(source: str, flags: int, cache: dict):
    key = (source, flags)
    if key not in cache:
        cache[key] = re.compile(f"(?:{source})", flags)
    return cache[key]


def _overlap(first: list, second: list, alphabet: str) -> bool:
    for char in alphabet:
        if any(m.fullmatch(char) for m in first) and any(
            m.fullmatch(char) for m in second
        ):
            return True
    return False


def _unit_matchers(item: _Item, flags: int, cache: dict) -> list:
    """Matchers for any single character ``item`` can consume (not just first)."""
    if item.group is None:
        return [] if item.empty_ok else [_compile(item.source, flags, cache)]
    found = []
    for alternative in item.group:
        for element in alternative:
            found.extend(_unit_matchers(element, flags, cache))
    return found


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------


def _nullable(item: _Item) -> bool:
    """Whether ``item`` can match the empty string."""
    if item.empty_ok or item.low == 0:
        return True
    return item.group is not None and any(
        all(_nullable(element) for element in alternative) for alternative in item.group
    )


def _variable(item: _Item) -> bool:
    r"""
    Whether ``item`` can match different lengths, so it can give characters back.

    Notes
    -----
    **Developer notes.** Any variation counts here, not only ``+`` and
    ``*``: inside a repeated group, ``\w{1,3}`` or ``\.?`` lets two
    repetitions trade a character as surely as ``\w+`` does (the round-26
    soundness fuzz found ``(?:\w?\.?)*``). A group with several branches is
    taken as variable. Possessive and atomic parts never give back.
    """
    if item.atomic or item.empty_ok:
        return False
    if item.high is None or item.high > item.low:
        return True
    return item.group is not None and (
        len(item.group) > 1
        or any(
            _variable(element) for alternative in item.group for element in alternative
        )
    )


def _ambiguous_after(
    alternative: list, position: int, flags: int, cache: dict, alphabet: str
) -> bool:
    r"""
    Whether the element at ``position`` can hand characters to what follows it.

    Notes
    -----
    **Developer notes.** Walk round the repetition from the element: the rest
    of this alternative, then — on the group's next repetition — its start,
    and finally the element itself again. Every element met that can start
    with a character the element consumes makes the boundary ambiguous; the
    first element that cannot be skipped (is not nullable) and does not
    overlap stops the walk, because it forces the boundary (``,`` after
    ``\w+`` in ``(\w+,)+``, or the leading ``.`` in ``(\.\w+)+``). Reaching
    the element itself means only skippable parts lie between two of its
    matches, so they can split a run between them (``(\w+\s?)+``).

    Skippable means *nullable*, not "has a zero minimum": a group whose every
    part is optional is skippable even with ``{1,9}`` (round-26 soundness
    fuzz). Positions, not equality: ``(?:\w+,\w+)+`` holds two equal ``\w+``
    (round-26 review).
    """
    units = _unit_matchers(alternative[position], flags, cache)
    walk = alternative[position + 1 :] + alternative[: position + 1]
    for element in walk:
        if _overlap(units, _first_matchers(element, flags, cache), alphabet):
            return True
        if not _nullable(element):
            return False
    return True


def _rules(alternatives: list, source: str, flags: int) -> list:
    cache: dict = {}
    alphabet = _probe_alphabet(source)
    risks: list[PatternRisk] = []
    seen: set = set()

    def report(key, risk) -> None:
        if key not in seen:
            seen.add(key)
            risks.append(risk)

    def repeated_group(element: _Item) -> None:
        for alternative in element.group:
            for position, inner in enumerate(alternative):
                if not _variable(inner):
                    continue
                if _ambiguous_after(alternative, position, flags, cache, alphabet):
                    report(
                        ("nested", element.source),
                        _nested(_quantified(element), _quantified(inner)),
                    )
                    return
        if element.atomic:
            return
        branches = [
            _sequence_first(alternative, flags, cache) for alternative in element.group
        ]
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                if (
                    branches[i]
                    and branches[j]
                    and _overlap(branches[i], branches[j], alphabet)
                ):
                    report(("alt", element.source), _alternation(element.source))
                    return

    def visit(sequence: list) -> None:
        previous = None
        for element in sequence:
            if element.group is not None:
                if element.repeats:
                    repeated_group(element)
                for alternative in element.group:
                    visit(alternative)
            if (
                previous is not None
                and previous.repeats
                and element.repeats
                and not (previous.atomic or element.atomic)
                and previous.group is None
                and element.group is None
                and _overlap(
                    _unit_matchers(previous, flags, cache),
                    _unit_matchers(element, flags, cache),
                    alphabet,
                )
            ):
                fragment = _quantified(previous) + _quantified(element)
                report(("adj", fragment), _adjacent(fragment))
            if not element.empty_ok:
                previous = element

    for alternative in alternatives:
        visit(alternative)
    return risks


def _quantified(item: _Item) -> str:
    if item.high is None:
        tail = "*" if item.low == 0 else "+" if item.low == 1 else f"{{{item.low},}}"
    elif item.low == item.high == 1:
        tail = ""
    else:
        tail = f"{{{item.low},{item.high}}}"
    return item.source + tail


def _nested(fragment: str, inner: str) -> PatternRisk:
    return PatternRisk(
        rule="nested-quantifier",
        severity="high",
        fragment=fragment,
        message=(
            f"a repetition of `{inner}` inside a repeated group can split the "
            "same text in exponentially many ways; a near-match takes time "
            "that doubles with each extra character"
        ),
        suggestions=(
            (
                "drop the outer repetition when the group adds nothing: `(x+)+` "
                "matches exactly what `x+` matches"
            ),
            (
                "end each repetition with a required separator the inner part cannot "
                "match, e.g. `(\\w+,)+` instead of `(\\w+,?)+`"
            ),
            (
                "make the repeated parts unable to match the same character, so "
                "every boundary between them is forced"
            ),
            (
                "on Python 3.11+, make the inner repetition possessive (`x++`) or "
                "atomic (`(?>x+)`) so it cannot give characters back"
            ),
            *_QUICK_OPTIONS,
        ),
    )


def _alternation(fragment: str) -> PatternRisk:
    return PatternRisk(
        rule="overlapping-alternation",
        severity="medium",
        fragment=fragment,
        message=(
            "two alternatives of a repeated group can start with the same "
            "character, so each repetition may be tried both ways"
        ),
        suggestions=(
            (
                "make the alternatives start differently, or factor the shared "
                "prefix: `(?:ab?)+` instead of `(?:a|ab)+`"
            ),
            (
                "on Python 3.11+, make the group atomic (`(?>a|ab)+`) so a "
                "repetition, once matched, is not tried the other way"
            ),
            *_QUICK_OPTIONS,
        ),
    )


def _adjacent(fragment: str) -> PatternRisk:
    return PatternRisk(
        rule="adjacent-quantifiers",
        severity="medium",
        fragment=fragment,
        message=(
            "two unbounded repetitions in a row can match the same characters, "
            "so a failing match tries every split point (polynomial time)"
        ),
        suggestions=(
            ("merge them into one repetition, e.g. `\\d{2,}` instead of `\\d+\\d+`"),
            "put a required character between them that neither can match",
            *_QUICK_OPTIONS,
        ),
    )


def analyse_pattern(source: str, flags: int = 0) -> tuple[PatternRisk, ...]:
    r"""
    Report the shapes in ``source`` that can backtrack catastrophically.

    Parameters
    ----------
    source : str
        A regular expression, as written in a pack.
    flags : int, default=0
        The :mod:`re` flags the pattern is compiled with (case folding changes
        which characters overlap).

    Returns
    -------
    tuple of PatternRisk
        Empty when no known risky shape is present. A source this parser
        cannot model gives one ``not-analysed`` finding rather than nothing.

    Examples
    --------
    >>> [risk.rule for risk in analyse_pattern(r"^(a+)+$")]
    ['nested-quantifier']
    >>> analyse_pattern(r"\bEMP-\d{6}\b")
    ()
    >>> analyse_pattern(r"(\w+,)+")
    ()
    """
    try:
        with warnings.catch_warnings():
            # The pack loader compiles (and warns about) the pattern itself;
            # here only "does it compile" matters (``[[]`` warns on 3.7+).
            warnings.simplefilter("ignore")
            re.compile(source, flags)
        prepared, overlap_flags = _prepare(source, flags)
        alternatives = _parse(prepared)
        return tuple(_rules(alternatives, prepared, overlap_flags))
    except (_UnparsedError, re.error, OverflowError, RecursionError) as exc:
        # RecursionError: groups nested deeper than the interpreter's stack
        # (a 2000-character source can hold ~1000); reported, never passed.
        return (
            PatternRisk(
                rule="not-analysed",
                severity="info",
                fragment=source[:60],
                message=(
                    f"this pattern uses syntax the risk check does not model ({exc}); "
                    "review it for nested or overlapping repetition by hand"
                ),
                suggestions=_QUICK_OPTIONS,
            ),
        )


@dataclass(frozen=True)
class PackFinding:
    """
    A finding located in a pack, with the pack's acceptance if it has one.

    Parameters
    ----------
    pack : str
        The pack's name.
    kind : str
        The pattern's kind.
    source : str
        Where the pack was loaded from (a file name, a bundle entry).
    risk : PatternRisk
        What was found.
    accepted : str or None
        The pack's ``risk_reason`` when the pattern says ``risk: accepted``;
        ``None`` when the finding is open.
    """

    pack: str
    kind: str
    source: str
    risk: PatternRisk
    accepted: str | None = None

    @property
    def where(self) -> str:
        """``source: pack NAME, pattern KIND`` — where to go to fix it."""
        return f"{self.source}: pack {self.pack}, pattern {self.kind}"

    def as_dict(self) -> dict:
        """Return the finding as JSON-safe data, for reports."""
        return {
            "pack": self.pack,
            "kind": self.kind,
            "source": self.source,
            "rule": self.risk.rule,
            "severity": self.risk.severity,
            "fragment": self.risk.fragment,
            "message": self.risk.message,
            "suggestions": list(self.risk.suggestions),
            "accepted": self.accepted,
        }


def pack_findings(packs, include_builtin: bool = False) -> list:
    """
    Analyse every pattern of ``packs``; return each finding, accepted or not.

    Parameters
    ----------
    packs : iterable of PackSpec
        Validated packs.
    include_builtin : bool, default=False
        Analyse built-in packs too. They are proven clean by the test suite,
        so a run checks only what a user supplied.

    Returns
    -------
    list of PackFinding
        In pack then pattern order. A pattern with ``risk: accepted`` still
        appears, with ``accepted`` set, so a review lists what was waived.
    """
    found = []
    for pack in packs:
        if pack.source == "<builtin>" and not include_builtin:
            continue
        for spec in pack.patterns:
            found.extend(
                PackFinding(pack.name, spec.kind, pack.source, risk, spec.risk_reason)
                for risk in analyse_pattern(spec.pattern, spec.flags)
            )
    return found


def enforce(findings, mode: str, label: str = "pattern risk") -> list:
    """
    Apply ``mode`` to the open findings: warn, refuse, or stay silent.

    Parameters
    ----------
    findings : iterable of PackFinding
        From :func:`pack_findings`. Accepted ones are never acted on.
    mode : str
        One of :data:`PATTERN_RISK_MODES`.
    label : str, default='pattern risk'
        Names what was loaded in a refusal.

    Returns
    -------
    list of PackFinding
        The open findings (whatever the mode), so a caller can report them.

    Raises
    ------
    PackError
        Under ``refuse`` when any open finding is present.
    ValueError
        If ``mode`` is not a known mode.

    Notes
    -----
    **Developer notes.** Warnings use :class:`PatternRiskWarning`, a
    :class:`UserWarning`, so Python shows them by default, a notebook shows
    them under the cell, ``-W error::...PatternRiskWarning`` makes them fatal
    in CI, and the command line prints them as one ``warning:`` block.
    """
    if mode not in PATTERN_RISK_MODES:
        msg = f"pattern_risk={mode!r} is not one of {', '.join(PATTERN_RISK_MODES)}"
        raise ValueError(msg)
    open_findings = [item for item in findings if item.accepted is None]
    if not open_findings or mode == "ignore":
        return open_findings
    if mode == "refuse":
        from ._packs import PackError  # ruff: ignore[import-outside-top-level]

        raise PackError(label, refusal_lines(open_findings))
    for item in open_findings:
        # stacklevel=2 names the library line, which is the same for every
        # caller: Python's default filter then shows a finding once per
        # process instead of once per cleaner a long session builds.
        warnings.warn(item.risk.describe(item.where), PatternRiskWarning, stacklevel=2)
    return open_findings


def refusal_lines(open_findings) -> list:
    """Return one problem line per open finding, then how to proceed."""
    lines = [
        f"{item.where}: {item.risk.rule} ({item.risk.severity}) in "
        f"`{item.risk.fragment}`: {item.risk.message}"
        for item in open_findings
    ]
    lines.append(
        "the mode is 'refuse': rewrite the pattern (`cleanprompt packs "
        "--check` shows suggested rewrites), accept it in its pack with "
        "`risk: accepted` and a `risk_reason:`, or choose --pattern-risk warn"
    )
    return lines
