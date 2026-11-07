"""
Curated pattern library for structural (non-linguistic) detection.

Each entry states what it is *for*, carries positive and negative examples that
the test suite executes, and may carry a validator that rejects a structural
match whose content is not actually of that kind.

Notes
-----
**User notes.** :func:`default_patterns` returns the set enabled when a policy
does not name ``kinds``. Every pattern is addressable by its ``kind``, so
``policy.evolve(kinds=("EMAIL", "URL"))`` selects exactly two.

**Developer notes.** Three rules apply to anything added here.

1. **State the intent.** A regular expression without a written intent cannot be
   reviewed, and cannot be distinguished from a typo. Both reproduced upstream
   pattern defects were typos that read as intentional: ``[$-_@.&+]`` is a
   character *range* from ``$`` (0x24) to ``_`` (0x5F) rather than the four
   literals it appears to be, and ``[A-Z|a-z]`` admits a literal ``|``.
2. **Carry negative examples.** ``examples_no`` is executed by the test suite.
   A pattern that cannot state what it must *not* match has not been thought
   through.
3. **Prefer a validator over a cleverer expression.** Structure and semantics
   are different questions. A card number is sixteen digits *and* passes Luhn; an
   IBAN is a country code and digits *and* passes mod-97. Encoding the second
   half into the regular expression makes it unreadable, while a small pure
   function is testable on its own.

Every expression here is linear-time on the inputs it accepts: there is no
nested quantifier over an overlapping character class, so none of them exhibits
catastrophic backtracking. Input length is separately bounded by
:class:`~scikitplot.cleanprompt._policy.Limits`.

See Also
--------
scikitplot.cleanprompt._detectors : Turns these specs into detectors.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

from ._exceptions import PatternError

__all__ = [
    "PATTERNS",
    "PatternSpec",
    "default_patterns",
    "get_pattern",
    "luhn_ok",
]


@dataclass(frozen=True)
class PatternSpec:
    """
    One named structural pattern.

    Parameters
    ----------
    kind : str
        Category label, used to build placeholders. Upper case by convention.
    pattern : str
        The regular expression source.
    intent : str
        What this pattern is for, in one sentence. Reviewed, not decorative.
    priority : int, default=50
        Arbitration weight under
        :attr:`~scikitplot.cleanprompt._policy.OverlapStrategy.PRIORITY_WINS`
        and as a tie-break elsewhere. Higher means more specific.
    flags : int, default=0
        Flags passed to :func:`re.compile`.
    validate : callable, optional
        ``validate(match) -> bool``. A structural match returning ``False`` is
        discarded. Must be pure and total.
    confidence : float, default=1.0
        Confidence reported for matches of this pattern.
    enabled_by_default : bool, default=True
        Whether :func:`default_patterns` includes it.
    examples_yes : tuple of str, default=()
        Strings that must produce at least one accepted match.
    examples_no : tuple of str, default=()
        Strings that must produce no accepted match.
    risk_reason : str, optional
        Set when the pack accepts this pattern's backtracking risk on purpose
        (``risk: accepted``), with the reason written beside it. The pattern
        risk check then lists it as accepted instead of warning.

    Raises
    ------
    PatternError
        If ``pattern`` does not compile, or ``intent`` is empty.
    """

    kind: str
    pattern: str
    intent: str
    priority: int = 50
    flags: int = 0
    validate: Callable[[re.Match], bool] | None = field(default=None, repr=False)
    confidence: float = 1.0
    enabled_by_default: bool = True
    examples_yes: tuple[str, ...] = ()
    examples_no: tuple[str, ...] = ()
    risk_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.intent:
            raise PatternError(
                f"pattern {self.kind!r} has no stated intent", name=self.kind
            )
        try:
            re.compile(self.pattern, self.flags)
        except (re.error, OverflowError) as exc:
            raise PatternError(
                f"pattern {self.kind!r} does not compile: {exc}",
                name=self.kind,
            ) from exc

    def compiled(self) -> re.Pattern:
        """
        Return the compiled expression.

        Returns
        -------
        re.Pattern
            The compiled pattern.

        Notes
        -----
        **Developer notes.** :func:`re.compile` maintains its own cache keyed on
        ``(pattern, flags)``, so repeated calls are cheap and no cache is kept
        here. Holding a compiled object on a frozen dataclass would make the
        instance unpicklable for no gain.
        """
        return re.compile(self.pattern, self.flags)


# ---------------------------------------------------------------------------
# validators
# ---------------------------------------------------------------------------


def luhn_ok(digits: str) -> bool:
    """
    Return whether ``digits`` satisfies the Luhn checksum.

    Parameters
    ----------
    digits : str
        A string of decimal digits, already stripped of separators.

    Returns
    -------
    bool
        ``True`` when the Luhn check passes and the length is plausible for a
        payment card (12 to 19 digits).

    Notes
    -----
    **Developer notes.** Luhn is a transcription-error check, not a validity
    proof: it rejects roughly 90 percent of random digit runs, which is enough
    to keep order numbers and long identifiers out of the card category while
    never rejecting a genuine card.

    Examples
    --------
    >>> luhn_ok("4242424242424242")
    True
    >>> luhn_ok("4242424242424241")
    False
    """
    if (
        not digits.isdigit()  # lint
        or not 12 <= len(digits) <= 19  # ruff: ignore[magic-value-comparison]
    ):
        return False
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        value = ord(char) - 48
        if index % 2 == parity:
            value *= 2
            if value > 9:  # ruff: ignore[magic-value-comparison]
                value -= 9
        total += value
    return total % 10 == 0


def _card_validate(match: re.Match) -> bool:
    """Accept a card-shaped match only when its digits satisfy Luhn."""
    return luhn_ok(re.sub(r"[^0-9]", "", match.group()))


def _iban_validate(match: re.Match) -> bool:
    """
    Accept an IBAN-shaped match only when it satisfies the mod-97 check.

    Notes
    -----
    **Developer notes.** ISO 13616: move the first four characters to the end,
    map letters to two-digit numbers with ``A=10``, and require the resulting
    integer to be congruent to 1 modulo 97.
    """
    raw = re.sub(r"\s", "", match.group()).upper()
    if not 15 <= len(raw) <= 34:  # ruff: ignore[magic-value-comparison]
        return False
    rotated = raw[4:] + raw[:4]
    buffer = []
    for char in rotated:
        if char.isdigit():
            buffer.append(char)
        elif "A" <= char <= "Z":
            buffer.append(str(ord(char) - 55))
        else:
            return False
    return int("".join(buffer)) % 97 == 1


def _mark_ranges() -> str:
    r"""
    Return every combining mark as the body of a regular-expression class.

    Notes
    -----
    **Developer notes.** ``\w`` matches letters and digits but not combining
    marks (categories ``Mn``, ``Mc``, ``Me``), and many scripts cannot write
    a word without them: Devanagari vowel signs and virama, Thai vowels,
    Hebrew points, a decomposed accent. A local part or domain label built on
    ``\w`` alone would stop inside the first such word, and a partial
    address is a leak (``CP-110``). Every mark lies below ``U+20000`` or in
    the variation-selector supplement, so those planes are scanned once at
    import (about 30 ms), as :mod:`._canonical` does for format characters.
    The set follows the running :mod:`unicodedata`, as ``\w`` does.
    """
    ranges: list[list[int]] = []
    for block in (range(0x20000), range(0xE0000, 0xE1000)):
        for point in block:
            if unicodedata.category(chr(point)) in ("Mn", "Mc", "Me"):
                if ranges and ranges[-1][1] == point - 1:
                    ranges[-1][1] = point
                else:
                    ranges.append([point, point])
    return "".join(
        (
            re.escape(chr(low))
            if low == high
            else f"{re.escape(chr(low))}-{re.escape(chr(high))}"
        )
        for low, high in ranges
    )


#: Letters, digits and combining marks of any script, without ``_``.
_LETTER = r"[^\W_]|[" + _mark_ranges() + r"]"

#: RFC 5322 ``atext`` outside letters and digits. ``|`` is atext; only the
#: domain excludes it (``CP-005``).
_ATEXT_SPECIALS = r"!#$%&'*+/=?^_`{|}~-"

#: One ``atext`` character of an internationalised local part (RFC 6531).
_ATEXT = r"(?:" + _LETTER + r"|[" + _ATEXT_SPECIALS + r"])"

#: One domain label (RFC 1035 shape, letters of any script as in an IDNA
#: U-label): starts with a letter or digit, ends with a letter, digit or
#: mark, at most 63 characters.
_LABEL = r"(?:[^\W_])(?:(?:" + _LETTER + r"|-){0,61}(?:" + _LETTER + r"))?"


#: Characters allowed anywhere inside a URL's path, query or fragment, from
#: the RFC 3986 unreserved and sub-delimiter sets.
_URL_BODY = r"A-Za-z0-9\-._~%!$&'()*+,;=:@/"

#: Characters a URL is allowed to *end* on. The difference from
#: :data:`_URL_BODY` is the sentence punctuation — ``. , ; ! ' ( )`` — which is
#: legal inside a URL but is, at the end of one, almost always the writer's
#: punctuation rather than part of the address.
#:
#: The cost is a URL that genuinely ends in a closing bracket, such as a
#: Wikipedia disambiguation link: its final ``)`` is left in the clear. That
#: direction is the safe one. The opposite reading swallowed the sentence's
#: full stop into the placeholder, which both corrupted the value stored in
#: the vault and removed the sentence boundary from the text handed to the
#: model.
_URL_TAIL = r"A-Za-z0-9\-_~%$&*+=:@/"


def _ipv4_validate(match: re.Match) -> bool:
    """Accept a dotted quad only when every octet is in ``0..255``."""
    parts = match.group().split(".")
    return len(parts) == 4 and all(  # ruff: ignore[magic-value-comparison]
        part.isdigit() and int(part) < 256  # ruff: ignore[magic-value-comparison]
        for part in parts  # ruff: ignore[magic-value-comparison]
    )


_ISO_DATE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")

#: Two years joined by a dash: a range, never a telephone number.
_YEAR_RANGE = re.compile(r"^(?:1[89]|2[01])\d\d ?- ?(?:1[89]|2[01])\d\d$")

#: A date of three numbers with one hyphen or full stop throughout, year last
#: (the only separators the phone expression shares with a date).
_NUMERIC_DATE = re.compile(r"^(\d{1,2})([.\-])(\d{1,2})\2[12]\d{3}$")

#: The largest day and month numbers.
_MAX_DAY, _MAX_MONTH = 31, 12


def _is_numeric_date(text: str) -> bool:
    """Return whether ``text`` is a day-month-year or month-day-year date."""
    found = _NUMERIC_DATE.match(text)
    if found is None:
        return False
    first, second = int(found.group(1)), int(found.group(3))
    if not (first and second):
        return False
    return (first <= _MAX_DAY and second <= _MAX_MONTH) or (
        first <= _MAX_MONTH and second <= _MAX_DAY
    )


#: Words that begin a sentence often enough that treating them as a name is
#: simply wrong. Used only by :func:`_title_case_validate`.
_TITLE_STOPWORDS = frozenset(
    {
        "A",
        "An",
        "And",
        "As",
        "At",
        "But",
        "By",
        "For",
        "From",
        "He",
        "Her",
        "His",
        "However",
        "I",
        "If",
        "In",
        "It",
        "Its",
        "Of",
        "On",
        "Or",
        "She",
        "That",
        "The",
        "Their",
        "Then",
        "There",
        "These",
        "They",
        "This",
        "Those",
        "To",
        "We",
        "What",
        "When",
        "Which",
        "While",
        "Who",
        "With",
        "You",
        "Your",
    }
)


def _title_case_validate(match: re.Match) -> bool:
    """
    Accept a word run only when every token is capitalised and meaningful.

    Notes
    -----
    **Developer notes.** The expression matches word runs; this decides whether
    the run is title case. Doing it here rather than in the pattern keeps the
    check Unicode-correct: :meth:`str.isupper` knows that ``Ü`` and ``Ş`` are
    upper case, whereas an ``[A-Z]`` character class does not, and the names
    this pattern exists to catch are frequently not ASCII.

    Two rejections. A run whose first token is a common sentence opener is
    ordinary sentence case, not a name. A single-token run is rejected outright:
    every sentence in English begins with a capital, so accepting single tokens
    would redact the first word of every sentence and destroy the text.
    """
    tokens = match.group().split()
    if len(tokens) < 2:  # ruff: ignore[magic-value-comparison]
        return False
    if tokens[0] in _TITLE_STOPWORDS:
        return False
    return all(token[:1].isupper() for token in tokens)


def _phone_validate(match: re.Match) -> bool:
    """
    Accept a phone-shaped match only when it is plausibly a phone number.

    Notes
    -----
    **Developer notes.** Structure alone is not enough, and this is the
    reproduced upstream defect ``CP-013``: the upstream expression matched
    ``2024-01-15`` and ``12345678``. Three rules, each deterministic:

    1. An ISO-8601-shaped run is a date, not a number. Rejected outright.
    2. Digit count must fall in ``7..15``. Seven is the shortest subscriber
       number in general use; fifteen is the E.164 maximum.
    3. A match without an international prefix must contain a separator, so a
       bare run of digits is never taken for a phone number.
    4. A year range (two years from 1800 to 2199 joined by a dash) and a
       day-month-year or month-day-year date with one separator throughout
       are not numbers either (``CP-111``). Read through the detection view,
       an en dash arrives as ``-``, so ``2019\u20132024`` is a range too.
    """
    text = match.group().strip()
    if _ISO_DATE.match(text) or _YEAR_RANGE.match(text) or _is_numeric_date(text):
        return False
    digits = re.sub(r"[^0-9]", "", text)
    if not 7 <= len(digits) <= 15:  # ruff: ignore[magic-value-comparison]
        return False
    return not (not text.startswith("+") and not re.search(r"[\s.\-()]", text))


def _ssn_validate(match: re.Match) -> bool:
    """
    Accept a US SSN shape only when its groups are structurally issuable.

    Notes
    -----
    **Developer notes.** The Social Security Administration never issues an area
    of ``000``, ``666`` or ``900``-``999``, a group of ``00``, or a serial of
    ``0000``. Rejecting those keeps common dummy values and formatted numeric
    identifiers out of the category.
    """
    digits = re.sub(r"[^0-9]", "", match.group())
    if len(digits) != 9:  # ruff: ignore[magic-value-comparison]
        return False
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    if area in {"000", "666"} or area[0] == "9":
        return False
    return group != "00" and serial != "0000"


# ---------------------------------------------------------------------------
# the library
# ---------------------------------------------------------------------------

_key = "KEY"
_SPECS: tuple[PatternSpec, ...] = (
    PatternSpec(
        kind="PRIVATE_KEY",
        pattern=(
            r"-----BEGIN (?:[A-Z][A-Z ]{0,40})?PRIVATE KEY-----"
            r"[\s\S]{0,8192}?"
            r"-----END (?:[A-Z][A-Z ]{0,40})?PRIVATE KEY-----"
        ),
        intent=(
            "A PEM-armoured private key block. Highest priority because the "
            "body contains base64 that other patterns would otherwise carve up."
        ),
        priority=100,
        examples_yes=(
            (
                f"-----BEGIN RSA PRIVATE {_key}-----\nMIIBOgIBAAJB\n"
                f"-----END RSA PRIVATE {_key}-----"
            ),
        ),
        examples_no=("-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----",),
    ),
    PatternSpec(
        kind="JWT",
        pattern=r"\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\b",
        intent=(
            "A JSON Web Token. Anchored on the base64 of '{\"' so that an "
            "arbitrary three-part dotted identifier is not mistaken for one."
        ),
        priority=95,
        examples_yes=("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.dBjftJeZ4CVP",),
        examples_no=("a.b.c", "one.two.three"),
    ),
    PatternSpec(
        kind="AWS_ACCESS_KEY",
        pattern=r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA|ANVA|ABIA|ACCA)[0-9A-Z]{16}\b",
        intent=(
            "An AWS access key identifier, recognised by its documented "
            "four-character resource-type prefix and sixteen-character body."
        ),
        priority=95,
        examples_yes=("AKIAIOSFODNN7EXAMPLE",),
        examples_no=("AKIA", "NOTAKEYHERE1234567890"),
    ),
    PatternSpec(
        kind="IBAN",
        pattern=r"\b[A-Z]{2}[0-9]{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b",
        intent=(
            "An international bank account number. The structural shape alone "
            "is far too permissive, so the mod-97 checksum decides."
        ),
        priority=90,
        validate=_iban_validate,
        examples_yes=("GB82 WEST 1234 5698 7654 32", "DE89370400440532013000"),
        examples_no=("GB00 WEST 1234 5698 7654 32", "AB12 CDEF"),
    ),
    PatternSpec(
        kind="CREDIT_CARD",
        pattern=r"\b(?:[0-9]{4}[ -]?){3}[0-9]{1,7}\b",
        intent=(
            "A payment card number in the common four-group layout. Luhn "
            "decides, so long order numbers and identifiers are not swept in."
        ),
        priority=90,
        validate=_card_validate,
        examples_yes=("4242 4242 4242 4242", "4242-4242-4242-4242", "4111111111111111"),
        examples_no=("1234 5678 9012 3456", "0000 0000 0000 0001"),
    ),
    PatternSpec(
        kind="EMAIL",
        pattern=(
            r"(?<!" + _ATEXT + r")" + _ATEXT + r"+(?:\." + _ATEXT + r"+)*"
            r"@" + _LABEL + r"(?:\." + _LABEL + r")+"
        ),
        intent=(
            "An email address, following the RFC 5322 dot-atom local part and "
            "an RFC 1035 label-based domain, with letters, digits and marks of "
            "any script as RFC 6531 and IDNA allow (CP-110). It starts where "
            "no local-part character precedes it, so a letter of another script "
            "in front is part of the address instead of hiding it. No label "
            "admits a pipe; the upstream class '[A-Z|a-z]' did, which is "
            "defect CP-005."
        ),
        priority=80,
        examples_yes=(
            "a.b+tag@example.co.uk",
            "user_name@sub.domain.org",
            "o'hara.test@example.com",
            "jos\u00e9@example.com",
            "ada@ex\u00e4mple.com",
            "\u0928\u092e\u0938\u094d\u0924\u0947@\u0909\u0926\u093e\u0939\u0930\u0923.\u092d\u093e\u0930\u0924",
            "\u7528\u6237@\u4f8b\u5b50.\u5e7f\u544a",
        ),
        # 'a@b.a|b' is deliberately NOT listed here: the pipe is excluded from
        # the domain, so the expression stops at 'a@b.a', which is itself a
        # well-formed address. The contract is "no match ever spans a pipe",
        # asserted directly in test__patterns.TestEmailDomainCharacters.
        examples_no=(
            "a@b",
            "not.an.email",
            "@example.com",
            "a@-bad.com",
            "a@b..c",
            "\u00e9t\u00e9 @ caf\u00e9",
        ),
    ),
    PatternSpec(
        kind="URL",
        pattern=(
            r"\b(?:https?|ftp)://"
            r"[A-Za-z0-9\-._~%]*[A-Za-z0-9\-_~%](?::[0-9]{1,5})?"
            r"(?:/(?:[" + _URL_BODY + r"]*[" + _URL_TAIL + r"])?)?"
            r"(?:\?(?:[" + _URL_BODY + r"?]*[" + _URL_TAIL + r"])?)?"
            r"(?:#(?:[" + _URL_BODY + r"?]*[" + _URL_TAIL + r"])?)?"
        ),
        intent=(
            "An absolute http, https or ftp URL, built from the RFC 3986 "
            "unreserved and sub-delimiter sets written out explicitly. The "
            "upstream class '[$-_@.&+]' was read by the engine as the range "
            "0x24 to 0x5F, which is defect CP-004. A URL may not end on "
            "sentence punctuation, so a trailing full stop, comma or bracket "
            "is left in the text instead of being swallowed into the "
            "placeholder (CP-029)."
        ),
        priority=70,
        examples_yes=(
            "https://example.com/path?q=1#frag",
            "http://10.0.0.1:8080/x",
            "ftp://files.example.org/pub",
            "https://example.com",
            "https://example.com/a/b/",
        ),
        examples_no=("example.com", "mailto:a@b.co", "://nope"),
    ),
    PatternSpec(
        kind="IPV6",
        pattern=(
            r"(?<![:.\w])(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}"
            r"(?![:\w])(?!\.\w)"
        ),
        intent=(
            "An IPv6 address in its common full or partially compressed forms. "
            "Placed above IPv4 so that an embedded dotted quad is not split. "
            "The trailing dot is rejected only when something follows it, so "
            "an address ending a sentence is still found (CP-028)."
        ),
        priority=65,
        examples_yes=("2001:0db8:85a3:0000:0000:8a2e:0370:7334", "fe80:1:2:3:4:5:6:7"),
        examples_no=("12:34", "not:an:address:zz"),
    ),
    PatternSpec(
        kind="IPV4",
        pattern=r"(?<!\d\.)(?<!\w)(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?!\.?\d)(?!\w)",
        intent=(
            "A dotted-quad IPv4 address. The octet range is checked by a "
            "validator rather than by an unreadable alternation. The guards "
            "reject a longer dotted run such as '1.2.3.4.5' without rejecting "
            "an address at the end of a sentence: the trailing dot only "
            "disqualifies the match when a digit follows it (CP-028)."
        ),
        priority=60,
        validate=_ipv4_validate,
        examples_yes=("192.168.1.10", "8.8.8.8"),
        examples_no=("999.1.1.1", "1.2.3", "1.2.3.4.5"),
    ),
    PatternSpec(
        kind="MAC",
        pattern=r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b",
        intent=(
            "A hardware (MAC) address in colon or hyphen notation. Ranked "
            "above IPV6 because the two shapes collide: six colon-separated "
            "pairs of hex digits is a valid IPv6 fragment and an exact MAC, "
            "and equal-length spans are arbitrated by priority."
        ),
        priority=70,
        examples_yes=("00:1B:44:11:3A:B7", "00-1b-44-11-3a-b7"),
        examples_no=("00:1B:44:11:3A", "zz:zz:zz:zz:zz:zz"),
    ),
    PatternSpec(
        kind="SSN_US",
        pattern=r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b",
        intent=(
            "A United States Social Security number in its hyphenated form. "
            "Only the hyphenated shape is matched, because a bare nine-digit "
            "run is indistinguishable from countless other identifiers."
        ),
        priority=85,
        validate=_ssn_validate,
        examples_yes=("123-45-6789",),
        examples_no=("000-45-6789", "666-45-6789", "123-00-6789", "123456789"),
    ),
    PatternSpec(
        kind="TITLE_CASE",
        pattern=(
            r"\b[^\W\d_][^\W\d_'\u2019.\-]*"
            r"(?:[ \t]+[^\W\d_][^\W\d_'\u2019.\-]*){1,5}\b"
        ),
        intent=(
            "A run of two or more capitalised words, which is what most "
            "personal and organisation names look like. DISABLED BY DEFAULT "
            "and deliberately so: it is the only pattern here whose precision "
            "is not high. It exists because without the optional 'ner' tier "
            "there is otherwise nothing at all that can find a person's name, "
            "and a redaction tool that silently cannot see names is worse than "
            "one that over-redacts visibly. Enable it explicitly, or use the "
            "suggestion surface instead, which shows candidates and lets a "
            "person choose."
        ),
        priority=20,
        validate=_title_case_validate,
        confidence=0.45,
        enabled_by_default=False,
        examples_yes=("Mustafa Kemal Atatürk", "Ada Lovelace", "Acme Corporation"),
        examples_no=(
            "The quick brown fox",
            "he went to town",
            "Turkey",
            "however Ada left",
        ),
    ),
    PatternSpec(
        kind="PHONE",
        pattern=(
            r"(?<![\w.])"
            r"(?:"
            # International: +CC, an optional bracketed group (a "(0)" trunk
            # or an area code), an area code of one to four digits, then
            # groups of up to eight (CP-111).
            r"\+[0-9]{1,3}[ .\-]?(?:\([0-9]{1,4}\)[ .\-]?)?"
            r"[0-9]{1,4}(?:[ .\-][0-9]{1,8}){1,5}"
            r"|"
            # National with a trunk 0 and a long subscriber number.
            r"0[0-9]{1,4}[ .\-][0-9]{5,8}"
            r"|"
            # National in groups of two to four digits.
            r"(?:\([0-9]{2,4}\)[ .\-]?)?[0-9]{2,4}(?:[ .\-][0-9]{2,4}){1,4}"
            r")"
            r"(?:[ ]?(?:ext\.?|x)[ ]?[0-9]{1,6})?"
            r"(?![\w])"
        ),
        intent=(
            "A telephone number written with separators, or with an "
            "international prefix, from its + or first digit to its last "
            "group and any extension (CP-111). A validator enforces a 7-to-15 "
            "digit count and rejects ISO-8601 dates (defect CP-013), "
            "day-month-year dates and year ranges."
        ),
        priority=40,
        validate=_phone_validate,
        confidence=0.85,
        examples_yes=(
            "+1 555 010 4477",
            "(020) 7946 0958",
            "555-010-4477",
            "+33 1 23 45 67 89",
            "+49 30 12345678",
            "+49 (0)30 1234 5678",
            "030 12345678",
            "+44 7700 900123",
            "+90 212 555 12 34",
            "+1 555 010 4477 ext. 123",
        ),
        examples_no=(
            "2024-01-15",
            "12345678",
            "1.26.4",
            "Revenue rose 1234",
            "2019-2024",
            "15-01-2024",
            "15.01.2024",
            "ISBN 978-3-16-148410-0",
        ),
    ),
)

#: Every pattern in the library, keyed by :attr:`PatternSpec.kind`.
PATTERNS: dict[str, PatternSpec] = {spec.kind: spec for spec in _SPECS}


def get_pattern(kind: str) -> PatternSpec:
    """
    Return one pattern specification by kind.

    Parameters
    ----------
    kind : str
        The category name.

    Returns
    -------
    PatternSpec
        The specification.

    Raises
    ------
    PatternError
        If no pattern of that kind exists. The message lists what does.
    """
    try:
        return PATTERNS[kind]
    except KeyError:
        msg = "unknown pattern kind {!r}; available kinds are {}".format(
            kind, ", ".join(sorted(PATTERNS))
        )
        raise PatternError(
            msg,
            name=kind,
        ) from None


def default_patterns() -> tuple[PatternSpec, ...]:
    """
    Return the patterns enabled when a policy does not name ``kinds``.

    Returns
    -------
    tuple of PatternSpec
        In descending priority, then alphabetical by kind, so the order is
        stable across runs.

    Examples
    --------
    >>> kinds = [spec.kind for spec in default_patterns()]
    >>> "EMAIL" in kinds and "URL" in kinds
    True
    """
    return tuple(
        sorted(
            (spec for spec in _SPECS if spec.enabled_by_default),
            key=lambda spec: (-spec.priority, spec.kind),
        )
    )
