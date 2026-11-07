r"""
Named validators that a pack may refer to, and nothing else.

A regular expression can say a value *looks like* an identifier. It cannot say
the value is one. A Luhn checksum, an IBAN mod-97, a Turkish identity-number
check or an NHS mod-11 can, and they are the difference between flagging every
eleven-digit run in a spreadsheet and flagging the identity numbers in it.

Notes
-----
**User notes.** In a pack you write the validator's *name*::

    patterns:
      - kind: TCKN
        pattern: '\b[1-9]\d{10}\b'
        validate: tckn

and the pattern only reports matches that pass the check. :data:`VALIDATORS`
lists the names you can use.

**Developer notes — why validators are named, never supplied.**

A pack is data, and a custom pack is data from somewhere this submodule does
not control. Letting a pack carry a callable — a dotted import path, an
expression, a snippet — would turn "load a configuration file" into "run
somebody's code", which is not a trade a privacy tool gets to make on a user's
behalf. So a pack can only *name* a function, and the name is resolved against
the fixed registry in this module. Adding a validator is a code change,
reviewed like any other, in exactly one place.

Every validator here is **pure and total**: it takes a match, returns a bool,
reads nothing else and raises nothing. A validator that raised would abort
redaction of the whole document, and a validator that returned ``True`` on an
error would silently widen what is hidden; neither is acceptable, so each
function guards its own input rather than relying on the pattern to have done
so.

See Also
--------
scikitplot.cleanprompt._packs : Resolves ``validate:`` names against this registry.
scikitplot.cleanprompt._patterns : The built-in patterns, which use the same checks.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable

from ._patterns import luhn_ok

if TYPE_CHECKING:  # pragma: no cover - annotations only
    Validator = Callable[[re.Match], bool]

__all__ = [
    "VALIDATORS",
    "get_validator",
    "validator_names",
]

#: Digits only, used by every validator that ignores separators.
_NON_DIGIT = re.compile(r"[^0-9]")


def _digits(match: re.Match) -> str:
    """Return the decimal digits of a match, separators removed."""
    return _NON_DIGIT.sub("", match.group())


def _luhn(match: re.Match) -> bool:
    """Accept a match whose digits satisfy the Luhn checksum (payment cards)."""
    return luhn_ok(_digits(match))


def _iban_mod97(match: re.Match) -> bool:
    """Accept an IBAN whose ISO 13616 mod-97 check equals 1."""
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


def _tckn(match: re.Match) -> bool:
    """
    Accept a Turkish identity number (T.C. Kimlik No) that passes its checks.

    Notes
    -----
    **Developer notes.** Eleven digits, the first non-zero. The tenth digit is
    ``(7 * (d1+d3+d5+d7+d9) - (d2+d4+d6+d8)) mod 10`` and the eleventh is the
    sum of the first ten mod 10. Both checks together reject about 99 percent
    of random eleven-digit runs, which is what keeps phone numbers and order
    references out of the category.
    """
    digits = _digits(match)
    if len(digits) != 11 or digits[0] == "0":  # ruff: ignore[magic-value-comparison]
        return False
    values = [ord(char) - 48 for char in digits]
    odd = sum(values[0:9:2])
    even = sum(values[1:8:2])
    if (7 * odd - even) % 10 != values[9]:
        return False
    return sum(values[:10]) % 10 == values[10]


def _nhs_mod11(match: re.Match) -> bool:
    """
    Accept a UK NHS number whose mod-11 check digit is correct.

    Notes
    -----
    **Developer notes.** Weights ten down to two over the first nine digits;
    ``11 - (sum mod 11)`` is the check digit, with 11 read as 0 and 10 meaning
    the number is invalid by construction.
    """
    digits = _digits(match)
    if len(digits) != 10:  # ruff: ignore[magic-value-comparison]
        return False
    total = sum(
        (10 - index) * (ord(char) - 48) for index, char in enumerate(digits[:9])
    )
    check = 11 - (total % 11)
    if check == 11:  # ruff: ignore[magic-value-comparison]
        check = 0
    if check == 10:  # ruff: ignore[magic-value-comparison]
        return False
    return check == ord(digits[9]) - 48


def _npi(match: re.Match) -> bool:
    """
    Accept a US National Provider Identifier.

    Notes
    -----
    **Developer notes.** Ten digits whose Luhn check passes once the constant
    prefix ``80840`` is prepended, as specified by the CMS standard.
    ``luhn_ok`` enforces a card-length window, so the check is computed here.
    """
    digits = _digits(match)
    if len(digits) != 10:  # ruff: ignore[magic-value-comparison]
        return False
    full = "80840" + digits
    total = 0
    parity = len(full) % 2
    for index, char in enumerate(full):
        value = ord(char) - 48
        if index % 2 == parity:
            value *= 2
            if value > 9:  # ruff: ignore[magic-value-comparison]
                value -= 9
        total += value
    return total % 10 == 0


def _aba_routing(match: re.Match) -> bool:
    """Accept a US ABA routing number whose 3-7-1 weighted checksum is zero."""
    digits = _digits(match)
    if len(digits) != 9:  # ruff: ignore[magic-value-comparison]
        return False
    weights = (3, 7, 1, 3, 7, 1, 3, 7, 1)
    return sum(w * (ord(c) - 48) for w, c in zip(weights, digits)) % 10 == 0


def _not_repeated_digit(match: re.Match) -> bool:
    """
    Reject a run made of one repeated digit.

    Notes
    -----
    **Developer notes.** ``0000000000`` and ``1111111111`` are placeholders in
    almost every dataset that has an identifier column. Hiding them is harmless
    and reporting them as identifiers is noise, so a pattern that would
    otherwise match any fixed-length run can use this to stay quiet about them.
    """
    digits = _digits(match)
    return bool(digits) and len(set(digits)) > 1


#: VIN letter values, 49 CFR 565.15(c), Table III. Digits keep their own value;
#: I, O and Q are not used in a VIN.
_VIN_VALUES = {
    **{str(digit): digit for digit in range(10)},
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8,
    "J": 1, "K": 2, "L": 3, "M": 4, "N": 5, "P": 7, "R": 9,
    "S": 2, "T": 3, "U": 4, "V": 5, "W": 6, "X": 7, "Y": 8, "Z": 9,
}  # fmt: skip

#: VIN position weights, 49 CFR 565.15(c), Table IV (position 9 is the check
#: digit and carries no weight).
_VIN_WEIGHTS = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)


def _vin_check(match: re.Match) -> bool:
    """
    Accept a vehicle identification number: by its check digit, or by its label.

    Notes
    -----
    **User notes.** A pattern that captures a ``label`` group (``VIN:``,
    ``chassis number``) is accepted on shape alone, check digit or not, as
    VINs are written outside North America; without a label the check digit
    and numeric tail must hold.

    **Developer notes.** The North American rule (49 CFR 565.15): seventeen
    characters, no I, O or Q; each character's value (Table III) times its
    position's weight (Table IV), summed, modulo 11, is the check digit in
    position 9, a remainder of 10 written ``X``. The last four characters are
    numeric for every vehicle (positions 14 to 17), which this also requires.
    A run of one repeated character (``11111111111111111`` passes the
    arithmetic) is a placeholder, not a vehicle, and is rejected. Case is
    ignored. Outside North America the check digit is optional, so a VIN
    there is found only next to its label.
    """
    groups = match.re.groupindex
    vin = (match.group("value") if "value" in groups else match.group()).upper()
    if len(vin) != len(_VIN_WEIGHTS) or len(set(vin)) == 1:
        return False
    if not all(char in _VIN_VALUES for char in vin):
        return False
    if "label" in groups and match.group("label"):
        return True
    if not vin[13:].isdigit():
        return False
    remainder = sum(_VIN_VALUES[c] * w for c, w in zip(vin, _VIN_WEIGHTS)) % 11
    return vin[8] == ("X" if remainder == 10 else str(remainder))  # noqa: PLR2004


def _de_tax_id(match: re.Match) -> bool:
    """
    Accept a German tax identification number (Steuer-ID) that passes its checks.

    Notes
    -----
    **User notes.** Eleven digits, written together or in groups
    (``12 345 678 901``); separators are ignored. The number must pass the
    published structure rule and check digit, so an order number or a phone
    number of the same length is left alone. Test numbers, which start with
    ``0``, are not personal and are not accepted.

    **Developer notes.** The rules are those of the tax administration's
    published check specification ("Pruefung der Steuer- und
    Steueridentifikationsnummer", ELSTER interface documentation):

    1. eleven digits, the eleventh a check digit, the first not ``0``;
    2. in the first ten digits exactly one digit occurs twice or three times
       and every other digit at most once; a digit occurring three times
       never stands in three directly consecutive positions;
    3. check digit (ISO 7064 MOD 11,10): ``product = 10``; for each of the
       first ten digits ``total = (digit + product) mod 10``, read as ``10``
       when it is ``0``, then ``product = (2 * total) mod 11``; the check
       digit is ``11 - product``, written ``0`` when that is ``10``.

    A pattern's ``value`` group is read when present, so a labelled pattern
    validates the number and not the label.
    """
    groups = match.re.groupindex
    text = match.group("value") if "value" in groups else match.group()
    digits = _NON_DIGIT.sub("", text)
    if len(digits) != 11 or digits[0] == "0":  # ruff: ignore[magic-value-comparison]
        return False
    head = digits[:10]
    counts = sorted((head.count(d) for d in set(head)), reverse=True)
    if counts[0] not in (2, 3) or (len(counts) > 1 and counts[1] > 1):
        return False
    if counts[0] == 3 and any(  # ruff: ignore[magic-value-comparison]
        head[i] == head[i + 1] == head[i + 2] for i in range(8)
    ):
        return False
    product = 10
    for char in head:
        total = (ord(char) - 48 + product) % 10 or 10
        product = (2 * total) % 11
    check = 11 - product
    return digits[10] == str(0 if check == 10 else check)  # noqa: PLR2004


#: Every validator a pack may name. The keys are the public vocabulary; they
#: are part of the pack schema and are never renamed.
VALIDATORS: dict[str, Callable[[re.Match], bool]] = {
    "aba_routing": _aba_routing,
    "de_tax_id": _de_tax_id,
    "iban_mod97": _iban_mod97,
    "luhn": _luhn,
    "nhs_mod11": _nhs_mod11,
    "not_repeated_digit": _not_repeated_digit,
    "npi": _npi,
    "tckn": _tckn,
    "vin_check": _vin_check,
}


def validator_names() -> tuple[str, ...]:
    """
    Return the names a pack may use in ``validate:``.

    Returns
    -------
    tuple of str
        Sorted.

    Examples
    --------
    >>> "luhn" in validator_names()
    True
    """
    return tuple(sorted(VALIDATORS))


def get_validator(name: str | None) -> Callable[[re.Match], bool] | None:
    r"""
    Resolve a validator name.

    Parameters
    ----------
    name : str or None
        A key of :data:`VALIDATORS`, or ``None`` for no validation.

    Returns
    -------
    callable or None
        The validator, or ``None`` when ``name`` is ``None``.

    Raises
    ------
    KeyError
        If the name is not registered. The message lists the names that are.

    Examples
    --------
    >>> get_validator(None) is None
    True
    >>> import re
    >>> get_validator("tckn")(re.search(r"\d+", "10000000146"))
    True
    """
    if name is None:
        return None
    try:
        return VALIDATORS[name]
    except KeyError:
        msg = f"unknown validator {name!r}; choose from {', '.join(validator_names())}"
        raise KeyError(msg) from None
