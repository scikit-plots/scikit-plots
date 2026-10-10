"""
Invented stand-ins that read as ordinary text (base tier).

Notes
-----
**User notes.** Instead of bracket tokens, put plausible invented values in::

    python -m scikitplot.cleanprompt encode --style surrogate --ner <<'END'
    Ada Lovelace mailed ada@example.com about the Analytical Society.
    END
    Marion Holt mailed marion.holt@example.invalid about Northwind Logistics.

Send that; ``decode`` turns the stand-ins back into your values. The mapping
lives in the same vault, so nothing else about the workflow changes.

**Developer notes — why this exists, beyond looking nicer.**

``[PERSON-1] emailed [EMAIL-1] about [ORG-1]`` is not a sentence. Two things go
wrong when it is sent to a language model.

*The model reasons about it worse.* The bracket tokens carry no grammatical
number, gender or animacy, they break the sentence's rhythm, and they invite
the model to talk about the redaction — "I notice this text has been
anonymised" — instead of the question asked.

*The model rewrites them.* This is the measured one. Replies come back with
``[email-1]``, ``[EMAIL_1]``, ``[EMAIL 1]``, escaped brackets and labels
wrapped across lines, because a bracket-delimited upper-case token is exactly
the sort of thing a model normalises. The lenient matcher in
:func:`~scikitplot.cleanprompt.restore` exists to repair that damage after the
fact. A surrogate prevents it: no model rewrites ``Marion Holt``, because there
is nothing about it to normalise.

So this is prevention where the lenient matcher is mitigation, and the two are
meant to be held together rather than chosen between.

**What is not surrogated, and why.**

Only kinds where a proper noun is the natural replacement and an unmistakably
non-real form exists. Credentials keep their placeholders:

.. code-block:: text

    surrogated   PERSON ORG GPE LOC FAC EMAIL PHONE URL
    placeholder  CREDIT_CARD IBAN SSN_US AWS_ACCESS_KEY JWT PRIVATE_KEY
                 MAC IPV4 IPV6 NORP and everything else

A plausible-looking card number, account number or access key is a hazard, not
a convenience: it can be mistaken for real by a person, it can be acted on by a
system, and by chance it could *be* real. ``[CREDIT_CARD-1]`` cannot be
mistaken for anything. ``NORP`` is excluded for a different reason — it is
adjectival ("Turkish", "Catholic"), and an invented demonym reads as nonsense
rather than as an ordinary word.

Where a surrogate is produced, it uses a form reserved so that it cannot
resolve to anything real: ``example.invalid`` for addresses and links (RFC 2606
reserves ``.invalid`` permanently) and the ``+1 555 0100``-``0199`` block for
telephone numbers, which is reserved for fiction in the North American plan.

**Names are invented, and recorded.** The banks below are made up. Any
resemblance to a real person or company is coincidence, and the vault always
says which stand-in means which value, so nothing depends on being able to tell
them apart by eye.

See Also
--------
scikitplot.cleanprompt._policy : ``TagStyle``, the placeholder grammar.
scikitplot.cleanprompt._engine : Where both styles are applied and reversed.
"""  # ruff: ignore[ambiguous-unicode-character-docstring]

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ._surrogate_sets import detected_by, entry_problem

__all__ = [
    "DEFAULT_STYLE",
    "STYLES",
    "SURROGATE_KINDS",
    "surrogate_for",
]

#: Rewriting styles offered to a caller.
STYLES = ("placeholder", "surrogate")

#: What a policy uses when nothing is chosen.
#:
#: ``placeholder`` remains the default. A bracket token is unmistakable: a
#: reader of the redacted text can see at a glance that a value was removed,
#: and cannot mistake the stand-in for the real thing. Surrogates trade that
#: obviousness for fluency, which is the right trade for a prompt and the wrong
#: one for, say, a document someone will read and act on.
DEFAULT_STYLE = "placeholder"

_FIRST = (
    "Marion",
    "Devin",
    "Priya",
    "Tomas",
    "Lines",
    "Rafael",
    "Noor",
    "Lukas",
    "Sasha",
    "Imani",
    "Bram",
    "Vera",
    "Otto",
    "Leila",
    "Hugo",
    "Mira",
    "Caspar",
    "Yara",
    "Nils",
    "Talia",
    "Emre",
    "Rosa",
    "Anton",
    "Wren",
)

_LAST = (
    "Holt",
    "Varga",
    "Okonkwo",
    "Bellamy",
    "Stroud",
    "Nakamura",
    "Orsini",
    "Deveraux",
    "Kaminski",
    "Ellery",
    "Thorne",
    "Lindqvist",
    "Ferreira",
    "Ashby",
    "Novak",
    "Quintero",
    "Rask",
    "Whitlow",
    "Barros",
    "Yilmaz",
    "Calloway",
    "Ferrand",
    "Osgood",
    "Tamm",
)

_ORG_FIRST = (
    "Northwind",
    "Cobalt",
    "Brightmoor",
    "Ardent",
    "Halcyon",
    "Ironvale",
    "Silverbeck",
    "Kestrel",
    "Marlowe",
    "Pentland",
    "Ravenna",
    "Thistledown",
)

_ORG_SECOND = (
    "Logistics",
    "Systems",
    "Holdings",
    "Analytics",
    "Foundry",
    "Partners",
    "Industries",
    "Laboratories",
    "Collective",
    "Works",
    "Group",
    "Union",
)

_PLACE = (
    "Ardenfield",
    "Coldharbour",
    "Mirefen",
    "Westrey",
    "Halloway",
    "Dunmoor",
    "Ellisport",
    "Kirkwall Green",
    "Sablebrook",
    "Tenby Cross",
    "Norreys",
    "Varinsk",
    "Little Ashcombe",
    "Pelham Reach",
    "Ostravale",
    "Brindlemere",
)

_FEATURE = (
    "the Kestrel Valley",
    "the Marrow Downs",
    "the Ashgate Marshes",
    "the Pellwood",
    "the Silverbeck Fells",
    "the Tarn Coast",
)

_FACILITY = (
    "Brightmoor Station",
    "the Ellisport Exchange",
    "Halloway Hall",
    "the Norreys Institute",
    "Sablebrook Depot",
    "the Pelham Rooms",
)

#: Kinds that receive an invented value, and the bank each draws on.
SURROGATE_KINDS = (
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "FAC",
    "EMAIL",
    "PHONE",
    "URL",
)


#: Surrogated kinds whose reserved form only the core builds.
_CORE_FORMS = ("EMAIL", "PHONE", "URL")


def _pair(first: tuple[str, ...], second: tuple[str, ...], index: int) -> str:
    """
    Return a deterministic two-part name for ``index``.

    Notes
    -----
    **Developer notes.** Both halves advance on every step, which is not what a
    plain mixed-radix counter does. Counting normally gives ``Marion Holt``,
    ``Devin Holt``, ``Priya Holt`` — consecutive people sharing a surname,
    which reads as a family and implies a relationship that is not in the data.
    Anyone reasoning over the redacted text, model or person, would draw a
    conclusion the original never supported.

    Offsetting the second index by a multiple of the first breaks that up while
    keeping every combination reachable: for a fixed first name the offset is
    constant, so the second index still runs through its whole range before
    repeating, and ``len(first) * len(second)`` pairs remain distinct.
    """
    a = index % len(first)
    b = (index // len(first) + a * 5) % len(second)
    return f"{first[a]} {second[b]}"


def _candidate(  # ruff: ignore[too-many-return-statements]
    kind: str,
    index: int,
    provider: Any = None,
) -> str | None:
    """
    Return the ``index``-th stand-in for ``kind``, or ``None``.

    Notes
    -----
    **Developer notes.** A provider (a
    :class:`~scikitplot.cleanprompt._surrogate_sets.SurrogateSet`) is asked
    first for name kinds only. ``EMAIL``, ``PHONE`` and ``URL`` never reach it:
    their reserved forms are the core's (``GENERATOR_DESIGN.md`` floor rule
    2). A provider answering ``None`` leaves the kind to the built-in names.
    """
    if provider is not None and kind not in _CORE_FORMS:
        proposed = provider.candidate(kind, index)
        if proposed is not None:
            return proposed
    if kind == "PERSON":
        return _pair(_FIRST, _LAST, index)
    if kind == "ORG":
        return _pair(_ORG_FIRST, _ORG_SECOND, index)
    if kind == "GPE":
        return _PLACE[index % len(_PLACE)]
    if kind == "LOC":
        return _FEATURE[index % len(_FEATURE)]
    if kind == "FAC":
        return _FACILITY[index % len(_FACILITY)]
    if kind == "EMAIL":
        person = _pair(_FIRST, _LAST, index).lower().replace(" ", ".")
        return f"{person}@example.invalid"
    if kind == "PHONE":
        # +1 555 0100..0199 is reserved for fiction in the North American plan,
        # so a surrogate telephone number cannot ring anybody.
        return f"+1 555 {100 + (index % 100):04d}"
    if kind == "URL":
        return "https://example.invalid/{}".format(
            _PLACE[index % len(_PLACE)].lower().replace(" ", "-")
        )
    return None


def surrogate_for(
    kind: str,
    ordinal: int,
    avoid: frozenset[str] | set[str] | None = None,
    source: str = "",
    forbidden: Callable[[str], bool] | None = None,
    *,
    provider: Any = None,
) -> str | None:
    """
    Return an invented stand-in for one value, or ``None`` to use a label.

    Parameters
    ----------
    kind : str
        The placeholder category, for example ``"PERSON"``.
    ordinal : int
        Which value of that kind this is, counting from one.
    avoid : set of str, optional
        Stand-ins already issued in this document. A surrogate must be unique,
        or two different values would restore to the same thing.
    source : str, default=''
        The text being redacted. A stand-in that already occurs in it is
        rejected, because restoration could not tell the two apart afterwards.
    forbidden : callable, optional
        Returns ``True`` for a candidate that must not be used — one that
        contains a value this conversation holds (``CP-071``).
    provider : SurrogateSet, optional
        Proposes name stand-ins in place of the built-in names
        (:mod:`~scikitplot.cleanprompt._surrogate_sets`). Every rule below
        still applies to what it proposes, and a proposal that is not a
        well-formed name is skipped.

    Returns
    -------
    str or None
        The stand-in, or ``None`` when this kind keeps its placeholder.

    Raises
    ------
    ValueError
        If ``ordinal`` is below one.

    Notes
    -----
    **Developer notes — the two collision rules are both load-bearing.**

    *Against the document.* If the text already says ``Marion Holt`` and a
    surrogate of that name is issued for somebody else, restoration would
    replace both occurrences with one value, corrupting text the user wrote.
    Checking against the source costs one substring search per candidate.

    *Against the other surrogates.* Two values sharing a stand-in cannot be
    told apart on the way back, which is the same defect as ``CP-001`` arriving
    from the other direction.

    *Against held values* (``CP-071``). The pools are fixed, so a stand-in can
    coincide with a real name or contain one: an email stand-in built from a
    pool name held as somebody's real first name put that name in front of the
    model, attached to the wrong person. The engine passes ``forbidden``, which
    matches every held value as a whole token however it is written.

    The search is bounded. After a hundred attempts it gives up and returns
    ``None``, which falls back to a placeholder — degraded but correct — rather
    than looping or inventing something unpredictable.

    Examples
    --------
    >>> surrogate_for("PERSON", 1)
    'Marion Holt'
    >>> surrogate_for("PERSON", 1, source="Marion Holt was here")
    'Devin Nakamura'
    >>> surrogate_for("CREDIT_CARD", 1) is None
    True
    """
    if ordinal < 1:
        raise ValueError(f"ordinal must be at least 1, got {ordinal!r}")
    if kind not in SURROGATE_KINDS:
        return None

    taken = avoid or frozenset()
    index = ordinal - 1
    for attempt in range(100):
        candidate = _candidate(kind, index + attempt, provider)
        if candidate is None:
            return None
        if (
            provider is not None
            and kind not in _CORE_FORMS
            and (entry_problem(candidate) or detected_by(candidate))
        ):
            # Floor rules 3 and 4, again at run time, on the combined stand-in:
            # a set's entries are checked when it loads, and this covers the
            # two-part names it forms and any later provider.
            continue
        if candidate in taken:
            continue
        if source and candidate in source:
            continue
        if forbidden is not None and forbidden(candidate):
            continue
        return candidate
    return None
