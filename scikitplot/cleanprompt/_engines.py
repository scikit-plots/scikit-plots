"""
Named-entity engines, and the one vocabulary they all speak.

Notes
-----
**User notes.** Three engines' worth of choice, selected with one option::

    --ner --ner-engine spacy   # the default: best quality where installed
    --ner --ner-engine nltk    # lighter, English only
    --ner --ner-engine both    # union of the two: higher recall, more noise
    --ner --ner-engine auto    # spaCy if usable, else NLTK, else nothing

``doctor`` reports which engines are usable and which one ``auto`` would pick.

**Developer notes — why a canonical vocabulary exists.**

The two engines do not agree on what to call things. Measured on the same
sentence:

.. code-block:: text

    spaCy : Mustafa Kemal Atatürk/PERSON  the Republic of Turkey/GPE
    NLTK  : Mustafa/PERSON  Kemal Atatürk/PERSON  Republic/ORGANIZATION  Turkey/GPE

spaCy says ``ORG`` where NLTK says ``ORGANIZATION``, and ``LOC`` where NLTK says
``LOCATION``. Passing those through raw would mean the placeholder depends on
*which engine happened to be installed*: the same value becomes ``[ORG-1]`` on
one machine and ``[ORGANIZATION-1]`` on another.

That is not cosmetic. A vault records the label it issued, so a vault written
with NLTK would not restore under spaCy — the placeholders in the text would no
longer match anything. Normalising to one vocabulary at the boundary makes the
engines interchangeable, which is the whole point of offering a choice.

The mapping is deliberately lossy in one direction only: several engine labels
may collapse onto one canonical label, and none is ever invented. An engine
label with no mapping becomes :data:`MISC` rather than being dropped, because
silently discarding an entity a model *did* find is the failure mode this
submodule exists to prevent.

**On ``both``.** The union raises recall and lowers precision, and that is the
trade it exists to make. Where the two engines overlap, the existing span
resolver settles it: spaCy's ``Mustafa Kemal Atatürk`` nests NLTK's two
fragments and absorbs them, so disagreement costs nothing. Where they overlap
only partially, the resolver merges, which cannot disclose.

This module imports neither engine. It describes them.

See Also
--------
scikitplot.cleanprompt._ner : The spaCy detector.
scikitplot.cleanprompt._nltk : The NLTK detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._capabilities import CapabilityStatus, probe
from ._exceptions import CapabilityError, PolicyError

if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from ._detectors import Detector

__all__ = [
    "CANONICAL_LABELS",
    "DEFAULT_ENGINE",
    "ENGINES",
    "ENGINE_MODES",
    "EngineSpec",
    "build_detectors",
    "canonical_label",
    "describe_engines",
    "resolve_engine",
    "trim_entity_span",
]

#: The canonical entity vocabulary. Every engine's labels map onto these, so a
#: placeholder means the same thing whichever engine produced it.
#:
#: Chosen to match spaCy's OntoNotes names because they are the more widely
#: recognised set, not because spaCy is privileged: NLTK's names map onto them,
#: and a third engine would map onto them too.
CANONICAL_LABELS: tuple[str, ...] = (
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "FAC",
    "NORP",
    "EVENT",
    "WORK_OF_ART",
    "PRODUCT",
    "LAW",
    "LANGUAGE",
    "MISC",
)

#: Engine label to canonical label. Anything absent becomes ``MISC``.
_LABEL_MAP: dict[str, str] = {
    # spaCy (OntoNotes) — mostly already canonical.
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "GPE",
    "LOC": "LOC",
    "FAC": "FAC",
    "NORP": "NORP",
    "EVENT": "EVENT",
    "WORK_OF_ART": "WORK_OF_ART",
    "PRODUCT": "PRODUCT",
    "LAW": "LAW",
    "LANGUAGE": "LANGUAGE",
    # spaCy multilingual (xx_ent_wiki_sm) uses a reduced set.
    "PER": "PERSON",
    "MISC": "MISC",
    # NLTK (ACE) — the names that would otherwise split the vocabulary.
    "ORGANIZATION": "ORG",
    "LOCATION": "LOC",
    "FACILITY": "FAC",
    "GSP": "GPE",  # geo-social political entity
}

#: Labels redacted by default, in canonical terms.
#:
#: ``DATE``, ``TIME``, ``CARDINAL``, ``ORDINAL``, ``MONEY``, ``PERCENT`` and
#: ``QUANTITY`` are deliberately absent, and a live run shows why: on one
#: ordinary sentence spaCy labelled ``4477`` — part of a telephone number —
#: as ``DATE``. Redacting those categories destroys the meaning a model needs
#: while hiding nothing personal, and invites exactly that class of false
#: positive.
DEFAULT_ENTITY_LABELS: frozenset[str] = frozenset(
    {
        "PERSON",
        "ORG",
        "GPE",
        "LOC",
        "FAC",
        "NORP",
        "EVENT",
        "WORK_OF_ART",
        "PRODUCT",
        "LAW",
        "LANGUAGE",
    }
)


@dataclass(frozen=True)
class EngineSpec:
    """
    Static description of one named-entity engine.

    Parameters
    ----------
    name : str
        Engine name, as used by ``--ner-engine``.
    tier : str
        Capability tier that must be available for it to run.
    summary : str
        One line describing what it is for.
    languages : tuple of str
        Language codes it can handle, or ``("*",)`` for "depends on the model".
    needs_model : bool
        Whether a separate model download is required.
    priority : int
        Detector arbitration weight. Below the structural patterns, so an
        address recognised by both is labelled ``EMAIL``, not ``ORG``.
    confidence : float
        Reported confidence for this engine's spans.
    """

    name: str
    tier: str
    summary: str
    languages: tuple[str, ...]
    needs_model: bool
    priority: int = 30
    confidence: float = 0.7

    def supports(self, language: str) -> bool:
        """Return whether this engine can handle ``language``."""
        return "*" in self.languages or language in self.languages


#: Every engine this submodule knows about.
ENGINES: dict[str, EngineSpec] = {
    "spacy": EngineSpec(
        name="spacy",
        tier="ner",
        summary="statistical models, many languages, best quality",
        languages=("*",),
        needs_model=True,
        priority=30,
        confidence=0.75,
    ),
    "nltk": EngineSpec(
        name="nltk",
        tier="nltk",
        summary="classic chunker, English only, small download",
        languages=("en",),
        needs_model=True,
        priority=25,
        confidence=0.55,
    ),
}

#: Selectable modes for ``--ner-engine``.
ENGINE_MODES: tuple[str, ...] = ("auto", "spacy", "nltk", "both", "none")

#: What ``--ner`` means when no engine is named.
#:
#: ``auto`` prefers spaCy and falls back to NLTK, which is "spaCy by default"
#: without making a machine that has only NLTK unable to detect names at all.
DEFAULT_ENGINE = "auto"


#: Bracket pairs a value may not carry unmatched.
_BRACKETS = (("(", ")"), ("[", "]"), ("{", "}"))


def trim_entity_span(text: str, start: int, end: int) -> tuple[int, int]:
    """
    Shrink an entity span until it carries no unmatched bracket.

    Parameters
    ----------
    text : str
        The original text.
    start, end : int
        The span an engine reported, as offsets into ``text``.

    Returns
    -------
    tuple of int
        The adjusted ``(start, end)``. Equal offsets mean the span held nothing
        once the brackets were removed and should be dropped.

    Notes
    -----
    **Developer notes — why entity spans need validating at all.**

    Every structural pattern carries a validator: a card number passes Luhn, an
    IBAN passes mod-97, a dotted quad has octets in range. Entity spans came
    from a model and were taken verbatim, with nothing checked. That asymmetry
    produced a measured defect.

    On ``Mustafa Kemal Atatürk[e] (c. 1881)`` spaCy returns the person as
    ``'Mustafa Kemal Atatürk[e'`` — it swallows the opening bracket and the
    footnote letter and leaves the closing one behind. Two things go wrong at
    once. The vault records the person's name *with* ``[e`` on the end, which
    is simply the wrong value and would be restored into any later text. And
    the prompt goes out carrying ``[PERSON-1]]``, a malformed placeholder,
    which a language model is then more likely to rewrite or drop — the very
    failure that :func:`~scikitplot.cleanprompt.restore` has to be lenient
    about further down the pipeline.

    The rule is structural, not a guess about meaning: *a value may not contain
    an unmatched bracket*. A span is truncated at its first unmatched opener
    and started after its last unmatched closer. Balanced brackets inside a
    span — ``Acme (Europe) Ltd`` — are left alone, because there is nothing
    malformed about them.

    Only brackets. Trailing full stops are not trimmed, because ``Inc.`` and
    ``St.`` end in one legitimately and deciding which is which would be the
    kind of guess this submodule does not make.

    Examples
    --------
    >>> text = "Mustafa Kemal Atatürk[e] founded it"
    >>> start, end = trim_entity_span(text, 0, 23)
    >>> text[start:end]
    'Mustafa Kemal Atatürk'
    >>> text = "Acme (Europe) Ltd is here"
    >>> start, end = trim_entity_span(text, 0, 17)
    >>> text[start:end]
    'Acme (Europe) Ltd'
    """
    while start < end:
        cut = _first_unmatched_open(text, start, end)
        if cut is None:
            break
        end = cut
    while start < end:
        cut = _last_unmatched_close(text, start, end)
        if cut is None:
            break
        start = cut + 1

    # Whitespace exposed by the trimming is not part of a value either.
    while start < end and text[end - 1].isspace():
        end -= 1
    while start < end and text[start].isspace():
        start += 1
    return start, end


def _first_unmatched_open(text: str, start: int, end: int) -> int | None:
    """Return the index of the first opener with no closer inside the span."""
    for opener, closer in _BRACKETS:
        depth = 0
        first = None
        for index in range(start, end):
            char = text[index]
            if char == opener:
                if depth == 0:
                    first = index
                depth += 1
            elif char == closer and depth:
                depth -= 1
        if depth and first is not None:
            return first
    return None


def _last_unmatched_close(text: str, start: int, end: int) -> int | None:
    """Return the index of the last closer with no opener inside the span."""
    for opener, closer in _BRACKETS:
        depth = 0
        last = None
        for index in range(end - 1, start - 1, -1):
            char = text[index]
            if char == closer:
                if depth == 0:
                    last = index
                depth += 1
            elif char == opener and depth:
                depth -= 1
        if depth and last is not None:
            return last
    return None


def canonical_label(label: str) -> str:
    """
    Map an engine's entity label onto the canonical vocabulary.

    Parameters
    ----------
    label : str
        The label an engine reported, for example ``"ORGANIZATION"``.

    Returns
    -------
    str
        The canonical label, or ``"MISC"`` when the engine's label has no
        mapping.

    Notes
    -----
    **Developer notes.** Unknown labels become ``MISC`` rather than being
    dropped. A model that found something and a mapping that has not caught up
    are two different problems, and discarding the entity would turn a
    vocabulary gap into a privacy hole.

    Examples
    --------
    >>> canonical_label("ORGANIZATION")
    'ORG'
    >>> canonical_label("ORG")
    'ORG'
    >>> canonical_label("SOMETHING_NEW")
    'MISC'
    """
    return _LABEL_MAP.get(label.upper(), "MISC")


def resolve_engine(mode: str = DEFAULT_ENGINE, language: str = "en") -> tuple[str, ...]:
    """
    Resolve an engine mode into the engines that will actually run.

    Parameters
    ----------
    mode : str, default='auto'
        One of :data:`ENGINE_MODES`.
    language : str, default='en'
        Language code, used to rule out engines that cannot handle it.

    Returns
    -------
    tuple of str
        Engine names to run, possibly empty.

    Raises
    ------
    PolicyError
        If ``mode`` is not a known mode.

    Notes
    -----
    **Developer notes.** ``auto`` resolves by *usability*, not by presence:
    an engine whose tier reports ``BROKEN`` is skipped exactly as one that
    reports ``ABSENT``, because an installed-but-failing engine detects nothing
    either way.

    An explicitly named engine is **not** silently dropped when unusable — the
    caller asked for it by name, and quietly running without it would be the
    silent degradation this submodule exists to prevent. It is returned, and the
    detector raises with an install hint at the point of use.

    Examples
    --------
    >>> resolve_engine("none")
    ()
    >>> resolve_engine("nltk", "en")
    ('nltk',)
    >>> resolve_engine("nltk", "de")
    ('nltk',)
    """
    if mode not in ENGINE_MODES:
        msg = "unknown NER engine mode {!r}; choose from {}".format(
            mode, ", ".join(ENGINE_MODES)
        )
        raise PolicyError(msg)
    if mode == "none":
        return ()
    if mode == "both":
        return ("spacy", "nltk")
    if mode in ENGINES:
        return (mode,)

    # auto: prefer spaCy, fall back to NLTK, in both cases only if usable here.
    for name in ("spacy", "nltk"):
        spec = ENGINES[name]
        if spec.supports(language) and probe(spec.tier).available:
            return (name,)
    return ()


def describe_engines(
    language: str = "en", mode: str = DEFAULT_ENGINE
) -> dict[str, Any]:
    """
    Report every engine's usability, and what ``mode`` would select.

    Parameters
    ----------
    language : str, default='en'
        Language code to report against.
    mode : str, default='auto'
        The mode being asked about.

    Returns
    -------
    dict
        JSON-safe report with one entry per engine plus a ``selected`` key.

    Notes
    -----
    **User notes.** The ``language_supported`` field is the one to read when an
    engine looks installed but finds nothing: NLTK's chunker is English-only, so
    asking it for German names is a configuration error rather than a quiet
    zero result.
    """
    engines: dict[str, Any] = {}
    for name, spec in ENGINES.items():
        report = probe(spec.tier)
        supported = spec.supports(language)
        if not supported:
            status = CapabilityStatus.MISCONFIGURED.value
            detail = "{} does not support language {!r}; it handles {}".format(
                name, language, ", ".join(spec.languages)
            )
        else:
            status = report.status.value
            detail = report.detail
        engines[name] = {
            "status": status,
            "usable": supported and report.available,
            "language_supported": supported,
            "languages": list(spec.languages),
            "needs_model": spec.needs_model,
            "summary": spec.summary,
            "install_hint": report.install_hint,
            "version": report.version,
            "detail": detail,
        }

    selected = resolve_engine(mode, language)
    return {
        "mode": mode,
        "language": language,
        "selected": list(selected),
        "engines": engines,
        "canonical_labels": list(CANONICAL_LABELS),
    }


def _no_engine_error(mode: str, language: str) -> CapabilityError:
    """
    Build the failure for "entity detection was asked for and none is usable".

    Parameters
    ----------
    mode : str
        The engine mode that resolved to nothing.
    language : str
        The language that was asked for.

    Returns
    -------
    CapabilityError
        Carrying every engine's status and the command that would fix the
        nearest one.

    Notes
    -----
    **Developer notes.** The message names *each* engine and its own reason,
    because the two fail for different reasons and a combined "install
    something" would send a user with NLTK installed off to install spaCy. The
    ``install_hint`` carries the first engine that a language supports, since
    that is the one a caller can act on; the full picture is in the message.
    """
    report = describe_engines(language=language, mode=mode)
    lines = []
    hint = ""
    for name, engine in report["engines"].items():
        lines.append("  {}: {} — {}".format(name, engine["status"], engine["detail"]))
        if engine["language_supported"] and not hint:
            hint = engine["install_hint"]
    return CapabilityError(
        "entity detection was requested (--ner) with engine mode {!r}, but no "
        "engine is usable for language {!r}:\n{}\n"
        "Install one, or pass --ner-engine none to proceed without entity "
        "detection. Structural detectors (email, card, IBAN, …) are "
        "unaffected.".format(mode, language, "\n".join(lines)),
        tier="ner",
        status=CapabilityStatus.ABSENT.value,
        install_hint=hint or 'pip install "spacy>=3.4,<4"',
    )


def build_detectors(  # ruff: ignore[too-many-positional-arguments]
    mode: str = DEFAULT_ENGINE,
    language: str = "en",
    model: str | None = None,
    labels: frozenset[str] | None = DEFAULT_ENTITY_LABELS,
    size: str = "sm",
    required: bool = False,
) -> list[Detector]:
    """
    Construct the detectors for an engine mode.

    Parameters
    ----------
    mode : str, default='auto'
        One of :data:`ENGINE_MODES`.
    language : str, default='en'
        Language code.
    model : str, optional
        Explicit spaCy model name, overriding language-based resolution.
    labels : frozenset of str, optional
        Canonical labels to redact. ``None`` redacts every label reported.
    size : str, default='sm'
        Preferred spaCy model size.
    required : bool, default=False
        Whether the caller explicitly asked for entity detection. When true, a
        mode that resolves to no engine raises instead of returning an empty
        list. See the notes.

    Returns
    -------
    list of Detector
        Constructed detectors, in engine priority order. Construction imports
        nothing: each detector loads its engine on first use.

    Raises
    ------
    PolicyError
        If ``mode`` is unknown.
    CapabilityError
        If ``required`` is true and no engine is usable. ``mode="none"`` is
        exempt: it is an explicit instruction to run no engine.

    Notes
    -----
    **Developer notes.** Nothing here imports spaCy or NLTK. A detector is a
    description until its first ``detect`` call, which is what keeps
    ``--help`` and ``doctor`` free of a multi-second model load.

    **Why ``required`` exists.** ``auto`` degrades by design: it is the default,
    and a default that refused to run on a base installation would make the
    base tier unusable. But ``--ner`` is not a default, it is a request. With
    ``auto`` resolving to nothing, honouring that request by adding no detector
    and exiting successfully tells a user their text was scanned for names when
    nothing looked — the silent degradation that prompted this rewrite, arrived
    at from the opposite direction. ``required`` separates the two: the same
    resolution, but an explicit ask that cannot be met fails loudly, with every
    engine's status and install command in the message.

    This lives here rather than in each caller because there are two callers
    (the CLI and :func:`~scikitplot.cleanprompt.encode`) and a third would make
    the same omission.
    """
    selected = resolve_engine(mode, language)
    if required and not selected and mode != "none":
        raise _no_engine_error(mode, language)

    detectors: list[Detector] = []
    for name in selected:
        if name == "spacy":
            from ._ner import spacy_detector  # ruff: ignore[import-outside-top-level]

            detectors.append(
                spacy_detector(model=model, language=language, labels=labels, size=size)
            )
        elif name == "nltk":
            from ._nltk import nltk_detector  # ruff: ignore[import-outside-top-level]

            detectors.append(nltk_detector(language=language, labels=labels))
    return detectors
