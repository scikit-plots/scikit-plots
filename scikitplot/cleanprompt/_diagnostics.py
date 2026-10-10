"""
What this installation can find — and what it is blind to.

One source of truth, consumed by ``doctor``, the interactive session and the web
interface, so all three tell the same story.

Notes
-----
**User notes.** :func:`diagnose` answers "what will actually be detected if I
paste something right now?". :func:`suggest_terms` answers the complementary
question: "what did you *nearly* catch that I might want hidden?".

**Developer notes — why this module exists.**

A user pasted an encyclopaedia paragraph about a public figure into the web
interface and got it back unchanged. The engine was right: that text contains no
email, phone, URL, card or IBAN, and every sensitive thing in it is a *named
entity*, which only the ``ner`` tier detects. spaCy was not installed, so
nothing was found.

The failure was not the result. The failure was that nothing in the interface
said so. For a redaction tool this is the most dangerous state there is: the
user believes the text is anonymised and sends it. A redaction that silently
does less than it claims is worse than one that raises, because the caller acts
on it.

So "nothing was found" is never reported on its own. It is always reported
together with *what was looking*, *what was not*, and *what to do about it*.

Two complementary answers, and the distinction is deliberate:

- :func:`diagnose` is about the **configuration** — which detectors are live,
  which are installable, which categories nothing currently covers.
- :func:`suggest_terms` is about **this text** — high-recall candidates the
  active detectors did not claim, offered for a human to choose from. It never
  redacts anything on its own. Guessing that a capitalised word is a name is a
  heuristic and has no place in the pipeline; *showing* the user the candidates
  and letting them decide is not a guess, it is inspection.

See Also
--------
scikitplot.cleanprompt._capabilities : Tier probing that feeds this.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._capabilities import TIERS, CapabilityStatus, probe
from ._patterns import PATTERNS
from ._policy import DEFAULT_POLICY, RedactionPolicy

if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from ._detectors import DetectorRegistry
    from ._types import RedactionResult

__all__ = [
    "BlindSpot",
    "Diagnosis",
    "Suggestion",
    "describe_outcome",
    "diagnose",
    "suggest_terms",
]


@dataclass(frozen=True)
class BlindSpot:
    """
    A category of sensitive information nothing currently detects.

    Parameters
    ----------
    category : str
        What is not being found, in the user's language.
    reason : str
        Why, in one sentence.
    remedy : str
        The exact action that would fix it.
    examples : tuple of str
        Concrete things that would slip through, so the gap is legible without
        knowing the pattern names.
    severity : str
        ``"high"`` when the gap covers common personal data, ``"medium"``
        otherwise.
    """

    category: str
    reason: str
    remedy: str
    examples: tuple[str, ...] = ()
    severity: str = "medium"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return {
            "category": self.category,
            "reason": self.reason,
            "remedy": self.remedy,
            "examples": list(self.examples),
            "severity": self.severity,
        }


@dataclass(frozen=True)
class Diagnosis:
    """
    What a given configuration can and cannot detect.

    Parameters
    ----------
    active_kinds : tuple of str
        Detection kinds that will run.
    inactive_kinds : tuple of str
        Kinds available in the library but not enabled by this configuration.
    tiers : dict
        Per-tier capability report, keyed by tier name.
    blind_spots : tuple of BlindSpot
        Categories nothing currently covers.
    ner_active : bool
        Whether named-entity detection will run.

    Notes
    -----
    **User notes.** ``blind_spots`` is the field to read. An empty tuple means
    every category this tool knows about is covered by something.
    """

    active_kinds: tuple[str, ...]
    inactive_kinds: tuple[str, ...]
    tiers: dict[str, Any]
    blind_spots: tuple[BlindSpot, ...]
    ner_active: bool

    @property
    def healthy(self) -> bool:
        """bool: ``True`` when no high-severity blind spot is present."""
        return not any(spot.severity == "high" for spot in self.blind_spots)

    def headline(self) -> str:
        """
        Return one sentence describing the detection posture.

        Returns
        -------
        str
            Suitable for a banner in a terminal or a web page.
        """
        if not self.active_kinds:
            return "No detectors are active: nothing will be redacted."
        count = len(self.active_kinds)
        if self.ner_active:
            return (
                f"{count} structural detectors plus named-entity detection are active."
            )
        return (
            f"{count} structural detectors are active. Names, organisations and "
            "places are NOT being detected."
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return {
            "headline": self.headline(),
            "healthy": self.healthy,
            "ner_active": self.ner_active,
            "active_kinds": list(self.active_kinds),
            "inactive_kinds": list(self.inactive_kinds),
            "tiers": self.tiers,
            "blind_spots": [spot.as_dict() for spot in self.blind_spots],
        }


@dataclass(frozen=True)
class Suggestion:
    """
    A candidate the active detectors did not claim.

    Parameters
    ----------
    text : str
        The candidate surface, exactly as it appears.
    count : int
        How many times it occurs.
    reason : str
        Why it is being offered.
    first_offset : int
        Where it first appears, so a caller can highlight it.

    Notes
    -----
    **Developer notes.** A suggestion is never redacted automatically. It is
    offered for a person to accept, and accepting it adds the term to the
    literal detector, which is exact. That keeps the guess on the human side of
    the boundary, where it belongs.
    """

    text: str
    count: int
    reason: str
    first_offset: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return {
            "text": self.text,
            "count": self.count,
            "reason": self.reason,
            "first_offset": self.first_offset,
        }


def diagnose(
    policy: RedactionPolicy | None = None,
    registry: DetectorRegistry | None = None,
) -> Diagnosis:
    """
    Report what a configuration can and cannot detect.

    Parameters
    ----------
    policy : RedactionPolicy, optional
        The policy to describe. Defaults to
        :data:`~scikitplot.cleanprompt._policy.DEFAULT_POLICY`.
    registry : DetectorRegistry, optional
        The detectors that will run. When omitted, the default registry for
        ``policy`` is assumed, which means structural patterns only.

    Returns
    -------
    Diagnosis
        The report. Never raises and never imports an optional dependency.

    Examples
    --------
    >>> report = diagnose()
    >>> "EMAIL" in report.active_kinds
    True
    >>> isinstance(report.headline(), str)
    True
    """
    active_policy = policy if policy is not None else DEFAULT_POLICY

    if registry is not None:
        active = tuple(sorted(registry.kinds()))
        # Entity detectors declare themselves with kind "NE". Matching on the
        # *kind* rather than on a name prefix is what keeps this true when a
        # second engine arrives: the spaCy detector is named "ner:<model>" and
        # the NLTK one is named "nltk", so a prefix test reported a registry
        # running NLTK as having no entity detection at all — a blind spot
        # claimed where none existed, which is as misleading as missing one.
        ner_active = any(detector.kind == "NE" for detector in registry)
    else:
        from ._patterns import (  # ruff: ignore[import-outside-top-level]
            default_patterns,
        )

        selected = active_policy.kinds
        if selected is None:
            active = tuple(sorted(spec.kind for spec in default_patterns()))
        else:
            active = tuple(sorted(selected))
        ner_active = False

    inactive = tuple(sorted(set(PATTERNS) - set(active)))
    tiers = {name: _tier_report(name) for name in TIERS}

    spots: list[BlindSpot] = []
    if not ner_active:
        remedy = _entity_remedy()
        spots.append(
            BlindSpot(
                category="Names, organisations, places and other named entities",
                reason=(
                    "named-entity detection is not running, and no regular "
                    "expression can recognise a person's name"
                ),
                remedy=remedy,
                examples=(
                    "a person's full name",
                    "a company or institution",
                    "a city or country",
                ),
                severity="high",
            )
        )
        if "TITLE_CASE" in PATTERNS and "TITLE_CASE" not in active:
            spots.append(
                BlindSpot(
                    category="Capitalised multi-word names (partial cover)",
                    reason=(
                        "the TITLE_CASE pattern is available but not enabled; "
                        "it finds runs of capitalised words without needing "
                        "spaCy, at the cost of false positives"
                    ),
                    remedy=(
                        "add TITLE_CASE to the detection kinds "
                        "(--kinds ... TITLE_CASE), or use --suggest to review "
                        "candidates and hide them explicitly"
                    ),
                    examples=("Mustafa Kemal", "Acme Corporation"),
                    severity="medium",
                )
            )

    missing = set(PATTERNS) - set(active)
    if missing:
        spots.append(
            BlindSpot(
                category="Disabled structural patterns",
                reason="{} built-in pattern(s) are not enabled: {}".format(
                    len(missing), ", ".join(sorted(missing))
                ),
                remedy="drop --kinds to enable every built-in pattern",
                examples=tuple(sorted(missing))[:4],
                severity="medium",
            )
        )

    return Diagnosis(
        active_kinds=active,
        inactive_kinds=inactive,
        tiers=tiers,
        blind_spots=tuple(spots),
        ner_active=ner_active,
    )


def _entity_remedy() -> str:
    """
    Return the shortest route to working entity detection, here, now.

    Returns
    -------
    str
        A command or instruction the reader can act on without further lookup.

    Notes
    -----
    **Developer notes.** The remedy is computed from what is actually installed,
    not written as a constant, for three reasons that each bit us once.

    *The cheapest fix first.* If either engine is already usable, the fix is a
    flag, not a download. Telling someone with spaCy installed to install spaCy
    is the kind of advice that teaches people to ignore advice.

    *The model name must be the one this submodule will ask for.* It was
    hard-coded as ``en_core_web_lg`` while the default had moved to
    ``en_core_web_sm``, so following the instruction produced a 560 MB download
    that still left the tier misconfigured. It now comes from the same resolver
    the detector uses, so the two cannot drift again.

    *Both engines are offered.* NLTK is a tenth of the size and needs no
    compiler, which on a constrained machine is the difference between some
    name detection and none.

    *Installed is not ready* (``CP-093``). "spaCy is installed; enable it" was
    the advice on a machine with spaCy and no model, and following it failed on
    the first sentence. spaCy's readiness is read from metadata, so it is
    checked here without importing anything; NLTK's data can only be checked
    by importing NLTK, which this function must not do, so the NLTK advice says
    that ``doctor`` checks the data.
    """
    from ._engines import engine_readiness  # ruff: ignore[import-outside-top-level]
    from ._languages import (  # ruff: ignore[import-outside-top-level]
        DEFAULT_LANGUAGE,
        resolve_model,
    )

    spacy_report = probe("ner")
    nltk_report = probe("nltk")

    spacy_fix = ""
    if spacy_report.status is CapabilityStatus.AVAILABLE:
        spacy = engine_readiness("spacy", DEFAULT_LANGUAGE)
        if spacy.ready:
            return (
                "spaCy is installed with its model; enable it for this run (--ner "
                "on the command line, or add spacy_detector() to the registry)"
            )
        spacy_fix = (
            "spaCy is installed but has no model; download one: "
            f"{spacy.remedy}, then run with --ner"
        )
        if nltk_report.status is not CapabilityStatus.AVAILABLE:
            return spacy_fix
    if nltk_report.status is CapabilityStatus.AVAILABLE:
        advice = (
            "NLTK is installed; enable it for this run "
            "(--ner --ner-engine nltk, or add nltk_detector() to the registry) — "
            "it needs its data packages, and `doctor --ner --ner-engine nltk` "
            "says which are missing"
        )
        return f"{advice}; or: {spacy_fix}" if spacy_fix else advice

    model, _note = resolve_model(DEFAULT_LANGUAGE, "sm")
    return (
        f"{spacy_report.install_hint}, then: python -m spacy download {model} — or, for a smaller install, "
        f"{nltk_report.install_hint} and then --ner --ner-engine nltk"
    )


def _tier_report(name: str) -> dict[str, Any]:
    """Return a JSON-safe report for one optional tier."""
    report = probe(name)
    return {
        "status": report.status.value,
        "available": report.available,
        "distribution": report.distribution,
        "version": report.version,
        "supported": report.supported,
        "detail": report.detail,
        "install_hint": report.install_hint,
        "purpose": TIERS[name].purpose,
    }


#: A single word token. Suggestions are built by grouping these, not by matching
#: whole runs with one expression.
#:
#: Matching runs directly was the first attempt and it was wrong: an expression
#: anchored at a run's start only ever sees the *leading* stretch, so
#: ``"was a Turkish field marshal"`` yielded nothing at all, because the run
#: begins with a lower-case word. Every capitalised token after a lower-case one
#: was invisible. Grouping tokens finds maximal capitalised runs wherever they
#: sit in the sentence.
#:
#: Capitalisation is tested with :meth:`str.isupper` rather than an ``[A-Z]``
#: class, because the names this exists to surface are frequently not ASCII.
_TOKEN = re.compile(
    r"[^\W\d_][^\W\d_'’.\-]*",  # ruff: ignore[ambiguous-unicode-character-string]
)

#: Words that start a sentence so often that offering them is noise.
_SENTENCE_OPENERS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "but",
        "by",
        "for",
        "from",
        "he",
        "her",
        "his",
        "i",
        "if",
        "in",
        "it",
        "its",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "then",
        "there",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "when",
        "which",
        "who",
        "with",
        "you",
        "your",
    }
)


def suggest_terms(
    text: str,
    result: RedactionResult | None = None,
    min_tokens: int = 1,
    max_suggestions: int = 40,
) -> tuple[Suggestion, ...]:
    """
    Offer candidate terms the active detectors did not claim.

    Parameters
    ----------
    text : str
        The original text.
    result : RedactionResult, optional
        A completed redaction. Anything already redacted is excluded, so the
        list only ever contains things still in the clear.
    min_tokens : int, default=1
        Least number of capitalised words a candidate must have. ``2`` narrows
        to multi-word names such as ``"Mustafa Kemal"``; ``1`` also offers
        single words such as ``"Turkey"``.
    max_suggestions : int, default=40
        Most candidates returned, ordered by frequency then first appearance.

    Returns
    -------
    tuple of Suggestion
        Candidates for a human to accept or ignore. **Nothing here is
        redacted.**

    Raises
    ------
    TypeError
        If ``text`` is not a string.

    Notes
    -----
    **User notes.** Accepting a suggestion adds it as an exact term, so the
    redaction stays precise. This is how to handle names when the ``ner`` tier
    is not installed: look at the list, tick what matters, and those terms are
    then matched literally.

    **Developer notes.** Deliberately high-recall and deliberately advisory.
    Recall matters because a missed candidate is never shown to the user at all;
    precision matters less because every candidate is reviewed by a person
    before anything happens to it. Sentence-opening function words are dropped
    because offering "The" and "This" trains people to stop reading the list.

    Examples
    --------
    >>> [s.text for s in suggest_terms("Ada Lovelace met Charles Babbage.")]
    ['Ada Lovelace', 'Charles Babbage']
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__!r}")
    if min_tokens < 1:
        raise ValueError(f"min_tokens must be >= 1, got {min_tokens!r}")

    taken: list[tuple[int, int]] = []
    if result is not None:
        for entry in result.entries:
            taken.extend(entry.occurrences)

    def is_taken(start: int, end: int) -> bool:
        return any(start < t_end and t_start < end for t_start, t_end in taken)

    counts: dict[str, int] = {}
    first: dict[str, int] = {}

    # Group consecutive capitalised tokens into maximal runs. A run is broken
    # by a lower-case token or by anything other than spaces and tabs between
    # two tokens, so "Republic of Turkey" yields "Republic" and "Turkey"
    # separately rather than one run spanning the lower-case "of".
    run: list[str] = []
    run_start = 0
    run_end = 0

    def flush() -> None:
        """Record the current run, if it qualifies."""
        if len(run) < min_tokens:
            return
        candidate = " ".join(run)
        if len(run) == 1 and candidate.lower() in _SENTENCE_OPENERS:
            return
        if is_taken(run_start, run_end):
            return
        counts[candidate] = counts.get(candidate, 0) + 1
        first.setdefault(candidate, run_start)

    previous_end = -1
    for match in _TOKEN.finditer(text):
        token = match.group()
        start, end = match.span()
        gap = text[previous_end:start] if previous_end >= 0 else ""
        contiguous = bool(run) and gap != "" and gap.strip(" \t") == ""
        if token[:1].isupper():
            if contiguous:
                run.append(token)
                run_end = end
            else:
                flush()
                run = [token]
                run_start, run_end = start, end
        else:
            flush()
            run = []
        previous_end = end
    flush()

    ordered = sorted(counts, key=lambda term: (-counts[term], first[term], term))
    return tuple(
        Suggestion(
            text=term,
            count=counts[term],
            reason="capitalised word run, not matched by any active detector",
            first_offset=first[term],
        )
        for term in ordered[:max_suggestions]
    )


def _join(items: list[str]) -> str:
    """Join a list into readable prose: "a", "a and b", "a, b and c"."""
    items = list(items)
    if len(items) <= 1:
        return items[0] if items else ""
    return "{} and {}".format(", ".join(items[:-1]), items[-1])


def describe_outcome(
    result: RedactionResult,
    diagnosis: Diagnosis,
    suggestions: tuple[Suggestion, ...] = (),
) -> dict[str, Any]:
    """
    Explain a redaction outcome, especially an empty one.

    Parameters
    ----------
    result : RedactionResult
        The completed pass.
    diagnosis : Diagnosis
        What was looking.
    suggestions : tuple of Suggestion, default=()
        Candidates still in the clear.

    Returns
    -------
    dict
        A JSON-safe explanation with ``level``, ``headline``, ``detail`` and
        ``actions``. ``level`` is ``"ok"``, ``"warning"`` or ``"alert"``.

    Notes
    -----
    **Developer notes.** ``level`` is ``"alert"`` for the case that motivated
    this module: nothing was found *and* something was not looking. That
    combination must never render as a neutral "no sensitive values detected",
    because the user reads that as "this text is safe to send".
    """
    actions: list[str] = []
    high = [spot for spot in diagnosis.blind_spots if spot.severity == "high"]

    if result.stats.entries == 0:
        if high:
            level = "alert"
            headline = (
                "Nothing was redacted — and part of the detection surface is "
                "switched off."
            )
            detail = (
                "No structural values (emails, phone numbers, URLs, card or "
                "account numbers) were present. {} That means this text has "
                "NOT been checked for them.".format(
                    " ".join(
                        f"{spot.category} is not covered: {spot.reason}."
                        for spot in high
                    )
                )
            )
            actions.extend(spot.remedy for spot in high)
        else:
            level = "ok"
            headline = "Nothing was redacted."
            detail = (
                "Every detector was active and none matched, so this text "
                "contains no value the configured detectors recognise."
            )
    else:
        level = "warning" if high else "ok"
        headline = result.summary()
        detail = "Placeholders are stable; restore the reply with the same vault."
        if high:
            detail += f" Note that {_join([spot.category.lower() for spot in high])} are still not being detected."
            actions.extend(spot.remedy for spot in high)

    if suggestions:
        actions.append(
            f"Review {len(suggestions)} suggested term(s) and hide any that matter."
        )

    return {
        "level": level,
        "headline": headline,
        "detail": detail,
        "actions": actions,
        "entries": result.stats.entries,
        "suggestions": [item.as_dict() for item in suggestions],
    }
