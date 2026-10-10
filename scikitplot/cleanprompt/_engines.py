"""
Named-entity engines, and the one vocabulary they all speak.

Notes
-----
**User notes.** Three engines' worth of choice, selected with one option::

    --ner --ner-engine spacy   # the default: best quality where installed
    --ner --ner-engine nltk    # lighter, English only
    --ner --ner-engine both    # union of the two: higher recall, more noise
    --ner --ner-engine auto    # spaCy if ready, else NLTK, else nothing

``doctor`` reports which engines are ready and which one ``auto`` would pick.

**User notes — installed is not ready.** An engine is *ready* when three
things hold: its package is installed, it handles the language asked for, and
its data is present — a spaCy model, or NLTK's data packages. Having the
package without the data is the commonest way entity detection fails, so
``doctor`` reports the three separately and names the one missing step::

    entity_engines.engines.spacy.installed     true
    entity_engines.engines.spacy.assets_ready  false
    entity_engines.engines.spacy.remedy        python -m spacy download en_core_web_sm

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

import importlib.util
import os
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
    "EngineReadiness",
    "EngineSpec",
    "build_detectors",
    "canonical_label",
    "describe_engines",
    "engine_readiness",
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


@dataclass(frozen=True)
class EngineReadiness:
    """
    Whether one engine can run here, now, and if not, the one step that fixes it.

    Parameters
    ----------
    name : str
        Engine name, as used by ``--ner-engine``.
    installed : bool
        Whether the engine's capability tier reports ``AVAILABLE``: the
        package is installed, inside the declared range, and importable
        according to its metadata.
    language_supported : bool
        Whether the engine handles the language asked for.
    assets_checked : bool
        Whether the engine's data was looked at. ``False`` only for NLTK when
        the caller chose not to import it (see the notes).
    assets_ready : bool or None
        Whether the model or data packages are present; ``None`` when they
        were not checked.
    ready : bool
        ``installed and language_supported and assets_ready is not False`` —
        nothing known stands in the way of running.
    status : str
        A :class:`~scikitplot.cleanprompt.CapabilityStatus` value describing
        the first obstacle, or ``"AVAILABLE"``.
    reason : str
        One sentence saying what was found.
    remedy : str
        The command or flag that removes the first obstacle; empty when ready.
    model : str or None
        For spaCy, the model this configuration loads; otherwise ``None``.
    missing : tuple of str
        For NLTK, the data packages that were not found.
    version : str or None
        The installed distribution version, when known.

    Notes
    -----
    **User notes.** Read ``ready`` to decide, and ``remedy`` to act. ``reason``
    is for a person: it says which of the three conditions failed.

    **Developer notes — why NLTK's data may be unchecked.** spaCy models are
    ordinary installed distributions, so their presence is read from metadata
    without importing anything. NLTK's data packages live in directories that
    only ``nltk.data.find`` can resolve — the search path depends on
    environment variables, the interpreter prefix and the platform, and
    copying that logic here would be a guess about NLTK's internals. So the
    check imports NLTK, and it runs only where the caller is about to load
    NLTK anyway: an explicit request, ``doctor``, or the web app with entity
    detection switched on. Everywhere else the field says ``None`` rather than
    pretending to know.
    """

    name: str
    installed: bool
    language_supported: bool
    assets_checked: bool
    assets_ready: bool | None
    ready: bool
    status: str
    reason: str
    remedy: str
    model: str | None = None
    missing: tuple[str, ...] = ()
    version: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return {
            "installed": self.installed,
            "language_supported": self.language_supported,
            "assets_checked": self.assets_checked,
            "assets_ready": self.assets_ready,
            "ready": self.ready,
            "status": self.status,
            "reason": self.reason,
            "remedy": self.remedy,
            "model": self.model,
            "missing": list(self.missing),
            "version": self.version,
        }


def _spacy_model_ready(model: str) -> bool:
    """
    Return whether the spaCy model ``model`` can be found, without importing it.

    Parameters
    ----------
    model : str
        A model package name (``en_core_web_sm``) or a directory path, the two
        forms :func:`spacy.load` accepts.

    Returns
    -------
    bool
        ``True`` when the model is an installed distribution, an importable
        top-level package, or an existing directory.

    Notes
    -----
    **Developer notes.** Three lookups, all free of side effects:
    distribution metadata (how ``python -m spacy download`` installs a model),
    :func:`importlib.util.find_spec` for a package installed some other way
    (it locates a top-level module without executing it), and the filesystem
    for a model saved with ``nlp.to_disk``. A name that fails all three is
    absent, which is what :func:`spacy.load` would conclude with an
    ``OSError`` a moment later.
    """
    from ._languages import installed_models  # ruff: ignore[import-outside-top-level]

    if model in installed_models():
        return True
    if os.sep in model or (os.altsep and os.altsep in model) or model.startswith("."):
        return os.path.isdir(model)
    if not model.isidentifier():
        return False
    try:
        return importlib.util.find_spec(model) is not None
    except (ImportError, ValueError):
        return False


def _nltk_missing_corpora() -> tuple[str, ...]:
    """
    Return NLTK's missing data packages, importing NLTK to find out.

    Returns
    -------
    tuple of str
        Package names that ``nltk.download`` would fetch; empty when complete.

    Raises
    ------
    ImportError
        If NLTK's metadata says installed but the import fails.
    """
    import nltk  # noqa: PLC0415 - only on an explicit, about-to-load path

    from ._nltk import missing_data  # ruff: ignore[import-outside-top-level]

    return tuple(missing_data(nltk))


def _corpora_command(missing: tuple[str, ...]) -> str:
    """Return the one command that downloads ``missing`` (see ``_nltk``)."""
    from ._nltk import download_command  # ruff: ignore[import-outside-top-level]

    return download_command(missing)


def engine_readiness(  # ruff: ignore[too-many-return-statements]
    name: str,
    language: str = "en",
    model: str | None = None,
    size: str = "sm",
    check_assets: bool = False,
) -> EngineReadiness:
    """
    Decide whether one engine can run for a language, and how to fix it if not.

    Parameters
    ----------
    name : str
        ``"spacy"`` or ``"nltk"``.
    language : str, default='en'
        Language code.
    model : str, optional
        Explicit spaCy model, overriding ``language`` and ``size``.
    size : str, default='sm'
        Preferred spaCy model size.
    check_assets : bool, default=False
        Also check NLTK's data packages, which imports NLTK. spaCy's model is
        always checked, because that check imports nothing.

    Returns
    -------
    EngineReadiness
        The decision, its reason and its remedy.

    Raises
    ------
    PolicyError
        If ``name`` is not a known engine, or ``size`` is not a known size.

    Notes
    -----
    **Developer notes — the order of the checks is the order of the remedies.**
    Language first, because no installation fixes an engine that cannot read
    the language. Then the package, because a model cannot be downloaded into
    a library that is not there. Then the data. The first failure decides the
    status and the remedy, so a user is told the next step rather than all of
    them at once.

    This is ``CP-093``. ``doctor --ner`` reported entity detection as active
    on a machine with spaCy installed and no model, and the next ``inspect
    --ner`` failed on the missing model; the same for NLTK without its data.
    Both decisions read the package and never the data. A diagnosis that
    disagrees with the run it describes is worse than none, because the run is
    where the user finds out — after deciding the text was safe to send.

    Examples
    --------
    >>> engine_readiness("nltk", "de").reason
    "nltk does not support language 'de'; it handles en"
    """
    spec = ENGINES.get(name)
    if spec is None:
        msg = "unknown NER engine {!r}; choose from {}".format(name, ", ".join(ENGINES))
        raise PolicyError(msg)

    resolved_model = None
    if name == "spacy":
        from ._languages import resolve_model  # ruff: ignore[import-outside-top-level]

        resolved_model, _note = resolve_model(language, size, explicit=model)

    report = probe(spec.tier)
    installed = report.available
    supported = spec.supports(language)
    common = {
        "name": name,
        "installed": installed,
        "language_supported": supported,
        "model": resolved_model,
        "version": report.version,
    }

    if not supported:
        return EngineReadiness(
            assets_checked=False,
            assets_ready=None,
            ready=False,
            status=CapabilityStatus.MISCONFIGURED.value,
            reason="{} does not support language {!r}; it handles {}".format(
                name, language, ", ".join(spec.languages)
            ),
            remedy=(
                f"choose another engine for {language!r} (--ner-engine spacy)"
                if name == "nltk"
                else ""
            ),
            **common,
        )
    if not installed:
        return EngineReadiness(
            assets_checked=False,
            assets_ready=None,
            ready=False,
            status=report.status.value,
            reason=report.detail,
            remedy=report.install_hint,
            **common,
        )

    if name == "spacy":
        present = _spacy_model_ready(resolved_model)
        download = f"python -m spacy download {resolved_model}"
        return EngineReadiness(
            assets_checked=True,
            assets_ready=present,
            ready=present,
            status=(
                CapabilityStatus.AVAILABLE.value
                if present
                else CapabilityStatus.MISCONFIGURED.value
            ),
            reason=(
                f"spaCy {report.version} with model {resolved_model!r}"
                if present
                else f"spaCy is installed but model {resolved_model!r} is not"
            ),
            remedy="" if present else download,
            **common,
        )

    if not check_assets:
        return EngineReadiness(
            assets_checked=False,
            assets_ready=None,
            ready=True,
            status=CapabilityStatus.AVAILABLE.value,
            reason=(
                f"NLTK {report.version} is installed; its data packages are "
                "checked when it loads (doctor checks them now)"
            ),
            remedy="",
            **common,
        )
    try:
        missing = _nltk_missing_corpora()
    except ImportError as exc:
        from ._nltk import _reinstall_nltk  # ruff: ignore[import-outside-top-level]

        return EngineReadiness(
            assets_checked=True,
            assets_ready=None,
            ready=False,
            status=CapabilityStatus.BROKEN.value,
            reason=(
                "NLTK reports itself installed but cannot be imported "
                f"({type(exc).__name__}: {exc})"
            ),
            remedy=_reinstall_nltk(),
            **common,
        )
    return EngineReadiness(
        assets_checked=True,
        assets_ready=not missing,
        ready=not missing,
        status=(
            CapabilityStatus.AVAILABLE.value
            if not missing
            else CapabilityStatus.MISCONFIGURED.value
        ),
        reason=(
            f"NLTK {report.version} with its data packages"
            if not missing
            else "NLTK is installed but {} data package(s) are missing: {}".format(
                len(missing), ", ".join(missing)
            )
        ),
        remedy="" if not missing else _corpora_command(missing),
        missing=missing,
        **common,
    )


def resolve_engine(  # ruff: ignore[too-many-positional-arguments]
    mode: str = DEFAULT_ENGINE,
    language: str = "en",
    model: str | None = None,
    size: str = "sm",
    check_assets: bool = False,
) -> tuple[str, ...]:
    """
    Resolve an engine mode into the engines that will actually run.

    Parameters
    ----------
    mode : str, default='auto'
        One of :data:`ENGINE_MODES`.
    language : str, default='en'
        Language code, used to rule out engines that cannot handle it.
    model : str, optional
        Explicit spaCy model, used when judging whether spaCy is ready.
    size : str, default='sm'
        Preferred spaCy model size, used the same way.
    check_assets : bool, default=False
        Also check NLTK's data packages when ``auto`` considers NLTK. See
        :func:`engine_readiness`.

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
    **Developer notes.** ``auto`` resolves by *readiness*, not by presence:
    an engine whose package is absent, broken, unable to read the language or
    missing its data is skipped, because it would detect nothing either way.
    Before ``CP-093`` it resolved by package presence alone, so a machine with
    spaCy installed and no model chose spaCy over a working NLTK and then
    failed on the first sentence.

    An explicitly named engine is **not** silently dropped when it is not
    ready — the caller asked for it by name, and quietly running without it
    would be the silent degradation this submodule exists to prevent. It is
    returned; :func:`build_detectors` with ``required=True`` refuses it with
    its own remedy, and without ``required`` the detector raises on first use.

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

    # auto: prefer spaCy, fall back to NLTK, in both cases only if ready here.
    for name in ("spacy", "nltk"):
        if engine_readiness(
            name, language, model=model, size=size, check_assets=check_assets
        ).ready:
            return (name,)
    return ()


def describe_engines(  # ruff: ignore[too-many-positional-arguments]
    language: str = "en",
    mode: str = DEFAULT_ENGINE,
    model: str | None = None,
    size: str = "sm",
    check_assets: bool = False,
) -> dict[str, Any]:
    """
    Report every engine's readiness, and what ``mode`` would select.

    Parameters
    ----------
    language : str, default='en'
        Language code to report against.
    mode : str, default='auto'
        The mode being asked about.
    model : str, optional
        Explicit spaCy model.
    size : str, default='sm'
        Preferred spaCy model size.
    check_assets : bool, default=False
        Also check NLTK's data packages, which imports NLTK. ``doctor`` passes
        ``True``.

    Returns
    -------
    dict
        JSON-safe report with one entry per engine, a ``selected`` list, and
        ``ready``: whether every selected engine is ready.

    Notes
    -----
    **User notes.** Per engine, ``installed``, ``language_supported`` and
    ``assets_ready`` are the three conditions, ``ready`` is their conjunction,
    and ``remedy`` is the next step. ``usable`` is kept as a synonym of
    ``ready``. The top-level ``ready`` is ``False`` when a selected engine is
    known not to be able to run, which is the case ``doctor`` must not call
    healthy.
    """
    engines: dict[str, Any] = {}
    for name, spec in ENGINES.items():
        readiness = engine_readiness(
            name, language, model=model, size=size, check_assets=check_assets
        )
        engines[name] = dict(
            readiness.as_dict(),
            usable=readiness.ready,
            languages=list(spec.languages),
            needs_model=spec.needs_model,
            summary=spec.summary,
            install_hint=probe(spec.tier).install_hint,
            detail=readiness.reason,
        )

    selected = resolve_engine(
        mode, language, model=model, size=size, check_assets=check_assets
    )
    return {
        "mode": mode,
        "language": language,
        "selected": list(selected),
        "ready": all(engines[name]["ready"] for name in selected),
        "engines": engines,
        "canonical_labels": list(CANONICAL_LABELS),
    }


def _not_ready_error(
    mode: str, language: str, report: dict[str, Any], names: tuple[str, ...]
) -> CapabilityError:
    """
    Build the failure for "entity detection was asked for and cannot run".

    Parameters
    ----------
    mode : str
        The engine mode asked for.
    language : str
        The language asked for.
    report : dict
        :func:`describe_engines` output for the same configuration.
    names : tuple of str
        The engines that were required: the selection for a named mode or
        ``both``, every engine for ``auto``.

    Returns
    -------
    CapabilityError
        Carrying every engine's status and reason, and the remedy of the
        nearest one.

    Notes
    -----
    **Developer notes.** The message names *each* engine and its own reason,
    because the two fail for different reasons and a combined "install
    something" would send a user with NLTK installed off to install spaCy. The
    ``install_hint`` carries the remedy of the first required engine that can
    handle the language, since that is the one a caller can act on.
    """
    lines = []
    hint = ""
    status = CapabilityStatus.ABSENT.value
    for name in names:
        engine = report["engines"][name]
        line = "  {}: {} — {}".format(name, engine["status"], engine["reason"])
        if engine["remedy"]:
            line += "\n      fix: {}".format(engine["remedy"])
        lines.append(line)
        if engine["language_supported"] and not hint and engine["remedy"]:
            hint = engine["remedy"]
            status = engine["status"]
    return CapabilityError(
        "entity detection was requested (--ner) with engine mode {!r}, but {} "
        "for language {!r}:\n{}\n"
        "Fix one, or pass --ner-engine none to proceed without entity "
        "detection. Structural detectors (email, card, IBAN, …) are "
        "unaffected.".format(
            mode,
            "no engine is ready" if mode == "auto" else "it is not ready",
            language,
            "\n".join(lines),
        ),
        tier="ner",
        status=status,
        # Computed, never written down (CP-025): the tier's own declaration.
        install_hint=hint or probe("ner").install_hint,
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
        Whether the caller explicitly asked for entity detection. When true,
        every engine that will run must be ready, data included, or this
        raises. See the notes.

    Returns
    -------
    list of Detector
        Constructed detectors, in engine priority order. Construction loads no
        model: each detector loads its engine on first use.

    Raises
    ------
    PolicyError
        If ``mode`` is unknown.
    CapabilityError
        If ``required`` is true and the request cannot be met: ``auto`` found
        no ready engine, or a named engine (or either of ``both``) is not
        ready. ``mode="none"`` is exempt: it is an explicit instruction to run
        no engine.

    Notes
    -----
    **Developer notes.** Without ``required`` nothing here imports spaCy or
    NLTK, which is what keeps ``--help`` free of a multi-second load. With
    ``required`` NLTK may be imported to check its data — only when NLTK is
    about to run, so the import is paid once and early rather than once and
    late.

    **Why ``required`` exists.** ``auto`` degrades by design: it is the default,
    and a default that refused to run on a base installation would make the
    base tier unusable. But ``--ner`` is not a default, it is a request. With
    ``auto`` resolving to nothing, honouring that request by adding no detector
    and exiting successfully tells a user their text was scanned for names when
    nothing looked (``CP-024``). And a request for an engine that is installed
    but has no data must fail *here*, with the remedy, not at the first
    sentence — otherwise ``doctor``, which builds through this function too,
    calls the configuration healthy (``CP-093``).

    This lives here rather than in each caller because there are four callers
    (the command line, :func:`~scikitplot.cleanprompt.encode`, the file
    runtime and the web app) and a fifth would make the same omission. The web
    app built its own spaCy detector until ``CP-094`` and ignored the engine,
    language and model size it was given.
    """
    selected = resolve_engine(
        mode, language, model=model, size=size, check_assets=required
    )
    if required and mode != "none":
        report = describe_engines(
            language, mode, model=model, size=size, check_assets=True
        )
        if not selected:
            raise _not_ready_error(mode, language, report, tuple(ENGINES))
        not_ready = tuple(
            name for name in selected if not report["engines"][name]["ready"]
        )
        if not_ready:
            raise _not_ready_error(mode, language, report, not_ready)

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
