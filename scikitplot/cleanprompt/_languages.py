"""
Language codes, and the spaCy models that serve them.

Notes
-----
**User notes.** Ask for a language, not a model name::

    --ner --lang de              # picks de_core_news_sm
    --ner --lang de --model-size lg
    --ner --lang tr              # no Turkish model: falls back to multilingual
    --ner --ner-model my_model   # explicit, overrides all of this

``doctor --format json`` lists every supported language and says which models
are actually installed on this machine.

**Developer notes.** spaCy's model names follow a pattern rather than a
registry: ``{lang}_core_{genre}_{size}``, where English uses the ``web`` genre
and the other languages use ``news``, plus a multilingual ``xx_ent_wiki_sm``
trained on Wikipedia. The table below records the genre and the sizes each
language actually publishes, because guessing produces names that look right
and do not exist — ``de_core_web_sm`` is a plausible string and a 404.

**Falling back is announced, never silent.** A language with no model resolves
to the multilingual model, which is a genuinely different thing: it recognises
only ``PER``, ``ORG``, ``LOC`` and ``MISC``, so it will not find the finer
categories. :func:`resolve_model` reports that it substituted, and the callers
surface it, because a user who asked for Turkish and quietly got a
lower-resolution multilingual model would read the thinner results as "there
was nothing to find".

This module contains data and string manipulation. It imports nothing.

See Also
--------
scikitplot.cleanprompt._ner : Loads the model this module names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._exceptions import PolicyError

__all__ = [
    "DEFAULT_LANGUAGE",
    "LANGUAGES",
    "MODEL_SIZES",
    "MULTILINGUAL_MODEL",
    "LanguageSpec",
    "installed_models",
    "resolve_model",
    "supported_languages",
]

#: Default language when none is given.
DEFAULT_LANGUAGE = "en"

#: The multilingual model, used for any language with no dedicated one.
#:
#: Trained on Wikipedia with a reduced label set (``PER``, ``ORG``, ``LOC``,
#: ``MISC``), so it is a fallback rather than an equal.
MULTILINGUAL_MODEL = "xx_ent_wiki_sm"

#: Model sizes, smallest first. ``trf`` is transformer-based: much better and
#: much heavier, and it pulls in torch.
MODEL_SIZES: tuple[str, ...] = ("sm", "md", "lg", "trf")


@dataclass(frozen=True)
class LanguageSpec:
    """
    One language spaCy publishes models for.

    Parameters
    ----------
    code : str
        ISO 639-1 code.
    name : str
        English name of the language, for help text and reports.
    genre : str
        ``"web"`` or ``"news"``; part of the model name.
    sizes : tuple of str
        Sizes actually published for this language.
    """

    code: str
    name: str
    genre: str
    sizes: tuple[str, ...]

    def model_for(self, size: str) -> str:
        """
        Return the model name for ``size``, falling back to the nearest.

        Parameters
        ----------
        size : str
            Preferred size.

        Returns
        -------
        str
            A model name this language actually publishes.

        Notes
        -----
        **Developer notes.** Falls back *downwards* to the largest available
        size no bigger than the request, and only upwards when nothing smaller
        exists. Asking for ``lg`` on a language that publishes only ``sm``
        should get ``sm``, not a name that does not resolve.
        """
        if size in self.sizes:
            return f"{self.code}_core_{self.genre}_{size}"
        wanted = MODEL_SIZES.index(size) if size in MODEL_SIZES else 0
        candidates = [s for s in self.sizes if MODEL_SIZES.index(s) <= wanted]
        chosen = candidates[-1] if candidates else self.sizes[0]
        return f"{self.code}_core_{self.genre}_{chosen}"


def _spec(
    code: str, name: str, genre: str = "news", sizes: str = "sm,md,lg"
) -> LanguageSpec:
    """Build a :class:`LanguageSpec` from a compact description."""
    return LanguageSpec(
        code=code, name=name, genre=genre, sizes=tuple(sizes.split(","))
    )


#: Languages spaCy publishes a named-entity model for.
LANGUAGES: dict[str, LanguageSpec] = {
    spec.code: spec
    for spec in (
        _spec("en", "English", genre="web", sizes="sm,md,lg,trf"),
        _spec("de", "German", sizes="sm,md,lg"),
        _spec("fr", "French", sizes="sm,md,lg"),
        _spec("es", "Spanish", sizes="sm,md,lg"),
        _spec("pt", "Portuguese", sizes="sm,md,lg"),
        _spec("it", "Italian", sizes="sm,md,lg"),
        _spec("nl", "Dutch", sizes="sm,md,lg"),
        _spec("el", "Greek", sizes="sm,md,lg"),
        _spec("nb", "Norwegian Bokmal", sizes="sm,md,lg"),
        _spec("da", "Danish", sizes="sm,md,lg"),
        _spec("sv", "Swedish", sizes="sm,md,lg"),
        _spec("fi", "Finnish", sizes="sm,md,lg"),
        _spec("pl", "Polish", sizes="sm,md,lg"),
        _spec("ro", "Romanian", sizes="sm,md,lg"),
        _spec("ru", "Russian", sizes="sm,md,lg"),
        _spec("uk", "Ukrainian", sizes="sm,md,lg"),
        _spec("hr", "Croatian", sizes="sm,md,lg"),
        _spec("sl", "Slovenian", sizes="sm,md,lg"),
        _spec("mk", "Macedonian", sizes="sm,md,lg"),
        _spec("lt", "Lithuanian", sizes="sm,md,lg"),
        _spec("ca", "Catalan", sizes="sm,md,lg,trf"),
        _spec("ja", "Japanese", sizes="sm,md,lg,trf"),
        _spec("ko", "Korean", sizes="sm,md,lg"),
        _spec("zh", "Chinese", sizes="sm,md,lg,trf"),
    )
}


def supported_languages() -> tuple[str, ...]:
    """
    Return every language code with a dedicated model.

    Returns
    -------
    tuple of str
        Sorted codes. ``"xx"`` is not listed: it is the fallback, not a
        language.

    Examples
    --------
    >>> "en" in supported_languages() and "de" in supported_languages()
    True
    """
    return tuple(sorted(LANGUAGES))


def resolve_model(
    language: str = DEFAULT_LANGUAGE,
    size: str = "sm",
    explicit: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Resolve a language and size into a spaCy model name.

    Parameters
    ----------
    language : str, default='en'
        Language code.
    size : str, default='sm'
        Preferred model size; one of :data:`MODEL_SIZES`.
    explicit : str, optional
        A model name that overrides everything else.

    Returns
    -------
    model : str
        The model name to load.
    note : dict
        What was decided: ``language``, ``requested_size``, ``resolved_size``,
        ``fallback`` and, when a substitution happened, ``reason``.

    Raises
    ------
    PolicyError
        If ``size`` is not a known size.

    Notes
    -----
    **User notes.** When ``fallback`` is ``True`` the multilingual model was
    substituted, and it recognises only broad categories. Treat thin results as
    a limitation of that model rather than as an empty document.

    Examples
    --------
    >>> resolve_model("en", "sm")[0]
    'en_core_web_sm'
    >>> resolve_model("de", "lg")[0]
    'de_core_news_lg'
    >>> model, note = resolve_model("tr")
    >>> model, note["fallback"]
    ('xx_ent_wiki_sm', True)
    >>> resolve_model("de", explicit="my_model")[0]
    'my_model'
    """
    if explicit:
        return explicit, {
            "language": language,
            "requested_size": size,
            "resolved_size": None,
            "fallback": False,
            "reason": "explicit model name supplied",
        }

    if size not in MODEL_SIZES:
        msg = "unknown model size {!r}; choose from {}".format(
            size, ", ".join(MODEL_SIZES)
        )
        raise PolicyError(msg)

    spec = LANGUAGES.get(language)
    if spec is None:
        return MULTILINGUAL_MODEL, {
            "language": language,
            "requested_size": size,
            "resolved_size": "sm",
            "fallback": True,
            "reason": (
                f"spaCy publishes no named-entity model for {language!r}; using the "
                "multilingual model, which recognises only broad categories "
                "(PER, ORG, LOC, MISC)"
            ),
        }

    model = spec.model_for(size)
    resolved = model.rsplit("_", 1)[-1]
    note: dict[str, Any] = {
        "language": language,
        "requested_size": size,
        "resolved_size": resolved,
        "fallback": False,
    }
    if resolved != size:
        note["reason"] = "{} publishes {}; using {}".format(
            spec.name, ", ".join(spec.sizes), resolved
        )
    return model, note


def installed_models() -> dict[str, str]:
    """
    Return the spaCy models installed here, mapped to their versions.

    Returns
    -------
    dict of str to str
        Model name to version. Empty when spaCy is not installed or no model
        is present.

    Notes
    -----
    **Developer notes.** Reads distribution metadata rather than importing
    spaCy or loading anything, so this is safe to call from ``doctor`` and
    costs a directory scan. A spaCy model is an ordinary installed
    distribution, which is what makes that possible.
    """
    try:
        from importlib.metadata import (  # ruff: ignore[import-outside-top-level]
            distributions,
        )
    except ImportError:  # pragma: no cover - the floor is 3.8
        return {}

    known = {MULTILINGUAL_MODEL}
    for spec in LANGUAGES.values():
        for size in spec.sizes:
            known.add(f"{spec.code}_core_{spec.genre}_{size}")

    found: dict[str, str] = {}
    try:
        for dist in distributions():
            name = (dist.metadata["Name"] or "").replace("-", "_")
            if name in known:
                found[name] = dist.version or "unknown"
    except Exception:  # noqa: BLE001 - a broken entry must not break the report
        return found
    return found


def language_report(
    language: str = DEFAULT_LANGUAGE,
    size: str = "sm",
) -> dict[str, Any]:
    """
    Describe language support for a report or a user interface.

    Parameters
    ----------
    language : str, default='en'
        Language being asked about.
    size : str, default='sm'
        Preferred model size.

    Returns
    -------
    dict
        JSON-safe description, including the resolved model, whether it is
        installed, and the command that would install it.
    """
    model, note = resolve_model(language, size)
    present = installed_models()
    return {
        "requested": language,
        "supported": language in LANGUAGES,
        "model": model,
        "model_installed": model in present,
        "model_version": present.get(model),
        "install_hint": f"python -m spacy download {model}",
        "resolution": note,
        "installed_models": present,
        "supported_languages": {
            code: spec.name for code, spec in sorted(LANGUAGES.items())
        },
    }
