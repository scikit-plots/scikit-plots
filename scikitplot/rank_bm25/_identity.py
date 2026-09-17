"""
Declared identities for a lexical build: the analyzer and the recipe.

A BM25 index stored frequencies, document lengths and its parameters, and
nothing that recorded *how* those numbers were produced. Two things were
therefore invisible after the fact.

The tokenizer was a bare callable on the instance. Two indexes built over the
same corpus with different tokenisation -- a different stopword list, stemming
on or off, case folded or not -- hold different frequencies and are
indistinguishable afterwards, so neither can be pinned, compared or rebuilt
deliberately.

The scoring formula was identified only by a class name. ``BM25Okapi`` implies a
standard, but the implementation makes specific choices: the idf is floored at
``epsilon * average_idf`` so a term in most documents cannot score negative, and
a repeated query term is accumulated rather than saturated. Those are legitimate
choices and they are not universal, so the name is a claim the numbers do not
support. A recipe is named and versioned instead.

Notes
-----
**Developer.** Identity here is a digest over a flat record of declared strings,
numbers and sorted string sets, so a small local derivation is enough and honest.
It deliberately does not reach for the corpus canonical encoder: this submodule
does not depend on another, and copying a general encoder to derive an identity
over six string fields would be duplication without a reason.

Sets are sorted before hashing, so an identity cannot depend on iteration order,
and every part is length-prefixed so no two different records can produce the
same bytes.

See Also
--------
scikitplot.rank_bm25._validation : The row-index and count contract.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Iterable, Mapping

__all__ = [
    "RECIPES",
    "Analyzer",
    "Recipe",
    "lexical_identity",
]

_SEP = "\x1f"


def _field(label: str, value: object) -> str:
    """Return one length-prefixed ``label=value`` part."""
    text = "" if value is None else str(value)
    return f"{label}:{len(text)}:{text}"


def _digest(parts: Iterable[str]) -> str:
    """Return the SHA-256 hex digest of ``parts`` joined with a separator."""
    return hashlib.sha256(_SEP.join(parts).encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class Analyzer:
    """
    How text becomes tokens, declared so a build can record it.

    Parameters
    ----------
    name : str
        Analyzer name, e.g. ``"whitespace"``.
    version : str, optional
        Version of this analyzer's behaviour. Bump it whenever the tokens it
        produces change, so an index built with the old behaviour is visibly a
        different build rather than silently the same one.
    language : str or None, optional
        Language tag the analysis assumes, when it assumes one.
    stopwords : frozenset of str, optional
        Tokens removed after splitting.
    stemmer : str or None, optional
        Name of the stemmer applied, if any. A name rather than a callable: a
        callable has no identity that survives the process.
    lowercase : bool, optional
        Whether text is case-folded before splitting.

    Notes
    -----
    **User.** Two indexes built with analyzers that share a fingerprint hold
    comparable frequencies. Two that do not, do not -- whatever their class
    names say.
    """

    name: str
    version: str = "1"
    language: str | None = None
    stopwords: frozenset[str] = frozenset()
    stemmer: str | None = None
    lowercase: bool = True

    def __post_init__(self) -> None:
        """
        Validate the declaration and take ownership of its collections.

        Notes
        -----
        **Developer.** ``frozen=True`` freezes the attribute bindings, not the
        objects behind them: a caller passing a ``set`` keeps a reference and
        can mutate it afterwards, moving the fingerprint of an object that
        declares itself immutable. Copying into a ``frozenset`` here is what
        makes the declaration true.
        """
        for field_name in ("name", "version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise TypeError(
                    f"Analyzer.{field_name} must be a non-empty string, got {value!r}."
                )
        for field_name in ("language", "stemmer"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise TypeError(
                    f"Analyzer.{field_name} must be a string or None, got "
                    f"{type(value).__name__}."
                )
        if not isinstance(self.lowercase, bool):
            raise TypeError("Analyzer.lowercase must be a bool.")
        words = self.stopwords
        if isinstance(words, str) or not hasattr(words, "__iter__"):
            raise TypeError(
                "Analyzer.stopwords must be an iterable of strings, not "
                f"{type(words).__name__}."
            )
        frozen = frozenset(words)
        if any(not isinstance(word, str) for word in frozen):
            raise TypeError("Analyzer.stopwords must contain only strings.")
        object.__setattr__(self, "stopwords", frozen)

    @property
    def fingerprint(self) -> str:
        """
        Full SHA-256 digest of the declared behaviour.

        Returns
        -------
        str
            A 64-character hex digest, stable across processes because every
            part is a declared string and sets are sorted before hashing.
        """
        stopwords = _SEP.join(sorted(self.stopwords))
        return _digest(
            [
                _field("analyzer", self.name),
                _field("version", self.version),
                _field("language", self.language),
                _field("stopwords", f"{len(self.stopwords)}:{stopwords}"),
                _field("stemmer", self.stemmer),
                _field("lowercase", int(bool(self.lowercase))),
            ]
        )

    def tokenize(self, text: str) -> list[str]:
        """
        Split ``text`` into tokens, applying what this analyzer declares.

        Parameters
        ----------
        text : str
            Text to analyse.

        Returns
        -------
        list of str
            Tokens, in order.

        Notes
        -----
        **Developer.** The declaration drives the behaviour rather than
        describing it from a distance. A declared field that did not affect the
        tokens would make the fingerprint a claim about nothing.

        ``stemmer`` is recorded but not applied here: this module carries no
        stemming implementation, and pretending to apply one would be worse than
        declaring that a build used it.
        """
        prepared = text.lower() if self.lowercase else text
        tokens = prepared.split()
        if self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        return tokens


@dataclasses.dataclass(frozen=True)
class Recipe:
    """
    A named, versioned scoring formula.

    Parameters
    ----------
    name : str
        Recipe name.
    version : str
        Version of the formula. Any change to the arithmetic bumps it.
    notes : str
        What this recipe does that a reader should not assume, in one sentence.
    """

    name: str
    version: str
    notes: str = ""

    @property
    def identifier(self) -> str:
        """``"name/version"``, the string a build records."""
        return f"{self.name}/{self.version}"

    @property
    def fingerprint(self) -> str:
        """Full SHA-256 digest of the recipe identity."""
        return _digest([_field("recipe", self.identifier)])


#: The scoring formulas this package implements, by identifier. A build records
#: the identifier, so "which BM25" is answerable from the artifact rather than
#: from the class that happened to produce it.
RECIPES: dict[str, Recipe] = {
    recipe.identifier: recipe
    for recipe in (
        Recipe(
            name="okapi-epsilon-floor",
            version="1",
            notes=(
                "Okapi BM25 with the idf floored at epsilon * average_idf, so a "
                "term appearing in most documents cannot contribute a negative "
                "score; a repeated query term is accumulated, not saturated."
            ),
        ),
        Recipe(
            name="bm25l-delta",
            version="1",
            notes=(
                "BM25L: a delta is added to the normalised term frequency, which "
                "lifts long documents relative to Okapi."
            ),
        ),
        Recipe(
            name="bm25plus-delta",
            version="1",
            notes=(
                "BM25+: a delta is added to the whole term contribution, so a "
                "document containing the term always outscores one that does not."
            ),
        ),
    )
}


def lexical_identity(
    recipe: Recipe,
    analyzer: Analyzer | None,
    parameters: Mapping[str, Any],
) -> str:
    """
    Return the identity of a lexical build.

    Parameters
    ----------
    recipe : Recipe
        The scoring formula used.
    analyzer : Analyzer or None
        How the text was tokenised. ``None`` records that the caller supplied
        tokens directly, which is a different build from any analysed one.
    parameters : mapping
        Scoring parameters, e.g. ``{"k1": 1.5, "b": 0.75}``.

    Returns
    -------
    str
        A 64-character hex digest covering what produced the numbers.

    Notes
    -----
    **Developer.** Parameters are sorted by name, so identity does not depend on
    the order a caller happened to pass them. The corpus content is deliberately
    not part of this: it answers "how was this built", and the documents answer
    "what was built from". Mixing them would mean a re-tokenisation and a new
    document looked like the same kind of change.
    """
    if analyzer is None:
        # A bare callable has no identity that survives the process: two
        # different tokenizers would otherwise produce one durable identity for
        # two different indexes. "unidentified" is recorded as its own marker so
        # the collision is visible in the value rather than hidden by it, and
        # persistence refuses it outright.
        analyzer_part = "unidentified-tokenizer"
    else:
        analyzer_part = analyzer.fingerprint
    parts = [
        _field("recipe", recipe.identifier),
        _field("analyzer", analyzer_part),
    ]
    parts.extend(_field(f"param.{key}", parameters[key]) for key in sorted(parameters))
    return _digest(parts)
