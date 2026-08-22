# scikitplot/corpus/_enrichers/_simple.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Zero-dependency token and frequency-keyword enrichment."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from .._schema import CorpusDocument

__all__ = [
    "SimpleEnricherSpec",
    "SimpleFrequencyEnricher",
]

_DEFAULT_TOKEN_PATTERN = r"[^\W\d_]+(?:['’][^\W\d_]+)*"  # ruff: ignore[ambiguous-unicode-character-string, hardcoded-password-string]


@dataclass(frozen=True)
class SimpleEnricherSpec:
    """
    Declarative configuration for :class:`SimpleFrequencyEnricher`.

    Parameters
    ----------
    min_token_length : int, default=3
        Minimum token length retained after tokenization.
    max_keywords : int, default=8
        Maximum number of frequency-ranked keywords stored per document.
    lowercase : bool, default=True
        Lowercase text before tokenization.
    token_pattern : str, optional
        Unicode-aware regular expression used to extract tokens.

    Notes
    -----
    The spec has no optional NLP dependencies and performs no resource lookup.
    It can be passed directly to ``FluentCorpus.enricher(...)``; runtime
    materialization converts it to :class:`SimpleFrequencyEnricher`.
    """

    min_token_length: int = 3
    max_keywords: int = 8
    lowercase: bool = True
    token_pattern: str = field(default=_DEFAULT_TOKEN_PATTERN, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.min_token_length, bool) or not isinstance(
            self.min_token_length, int
        ):
            raise TypeError("min_token_length must be an integer")
        if self.min_token_length < 1:
            raise ValueError("min_token_length must be >= 1")

        if isinstance(self.max_keywords, bool) or not isinstance(
            self.max_keywords, int
        ):
            raise TypeError("max_keywords must be an integer")
        if self.max_keywords < 1:
            raise ValueError("max_keywords must be >= 1")

        if not isinstance(self.lowercase, bool):
            raise TypeError("lowercase must be a bool")
        if not isinstance(self.token_pattern, str) or not self.token_pattern:
            raise ValueError("token_pattern must be a non-empty regular expression")
        try:
            pattern = re.compile(self.token_pattern)
        except re.error as exc:
            raise ValueError(f"invalid token_pattern: {exc}") from exc
        if pattern.search("") is not None:
            raise ValueError("token_pattern must not match the empty string")


@dataclass(frozen=True, init=False)
class SimpleFrequencyEnricher:
    """
    Populate tokens and deterministic frequency-ranked keywords.

    The implementation is intentionally small and dependency-free. It uses
    ``normalized_text`` when available, otherwise ``text``. Existing tokens or
    keywords are preserved when ``overwrite=False``; if tokens already exist
    but keywords do not, those tokens are reused to derive keywords.

    Parameters
    ----------
    spec : SimpleEnricherSpec or None, optional
        Declarative configuration. If omitted, keyword overrides can be passed
        directly for convenience.
    min_token_length : int or None, keyword-only
        Convenience override used only when ``spec`` is omitted.
    max_keywords : int or None, keyword-only
        Convenience override used only when ``spec`` is omitted.
    lowercase : bool or None, keyword-only
        Convenience override used only when ``spec`` is omitted.
    token_pattern : str or None, keyword-only
        Convenience override used only when ``spec`` is omitted.

    Examples
    --------
    >>> enricher = SimpleFrequencyEnricher(min_token_length=2, max_keywords=3)
    >>> enricher.spec.max_keywords
    3
    """

    spec: SimpleEnricherSpec

    def __init__(
        self,
        spec: SimpleEnricherSpec | None = None,
        *,
        min_token_length: int | None = None,
        max_keywords: int | None = None,
        lowercase: bool | None = None,
        token_pattern: str | None = None,
    ) -> None:
        if spec is not None and not isinstance(spec, SimpleEnricherSpec):
            raise TypeError("spec must be SimpleEnricherSpec or None")

        overrides = {
            "min_token_length": min_token_length,
            "max_keywords": max_keywords,
            "lowercase": lowercase,
            "token_pattern": token_pattern,
        }
        supplied = {
            name: value for name, value in overrides.items() if value is not None
        }

        if spec is not None and supplied:
            names = ", ".join(sorted(supplied))
            raise ValueError(
                f"pass either spec or direct overrides, not both (overrides: {names})"
            )

        if spec is None:
            defaults = SimpleEnricherSpec()
            spec = SimpleEnricherSpec(
                min_token_length=(
                    defaults.min_token_length
                    if min_token_length is None
                    else min_token_length
                ),
                max_keywords=(
                    defaults.max_keywords if max_keywords is None else max_keywords
                ),
                lowercase=defaults.lowercase if lowercase is None else lowercase,
                token_pattern=(
                    defaults.token_pattern if token_pattern is None else token_pattern
                ),
            )

        object.__setattr__(self, "spec", spec)

    @property
    def min_token_length(self) -> int:
        """Minimum token length from :attr:`spec`."""
        return self.spec.min_token_length

    @property
    def max_keywords(self) -> int:
        """Maximum keyword count from :attr:`spec`."""
        return self.spec.max_keywords

    def _tokenize(self, text: str) -> list[str]:
        pattern = re.compile(self.spec.token_pattern)
        tokens: list[str] = []
        for match in pattern.finditer(text):
            token = match.group(0)
            if self.spec.lowercase:
                token = token.casefold()
            if len(token) >= self.spec.min_token_length:
                tokens.append(token)
        return tokens

    def enrich_documents(
        self,
        documents: Iterable[CorpusDocument],
        *,
        overwrite: bool = False,
    ) -> list[CorpusDocument]:
        """Return copy-on-write documents with tokens and keywords populated."""
        enriched: list[CorpusDocument] = []

        for index, doc in enumerate(documents):
            if not isinstance(doc, CorpusDocument):
                raise TypeError(
                    f"documents[{index}] must be CorpusDocument, "
                    f"got {type(doc).__name__}"
                )

            if not overwrite and doc.tokens is not None:
                tokens = list(doc.tokens)
            else:
                source_text = doc.normalized_text or doc.text
                tokens = self._tokenize(source_text)

            if not overwrite and doc.keywords is not None:
                keywords = list(doc.keywords)
            else:
                counts = Counter(tokens)
                keywords = [
                    token
                    for token, _count in sorted(
                        counts.items(),
                        key=lambda item: (-item[1], item[0]),
                    )[: self.spec.max_keywords]
                ]

            if not overwrite and doc.tokens is not None and doc.keywords is not None:
                enriched.append(doc)
                continue

            enriched.append(
                doc.replace(
                    tokens=tokens or None,
                    keywords=keywords or None,
                )
            )

        return enriched
