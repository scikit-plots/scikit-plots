# scikitplot/levenshtein/_core.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Backend-neutral Levenshtein distance helpers.

The public facade deliberately keeps optional implementations lazy.  ``auto``
prefers Scikit-Plots' bundled Cython backend, then the MIT-licensed RapidFuzz
implementation, and finally a dependency-free dynamic-programming fallback.
The GPL ``Levenshtein`` distribution is supported only when explicitly named.
"""

from __future__ import annotations

import dataclasses
import heapq
import importlib
import importlib.metadata
import importlib.util
import logging
from typing import Any, Callable, Iterable, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "BackendInfo",
    "Match",
    "available_backends",
    "backend_info",
    "closest",
    "distance",
    "make_corpus_scorer",
    "normalized_distance",
    "normalized_similarity",
    "rank",
    "similarity",
]


@dataclasses.dataclass(frozen=True)
class BackendInfo:
    """Read-only description of one implementation backend."""

    name: str
    available: bool
    version: str | None
    license: str
    implementation: str
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Match:
    """One ranked fuzzy-match result with actual backend provenance."""

    choice: Any
    distance: int
    similarity: float
    index: int
    backend: str = "python"

    def to_dict(self) -> dict[str, Any]:
        """Return scalar match metadata; ``choice`` is preserved as-is."""
        return {
            "choice": self.choice,
            "distance": self.distance,
            "similarity": self.similarity,
            "index": self.index,
            "backend": self.backend,
        }


_BACKEND_ALIASES = {
    "internal": "internal",
    "cexternal": "internal",
    "cython": "internal",
    "rapidfuzz": "rapidfuzz",
    "rapid-fuzz": "rapidfuzz",
    "levenshtein": "levenshtein",
    "python-levenshtein": "levenshtein",
    "python": "python",
    "pure-python": "python",
    "auto": "auto",
}


def _distribution_version(
    *names: str,
) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except (  # ruff: ignore[try-except-in-loop]
            importlib.metadata.PackageNotFoundError
        ):
            continue
        # metadata backends can fail oddly
        except Exception:  # ruff: ignore[blind-except]
            return None
    return None


def _find_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # ruff: ignore[blind-except]
        return False


def _backend_info(name: str) -> BackendInfo:
    if name == "internal":
        try:
            module = importlib.import_module(
                "scikitplot.cexternals._editdistance",
            )
        except Exception as exc:  # ruff: ignore[blind-except]
            return BackendInfo(
                name="internal",
                available=False,
                version=None,
                license="MIT",
                implementation="scikitplot.cexternals._editdistance",
                reason=f"{type(exc).__name__}: {exc}",
            )
        if not callable(getattr(module, "distance", None)):
            return BackendInfo(
                name="internal",
                available=False,
                version=None,
                license="MIT",
                implementation="scikitplot.cexternals._editdistance",
                reason="distance callable is unavailable",
            )
        return BackendInfo(
            name="internal",
            available=True,
            version=None,
            license="MIT",
            implementation="scikitplot.cexternals._editdistance",
        )
    if name == "rapidfuzz":
        available = _find_module("rapidfuzz.distance")
        return BackendInfo(
            name="rapidfuzz",
            available=available,
            version=_distribution_version("RapidFuzz", "rapidfuzz"),
            license="MIT",
            implementation="rapidfuzz.distance.Levenshtein",
            reason=None if available else "module rapidfuzz.distance not found",
        )
    if name == "levenshtein":
        available = _find_module("Levenshtein")
        return BackendInfo(
            name="levenshtein",
            available=available,
            version=_distribution_version("Levenshtein", "python-Levenshtein"),
            license="GPL-2.0-or-later",
            implementation="Levenshtein",
            reason=None if available else "module Levenshtein not found",
        )
    if name == "python":
        return BackendInfo(
            name="python",
            available=True,
            version=None,
            license="BSD-3-Clause",
            implementation="scikitplot.levenshtein pure Python fallback",
        )
    raise ValueError(f"unknown Levenshtein backend {name!r}")


def backend_info(
    name: str = "auto",
) -> BackendInfo:
    """
    Return capability information for one backend.

    ``auto`` returns information for the backend that would currently be used.
    """
    canonical = _BACKEND_ALIASES.get(str(name).lower())
    if canonical is None:
        raise ValueError(
            f"unknown backend {name!r}; expected one of {sorted(_BACKEND_ALIASES)}",
        )
    if canonical == "auto":
        canonical = _resolve_backend_name("auto", strict=False)
    return _backend_info(canonical)


def available_backends(
    *,
    include_gpl: bool = True,
) -> tuple[BackendInfo, ...]:
    """Return backend capability records in preferred order."""
    names = ["internal", "rapidfuzz"]
    if include_gpl:
        names.append("levenshtein")
    names.append("python")
    return tuple(_backend_info(name) for name in names)


def _resolve_backend_name(
    name: str,
    *,
    strict: bool,
) -> str:
    canonical = _BACKEND_ALIASES.get(str(name).lower())
    if canonical is None:
        raise ValueError(
            f"unknown backend {name!r}; expected one of {sorted(_BACKEND_ALIASES)}",
        )
    if canonical == "auto":
        # GPL Levenshtein is intentionally not selected automatically.
        for candidate in ("internal", "rapidfuzz", "python"):
            if _backend_info(candidate).available:
                logger.debug("Levenshtein auto backend selected %r", candidate)
                return candidate
        return "python"  # defensive; pure Python is always available

    info = _backend_info(canonical)
    if info.available:
        return canonical
    message = f"Levenshtein backend {canonical!r} is unavailable: {info.reason}"
    if strict:
        raise ImportError(message)
    logger.warning("%s; falling back to backend='auto'.", message)
    return _resolve_backend_name("auto", strict=True)


def _as_sequence(
    value: Any,
) -> Sequence[Any]:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value
    if isinstance(value, Sequence):
        return value
    return tuple(value)


def _python_distance(
    a: Any,
    b: Any,
) -> int:
    """Memory-bounded Wagner-Fischer distance for arbitrary sequences."""
    left = _as_sequence(a)
    right = _as_sequence(b)
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, x in enumerate(left, start=1):
        current = [i]
        for j, y in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (x != y)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def _distance_impl(
    backend: str,
) -> Callable[[Any, Any], int]:
    if backend == "internal":
        module = importlib.import_module(
            "scikitplot.cexternals._editdistance",
        )
        return module.distance
    if backend == "rapidfuzz":
        module = importlib.import_module(
            "rapidfuzz.distance",
        )
        return module.Levenshtein.distance
    if backend == "levenshtein":
        module = importlib.import_module(
            "Levenshtein",
        )
        return module.distance
    if backend == "python":
        return _python_distance
    raise AssertionError(backend)


def _distance_with_backend(
    a: Any,
    b: Any,
    *,
    backend: str = "auto",
    strict: bool = False,
) -> tuple[int, str]:
    """Return distance plus the implementation that actually produced it."""
    selected = _resolve_backend_name(backend, strict=strict)
    try:
        return int(_distance_impl(selected)(a, b)), selected
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception as exc:
        if strict or selected == "python":
            raise
        logger.warning(
            "Levenshtein backend %r failed at runtime (%s: %s); "
            "falling back to pure Python.",
            selected,
            type(exc).__name__,
            exc,
        )
        return _python_distance(a, b), "python"


def distance(
    a: Any,
    b: Any,
    *,
    backend: str = "auto",
    strict: bool = False,
) -> int:
    """
    Return the Levenshtein edit distance between two sequences.

    Parameters
    ----------
    a, b : sequence-like
        Strings, bytes, or sequence-like values.  The pure-Python and bundled
        internal backends support general sequences. External libraries may
        impose narrower type constraints.
    backend : str, optional
        ``"auto"`` (default), ``"internal"``, ``"rapidfuzz"``,
        ``"levenshtein"`` or ``"python"``.
    strict : bool, optional
        If an explicitly selected backend is unavailable, raise instead of
        warning and falling back to the automatic safe chain.
    """
    value, _used = _distance_with_backend(
        a,
        b,
        backend=backend,
        strict=strict,
    )
    return value


def normalized_distance(
    a: Any,
    b: Any,
    *,
    backend: str = "auto",
    strict: bool = False,
) -> float:
    """Return distance normalised to ``[0, 1]`` by maximum sequence length."""
    left = _as_sequence(a)
    right = _as_sequence(b)
    denom = max(len(left), len(right))
    if denom == 0:
        return 0.0
    return (
        distance(
            left,
            right,
            backend=backend,
            strict=strict,
        )
        / denom
    )


def normalized_similarity(
    a: Any,
    b: Any,
    *,
    backend: str = "auto",
    strict: bool = False,
) -> float:
    """Return normalised Levenshtein similarity in ``[0, 1]``."""
    return 1.0 - normalized_distance(
        a,
        b,
        backend=backend,
        strict=strict,
    )


def similarity(
    a: Any,
    b: Any,
    *,
    backend: str = "auto",
    strict: bool = False,
) -> int:
    """Return ``max(len(a), len(b)) - distance(a, b)``."""
    left = _as_sequence(a)
    right = _as_sequence(b)
    return max(len(left), len(right)) - distance(
        left, right, backend=backend, strict=strict
    )


def rank(
    query: Any,
    choices: Iterable[Any],
    *,
    key: Callable[[Any], Any] | None = None,
    limit: int | None = None,
    score_cutoff: float = 0.0,
    backend: str = "auto",
    strict: bool = False,
) -> list[Match]:
    """
    Rank choices from most to least similar to *query*.

    ``score_cutoff`` is backend-neutral normalized similarity in ``[0, 1]``.
    It is applied after exact distance computation so every backend preserves
    identical ranking semantics and provenance.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0 or None")
    if not 0.0 <= score_cutoff <= 1.0:
        raise ValueError("score_cutoff must be within [0, 1]")
    query_seq = _as_sequence(query)

    def iter_matches() -> Iterable[Match]:
        for index, choice in enumerate(choices):
            value = key(choice) if key is not None else choice
            value_seq = _as_sequence(value)
            d, used_backend = _distance_with_backend(
                query_seq, value_seq, backend=backend, strict=strict
            )
            denom = max(len(query_seq), len(value_seq))
            sim = 1.0 if denom == 0 else 1.0 - (d / denom)
            if sim < score_cutoff:
                continue
            yield Match(
                choice=choice,
                distance=d,
                similarity=sim,
                index=index,
                backend=used_backend,
            )

    def order_key(m: Match) -> tuple[float, int, int]:
        return (-m.similarity, m.distance, m.index)

    matches = iter_matches()
    if limit is not None:
        if limit == 0:
            return []
        # ``heapq.nsmallest`` keeps only O(limit) candidates instead of
        # materialising an entire corpus-sized match list before truncation.
        return heapq.nsmallest(limit, matches, key=order_key)
    out = list(matches)
    out.sort(key=order_key)
    return out


def closest(
    query: Any,
    choices: Iterable[Any],
    *,
    key: Callable[[Any], Any] | None = None,
    score_cutoff: float = 0.0,
    backend: str = "auto",
    strict: bool = False,
) -> Match | None:
    """Return the closest choice, or ``None`` for an empty iterable."""
    ranked = rank(
        query,
        choices,
        key=key,
        limit=1,
        score_cutoff=score_cutoff,
        backend=backend,
        strict=strict,
    )
    return ranked[0] if ranked else None


def make_corpus_scorer(
    *,
    backend: str = "auto",
    strict: bool = False,
    use_normalized_text: bool = True,
    score_cutoff: float = 0.0,
) -> Callable[[str, list[Any], Any], list[Any]]:
    """
    Return a scorer compatible with ``corpus.CustomRetrievalIndex``.

    The Corpus import is intentionally lazy so :mod:`scikitplot.levenshtein`
    remains independently importable.
    """

    def scorer(query: str, documents: list[Any], config: Any) -> list[Any]:
        from scikitplot.corpus._similarity import RetrievalHit  # noqa: PLC0415

        def _doc_key(doc: Any) -> Any:
            if use_normalized_text:
                return getattr(doc, "normalized_text", None) or getattr(doc, "text", "")
            return getattr(doc, "text", "")

        top_k = int(getattr(config, "top_k", 10))
        ranked = rank(
            query,
            documents,
            key=_doc_key,
            limit=top_k,
            score_cutoff=score_cutoff,
            backend=backend,
            strict=strict,
        )
        return [
            RetrievalHit(
                doc=item.choice,
                score=item.similarity,
                match_mode="levenshtein",
                backend=f"levenshtein:{item.backend}",
                native_score=item.similarity,
                native_metric="normalized_levenshtein_similarity",
                rank=position,
            )
            for position, item in enumerate(ranked)
        ]

    return scorer
