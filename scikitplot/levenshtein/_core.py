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
    """Memory-bounded Wagner-Fischer distance for arbitrary sequences.

    Elements match under container equality (identity, then ``==``), the rule
    ``left == right`` already applies, so the same NaN object matches itself
    here exactly as it does in every accelerated backend.
    """
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
            replace = previous[j - 1] + (0 if (x is y or x == y) else 1)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


#: Argument pairs every backend already compares exactly, element by element.
_EXACT_PAIRS = ((str, str), (bytes, bytes), (bytearray, bytearray), (bytes, bytearray))


def _exact_codes(
    a: Any,
    b: Any,
) -> tuple[Any, Any]:
    """Return ``(a, b)`` in a form every backend compares by element equality.

    Parameters
    ----------
    a, b : sequence-like
        The two inputs of one distance computation.

    Returns
    -------
    tuple
        ``(a, b)`` unchanged for string or byte pairs; otherwise two tuples of
        small integers, one code per distinct element under container
        equality (identity, then ``==``) shared by both sides.

    Notes
    -----
    **Developer.** Accelerated backends compare element *hashes*: the bundled
    kernel always did, and RapidFuzz does for values outside its native integer
    range. CPython collides ``hash(-1) == hash(-2)`` and
    ``hash(0) == hash(2**61 - 1)``, so ``distance([0], [2**61 - 1])`` was 0 on
    the default backend and 1 on the Python one. Coding through one dict makes
    equality exact for every backend, at O(len(a) + len(b)) beside an
    O(len(a) * len(b)) computation. An unhashable element leaves the inputs
    as they are; accelerated backends then raise and the documented fallback to
    the Python implementation applies, unchanged.
    """
    pair = (type(a), type(b))
    if pair in _EXACT_PAIRS or pair[::-1] in _EXACT_PAIRS:
        return a, b
    codes: dict[Any, int] = {}
    try:
        left = tuple(codes.setdefault(item, len(codes)) for item in a)
        right = tuple(codes.setdefault(item, len(codes)) for item in b)
    except TypeError:  # an unhashable element
        return a, b
    return left, right


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


#: Safe automatic preference order. The GPL ``levenshtein`` backend is never
#: added to a plan unless it was requested explicitly.
_SAFE_ORDER = ("internal", "rapidfuzz", "python")


class _ExecutionPlan:
    """
    The backends one top-level operation uses, resolved once (LV-001, LV-002).

    Parameters
    ----------
    selected : str
        The resolved backend, tried first.
    strict : bool
        Raise on a runtime failure instead of falling back.

    Notes
    -----
    **Developer.** ``distance()``, ``rank()`` and ``closest()`` each build one
    plan, so backend capability is resolved once per call, not once per
    candidate. When the current backend raises at runtime (and ``strict`` is
    false), the plan warns once, drops that backend for the rest of the call
    and tries the next available backend of :data:`_SAFE_ORDER`, ending with
    pure Python, which cannot fail for a supported input. Every result still
    names the backend that produced it.
    """

    def __init__(self, selected: str, *, strict: bool) -> None:
        self._queue = [selected]
        self._strict = strict
        self._extended = False

    @classmethod
    def build(cls, backend: str, *, strict: bool) -> _ExecutionPlan:
        """Resolve ``backend`` once (warning once if it is unavailable)."""
        return cls(_resolve_backend_name(backend, strict=strict), strict=strict)

    def _fall_back(self, failed: str, exc: Exception) -> None:
        if not self._extended:
            self._queue.extend(
                name
                for name in _SAFE_ORDER
                if name not in self._queue
                and (name == "python" or _backend_info(name).available)
            )
            self._extended = True
        self._queue.remove(failed)
        logger.warning(
            "Levenshtein backend %r failed at runtime (%s: %s); using %r for the "
            "rest of this call.",
            failed,
            type(exc).__name__,
            exc,
            self._queue[0],
        )

    def distance(self, a: Any, b: Any) -> tuple[int, str]:
        """Return the distance and the backend that produced it."""
        # Materialise iterators once, then hand every backend the same exact form.
        a, b = _exact_codes(_as_sequence(a), _as_sequence(b))
        while True:
            name = self._queue[0]
            if name == "python":
                return _python_distance(a, b), "python"
            try:
                return int(_distance_impl(name)(a, b)), name
            except (KeyboardInterrupt, SystemExit, MemoryError):
                raise
            except Exception as exc:
                if self._strict:
                    raise
                self._fall_back(name, exc)


def _distance_with_backend(
    a: Any,
    b: Any,
    *,
    backend: str = "auto",
    strict: bool = False,
) -> tuple[int, str]:
    """Return distance plus the implementation that actually produced it."""
    return _ExecutionPlan.build(backend, strict=strict).distance(a, b)


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

    The backend is resolved once per call. A backend that fails at runtime
    warns once and is replaced, for the rest of the call, by the next available
    one of ``internal``, ``rapidfuzz``, ``python``; each match records the
    backend that computed it. An unknown or (with ``strict=True``) unavailable
    backend raises even when ``choices`` is empty.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0 or None")
    if not 0.0 <= score_cutoff <= 1.0:
        raise ValueError("score_cutoff must be within [0, 1]")
    query_seq = _as_sequence(query)
    # One plan for the whole call: the backend is resolved once, and a backend
    # that fails at runtime warns once (LV-002, LV-001).
    plan = _ExecutionPlan.build(backend, strict=strict)

    def iter_matches() -> Iterable[Match]:
        for index, choice in enumerate(choices):
            value = key(choice) if key is not None else choice
            value_seq = _as_sequence(value)
            d, used_backend = plan.distance(query_seq, value_seq)
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
