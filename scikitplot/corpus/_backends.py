# scikitplot/corpus/_backends.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Ordered optional-backend orchestration for :mod:`scikitplot.corpus`.

The reader layer contains several operations that may be served by more than
one optional implementation: speech recognition, PDF text extraction, XML
parsing, duration probing, OCR, and future user-provided engines.  The details
of each backend remain owned by the reader that understands the format; this
module centralises only the *orchestration contract*:

* ordered attempts;
* per-backend fallback exception boundaries;
* consistent logging;
* serialisable failure diagnostics;
* explicit SUCCESS / EMPTY / DEGRADED / FAILED status;
* no third-party imports at module import time.

A backend that returned a valid but empty result is different from a backend
that failed.  Callers express that distinction with ``accept`` and ``is_empty``
predicates instead of relying on truthiness.
"""  # noqa: D205, D400

from __future__ import annotations

import dataclasses
import logging
from enum import Enum
from typing import Any, Callable, Generic, Sequence, TypeVar

from ._diagnostics import ErrorCategory, ErrorRecord

__all__ = [
    "BackendCandidate",
    "BackendOutcome",
    "BackendStatus",
    "run_backend_chain",
]

_T = TypeVar("_T")


def _accept_any(_value: Any) -> bool:
    return True


def _never_empty(_value: Any) -> bool:
    return False


def _summary(exc: BaseException) -> str:
    """Return a compact one-line exception description for logs."""
    message = " ".join(str(exc).split())
    return message or repr(exc)


def _log(logger: logging.Logger, level: int, message: str, *args: Any) -> None:
    """Use named logger methods for common levels, preserving test hooks."""
    if level == logging.WARNING:
        logger.warning(message, *args)
    elif level == logging.DEBUG:
        logger.debug(message, *args)
    elif level == logging.INFO:
        logger.info(message, *args)
    elif level == logging.ERROR:
        logger.error(message, *args)
    else:
        logger.log(level, message, *args)


class BackendStatus(str, Enum):
    """Outcome of one ordered backend operation."""

    SUCCESS = "success"
    """A backend completed and produced a non-empty accepted result."""

    EMPTY = "empty"
    """A backend completed successfully and produced a valid empty result."""

    DEGRADED = "degraded"
    """A fallback succeeded after at least one earlier backend failed."""

    FAILED = "failed"
    """No backend completed successfully and at least one backend failed."""


@dataclasses.dataclass(frozen=True)
class BackendCandidate(Generic[_T]):
    """One lazy backend attempt.

    Parameters
    ----------
    name : str
        Stable backend identifier used in logs and diagnostics.
    run : callable
        Zero-argument callable that performs the backend work.  Third-party
        imports should remain inside this callable.
    accept : callable, optional
        Predicate deciding whether a returned value is usable.  Returning
        ``False`` is a *miss*, not a failure, and advances to the next backend.
        The default accepts every returned value, including ``None`` and empty
        containers.
    fallback_exceptions : tuple of exception classes, optional
        Exceptions that mean "this backend could not serve the request; try the
        next one".  Exceptions outside this tuple propagate immediately.
        Defaults to ordinary :class:`Exception`, intentionally excluding
        ``KeyboardInterrupt`` and ``SystemExit``.
    failure_level : int, optional
        Logging level for backend exceptions.  Default: ``logging.WARNING``.
    miss_level : int, optional
        Logging level when ``accept`` rejects a normal return value.  Default:
        ``logging.DEBUG``.
    degrades_on_failure : bool, optional
        Whether failure of this preferred backend makes a later successful
        fallback ``DEGRADED``.  Set ``False`` when absence/failure is an
        expected capability probe and the fallback is semantically equivalent.
    """

    name: str
    run: Callable[[], _T]
    accept: Callable[[_T], bool] = _accept_any
    fallback_exceptions: tuple[type[BaseException], ...] = (Exception,)
    failure_level: int = logging.WARNING
    miss_level: int = logging.DEBUG
    degrades_on_failure: bool = True


@dataclasses.dataclass(frozen=True)
class BackendOutcome(Generic[_T]):
    """Structured, serialisable outcome from :func:`run_backend_chain`."""

    status: BackendStatus
    value: _T
    backend: str | None
    attempted: tuple[str, ...]
    missed: tuple[str, ...]
    errors: tuple[ErrorRecord, ...]

    @property
    def succeeded(self) -> bool:
        """Whether some backend completed successfully."""
        return self.status in (
            BackendStatus.SUCCESS,
            BackendStatus.EMPTY,
            BackendStatus.DEGRADED,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation without live exceptions."""
        return {
            "status": self.status.value,
            "backend": self.backend,
            "attempted": list(self.attempted),
            "missed": list(self.missed),
            "errors": [record.to_dict() for record in self.errors],
        }


def run_backend_chain(
    candidates: Sequence[BackendCandidate[_T]],
    *,
    default: _T,
    logger: logging.Logger,
    component: str,
    operation: str,
    subject: str | None = None,
    error_code: str = "OPTIONAL_BACKEND_FAILED",
    error_category: ErrorCategory | str = ErrorCategory.CAPABILITY,
    stage: str | None = None,
    is_empty: Callable[[_T], bool] = _never_empty,
    passthrough_exceptions: tuple[type[BaseException], ...] = (MemoryError,),
) -> BackendOutcome[_T]:
    """Try *candidates* in order and return a structured outcome.

    This function deliberately does **not** decide whether a failed chain should
    raise.  That is operation policy: ASR may be fail-soft by default and
    strict on request, while a parser may always raise.  Callers inspect the
    returned status/errors and apply their own public contract.

    Parameters
    ----------
    candidates : sequence of BackendCandidate
        Ordered backend attempts.
    default : object
        Value returned when no backend produces an accepted result.
    logger : logging.Logger
        Logger receiving per-backend diagnostics.
    component : str
        Human-facing owner, e.g. ``"AudioReader"``.
    operation : str
        Operation label, e.g. ``"transcription"``.
    subject : str or None, optional
        Filename or other specific subject for diagnostics.
    error_code : str, optional
        Stable :class:`ErrorRecord` code used for backend exceptions.
    error_category : ErrorCategory or str, optional
        Diagnostic category.  Defaults to ``CAPABILITY``.
    stage : str or None, optional
        Pipeline stage stored in structured errors.
    is_empty : callable, optional
        Predicate identifying a successful but empty accepted value.
    passthrough_exceptions : tuple, optional
        Exceptions never converted into fallback diagnostics.  Defaults to
        :class:`MemoryError`.

    Returns
    -------
    BackendOutcome
        Safe result plus backend provenance and structured errors.  No live
        exception objects are retained.
    """
    attempted: list[str] = []
    missed: list[str] = []
    errors: list[ErrorRecord] = []
    degrading_failure = False
    total = len(candidates)

    for index, candidate in enumerate(candidates):
        attempted.append(candidate.name)
        next_name = candidates[index + 1].name if index + 1 < total else None
        try:
            value = candidate.run()
        except passthrough_exceptions:
            raise
        except candidate.fallback_exceptions as exc:
            if candidate.degrades_on_failure:
                degrading_failure = True
            errors.append(
                ErrorRecord.from_exception(
                    exc,
                    code=error_code,
                    category=error_category,
                    stage=stage,
                    source_id=subject,
                    details={
                        "component": component,
                        "operation": operation,
                        "backend": candidate.name,
                        "degrades": candidate.degrades_on_failure,
                        "is_import_error": isinstance(exc, ImportError),
                    },
                )
            )
            suffix = (
                f"trying backend {next_name!r}."
                if next_name is not None
                else "no backend remains."
            )
            _log(
                logger,
                candidate.failure_level,
                "%s: backend %r failed during %s%s (%s: %s); %s",
                component,
                candidate.name,
                operation,
                f" for {subject}" if subject else "",
                type(exc).__name__,
                _summary(exc),
                suffix,
            )
            continue

        if not candidate.accept(value):
            missed.append(candidate.name)
            suffix = (
                f"trying backend {next_name!r}."
                if next_name is not None
                else "no backend remains."
            )
            _log(
                logger,
                candidate.miss_level,
                "%s: backend %r returned no usable result during %s%s; %s",
                component,
                candidate.name,
                operation,
                f" for {subject}" if subject else "",
                suffix,
            )
            continue

        if degrading_failure:
            status = BackendStatus.DEGRADED
        elif is_empty(value):
            status = BackendStatus.EMPTY
        else:
            status = BackendStatus.SUCCESS
        return BackendOutcome(
            status=status,
            value=value,
            backend=candidate.name,
            attempted=tuple(attempted),
            missed=tuple(missed),
            errors=tuple(errors),
        )

    return BackendOutcome(
        status=BackendStatus.FAILED if errors else BackendStatus.EMPTY,
        value=default,
        backend=None,
        attempted=tuple(attempted),
        missed=tuple(missed),
        errors=tuple(errors),
    )
