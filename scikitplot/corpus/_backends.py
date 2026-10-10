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
* explicit SUCCESS / EMPTY / DEGRADED / UNAVAILABLE / EXHAUSTED / FAILED status;
* no third-party imports at module import time.

A backend that returned a valid but empty result is different from a backend
that failed.  Callers express that distinction with ``accept`` and ``is_empty``
predicates instead of relying on truthiness.
"""  # noqa: D205, D400

from __future__ import annotations

import dataclasses
import logging
from enum import Enum
from typing import Any, Callable, Generic, Mapping, Sequence, TypeVar

from ._capabilities import (
    CapabilityRegistry,
    capability_report,
    component_capabilities,
)
from ._diagnostics import ErrorCategory, ErrorRecord
from ._schema import ErrorPolicy

__all__ = [
    "BackendCandidate",
    "BackendOutcome",
    "BackendPlan",
    "BackendPolicy",
    "BackendSkip",
    "BackendStatus",
    "backend_policy",
    "plan_backend_chain",
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

    UNAVAILABLE = "unavailable"
    """Policy/readiness left no backend eligible to execute."""

    EXHAUSTED = "exhausted"
    """Backends ran normally but none produced an accepted result."""

    DEGRADED = "degraded"
    """A fallback succeeded after at least one earlier backend failed."""

    FAILED = "failed"
    """No backend completed successfully and at least one backend failed."""


@dataclasses.dataclass(frozen=True)
class BackendPolicy:
    """Policy for selecting and exhausting optional backend chains.

    The policy controls *orchestration*, not backend implementation details.
    Readers continue to decide what counts as a usable result and which
    exceptions are safe to fall back from.

    Parameters
    ----------
    name : str
        Stable policy label used in diagnostics.
    order : tuple[str, ...], optional
        Preferred backend names. Unlisted candidates follow in their declared
        order unless ``include_unlisted`` is false.
    include_unlisted : bool, optional
        Append candidates absent from ``order``. Default: ``True``.
    allow_fallback : bool, optional
        When false, only the first selected candidate is attempted.
    require_ready : bool, optional
        Skip candidates whose registered capability is not definitely ready.
        This is useful for deterministic/offline runs; the default resilient
        policy may attempt components whose model readiness is unknowable until
        first use.
    allow_network : bool, optional
        Permit candidates declaring a network requirement.
    allow_download : bool, optional
        Permit candidates that may download assets on first use. Backends that
        declare ``offline_capable=True`` may still run when downloads are
        forbidden, but they must enforce a local-only execution mode.
    on_exhausted : ErrorPolicy, optional
        Public operation behavior after every backend fails.  The generic runner
        records the policy but does not raise; readers apply it because only the
        reader knows the correct exception/message contract.
    """

    name: str = "resilient"
    order: tuple[str, ...] = ()
    include_unlisted: bool = True
    allow_fallback: bool = True
    require_ready: bool = False
    allow_network: bool = True
    allow_download: bool = True
    on_exhausted: ErrorPolicy = ErrorPolicy.COLLECT

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("BackendPolicy.name must be a non-empty string")
        if isinstance(self.order, str):
            raise TypeError("BackendPolicy.order must be a sequence, not str")
        if not isinstance(self.order, tuple):
            try:
                object.__setattr__(self, "order", tuple(self.order))
            except TypeError as exc:
                raise TypeError(
                    "BackendPolicy.order must be a sequence of backend names"
                ) from exc
        if any(not isinstance(name, str) or not name.strip() for name in self.order):
            raise ValueError("BackendPolicy.order entries must be non-empty strings")
        if len(set(self.order)) != len(self.order):
            raise ValueError("BackendPolicy.order must not contain duplicate names")
        for field_name in (
            "include_unlisted",
            "allow_fallback",
            "require_ready",
            "allow_network",
            "allow_download",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"BackendPolicy.{field_name} must be bool")
        if not isinstance(self.on_exhausted, ErrorPolicy):
            try:
                object.__setattr__(
                    self, "on_exhausted", ErrorPolicy(str(self.on_exhausted))
                )
            except ValueError as exc:
                raise ValueError(
                    "BackendPolicy.on_exhausted must be a valid ErrorPolicy value"
                ) from exc

    @classmethod
    def resilient(cls) -> BackendPolicy:
        """Fallback through declared candidates and collect final failure."""
        return cls()

    @classmethod
    def strict(cls) -> BackendPolicy:
        """Fallback through candidates, then require the caller to raise."""
        return cls(name="strict", on_exhausted=ErrorPolicy.RAISE)

    @classmethod
    def offline(cls) -> BackendPolicy:
        """Use only local components and forbid network/download side effects.

        Readiness may be ``UNKNOWN`` for model-backed engines whose cache can
        only be proven by attempting a local-only load. Such engines are
        eligible only when their candidate declares ``offline_capable=True``.
        """
        return cls(
            name="offline",
            require_ready=False,
            allow_network=False,
            allow_download=False,
            on_exhausted=ErrorPolicy.COLLECT,
        )

    @classmethod
    def first_available(cls) -> BackendPolicy:
        """Attempt only the first selected candidate; do not cascade."""
        return cls(name="first-available", allow_fallback=False)

    def with_order(
        self,
        *names: str,
        include_unlisted: bool | None = None,
    ) -> BackendPolicy:
        """Return a copy with explicit backend preference order."""
        return dataclasses.replace(
            self,
            order=tuple(names),
            include_unlisted=(
                self.include_unlisted if include_unlisted is None else include_unlisted
            ),
        )

    @property
    def raise_on_exhausted(self) -> bool:
        """Whether the operation should raise after chain exhaustion."""
        return self.on_exhausted is ErrorPolicy.RAISE

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible policy representation."""
        return {
            "name": self.name,
            "order": list(self.order),
            "include_unlisted": self.include_unlisted,
            "allow_fallback": self.allow_fallback,
            "require_ready": self.require_ready,
            "allow_network": self.allow_network,
            "allow_download": self.allow_download,
            "on_exhausted": self.on_exhausted.value,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> BackendPolicy:
        """Build a policy from JSON/YAML-shaped configuration.

        The optional ``preset`` key selects ``resilient`` (default), ``strict``,
        ``offline`` or ``first`` first; remaining fields explicitly override the
        preset. Unknown fields are rejected so misspelled policy controls never
        look applied while doing nothing.
        """
        if not isinstance(config, Mapping):
            raise TypeError("BackendPolicy config must be a mapping")
        data = dict(config)
        preset = data.pop("preset", "resilient")
        base = backend_policy(str(preset))
        allowed = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(
                f"unknown BackendPolicy config field(s) {unknown}; "
                f"expected subset of {sorted(allowed)} plus 'preset'"
            )
        if "order" in data:
            raw_order = data["order"]
            if isinstance(raw_order, str):
                data["order"] = (raw_order,)
            else:
                data["order"] = tuple(raw_order)
        if "on_exhausted" in data and not isinstance(data["on_exhausted"], ErrorPolicy):
            try:
                data["on_exhausted"] = ErrorPolicy(str(data["on_exhausted"]))
            except ValueError as exc:
                raise ValueError(
                    "BackendPolicy.on_exhausted must be a valid ErrorPolicy value"
                ) from exc
        return dataclasses.replace(base, **data)


_POLICY_PRESETS: Mapping[str, Callable[[], BackendPolicy]] = {
    "resilient": BackendPolicy.resilient,
    "default": BackendPolicy.resilient,
    "strict": BackendPolicy.strict,
    "offline": BackendPolicy.offline,
    "first": BackendPolicy.first_available,
    "first-available": BackendPolicy.first_available,
}


def backend_policy(
    value: BackendPolicy | str | Mapping[str, Any] | None,
) -> BackendPolicy:
    """Resolve an object, named preset or mapping without hidden global state."""
    if value is None:
        return BackendPolicy.resilient()
    if isinstance(value, BackendPolicy):
        return value
    if isinstance(value, Mapping):
        return BackendPolicy.from_config(value)
    key = str(value).strip().lower()
    try:
        return _POLICY_PRESETS[key]()
    except KeyError as exc:
        raise ValueError(
            f"unknown backend policy {value!r}; expected one of {sorted(_POLICY_PRESETS)}"
        ) from exc


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
    capability : str or None, optional
        Capability-registry key used for readiness preflight and provenance.
    requires_network : bool, optional
        Whether normal execution requires network access.
    may_download : bool, optional
        Whether first use may need to download assets.  This is a risk/property
        flag, not proof that a download is needed for the current request.
    offline_capable : bool, optional
        Whether the candidate can *enforce* local-only execution when policy
        forbids downloads/network.  Such a candidate may be attempted when
        asset readiness is ``UNKNOWN``; its runner must then select a local-only
        mode rather than silently contacting the network.
    """

    name: str
    run: Callable[[], _T]
    accept: Callable[[_T], bool] = _accept_any
    fallback_exceptions: tuple[type[BaseException], ...] = (Exception,)
    failure_level: int = logging.WARNING
    miss_level: int = logging.DEBUG
    degrades_on_failure: bool = True
    capability: str | None = None
    requires_network: bool = False
    may_download: bool = False
    offline_capable: bool = False


@dataclasses.dataclass(frozen=True)
class BackendSkip:
    """Why a backend was not executed under the resolved policy.

    The record is deliberately small and JSON-compatible.  It contains no live
    exception or heavyweight capability object, so reader diagnostics can keep
    it safely after the operation completes.
    """

    backend: str
    reason: str
    capability: str | None = None
    capability_status: str | None = None
    ready: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "backend": self.backend,
            "reason": self.reason,
            "capability": self.capability,
            "capability_status": self.capability_status,
            "ready": self.ready,
        }


@dataclasses.dataclass(frozen=True)
class BackendPlan:
    """Side-effect-free preflight of candidate ordering and policy eligibility.

    ``BackendPlan`` is the discovery counterpart to :class:`BackendOutcome`.
    It executes no backend callable and performs only lightweight capability
    probes.  The runtime runner consumes this exact plan-building logic, so
    preflight and execution cannot silently diverge in ordering or policy.
    """

    policy: str
    declared: tuple[str, ...]
    ordered: tuple[str, ...]
    eligible: tuple[str, ...]
    selected: tuple[str, ...]
    skipped: tuple[str, ...] = ()
    selected_capabilities: tuple[str, ...] = ()
    skip_details: tuple[BackendSkip, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "policy": self.policy,
            "declared": list(self.declared),
            "ordered": list(self.ordered),
            "eligible": list(self.eligible),
            "selected": list(self.selected),
            "skipped": list(self.skipped),
            "selected_capabilities": list(self.selected_capabilities),
            "skip_details": [record.to_dict() for record in self.skip_details],
        }

    def capability_view(
        self, *, registry: CapabilityRegistry | None = None
    ) -> dict[str, dict[str, Any]]:
        """Return readiness records with this plan's selection marked."""
        names = set(self.selected_capabilities)
        names.update(
            detail.capability
            for detail in self.skip_details
            if detail.capability is not None
        )
        if not names:
            return {}
        return component_capabilities(
            sorted(names),
            selected=self.selected_capabilities,
            registry=registry,
        )


def plan_backend_chain(  # ruff: ignore[too-many-branches]
    candidates: Sequence[BackendCandidate[Any]],
    *,
    policy: BackendPolicy | str | Mapping[str, Any] | None = None,
    capability_registry: CapabilityRegistry | None = None,
) -> BackendPlan:
    """Resolve backend order/readiness without executing a backend.

    This is suitable for diagnostics, galleries, CI preflight and user-facing
    explain tooling.  Capability probes must remain side-effect-free; unknown
    model-cache state therefore stays ``UNKNOWN`` instead of triggering a
    download merely to answer a readiness question.
    """
    resolved_policy = backend_policy(policy)
    declared = list(candidates)
    by_name = {candidate.name: candidate for candidate in declared}
    if len(by_name) != len(declared):
        duplicates = sorted(
            name for name in by_name if sum(c.name == name for c in declared) > 1
        )
        raise ValueError(
            f"backend candidate names must be unique; duplicates={duplicates}"
        )

    unknown_order = [name for name in resolved_policy.order if name not in by_name]
    if unknown_order:
        raise ValueError(
            f"backend policy {resolved_policy.name!r} references unknown candidates "
            f"{unknown_order}; available={list(by_name)}"
        )

    ordered: list[BackendCandidate[Any]] = []
    seen: set[str] = set()
    for name in resolved_policy.order:
        ordered.append(by_name[name])
        seen.add(name)
    if resolved_policy.include_unlisted:
        ordered.extend(c for c in declared if c.name not in seen)
    elif not resolved_policy.order:
        ordered = declared

    skipped: list[str] = []
    skip_details: list[BackendSkip] = []
    eligible: list[BackendCandidate[Any]] = []
    for candidate in ordered:
        reason: str | None = None
        report = None
        needs_capability_probe = candidate.capability is not None and (
            resolved_policy.require_ready
            or (candidate.may_download and not resolved_policy.allow_download)
        )
        if needs_capability_probe:
            try:
                report = capability_report(
                    candidate.capability,
                    selected=True,
                    registry=capability_registry,
                )
            except KeyError:
                if resolved_policy.require_ready:
                    reason = "capability is not registered"

        if (
            reason is None
            and candidate.requires_network
            and not resolved_policy.allow_network
        ):
            reason = "network disabled by backend policy"
        elif (
            reason is None
            and candidate.may_download
            and not resolved_policy.allow_download
        ):
            if not candidate.offline_capable:
                reason = (
                    "asset download disabled by backend policy and backend "
                    "cannot enforce local-only execution"
                )
            elif report is not None and report.installed is False:
                reason = (
                    f"capability is not installed "
                    f"({report.status.value}:{report.reason_code})"
                )
            elif report is not None and report.ready is False:
                reason = (
                    f"required local asset is not ready "
                    f"({report.status.value}:{report.reason_code})"
                )
            # ready=None is allowed for an offline-capable backend. The backend
            # itself must enforce a local-only attempt, turning cache miss into
            # an ordinary runtime failure rather than network activity.
        elif (
            reason is None
            and resolved_policy.require_ready
            and candidate.capability is not None
        ):
            if report is None:
                reason = "capability is not registered"
            elif report.ready is not True:
                reason = (
                    f"capability readiness is {report.ready!r} "
                    f"({report.status.value}:{report.reason_code})"
                )

        if reason is not None:
            skipped.append(candidate.name)
            skip_details.append(
                BackendSkip(
                    backend=candidate.name,
                    reason=reason,
                    capability=candidate.capability,
                    capability_status=(
                        report.status.value if report is not None else None
                    ),
                    ready=(report.ready if report is not None else None),
                )
            )
            continue
        eligible.append(candidate)

    selected = (
        eligible[:1] if (not resolved_policy.allow_fallback and eligible) else eligible
    )
    selected_capabilities = tuple(
        candidate.capability
        for candidate in selected
        if candidate.capability is not None
    )
    return BackendPlan(
        policy=resolved_policy.name,
        declared=tuple(candidate.name for candidate in declared),
        ordered=tuple(candidate.name for candidate in ordered),
        eligible=tuple(candidate.name for candidate in eligible),
        selected=tuple(candidate.name for candidate in selected),
        skipped=tuple(skipped),
        selected_capabilities=selected_capabilities,
        skip_details=tuple(skip_details),
    )


@dataclasses.dataclass(frozen=True)
class BackendOutcome(Generic[_T]):
    """Structured, serialisable outcome from :func:`run_backend_chain`."""

    status: BackendStatus
    value: _T
    backend: str | None
    attempted: tuple[str, ...]
    missed: tuple[str, ...]
    errors: tuple[ErrorRecord, ...]
    skipped: tuple[str, ...] = ()
    policy: str = "resilient"
    selected_capabilities: tuple[str, ...] = ()
    active_capability: str | None = None
    skip_details: tuple[BackendSkip, ...] = ()

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
            "skipped": list(self.skipped),
            "policy": self.policy,
            "selected_capabilities": list(self.selected_capabilities),
            "active_capability": self.active_capability,
            "skip_details": [record.to_dict() for record in self.skip_details],
            "errors": [record.to_dict() for record in self.errors],
        }

    def capability_view(
        self, *, registry: CapabilityRegistry | None = None
    ) -> dict[str, dict[str, Any]]:
        """Return readiness records with selected/active runtime provenance."""
        names = set(self.selected_capabilities)
        names.update(
            detail.capability
            for detail in self.skip_details
            if detail.capability is not None
        )
        if self.active_capability is not None:
            names.add(self.active_capability)
        if not names:
            return {}
        active = (self.active_capability,) if self.active_capability else ()
        return component_capabilities(
            sorted(names),
            selected=self.selected_capabilities,
            active=active,
            registry=registry,
        )


def run_backend_chain(  # ruff: ignore[undocumented-param]
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
    policy: BackendPolicy | str | Mapping[str, Any] | None = None,
    capability_registry: CapabilityRegistry | None = None,
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
    resolved_policy = backend_policy(policy)
    plan = plan_backend_chain(
        candidates,
        policy=resolved_policy,
        capability_registry=capability_registry,
    )
    by_name = {candidate.name: candidate for candidate in candidates}
    eligible = [by_name[name] for name in plan.selected]
    skipped = list(plan.skipped)
    skip_details = list(plan.skip_details)
    selected_capabilities = plan.selected_capabilities

    attempted: list[str] = []
    missed: list[str] = []
    errors: list[ErrorRecord] = []
    degrading_failure = False
    total = len(eligible)

    for index, candidate in enumerate(eligible):
        attempted.append(candidate.name)
        next_name = eligible[index + 1].name if index + 1 < total else None
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
            skipped=tuple(skipped),
            policy=resolved_policy.name,
            selected_capabilities=selected_capabilities,
            active_capability=candidate.capability,
            skip_details=tuple(skip_details),
        )

    if errors:
        final_status = BackendStatus.FAILED
    elif attempted or missed:
        # At least one backend ran, but none returned an accepted value. This
        # is materially different from a successful backend returning a valid
        # empty value (handled inside the loop above).
        final_status = BackendStatus.EXHAUSTED
    else:
        # No backend could run: policy/readiness/capability selection exhausted
        # the chain before execution. Fail-soft callers may still return the
        # supplied default, but the report must not claim success/emptiness.
        final_status = BackendStatus.UNAVAILABLE

    return BackendOutcome(
        status=final_status,
        value=default,
        backend=None,
        attempted=tuple(attempted),
        missed=tuple(missed),
        errors=tuple(errors),
        skipped=tuple(skipped),
        policy=resolved_policy.name,
        selected_capabilities=selected_capabilities,
        active_capability=None,
        skip_details=tuple(skip_details),
    )
