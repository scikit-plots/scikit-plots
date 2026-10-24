# scikitplot/feature_selection/_assessments.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Policy-backed feature assessments: "am I skewed / discrete / constant ...?".

An *assessment* is a question with a name (``"skewed"``), answered for one
feature by a *provider* (a function with documented parameters) and recorded
as an :class:`Assessment`: a four-valued state, reason codes, evidence, and
the provider name, version and parameters that produced it.

Notes
-----
**User note.** Thresholds live in the policy, not in code::

    FeaturePolicy(
        assessments={"skewed": {"provider": "skewed", "params": {"threshold": 0.5}}}
    )

Add a project-specific question by registering a provider once, then naming
it in any policy::

    @register_assessment("looks_like_percentage", defaults={"tolerance": 0.0})
    def _pct(context, tolerance):
        s = context.stats
        if not s.is_numeric:
            return context.result("not_applicable", "not_numeric")
        ok = s.min is not None and s.min >= -tolerance and s.max <= 100 + tolerance
        return context.result("true" if ok else "false", min=s.min, max=s.max)

**Developer note.** One shared :data:`default_registry` serves every entry
point; built-ins are inserted only where a name is still free, so a user's
registration made before first use is never displaced (Rule 116). Providers
return :class:`AssessmentOutcome`; :func:`run_assessment` stamps provenance
and refuses anything else, so a provider cannot forge its own name or
version.
"""

from __future__ import annotations

import dataclasses
import numbers
import threading
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from ._errors import PolicyError
from ._policy import AssessmentSpec, _param_value
from ._stats import FeatureStats, numeric_values
from ._types import AssessmentState, DtypeFamily, FeatureKind, coerce_enum

__all__ = [
    "Assessment",
    "AssessmentContext",
    "AssessmentOutcome",
    "AssessmentProvider",
    "AssessmentRegistry",
    "default_registry",
    "list_assessment_providers",
    "register_assessment",
]

_EVIDENCE_TYPES = (str, int, float, bool, type(None))
_CATEGORICAL_KINDS = (
    FeatureKind.BINARY.value,
    FeatureKind.CATEGORICAL_NOMINAL.value,
    FeatureKind.CATEGORICAL_ORDINAL.value,
)


def _freeze_defaults(defaults: Mapping[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    """Validate provider defaults; unlike policy params, ``None`` is kept as a default."""
    out = []
    for key, value in (defaults or {}).items():
        if not isinstance(key, str) or not key:
            raise TypeError(f"parameter names must be non-empty strings, got {key!r}")
        out.append((key, _param_value(f"defaults.{key}", value)))
    return tuple(sorted(out))


def _clean_evidence(evidence: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    out = []
    for key, value in evidence.items():
        if not isinstance(key, str):
            raise TypeError(f"evidence keys must be strings, got {key!r}")
        if isinstance(value, np.generic):
            value = value.item()  # noqa: PLW2901
        if isinstance(value, float) and value != value:  # noqa: PLR0124
            value = None  # noqa: PLW2901
        if not isinstance(value, _EVIDENCE_TYPES):
            raise TypeError(
                f"evidence {key!r} must be a str/int/float/bool/None, got"
                f" {type(value).__name__}"
            )
        if isinstance(value, float) and value in (float("inf"), float("-inf")):
            value = None  # noqa: PLW2901
        if value is None:  # unavailable evidence is omitted (TOML has no null)
            continue
        out.append((key, value))
    return tuple(sorted(out))


@dataclasses.dataclass(frozen=True)
class AssessmentOutcome:
    """
    What a provider returns: a state, reason codes and evidence.

    Parameters
    ----------
    state : str
        An :class:`~scikitplot.feature_selection.AssessmentState` value.
    reasons : tuple of str, default=()
        Machine-readable reason codes (``"abs_skewness_ge_threshold"``).
    evidence : tuple of (str, scalar) pairs, default=()
        Values the decision rests on (``skewness=2.31``).
    """

    state: str
    reasons: tuple[str, ...] = ()
    evidence: tuple[tuple[str, Any], ...] = ()


@dataclasses.dataclass(frozen=True)
class Assessment:
    """
    The recorded answer to one assessment for one feature.

    Parameters
    ----------
    name : str
        Assessment name, as in the policy (``"skewed"``).
    state : str
        Final :class:`~scikitplot.feature_selection.AssessmentState` value.
    provider : str
        Provider that answered.
    version : str
        Provider version.
    params : tuple of (str, value) pairs
        Parameters used.
    reasons : tuple of str
        Reason codes.
    evidence : tuple of (str, scalar) pairs
        Evidence values.
    origin : {"inferred", "manual"}, default="inferred"
        ``"manual"`` when a policy override set :attr:`state`.
    inferred_state : str or None, default=None
        For a manual override, the state the provider inferred; kept so the
        override never erases evidence.

    See Also
    --------
    FeatureSpec.assess : Read an assessment of a fitted feature.

    Examples
    --------
    >>> import pandas as pd
    >>> from scikitplot.feature_selection import profile_features
    >>> schema = profile_features(pd.DataFrame({"x": [1.0, 1.0, 2.0, 50.0] * 5}))
    >>> a = schema["x"].assess("skewed")
    >>> a.state, a.reasons
    ('true', ('abs_skewness_ge_threshold',))
    """

    name: str
    state: str
    provider: str
    version: str
    params: tuple[tuple[str, Any], ...] = ()
    reasons: tuple[str, ...] = ()
    evidence: tuple[tuple[str, Any], ...] = ()
    origin: str = "inferred"
    inferred_state: str | None = None

    @property
    def is_true(self) -> bool:
        """bool: The state is ``"true"`` (``unknown`` is not true)."""
        return self.state == AssessmentState.TRUE.value

    @property
    def evidence_dict(self) -> dict[str, Any]:
        """dict: Evidence as a mapping."""
        return dict(self.evidence)

    def to_dict(self) -> dict[str, Any]:
        """Return the shared-data-model form."""
        return {
            "name": self.name,
            "state": self.state,
            "provider": self.provider,
            "version": self.version,
            "params": {
                k: list(v) if isinstance(v, tuple) else v for k, v in self.params
            },
            "reasons": list(self.reasons),
            "evidence": dict(self.evidence),
            "origin": self.origin,
            "inferred_state": self.inferred_state,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Assessment:
        """
        Rebuild an assessment from :meth:`to_dict` output, checking every field.

        Parameters
        ----------
        data : mapping
            Stored form.

        Returns
        -------
        Assessment
            The assessment.

        Raises
        ------
        ValueError
            If a field is unknown, missing or wrongly typed.
        """
        allowed = {f.name for f in dataclasses.fields(cls)}
        if set(data) - allowed or not {"name", "state", "provider", "version"} <= set(
            data
        ):
            raise ValueError(f"off-contract assessment record: keys {sorted(data)}")
        for key in ("name", "state", "provider", "version", "origin"):
            if key in data and not isinstance(data[key], str):
                raise ValueError(f"assessment {key!r} must be a string")
        AssessmentState(data["state"])
        if data.get("inferred_state") is not None:
            AssessmentState(data["inferred_state"])
        if data.get("origin", "inferred") not in ("inferred", "manual"):
            raise ValueError("assessment origin must be 'inferred' or 'manual'")
        reasons = data.get("reasons", [])
        if not isinstance(reasons, list) or not all(
            isinstance(r, str) for r in reasons
        ):
            raise ValueError("assessment reasons must be a list of strings")
        evidence = data.get("evidence", {})
        params = data.get("params", {})
        if not isinstance(evidence, Mapping) or not isinstance(params, Mapping):
            raise ValueError(  # ruff: ignore[type-check-without-type-error]
                "assessment evidence and params must be mappings",
            )
        for key, value in params.items():
            items = value if isinstance(value, list) else [value]
            if not isinstance(key, str) or not all(
                isinstance(v, (str, int, float, bool)) for v in items
            ):
                raise ValueError(
                    f"assessment param {key!r} must be a scalar or a list of scalars"
                )
        return cls(
            name=data["name"],
            state=data["state"],
            provider=data["provider"],
            version=data["version"],
            params=tuple(
                sorted(
                    (k, tuple(v) if isinstance(v, list) else v)
                    for k, v in params.items()
                )
            ),
            reasons=tuple(reasons),
            evidence=_clean_evidence(evidence),
            origin=data.get("origin", "inferred"),
            inferred_state=data.get("inferred_state"),
        )


@dataclasses.dataclass(frozen=True)
class AssessmentContext:
    """
    What a provider may read about one feature.

    Parameters
    ----------
    name : str
        Column name.
    stats : FeatureStats
        Training statistics.
    series : pandas.Series or None
        The training column; None when re-assessing from stored stats.
    declared_kind : str or None
        Kind declared in the policy, if any.
    """

    name: str
    stats: FeatureStats
    series: pd.Series | None = None
    declared_kind: str | None = None

    @property
    def family(self) -> DtypeFamily:
        """DtypeFamily: Physical family of the column."""
        return DtypeFamily(self.stats.family)

    def result(self, state: Any, *reasons: str, **evidence: Any) -> AssessmentOutcome:
        """
        Build an :class:`AssessmentOutcome`.

        Parameters
        ----------
        state : str or AssessmentState
            The answer.
        *reasons : str
            Reason codes.
        **evidence : scalar
            Evidence values.

        Returns
        -------
        AssessmentOutcome
            The outcome.
        """
        member = coerce_enum(state, AssessmentState, field="state")
        for reason in reasons:
            if not isinstance(reason, str):
                raise TypeError(f"reason codes must be strings, got {reason!r}")
        return AssessmentOutcome(
            member.value, tuple(reasons), _clean_evidence(evidence)
        )


@dataclasses.dataclass(frozen=True)
class AssessmentProvider:
    """
    A named, versioned function that answers one kind of question.

    Parameters
    ----------
    name : str
        Registry name.
    func : callable
        ``func(context, **params) -> AssessmentOutcome``.
    defaults : tuple of (str, value) pairs
        Every accepted parameter with its default; a value's type is the
        parameter's type (``None`` accepts any scalar).
    version : str, default="1"
        Bumped when the rule's meaning changes.
    description : str, default=""
        One line for listings.
    validate : callable or None, default=None
        ``validate(params)`` raising :class:`PolicyError` for out-of-range
        values; called once per profile, before any column is assessed.
    """

    name: str
    func: Callable[..., AssessmentOutcome]
    defaults: tuple[tuple[str, Any], ...] = ()
    version: str = "1"
    description: str = ""
    validate: Callable[[Mapping[str, Any]], None] | None = None

    def resolve_params(self, given: Mapping[str, Any], *, field: str) -> dict[str, Any]:
        """
        Merge ``given`` over the defaults, checking names and types.

        Parameters
        ----------
        given : mapping
            Parameters from the policy.
        field : str
            Name used in error messages.

        Returns
        -------
        dict
            Complete parameters.

        Raises
        ------
        PolicyError
            For an unknown name, a wrong type or (through ``validate``) a bad
            range.
        """
        defaults = dict(self.defaults)
        unknown = sorted(set(given) - set(defaults))
        if unknown:
            raise PolicyError(
                f"{field}: provider {self.name!r} has no parameter(s) {unknown};"
                f" accepted: {sorted(defaults)}"
            )
        params = dict(defaults)
        for key, value in given.items():
            default = defaults[key]
            if isinstance(default, bool):
                ok = isinstance(value, bool)
            elif isinstance(default, int):
                ok = isinstance(value, numbers.Integral) and not isinstance(value, bool)
            elif isinstance(default, float):
                ok = isinstance(value, numbers.Real) and not isinstance(value, bool)
                value = float(value) if ok else value  # noqa: PLW2901
            elif isinstance(default, str):
                ok = isinstance(value, str)
            elif isinstance(default, tuple):
                ok = isinstance(value, tuple)
            else:  # default None: any scalar
                ok = True
            if not ok:
                raise PolicyError(
                    f"{field}.params.{key} must be {type(default).__name__}, got"
                    f" {type(value).__name__}: {value!r}"
                )
            params[key] = value
        if self.validate is not None:
            try:
                self.validate(params)
            except PolicyError as exc:
                raise PolicyError(f"{field}: {exc}") from None
        return params


class AssessmentRegistry:
    """
    Name table of assessment providers.

    Notes
    -----
    **Developer note.** Thread-safe. :meth:`register` refuses to replace an
    existing name unless ``replace=True``; :meth:`_register_builtin` inserts
    only absent names.
    """

    def __init__(self) -> None:  # noqa: D107
        self._lock = threading.Lock()
        self._providers: dict[str, AssessmentProvider] = {}

    def register(self, provider: AssessmentProvider, *, replace: bool = False) -> None:
        """
        Add ``provider`` under its name.

        Parameters
        ----------
        provider : AssessmentProvider
            The provider.
        replace : bool, default=False
            Allow replacing an existing entry.

        Raises
        ------
        ValueError
            If the name is taken and ``replace`` is False.
        """
        if not isinstance(provider, AssessmentProvider):
            raise TypeError("provider must be an AssessmentProvider")
        with self._lock:
            if provider.name in self._providers and not replace:
                raise ValueError(
                    f"an assessment provider named {provider.name!r} is already"
                    " registered; pass replace=True to override it"
                )
            self._providers[provider.name] = provider

    def _register_builtin(self, provider: AssessmentProvider) -> None:
        with self._lock:
            self._providers.setdefault(provider.name, provider)

    def get(self, name: str) -> AssessmentProvider:
        """
        Return the provider registered as ``name``.

        Parameters
        ----------
        name : str
            Provider name.

        Returns
        -------
        AssessmentProvider
            The provider.

        Raises
        ------
        PolicyError
            If no provider has that name.
        """
        with self._lock:
            try:
                return self._providers[name]
            except KeyError:
                raise PolicyError(
                    f"no assessment provider named {name!r}; registered:"
                    f" {sorted(self._providers)}"
                ) from None

    def copy(self) -> AssessmentRegistry:
        """
        Return an independent registry with the same providers.

        Returns
        -------
        AssessmentRegistry
            A copy; registering in it does not change this registry.

        Examples
        --------
        >>> from scikitplot.feature_selection import default_registry
        >>> private = default_registry.copy()
        >>> private.names() == default_registry.names()
        True
        """
        clone = AssessmentRegistry()
        with self._lock:
            clone._providers = dict(self._providers)
        return clone

    def names(self) -> tuple[str, ...]:
        """Tuple of str: Registered names, sorted."""
        with self._lock:
            return tuple(sorted(self._providers))

    def unregister(self, name: str) -> None:
        """
        Remove ``name`` (mainly for tests); missing names are ignored.

        Parameters
        ----------
        name : str
            Provider name.
        """
        with self._lock:
            self._providers.pop(name, None)


#: The registry used by every entry point unless one is passed explicitly.
default_registry = AssessmentRegistry()


def register_assessment(
    name: str,
    func: Callable[..., AssessmentOutcome] | None = None,
    *,
    defaults: Mapping[str, Any] | None = None,
    version: str = "1",
    description: str = "",
    validate: Callable[[Mapping[str, Any]], None] | None = None,
    replace: bool = False,
    registry: AssessmentRegistry | None = None,
) -> Any:
    """
    Register an assessment provider; usable as a decorator.

    Parameters
    ----------
    name : str
        Provider name used by policies.
    func : callable or None, default=None
        ``func(context, **params) -> AssessmentOutcome``; omit to decorate.
    defaults : mapping or None, default=None
        Accepted parameters and their defaults.
    version : str, default="1"
        Provider version, recorded in every assessment.
    description : str, default=""
        One line for :func:`list_assessment_providers`.
    validate : callable or None, default=None
        Range check for parameters; raise :class:`PolicyError`.
    replace : bool, default=False
        Replace an existing provider of the same name.
    registry : AssessmentRegistry or None, default=None
        Target registry; :data:`default_registry` when None.

    Returns
    -------
    callable
        ``func`` (so the decorator form leaves it usable).

    Examples
    --------
    >>> from scikitplot.feature_selection import (
    ...     AssessmentRegistry,
    ...     register_assessment,
    ... )
    >>> reg = AssessmentRegistry()
    >>> @register_assessment("wide_range", defaults={"span": 100.0}, registry=reg)
    ... def wide_range(context, span):
    ...     s = context.stats
    ...     if not s.is_numeric or s.min is None:
    ...         return context.result("not_applicable", "not_numeric")
    ...     return context.result("true" if s.max - s.min >= span else "false")
    >>> reg.names()
    ('wide_range',)
    """
    _ = name.strip() if isinstance(name, str) else None
    if not _:
        raise ValueError("name must be a non-empty string")
    target = default_registry if registry is None else registry

    def decorate(
        fn: Callable[..., AssessmentOutcome],
    ) -> Callable[..., AssessmentOutcome]:
        if not callable(fn):
            raise TypeError("func must be callable")
        frozen = _freeze_defaults(defaults)
        target.register(
            AssessmentProvider(
                name=name,
                func=fn,
                defaults=frozen,
                version=str(version),
                description=description,
                validate=validate,
            ),
            replace=replace,
        )
        return fn

    return decorate if func is None else decorate(func)


def list_assessment_providers(
    registry: AssessmentRegistry | None = None,
) -> pd.DataFrame:
    """
    Tabulate registered providers.

    Parameters
    ----------
    registry : AssessmentRegistry or None, default=None
        Registry to list; :data:`default_registry` when None.

    Returns
    -------
    pandas.DataFrame
        Columns ``name``, ``version``, ``parameters``, ``description``.

    Examples
    --------
    >>> from scikitplot.feature_selection import list_assessment_providers
    >>> "skewed" in set(list_assessment_providers()["name"])
    True
    """
    target = default_registry if registry is None else registry
    rows = []
    for name in target.names():
        provider = target.get(name)
        rows.append(
            {
                "name": name,
                "version": provider.version,
                "parameters": dict(provider.defaults),
                "description": provider.description,
            }
        )
    return pd.DataFrame(rows, columns=["name", "version", "parameters", "description"])


def run_assessment(
    name: str,
    spec: AssessmentSpec,
    params: Mapping[str, Any],
    provider: AssessmentProvider,
    context: AssessmentContext,
) -> Assessment:
    """
    Call ``provider`` and stamp the provenance of its answer.

    Parameters
    ----------
    name : str
        Assessment name.
    spec : AssessmentSpec
        Spec from the policy.
    params : mapping
        Resolved parameters.
    provider : AssessmentProvider
        Provider.
    context : AssessmentContext
        Feature context.

    Returns
    -------
    Assessment
        The recorded assessment.

    Raises
    ------
    TypeError
        If the provider returns anything but an :class:`AssessmentOutcome`.
    """
    outcome = provider.func(context, **params)
    if not isinstance(outcome, AssessmentOutcome):
        raise TypeError(
            f"assessment provider {provider.name!r} returned"
            f" {type(outcome).__name__}; return context.result(...)"
        )
    AssessmentState(outcome.state)
    return Assessment(
        name=name,
        state=outcome.state,
        provider=spec.provider,
        version=provider.version,
        params=tuple(sorted((k, v) for k, v in params.items() if v is not None)),
        reasons=outcome.reasons,
        evidence=outcome.evidence,
    )


# --------------------------------------------------------------------------- #
# Built-in providers
# --------------------------------------------------------------------------- #

_TRUE, _FALSE = AssessmentState.TRUE, AssessmentState.FALSE
_UNKNOWN, _NA = AssessmentState.UNKNOWN, AssessmentState.NOT_APPLICABLE


def _ratio(field: str, value: Any, *, allow_zero: bool = False) -> None:
    low_ok = value >= 0.0 if allow_zero else value > 0.0
    if not (low_ok and value <= 1.0):
        bound = "[0, 1]" if allow_zero else "(0, 1]"
        raise PolicyError(f"{field} must be in {bound}, got {value}")


def _positive_int(field: str, value: Any, low: int = 1) -> None:
    if value < low:
        raise PolicyError(f"{field} must be >= {low}, got {value}")


def _declared_categorical(context: AssessmentContext) -> bool:
    return context.declared_kind in _CATEGORICAL_KINDS


def _constant(context: AssessmentContext) -> AssessmentOutcome:
    s = context.stats
    if s.n_rows == 0:
        return context.result(_UNKNOWN, "empty_frame")
    if s.n_non_null == 0:
        return context.result(_TRUE, "all_missing", n_unique=0)
    if s.n_unique is None:
        return context.result(_UNKNOWN, "unhashable_values")
    if s.n_unique <= 1:
        return context.result(_TRUE, "single_value", n_unique=s.n_unique)
    return context.result(_FALSE, "several_values", n_unique=s.n_unique)


def _near_constant(context: AssessmentContext, min_top_ratio: float, min_count: int):
    s = context.stats
    if s.n_unique is None:
        return context.result(_UNKNOWN, "unhashable_values")
    if s.n_non_null < max(min_count, 1):
        return context.result(_UNKNOWN, "too_few_values", n_non_null=s.n_non_null)
    ratio = s.top_ratio
    state = _TRUE if ratio >= min_top_ratio else _FALSE
    reason = "top_ratio_ge_threshold" if state is _TRUE else "top_ratio_lt_threshold"
    return context.result(state, reason, top_ratio=ratio, threshold=min_top_ratio)


def _binary(context: AssessmentContext) -> AssessmentOutcome:
    s = context.stats
    if s.n_unique is None:
        return context.result(_UNKNOWN, "unhashable_values")
    if s.n_non_null == 0:
        return context.result(_UNKNOWN, "all_missing")
    _n_unique = s.n_unique == 2  # ruff: ignore[magic-value-comparison]
    state = _TRUE if _n_unique else _FALSE
    return context.result(
        state, "two_values" if state is _TRUE else "not_two_values", n_unique=s.n_unique
    )


def _discrete(
    context: AssessmentContext,
    max_unique_count: int,
    max_unique_ratio: Any,
    require_integer: bool,
) -> AssessmentOutcome:
    s = context.stats
    if not s.is_numeric:
        return context.result(_NA, "not_numeric", family=s.family)
    if not s.n_finite:
        return context.result(_UNKNOWN, "no_finite_values")
    if require_integer and not s.integer_valued:
        return context.result(_FALSE, "non_integer_values")
    ratio = s.unique_ratio
    within_count = s.n_unique <= max_unique_count
    within_ratio = max_unique_ratio is None or (
        ratio is not None and ratio <= max_unique_ratio
    )
    if within_count and within_ratio:
        return context.result(
            _TRUE,
            "few_distinct_values",
            n_unique=s.n_unique,
            max_unique_count=max_unique_count,
        )
    reason = "too_many_distinct_values" if not within_count else "unique_ratio_too_high"
    return context.result(_FALSE, reason, n_unique=s.n_unique, unique_ratio=ratio)


def _validate_discrete(params: Mapping[str, Any]) -> None:
    _positive_int("max_unique_count", params["max_unique_count"])
    ratio = params["max_unique_ratio"]
    if ratio is not None:
        if isinstance(ratio, bool) or not isinstance(ratio, numbers.Real):
            raise PolicyError(f"max_unique_ratio must be a number, got {ratio!r}")
        _ratio("max_unique_ratio", ratio)


def _categorical(context: AssessmentContext, include_binary: bool) -> AssessmentOutcome:
    s = context.stats
    if _declared_categorical(context):
        return context.result(_TRUE, "declared_kind", kind=context.declared_kind)
    family = context.family
    if family in (DtypeFamily.CATEGORICAL, DtypeFamily.STRING):
        return context.result(_TRUE, "categorical_family", family=family.value)
    if family is DtypeFamily.OBJECT:
        return context.result(_UNKNOWN, "mixed_object_values")
    if family is DtypeFamily.BOOLEAN:
        return context.result(_TRUE if include_binary else _FALSE, "boolean_family")
    _n_unique = s.n_unique == 2  # ruff: ignore[magic-value-comparison]
    if include_binary and _n_unique:
        return context.result(_TRUE, "binary_values", family=family.value)
    return context.result(_FALSE, "non_categorical_family", family=family.value)


def _skewed(  # ruff: ignore[too-many-return-statements]
    context: AssessmentContext,
    threshold: float,
    min_count: int,
):
    s = context.stats
    if not s.is_numeric:
        return context.result(_NA, "not_numeric", family=s.family)
    if _declared_categorical(context):
        return context.result(_NA, "declared_categorical", kind=context.declared_kind)
    _n_unique = s.n_unique <= 2  # ruff: ignore[magic-value-comparison]
    if s.n_unique is not None and _n_unique:
        return context.result(_NA, "two_or_fewer_values", n_unique=s.n_unique)
    if (s.n_finite or 0) < min_count:
        return context.result(
            _UNKNOWN, "too_few_finite_values", n_finite=s.n_finite, min_count=min_count
        )
    if s.skewness is None:
        return context.result(_UNKNOWN, "undefined_skewness")
    direction = "right" if s.skewness > 0 else "left" if s.skewness < 0 else "none"
    if abs(s.skewness) >= threshold:
        return context.result(
            _TRUE,
            "abs_skewness_ge_threshold",
            skewness=s.skewness,
            threshold=threshold,
            direction=direction,
        )
    return context.result(
        _FALSE,
        "abs_skewness_lt_threshold",
        skewness=s.skewness,
        threshold=threshold,
        direction=direction,
    )


def _validate_skewed(params: Mapping[str, Any]) -> None:
    if params["threshold"] < 0:
        raise PolicyError(f"threshold must be >= 0, got {params['threshold']}")
    _positive_int("min_count", params["min_count"], low=3)


def _high_cardinality(
    context: AssessmentContext, min_unique_count: int, min_unique_ratio: Any
):
    s = context.stats
    family = context.family
    if family in (DtypeFamily.DATETIME, DtypeFamily.TIMEDELTA):
        return context.result(_NA, "temporal_family", family=family.value)
    if s.is_numeric and not _declared_categorical(context):
        return context.result(_NA, "numeric_not_declared_categorical")
    if s.n_unique is None:
        return context.result(_UNKNOWN, "unhashable_values")
    if s.n_non_null == 0:
        return context.result(_UNKNOWN, "all_missing")
    ratio = s.unique_ratio
    by_count = s.n_unique >= min_unique_count
    by_ratio = min_unique_ratio is not None and ratio >= min_unique_ratio
    if by_count or by_ratio:
        reason = (
            "unique_count_ge_threshold" if by_count else "unique_ratio_ge_threshold"
        )
        return context.result(_TRUE, reason, n_unique=s.n_unique, unique_ratio=ratio)
    return context.result(
        _FALSE, "few_categories", n_unique=s.n_unique, unique_ratio=ratio
    )


def _validate_high_cardinality(params: Mapping[str, Any]) -> None:
    _positive_int("min_unique_count", params["min_unique_count"])
    ratio = params["min_unique_ratio"]
    if ratio is not None:
        if isinstance(ratio, bool) or not isinstance(ratio, numbers.Real):
            raise PolicyError(f"min_unique_ratio must be a number, got {ratio!r}")
        _ratio("min_unique_ratio", ratio)


def _high_missing(context: AssessmentContext, min_missing_ratio: float):
    s = context.stats
    if s.n_rows == 0:
        return context.result(_UNKNOWN, "empty_frame")
    ratio = s.missing_ratio
    state = _TRUE if ratio >= min_missing_ratio else _FALSE
    return context.result(
        state,
        (
            "missing_ratio_ge_threshold"
            if state is _TRUE
            else "missing_ratio_lt_threshold"
        ),
        missing_ratio=ratio,
        threshold=min_missing_ratio,
    )


def _has_missing(context: AssessmentContext) -> AssessmentOutcome:
    s = context.stats
    if s.n_rows == 0:
        return context.result(_UNKNOWN, "empty_frame")
    return context.result(
        _TRUE if s.n_missing else _FALSE,
        "missing_present" if s.n_missing else "no_missing",
        n_missing=s.n_missing,
    )


def _identifier_like(
    context: AssessmentContext,
    min_unique_ratio: float,
    min_count: int,
    include_integers: bool,
):
    s = context.stats
    family = context.family
    if family in (DtypeFamily.BOOLEAN, DtypeFamily.DATETIME, DtypeFamily.TIMEDELTA):
        return context.result(_NA, "family_cannot_identify", family=family.value)
    if s.is_numeric and not _declared_categorical(context):
        if not include_integers:
            return context.result(_NA, "numeric_not_checked")
        if s.integer_valued is False:
            return context.result(_NA, "continuous_values_are_naturally_unique")
    if s.n_unique is None:
        return context.result(_UNKNOWN, "unhashable_values")
    if s.n_non_null < min_count:
        return context.result(
            _UNKNOWN, "too_few_values", n_non_null=s.n_non_null, min_count=min_count
        )
    ratio = s.unique_ratio
    state = _TRUE if ratio >= min_unique_ratio else _FALSE
    return context.result(
        state,
        "unique_ratio_ge_threshold" if state is _TRUE else "unique_ratio_lt_threshold",
        unique_ratio=ratio,
        threshold=min_unique_ratio,
    )


def _integer_valued(context: AssessmentContext) -> AssessmentOutcome:
    s = context.stats
    if not s.is_numeric:
        return context.result(_NA, "not_numeric", family=s.family)
    if s.integer_valued is None:
        return context.result(_UNKNOWN, "no_finite_values")
    return context.result(
        _TRUE if s.integer_valued else _FALSE,
        "whole_numbers" if s.integer_valued else "fractional_values",
    )


def _non_negative(context: AssessmentContext) -> AssessmentOutcome:
    s = context.stats
    if not s.is_numeric:
        return context.result(_NA, "not_numeric", family=s.family)
    if s.min is None:
        return context.result(_UNKNOWN, "no_finite_values")
    state = _TRUE if s.min >= 0 else _FALSE
    return context.result(
        state, "min_ge_zero" if state is _TRUE else "negative_values", min=s.min
    )


def _has_infinite(context: AssessmentContext) -> AssessmentOutcome:
    s = context.stats
    if not s.is_numeric:
        return context.result(_NA, "not_numeric", family=s.family)
    return context.result(
        _TRUE if s.n_inf else _FALSE,
        "infinite_present" if s.n_inf else "no_infinite",
        n_inf=s.n_inf,
    )


def _has_outliers(
    context: AssessmentContext,
    iqr_factor: float,
    min_outlier_ratio: float,
    min_count: int,
) -> AssessmentOutcome:
    s = context.stats
    if not s.is_numeric:
        return context.result(_NA, "not_numeric", family=s.family)
    _n_unique = s.n_unique <= 2  # ruff: ignore[magic-value-comparison]
    if s.n_unique is not None and _n_unique:
        return context.result(_NA, "two_or_fewer_values", n_unique=s.n_unique)
    if context.series is None:
        return context.result(_UNKNOWN, "training_data_unavailable")
    if (s.n_finite or 0) < min_count:
        return context.result(_UNKNOWN, "too_few_finite_values", n_finite=s.n_finite)
    values = numeric_values(context.series)
    values = values[np.isfinite(values)]
    iqr = s.q3 - s.q1
    lower, upper = s.q1 - iqr_factor * iqr, s.q3 + iqr_factor * iqr
    n_out = int(np.count_nonzero((values < lower) | (values > upper)))
    ratio = n_out / values.size
    state = _TRUE if n_out and ratio >= min_outlier_ratio else _FALSE
    return context.result(
        state,
        (
            "outlier_ratio_ge_threshold"
            if state is _TRUE
            else "outlier_ratio_lt_threshold"
        ),
        n_outliers=n_out,
        outlier_ratio=ratio,
        lower_fence=lower,
        upper_fence=upper,
    )


def _validate_outliers(params: Mapping[str, Any]) -> None:
    if params["iqr_factor"] <= 0:
        raise PolicyError(f"iqr_factor must be > 0, got {params['iqr_factor']}")
    _ratio("min_outlier_ratio", params["min_outlier_ratio"], allow_zero=True)
    _positive_int("min_count", params["min_count"], low=4)


def _builtin(name, func, defaults=None, description="", validate=None) -> None:
    default_registry._register_builtin(
        AssessmentProvider(
            name=name,
            func=func,
            defaults=_freeze_defaults(defaults),
            description=description,
            validate=validate,
        )
    )


_builtin("constant", _constant, description="At most one distinct non-missing value.")
_builtin(
    "near_constant",
    _near_constant,
    {"min_top_ratio": 0.99, "min_count": 1},
    "Most frequent value covers at least min_top_ratio of non-missing values.",
    lambda p: (
        _ratio("min_top_ratio", p["min_top_ratio"]),
        _positive_int("min_count", p["min_count"]),
    ),
)
_builtin("binary", _binary, description="Exactly two distinct non-missing values.")
_builtin(
    "discrete",
    _discrete,
    {"max_unique_count": 20, "max_unique_ratio": None, "require_integer": True},
    "Numeric with few distinct (whole) values.",
    _validate_discrete,
)
_builtin(
    "categorical",
    _categorical,
    {"include_binary": True},
    "Categorical/string family, declared categorical kind, or (optionally) binary.",
)
_builtin(
    "skewed",
    _skewed,
    {"threshold": 1.0, "min_count": 8},
    "Absolute bias-corrected sample skewness at least threshold.",
    _validate_skewed,
)
_builtin(
    "high_cardinality",
    _high_cardinality,
    {"min_unique_count": 50, "min_unique_ratio": None},
    "Non-numeric (or declared categorical) with many distinct values.",
    _validate_high_cardinality,
)
_builtin(
    "high_missing",
    _high_missing,
    {"min_missing_ratio": 0.5},
    "Share of missing rows at least min_missing_ratio.",
    lambda p: _ratio("min_missing_ratio", p["min_missing_ratio"], allow_zero=True),
)
_builtin("has_missing", _has_missing, description="At least one missing value.")
_builtin(
    "identifier_like",
    _identifier_like,
    {"min_unique_ratio": 0.99, "min_count": 20, "include_integers": False},
    "String or categorical (optionally integer) values that are (almost) all distinct.",
    lambda p: (
        _ratio("min_unique_ratio", p["min_unique_ratio"]),
        _positive_int("min_count", p["min_count"]),
    ),
)
_builtin("integer_valued", _integer_valued, description="Every finite value is whole.")
_builtin("non_negative", _non_negative, description="Smallest finite value is >= 0.")
_builtin("has_infinite", _has_infinite, description="At least one infinite value.")
_builtin(
    "has_outliers",
    _has_outliers,
    {"iqr_factor": 1.5, "min_outlier_ratio": 0.01, "min_count": 8},
    "Share of values beyond Tukey's fences (q1 - k*IQR, q3 + k*IQR).",
    _validate_outliers,
)
