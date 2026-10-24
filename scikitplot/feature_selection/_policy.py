# scikitplot/feature_selection/_policy.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Declarative feature policy: roles, kinds, tags, assessments, binning, drift.

A :class:`FeaturePolicy` is *user intent*. It holds no data-derived value, so
the same policy can be shared between projects, stored next to code as TOML,
YAML or JSON, and reviewed like any other configuration. Fitting a profiler on
training data turns a policy into a :class:`~scikitplot.feature_selection.FeatureSchema`
(fitted state), which records the digest of the policy it came from.

Example TOML document::

    schema_version = 1
    on_missing_column = "raise"

    [roles]
    target = ["churn"]
    identifier = { columns = ["customer_id"] }

    [kinds]
    zip_code = "categorical_nominal"

    [tags.one_hot]
    columns = ["city"]
    patterns = ["cat_*"]

    [assessments.skewed]
    provider = "skewed"
    params = { threshold = 0.75, min_count = 20 }

    [assessments.very_skewed]
    provider = "skewed"
    params = { threshold = 2.0 }

    [overrides.income]
    skewed = true

    [binning]
    rule = "knuth"
    max_bins = 50

Notes
-----
**User note.** Every section is optional; an empty policy is valid and uses the
documented defaults. ``assessments`` *merges* over :data:`DEFAULT_ASSESSMENTS`
by name: give a name to change its parameters, set ``enabled = false`` to switch
it off, or add a new name that reuses a provider with other parameters.
Executable code is never part of a policy: a custom provider is registered at
run time with :func:`~scikitplot.feature_selection.register_assessment` and the
policy refers to it by name.

**Developer note.** All containers are stored as sorted tuples of pairs, so a
policy is hashable, picklable, comparable and safe to copy (Rule 112).
:meth:`FeaturePolicy.from_dict` checks each field against an explicit contract
before constructing anything (Rule 106): unknown keys are refused, ``bool`` is
not accepted as a number, and YAML/TOML dates are not strings.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import hashlib
import json
import numbers
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import _serialize
from ._errors import IncompatibleSchemaError, PolicyError
from ._types import FeatureKind, FeatureRole, coerce_enum

__all__ = [
    "DEFAULT_ASSESSMENTS",
    "AssessmentSpec",
    "BinningPolicy",
    "ColumnRule",
    "DriftPolicy",
    "FeaturePolicy",
    "dump_policy",
    "load_policy",
]

#: Generation of the policy document format; bumped only on incompatible change.
POLICY_SCHEMA_VERSION = 1

#: Histogram bin-count rules understood by :class:`BinningPolicy`.
BIN_RULES: tuple[str, ...] = (
    "auto",
    "doane",
    "fd",
    "fixed",
    "knuth",
    "rice",
    "scott",
    "sqrt",
    "stone",
    "sturges",
)

#: Edge placement strategies understood by :class:`BinningPolicy`.
BIN_STRATEGIES: tuple[str, ...] = ("uniform", "quantile")

_SCALARS = (str, int, float, bool, type(None))


# --------------------------------------------------------------------------- #
# Field validators (Rule 68: branch on type first, then check range)
# --------------------------------------------------------------------------- #


def _require_bool(field: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise PolicyError(
            f"{field} must be a bool, got {type(value).__name__}: {value!r}"
        )
    return value


def _require_str(field: str, value: Any) -> str:
    if not isinstance(value, str):
        raise PolicyError(
            f"{field} must be a string, got {type(value).__name__}: {value!r}"
        )
    if not value:
        raise PolicyError(f"{field} must not be empty")
    return value


def _require_int(
    field: str, value: Any, *, low: int | None = None, high: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise PolicyError(
            f"{field} must be an int, got {type(value).__name__}: {value!r}"
        )
    value = int(value)
    if low is not None and value < low:
        raise PolicyError(f"{field} must be >= {low}, got {value}")
    if high is not None and value > high:
        raise PolicyError(f"{field} must be <= {high}, got {value}")
    return value


def _require_float(
    field: str,
    value: Any,
    *,
    low: float | None = None,
    high: float | None = None,
    low_open: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise PolicyError(
            f"{field} must be a number, got {type(value).__name__}: {value!r}"
        )
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):  # noqa: PLR0124
        raise PolicyError(f"{field} must be finite, got {value!r}")
    if low is not None and (value <= low if low_open else value < low):
        raise PolicyError(
            f"{field} must be {'>' if low_open else '>='} {low}, got {value}"
        )
    if high is not None and value > high:
        raise PolicyError(f"{field} must be <= {high}, got {value}")
    return value


def _require_choice(field: str, value: Any, choices: Sequence[str]) -> str:
    value = _require_str(field, value)
    if value not in choices:
        raise PolicyError(f"{field}={value!r} is not one of {tuple(choices)}")
    return value


def _require_mapping(field: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyError(f"{field} must be a mapping, got {type(value).__name__}")
    for key in value:
        if not isinstance(key, str):
            raise PolicyError(f"{field} keys must be strings, got {key!r}")
    return value


def _refuse_unknown(field: str, given: Iterable[str], allowed: Iterable[str]) -> None:
    unknown = sorted(set(given) - set(allowed))
    if unknown:
        raise PolicyError(
            f"{field} has unknown key(s) {unknown}; allowed: {sorted(set(allowed))}"
        )


def _str_tuple(field: str, value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (_require_str(field, value),)
    if not isinstance(value, (list, tuple)):
        raise PolicyError(
            f"{field} must be a string or a list of strings, got {type(value).__name__}"
        )
    return tuple(_require_str(f"{field}[{i}]", item) for i, item in enumerate(value))


def _plain(value: Any) -> Any:
    """Return NumPy and other numeric scalars as the built-in bool/int/float."""
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    return value


def _param_value(field: str, value: Any) -> Any:
    value = _plain(value)
    if isinstance(value, (list, tuple)):
        value = tuple(_plain(v) for v in value)
    if isinstance(value, _SCALARS):
        if isinstance(value, float) and value != value:  # noqa: PLR0124
            raise PolicyError(f"{field} must not be NaN")
        return value
    if isinstance(value, (list, tuple)):
        items = tuple(value)
        for i, item in enumerate(items):
            if not isinstance(item, _SCALARS) or item is None:
                raise PolicyError(
                    f"{field}[{i}] must be a str/int/float/bool, got"
                    f" {type(item).__name__}"
                )
        return items
    raise PolicyError(
        f"{field} must be a scalar or a list of scalars, got {type(value).__name__}"
        f" ({value!r})"
    )


def _thaw(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


# --------------------------------------------------------------------------- #
# ColumnRule
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ColumnRule:
    """
    Select columns by exact name and/or shell-style pattern.

    Parameters
    ----------
    columns : str or sequence of str, default=()
        Exact column names.
    patterns : str or sequence of str, default=()
        :func:`fnmatch.fnmatchcase` patterns (``"cat_*"``, ``"x_[0-9]"``).
        Matching is case-sensitive on every platform.

    See Also
    --------
    FeaturePolicy : Uses column rules for roles and tags.

    Notes
    -----
    **User note.** A list is shorthand for ``columns``; ``ColumnRule(["a"])``
    and ``{"columns": ["a"]}`` in a config file are the same rule.

    **Developer note.** Patterns use ``fnmatchcase`` (not ``fnmatch``), so a
    rule selects the same columns on Windows and POSIX.

    Examples
    --------
    >>> from scikitplot.feature_selection import ColumnRule
    >>> ColumnRule(columns=["age"], patterns=["cat_*"]).resolve(
    ...     ["age", "cat_a", "cat_b", "income"]
    ... )
    ('age', 'cat_a', 'cat_b')
    """

    columns: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:  # noqa: D105
        object.__setattr__(self, "columns", _str_tuple("columns", self.columns))
        object.__setattr__(self, "patterns", _str_tuple("patterns", self.patterns))

    @classmethod
    def from_value(cls, value: Any, *, field: str = "rule") -> ColumnRule:
        """
        Build a rule from its config form: a string, a list, a mapping or a rule.

        Parameters
        ----------
        value : str, list of str, mapping or ColumnRule
            ``{"columns": [...], "patterns": [...]}`` or the list shorthand.
        field : str, default="rule"
            Name used in error messages.

        Returns
        -------
        ColumnRule
            The rule.

        Raises
        ------
        PolicyError
            If the value is off contract.
        """
        if isinstance(value, ColumnRule):
            return value
        if isinstance(value, (str, list, tuple)):
            return cls(columns=_str_tuple(field, value))
        mapping = _require_mapping(field, value)
        _refuse_unknown(field, mapping, ("columns", "patterns"))
        return cls(
            columns=_str_tuple(f"{field}.columns", mapping.get("columns", ())),
            patterns=_str_tuple(f"{field}.patterns", mapping.get("patterns", ())),
        )

    @property
    def is_empty(self) -> bool:
        """bool: True when the rule names no column and no pattern."""
        return not self.columns and not self.patterns

    def resolve(
        self, names: Sequence[str], *, on_missing: str = "raise", field: str = "rule"
    ) -> tuple[str, ...]:
        """
        Return the names this rule selects, in the order of ``names``.

        Parameters
        ----------
        names : sequence of str
            Available column names.
        on_missing : {"raise", "ignore"}, default="raise"
            What to do with an exact name that is not in ``names``.
        field : str, default="rule"
            Name used in error messages.

        Returns
        -------
        tuple of str
            Selected names, each once, in ``names`` order.

        Raises
        ------
        PolicyError
            If ``on_missing="raise"`` and an exact name is absent.
        """
        available = set(names)
        missing = [c for c in self.columns if c not in available]
        if missing and on_missing == "raise":
            raise PolicyError(
                f"{field} names column(s) {missing} that are not in the data; fix the"
                " policy or set on_missing_column='ignore'"
            )
        wanted = set(self.columns)
        return tuple(
            name
            for name in names
            if name in wanted
            or any(fnmatch.fnmatchcase(name, p) for p in self.patterns)
        )

    def to_dict(self) -> dict[str, list[str]]:
        """Return the config form ``{"columns": [...], "patterns": [...]}``."""
        return {"columns": list(self.columns), "patterns": list(self.patterns)}


# --------------------------------------------------------------------------- #
# AssessmentSpec
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class AssessmentSpec:
    """
    Which provider answers one assessment question, with which parameters.

    Parameters
    ----------
    provider : str
        Name of a registered provider (see
        :func:`~scikitplot.feature_selection.list_assessment_providers`).
    params : mapping of str to scalar or list of scalars, default=()
        Provider parameters; unknown names are refused when profiling. A
        ``None`` value means "use the provider default" and is not stored, so
        every format (TOML has no null) holds the same spec.
    enabled : bool, default=True
        False switches the assessment off without removing it.

    Notes
    -----
    **Developer note.** ``params`` is stored as a sorted tuple of
    ``(name, value)`` pairs so the spec stays hashable; use :attr:`params_dict`
    to read it as a mapping.
    """

    provider: str
    params: tuple[tuple[str, Any], ...] = ()
    enabled: bool = True

    def __post_init__(self) -> None:  # noqa: D105
        _require_str("provider", self.provider)
        _require_bool("enabled", self.enabled)
        raw = self.params
        len_item = 2
        if isinstance(raw, Mapping):
            items = list(raw.items())
        elif isinstance(raw, (list, tuple)):
            items = [tuple(p) for p in raw]
            if any(len(p) != len_item for p in items):
                raise PolicyError("params must be a mapping or (name, value) pairs")
        else:
            raise PolicyError(f"params must be a mapping, got {type(raw).__name__}")
        frozen = []
        for key, value in items:
            _require_str("params key", key)
            if value is None:  # None means "provider default" (TOML has no null)
                continue
            frozen.append((key, _param_value(f"params.{key}", value)))
        if len({k for k, _ in frozen}) != len(frozen):
            raise PolicyError("params has duplicate names")
        object.__setattr__(self, "params", tuple(sorted(frozen)))

    @property
    def params_dict(self) -> dict[str, Any]:
        """dict: The parameters as a mapping (tuples stay tuples)."""
        return dict(self.params)

    @classmethod
    def from_value(cls, value: Any, *, field: str = "assessment") -> AssessmentSpec:
        """
        Build a spec from its config form.

        Parameters
        ----------
        value : mapping or AssessmentSpec
            ``{"provider": ..., "params": {...}, "enabled": ...}``.
        field : str, default="assessment"
            Name used in error messages.

        Returns
        -------
        AssessmentSpec
            The spec.
        """
        if isinstance(value, AssessmentSpec):
            return value
        mapping = _require_mapping(field, value)
        _refuse_unknown(field, mapping, ("provider", "params", "enabled"))
        if "provider" not in mapping:
            raise PolicyError(f"{field} needs a 'provider'")
        params = mapping.get("params", {})
        _require_mapping(f"{field}.params", params)
        return cls(
            provider=_require_str(f"{field}.provider", mapping["provider"]),
            params=dict(params),
            enabled=_require_bool(f"{field}.enabled", mapping.get("enabled", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the config form."""
        return {
            "provider": self.provider,
            "params": {k: _thaw(v) for k, v in self.params},
            "enabled": self.enabled,
        }


#: Built-in assessment names, each answered by the provider of the same name
#: with that provider's default parameters. The defaults are defined once, on
#: the providers (see :func:`~scikitplot.feature_selection.list_assessment_providers`);
#: every fitted :class:`~scikitplot.feature_selection.Assessment` records the
#: parameters it actually used.
DEFAULT_ASSESSMENTS: Mapping[str, AssessmentSpec] = {
    name: AssessmentSpec(name)
    for name in (
        "binary",
        "categorical",
        "constant",
        "discrete",
        "has_infinite",
        "has_missing",
        "has_outliers",
        "high_cardinality",
        "high_missing",
        "identifier_like",
        "integer_valued",
        "near_constant",
        "non_negative",
        "skewed",
    )
}


# --------------------------------------------------------------------------- #
# Binning and drift sections
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class BinningPolicy:
    """
    How bin recommendations are computed for numeric features.

    Parameters
    ----------
    enabled : bool, default=True
        Compute a :class:`~scikitplot.feature_selection.BinningRecommendation`
        for each feature while profiling.
    rule : {"auto", "knuth", "fd", "scott", "sturges", "doane", "rice", "sqrt", "stone", "fixed"}, default="auto"
        Bin-count rule. ``"knuth"`` maximises Knuth's Bayesian posterior over
        every integer count in ``[min_bins, max_bins]`` (an exhaustive, exact
        search); ``"fixed"`` uses ``n_bins``; the others are the rules of
        :func:`numpy.histogram_bin_edges`.
    n_bins : int or None, default=None
        Bin count for ``rule="fixed"``; must be None for other rules.
    min_bins, max_bins : int, default=2, 64
        Bounds applied to every rule's count (a clipped count is reported).
    min_count : int, default=30
        Fewest finite values for a recommendation; below it the suitability is
        ``unknown``.
    strategy : {"uniform", "quantile"}, default="uniform"
        Edge placement. The data-driven rules assume equal widths; with
        ``"quantile"`` their count is used with quantile edges, and tied
        quantiles can reduce the count (reported).

    Notes
    -----
    **User note.** A recommendation is advice: it never changes your data.
    Turn it into a transformer with
    :meth:`~scikitplot.feature_selection.FeatureSchema.binning_plan`.
    """

    enabled: bool = True
    rule: str = "auto"
    n_bins: int | None = None  # Optional: pydantic evaluates it on Python 3.9
    min_bins: int = 2
    max_bins: int = 64
    min_count: int = 30
    strategy: str = "uniform"

    def __post_init__(self) -> None:  # noqa: D105
        _require_bool("binning.enabled", self.enabled)
        _require_choice("binning.rule", self.rule, BIN_RULES)
        _require_choice("binning.strategy", self.strategy, BIN_STRATEGIES)
        set_ = object.__setattr__
        min_bins = _require_int("binning.min_bins", self.min_bins, low=1)
        max_bins = _require_int("binning.max_bins", self.max_bins, low=1, high=100_000)
        set_(self, "min_bins", min_bins)
        set_(self, "max_bins", max_bins)
        if min_bins > max_bins:
            raise PolicyError(f"binning.min_bins ({min_bins}) > max_bins ({max_bins})")
        set_(
            self, "min_count", _require_int("binning.min_count", self.min_count, low=2)
        )
        if self.rule == "fixed":
            if self.n_bins is None:
                raise PolicyError("binning.rule='fixed' needs binning.n_bins")
            set_(
                self,
                "n_bins",
                _require_int(
                    "binning.n_bins", self.n_bins, low=min_bins, high=max_bins
                ),
            )
        elif self.n_bins is not None:
            raise PolicyError(
                f"binning.n_bins is only used with rule='fixed' (rule={self.rule!r})"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the config form."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DriftPolicy:
    """
    How a drift reference is frozen and judged.

    Parameters
    ----------
    n_bins : int, default=10
        Quantile bins of a numeric reference (ties can merge bins).
    max_categories : int, default=50
        Most frequent training categories kept by name; the rest share one
        ``"other"`` bucket.
    epsilon : float, default=1e-4
        Floor applied to bucket proportions before the logarithm of the PSI,
        so an empty bucket gives a large finite value instead of infinity.
    psi_warn, psi_alert : float, default=0.1, 0.25
        Population stability index thresholds for the ``warn`` and ``drift``
        statuses (the conventional 0.1 / 0.25 bands).
    min_count : int, default=30
        Fewest current non-missing values to judge a feature; below it the
        status is ``insufficient``.
    """

    n_bins: int = 10
    max_categories: int = 50
    epsilon: float = 1e-4
    psi_warn: float = 0.1
    psi_alert: float = 0.25
    min_count: int = 30

    def __post_init__(self) -> None:  # noqa: D105
        object.__setattr__(
            self,
            "n_bins",
            _require_int("drift.n_bins", self.n_bins, low=1, high=10_000),
        )
        object.__setattr__(
            self,
            "max_categories",
            _require_int("drift.max_categories", self.max_categories, low=1),
        )
        object.__setattr__(
            self,
            "epsilon",
            _require_float(
                "drift.epsilon", self.epsilon, low=0.0, high=0.5, low_open=True
            ),
        )
        warn = _require_float("drift.psi_warn", self.psi_warn, low=0.0)
        alert = _require_float("drift.psi_alert", self.psi_alert, low=0.0)
        if warn > alert:
            raise PolicyError(f"drift.psi_warn ({warn}) > drift.psi_alert ({alert})")
        object.__setattr__(self, "psi_warn", warn)
        object.__setattr__(self, "psi_alert", alert)
        object.__setattr__(
            self, "min_count", _require_int("drift.min_count", self.min_count, low=1)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the config form."""
        return dataclasses.asdict(self)


def _section(cls: type, field: str, value: Any) -> Any:
    if isinstance(value, cls):
        return value
    mapping = _require_mapping(field, value)
    names = [f.name for f in dataclasses.fields(cls)]
    _refuse_unknown(field, mapping, names)
    return cls(**dict(mapping))


# --------------------------------------------------------------------------- #
# FeaturePolicy
# --------------------------------------------------------------------------- #

_POLICY_KEYS = (
    "schema_version",
    "on_missing_column",
    "roles",
    "kinds",
    "tags",
    "assessments",
    "overrides",
    "binning",
    "drift",
)


@dataclasses.dataclass(frozen=True)
class FeaturePolicy:
    """
    Complete, declarative description of how to profile a table.

    Parameters
    ----------
    roles : mapping of role to ColumnRule-like, default=()
        Columns with a role other than ``"feature"`` (``target``, ``drop``,
        ``identifier``, ``weight``, ``group``, ``time``). A column matched by
        two roles is an error.
    kinds : mapping of column to kind, default=()
        Explicit semantic kinds that replace inference (for example a numeric
        ``zip_code`` declared ``"categorical_nominal"``, or ``"text"``).
    tags : mapping of tag to ColumnRule-like, default=()
        Free user labels (``"one_hot"``, ``"engineered"``, ``"sensitive"``); a
        column can carry many tags. Tags never change inference.
    assessments : mapping of name to AssessmentSpec-like, default=()
        Merged over :data:`DEFAULT_ASSESSMENTS` by name.
    overrides : mapping of column to mapping of assessment to bool, default=()
        Manual answers. The inferred answer is kept next to the override.
    binning : BinningPolicy or mapping, default=BinningPolicy()
        Bin recommendation settings.
    drift : DriftPolicy or mapping, default=DriftPolicy()
        Drift reference settings.
    on_missing_column : {"raise", "ignore"}, default="raise"
        What to do when ``roles``, ``kinds``, ``tags`` or ``overrides`` name a
        column that is not in the data.

    See Also
    --------
    load_policy : Read a policy from TOML, YAML or JSON.
    dump_policy : Write a policy, verified by reading it back.
    profile_features : Fit a schema from a DataFrame and a policy.

    Notes
    -----
    **User note.** Keep one policy per project in version control. Its
    :attr:`digest` is recorded in every fitted schema, so you can tell which
    configuration produced a result.

    **Developer note.** Stored containers are sorted tuples of pairs; the
    ``*_map`` properties return fresh dictionaries.

    Examples
    --------
    >>> from scikitplot.feature_selection import FeaturePolicy
    >>> policy = FeaturePolicy(
    ...     roles={"target": ["churn"]},
    ...     assessments={
    ...         "skewed": {"provider": "skewed", "params": {"threshold": 0.75}}
    ...     },
    ... )
    >>> policy.assessments_map["skewed"].params_dict["threshold"]
    0.75
    >>> FeaturePolicy.from_dict(policy.to_dict()) == policy
    True
    """

    roles: tuple[tuple[str, ColumnRule], ...] = ()
    kinds: tuple[tuple[str, str], ...] = ()
    tags: tuple[tuple[str, ColumnRule], ...] = ()
    assessments: tuple[tuple[str, AssessmentSpec], ...] = ()
    overrides: tuple[tuple[str, tuple[tuple[str, bool], ...]], ...] = ()
    binning: BinningPolicy = dataclasses.field(default_factory=BinningPolicy)
    drift: DriftPolicy = dataclasses.field(default_factory=DriftPolicy)
    on_missing_column: str = "raise"

    def __post_init__(self) -> None:  # noqa: D105
        set_ = object.__setattr__
        set_(self, "roles", self._rules("roles", self.roles, key_enum=FeatureRole))
        set_(self, "tags", self._rules("tags", self.tags))
        kinds = {}
        for column, kind in self._pairs("kinds", self.kinds):
            _require_str("kinds key", column)
            try:
                kinds[column] = coerce_enum(
                    kind, FeatureKind, field=f"kinds.{column}"
                ).value
            except (TypeError, ValueError) as exc:
                raise PolicyError(str(exc)) from None
        set_(self, "kinds", tuple(sorted(kinds.items())))
        merged = dict(DEFAULT_ASSESSMENTS)
        for name, spec in self._pairs("assessments", self.assessments):
            _require_str("assessments key", name)
            merged[name] = AssessmentSpec.from_value(spec, field=f"assessments.{name}")
        set_(self, "assessments", tuple(sorted(merged.items())))
        overrides = {}
        for column, answers in self._pairs("overrides", self.overrides):
            _require_str("overrides key", column)
            pairs = self._pairs(f"overrides.{column}", answers)
            frozen = []
            for name, answer in pairs:
                _require_str(f"overrides.{column} key", name)
                if name not in merged:
                    raise PolicyError(
                        f"overrides.{column}.{name}: no assessment named {name!r};"
                        f" known: {sorted(merged)}"
                    )
                frozen.append(
                    (name, _require_bool(f"overrides.{column}.{name}", answer))
                )
            overrides[column] = tuple(sorted(frozen))
        set_(self, "overrides", tuple(sorted(overrides.items())))
        set_(self, "binning", _section(BinningPolicy, "binning", self.binning))
        set_(self, "drift", _section(DriftPolicy, "drift", self.drift))
        _require_choice(
            "on_missing_column", self.on_missing_column, ("raise", "ignore")
        )

    # -- normalisation helpers ------------------------------------------------

    @staticmethod
    def _pairs(
        field: str,
        value: Any,
    ) -> list[tuple[Any, Any]]:
        len_item = 2
        if isinstance(value, Mapping):
            pairs = list(value.items())
        elif isinstance(value, (list, tuple)):
            pairs = [tuple(p) for p in value]
            if any(len(p) != len_item for p in pairs):
                raise PolicyError(f"{field} must be a mapping")
        else:
            raise PolicyError(f"{field} must be a mapping, got {type(value).__name__}")
        keys = [k for k, _ in pairs]
        if len(set(map(repr, keys))) != len(keys):
            raise PolicyError(f"{field} has duplicate keys")
        return pairs

    @classmethod
    def _rules(
        cls, field: str, value: Any, key_enum: type | None = None
    ) -> tuple[tuple[str, ColumnRule], ...]:
        out = {}
        for key, rule in cls._pairs(field, value):
            _require_str(f"{field} key", key)
            if key_enum is not None:
                try:
                    key = coerce_enum(  # ruff: ignore[redefined-loop-name]
                        key, key_enum, field=f"{field} key"
                    ).value  # noqa: PLW2901
                except (TypeError, ValueError) as exc:
                    raise PolicyError(str(exc)) from None
            out[key] = ColumnRule.from_value(rule, field=f"{field}.{key}")
        return tuple(sorted(out.items()))

    # -- read access ----------------------------------------------------------

    @property
    def roles_map(self) -> dict[str, ColumnRule]:
        """dict: Role name to column rule."""
        return dict(self.roles)

    @property
    def kinds_map(self) -> dict[str, str]:
        """dict: Column to declared kind."""
        return dict(self.kinds)

    @property
    def tags_map(self) -> dict[str, ColumnRule]:
        """dict: Tag to column rule."""
        return dict(self.tags)

    @property
    def assessments_map(self) -> dict[str, AssessmentSpec]:
        """dict: Assessment name to spec, defaults included."""
        return dict(self.assessments)

    @property
    def overrides_map(self) -> dict[str, dict[str, bool]]:
        """dict: Column to manual assessment answers."""
        return {column: dict(answers) for column, answers in self.overrides}

    # -- resolution against a table -------------------------------------------

    def resolve_roles(self, names: Sequence[str]) -> dict[str, str]:
        """
        Return the role of every column in ``names``.

        Parameters
        ----------
        names : sequence of str
            Column names of the data.

        Returns
        -------
        dict
            Column to role value; unlisted columns are ``"feature"``.

        Raises
        ------
        PolicyError
            If a column matches two roles, or a named column is missing and
            ``on_missing_column="raise"``.
        """
        roles = dict.fromkeys(names, FeatureRole.FEATURE.value)
        claimed: dict[str, str] = {}
        for role, rule in self.roles:
            for column in rule.resolve(
                names, on_missing=self.on_missing_column, field=f"roles.{role}"
            ):
                if column in claimed and claimed[column] != role:
                    raise PolicyError(
                        f"column {column!r} matches roles {claimed[column]!r} and"
                        f" {role!r}; a column has exactly one role"
                    )
                claimed[column] = role
                roles[column] = role
        return roles

    def resolve_tags(self, names: Sequence[str]) -> dict[str, tuple[str, ...]]:
        """
        Return the tags of every column in ``names``.

        Parameters
        ----------
        names : sequence of str
            Column names of the data.

        Returns
        -------
        dict
            Column to sorted tuple of tags (empty when untagged).
        """
        tags: dict[str, list[str]] = {name: [] for name in names}
        for tag, rule in self.tags:
            for column in rule.resolve(
                names, on_missing=self.on_missing_column, field=f"tags.{tag}"
            ):
                tags[column].append(tag)
        return {name: tuple(sorted(values)) for name, values in tags.items()}

    def check_columns(self, names: Sequence[str]) -> None:
        """
        Refuse ``kinds``/``overrides`` keys absent from ``names`` (unless ignored).

        Parameters
        ----------
        names : sequence of str
            Column names of the data.

        Raises
        ------
        PolicyError
            If a key is missing and ``on_missing_column="raise"``.
        """
        if self.on_missing_column == "ignore":
            return
        available = set(names)
        for field, keys in (
            ("kinds", [k for k, _ in self.kinds]),
            ("overrides", [k for k, _ in self.overrides]),
        ):
            missing = [k for k in keys if k not in available]
            if missing:
                raise PolicyError(
                    f"{field} names column(s) {missing} that are not in the data; fix"
                    " the policy or set on_missing_column='ignore'"
                )

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Return the canonical config form, defaults included.

        Returns
        -------
        dict
            A tree of the shared data model with ``schema_version``.
        """
        return {
            "schema_version": POLICY_SCHEMA_VERSION,
            "on_missing_column": self.on_missing_column,
            "roles": {k: r.to_dict() for k, r in self.roles},
            "kinds": dict(self.kinds),
            "tags": {k: r.to_dict() for k, r in self.tags},
            "assessments": {k: s.to_dict() for k, s in self.assessments},
            "overrides": {k: dict(v) for k, v in self.overrides},
            "binning": self.binning.to_dict(),
            "drift": self.drift.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeaturePolicy:
        """
        Build a policy from its config form, checking every field first.

        Parameters
        ----------
        data : mapping
            A document as produced by :meth:`to_dict` or written by hand; every
            key is optional except that ``schema_version``, when given, must
            be the supported generation.

        Returns
        -------
        FeaturePolicy
            The policy.

        Raises
        ------
        PolicyError
            For an unknown key, a wrong type or an out-of-range value.
        IncompatibleSchemaError
            For another ``schema_version``.
        """
        mapping = _require_mapping("policy", data)
        _refuse_unknown("policy", mapping, _POLICY_KEYS)
        if "schema_version" in mapping:
            version = _require_int("schema_version", mapping["schema_version"])
            if version != POLICY_SCHEMA_VERSION:
                raise IncompatibleSchemaError(
                    f"policy schema_version {version} is not supported (this version"
                    f" reads {POLICY_SCHEMA_VERSION})"
                )
        kwargs: dict[str, Any] = {}
        for key in ("roles", "kinds", "tags", "assessments", "overrides"):
            if key in mapping:
                kwargs[key] = dict(_require_mapping(key, mapping[key]))
        if "overrides" in kwargs:
            kwargs["overrides"] = {
                column: dict(_require_mapping(f"overrides.{column}", answers))
                for column, answers in kwargs["overrides"].items()
            }
        for key in ("binning", "drift"):
            if key in mapping:
                kwargs[key] = dict(_require_mapping(key, mapping[key]))
        if "on_missing_column" in mapping:
            kwargs["on_missing_column"] = mapping["on_missing_column"]
        return cls(**kwargs)

    def dumps(self, format: str = "toml") -> str:  # noqa: A002
        """
        Serialise the policy, verified by reading it back.

        Parameters
        ----------
        format : {"toml", "yaml", "json"}, default="toml"
            Output format.

        Returns
        -------
        str
            The document.

        Raises
        ------
        PolicyError
            If the document would not load back equal to this policy.
        """
        if format == "jsonl":
            raise PolicyError("a policy is one document; use json, yaml or toml")
        text = _serialize.dumps(self.to_dict(), format)
        if FeaturePolicy.loads(text, format) != self:
            raise PolicyError(f"policy does not round-trip through {format}")
        return text

    @classmethod
    def loads(cls, text: str, format: str = "toml") -> FeaturePolicy:  # noqa: A002
        """
        Parse a policy document.

        Parameters
        ----------
        text : str
            The document.
        format : {"toml", "yaml", "json"}, default="toml"
            Input format.

        Returns
        -------
        FeaturePolicy
            The policy.
        """
        try:
            tree = _serialize.loads(text, format)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PolicyError):
                raise
            raise PolicyError(f"cannot parse {format} policy: {exc}") from exc
        if tree is None:
            tree = {}
        return cls.from_dict(tree)

    @property
    def digest(self) -> str:
        """str: SHA-256 of the canonical JSON form; equal policies share it."""
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(
            canonical.encode("utf-8", errors="surrogatepass")
        ).hexdigest()


def load_policy(source: Any, format: str | None = None) -> FeaturePolicy:  # noqa: A002
    """
    Load a :class:`FeaturePolicy` from a file, a mapping or a policy.

    Parameters
    ----------
    source : str, os.PathLike, mapping or FeaturePolicy
        A path to a ``.toml``, ``.yaml``/``.yml`` or ``.json`` file, a mapping
        in config form, or a policy (returned unchanged).
    format : str or None, default=None
        Explicit file format, else derived from the suffix.

    Returns
    -------
    FeaturePolicy
        The policy.

    Raises
    ------
    PolicyError, IncompatibleSchemaError
        If the document is off contract.

    Examples
    --------
    >>> from scikitplot.feature_selection import load_policy
    >>> load_policy({"roles": {"target": ["y"]}}).roles_map["target"].columns
    ('y',)
    """
    if isinstance(source, FeaturePolicy):
        return source
    if isinstance(source, Mapping):
        return FeaturePolicy.from_dict(source)
    fmt = _serialize.resolve_format(source, format)
    if fmt == "jsonl":
        raise PolicyError("a policy is one document; use json, yaml or toml")
    import pathlib  # noqa: PLC0415

    text = pathlib.Path(source).read_text(encoding="utf-8")
    return FeaturePolicy.loads(text, fmt)


def dump_policy(
    policy: FeaturePolicy,
    path: Any,
    format: str | None = None,
) -> Any:  # noqa: A002
    """
    Write ``policy`` to ``path`` atomically, verified by reading it back.

    Parameters
    ----------
    policy : FeaturePolicy
        The policy to write.
    path : str or os.PathLike
        Target file.
    format : str or None, default=None
        Explicit format, else derived from the suffix.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    if not isinstance(policy, FeaturePolicy):
        raise TypeError(f"policy must be a FeaturePolicy, got {type(policy).__name__}")
    fmt = _serialize.resolve_format(path, format)
    policy.dumps(fmt)  # verify before touching the file
    return _serialize.write(policy.to_dict(), path, fmt)
