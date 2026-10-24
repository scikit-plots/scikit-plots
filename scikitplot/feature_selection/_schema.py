# scikitplot/feature_selection/_schema.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The fitted feature schema: one self-describing :class:`FeatureSpec` per column.

:func:`profile_features` fits a :class:`FeatureSchema` on *training* data. The
schema is immutable and serialisable; validation, test and production data are
compared against it (:meth:`FeatureSchema.audit`) and never re-profiled into
it.

Each feature knows, independently:

``role``
    ``feature``, ``target``, ``drop``, ``identifier``, ``weight``, ``group``
    or ``time``, from the policy.
``family``
    Physical storage (``integer``, ``float``, ``string``, ``categorical`` ...).
``kind``
    Semantic kind, inferred from the family and the ``binary`` / ``discrete``
    assessments, or declared.
``assessments``
    Four-valued, evidence-carrying answers ("am I skewed?").
``tags``
    User labels from the policy (``one_hot``, ``engineered`` ...).
``binning``
    A frozen bin recommendation for numeric features.

Notes
-----
**User note.** Ask a feature about itself::

    schema["income"].is_("skewed")  # True only for state "true"
    schema["income"].assess("skewed")  # full answer with evidence
    print(schema["income"].explain())  # human summary

and select groups for analysis or a ``ColumnTransformer``::

    schema.select(kind="numeric_continuous", is_not=("skewed",))
    schema.groups(by="kind")

**Developer note.** Kind inference reads the final (override-applied) states of
the ``binary`` and ``discrete`` assessments; there is no second threshold
table, so changing the ``discrete`` parameters in the policy changes kinds and
the ``discrete`` answers together.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pandas as pd

from . import _serialize
from ._assessments import (
    Assessment,
    AssessmentContext,
    AssessmentRegistry,
    default_registry,
    run_assessment,
)
from ._binning import BinningRecommendation, recommend_bins
from ._errors import IncompatibleSchemaError, SchemaMismatchError
from ._policy import FeaturePolicy, load_policy
from ._stats import FeatureStats, compute_stats
from ._types import (
    AssessmentState,
    DtypeFamily,
    FeatureKind,
    FeatureRole,
    coerce_enum,
)

__all__ = ["FeatureSchema", "FeatureSpec", "profile_features"]

#: Generation of the persisted schema format.
SCHEMA_VERSION = 1


def check_frame(X: Any, *, what: str = "X") -> pd.DataFrame:
    """
    Return ``X`` if it is a DataFrame with unique string column labels.

    Parameters
    ----------
    X : object
        Candidate frame.
    what : str, default="X"
        Name used in messages.

    Returns
    -------
    pandas.DataFrame
        ``X`` itself.

    Raises
    ------
    TypeError
        If ``X`` is not a DataFrame or a label is not a string.
    ValueError
        If labels are duplicated.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"{what} must be a pandas DataFrame with named columns, got"
            f" {type(X).__name__}; wrap arrays with pd.DataFrame(X, columns=[...])"
        )
    bad = [c for c in X.columns if not isinstance(c, str)]
    if bad:
        raise TypeError(
            f"{what} column labels must all be strings, got {bad[:5]!r}; rename with"
            " X.columns = X.columns.astype(str)"
        )
    duplicated = X.columns[X.columns.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"{what} has duplicated column labels {duplicated[:5]!r}")
    return X


def _infer_kind(  # ruff: ignore[too-many-return-statements]
    stats: FeatureStats,
    states: Mapping[str, str],
    manual: Mapping[str, str] | None = None,
) -> str:
    family = DtypeFamily(stats.family)
    manual = manual or {}
    binary_refused = manual.get("binary") == AssessmentState.FALSE.value
    if stats.n_non_null == 0:
        return FeatureKind.UNKNOWN.value
    if family is DtypeFamily.DATETIME:
        return FeatureKind.DATETIME.value
    if family is DtypeFamily.TIMEDELTA:
        return FeatureKind.TIMEDELTA.value
    if family is DtypeFamily.OBJECT:  # values of mixed or opaque types
        return FeatureKind.OTHER.value
    if family is DtypeFamily.BOOLEAN and binary_refused:  # a manual "not binary"
        return FeatureKind.CATEGORICAL_NOMINAL.value
    if (
        family is DtypeFamily.BOOLEAN
        or states.get("binary") == AssessmentState.TRUE.value
    ):
        return FeatureKind.BINARY.value
    if family is DtypeFamily.CATEGORICAL:
        return (
            FeatureKind.CATEGORICAL_ORDINAL.value
            if stats.ordered
            else FeatureKind.CATEGORICAL_NOMINAL.value
        )
    if family is DtypeFamily.STRING:
        return FeatureKind.CATEGORICAL_NOMINAL.value
    if family in (DtypeFamily.INTEGER, DtypeFamily.FLOAT):
        if states.get("discrete") == AssessmentState.TRUE.value:
            return FeatureKind.NUMERIC_DISCRETE.value
        return FeatureKind.NUMERIC_CONTINUOUS.value
    return FeatureKind.OTHER.value


def _override(assessment: Assessment, answer: bool) -> Assessment:
    state = AssessmentState.TRUE.value if answer else AssessmentState.FALSE.value
    return dataclasses.replace(
        assessment,
        state=state,
        origin="manual",
        inferred_state=assessment.state,
        reasons=("manual_override", *assessment.reasons),
    )


# --------------------------------------------------------------------------- #
# FeatureSpec
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class FeatureSpec:
    """
    Everything a fitted schema knows about one column.

    Parameters
    ----------
    name : str
        Current column name.
    position : int
        Column position in the training frame.
    role : str
        :class:`~scikitplot.feature_selection.FeatureRole` value.
    family : str
        :class:`~scikitplot.feature_selection.DtypeFamily` value.
    kind : str
        :class:`~scikitplot.feature_selection.FeatureKind` value.
    kind_origin : {"inferred", "declared"}
        Where the kind came from.
    stats : FeatureStats
        Training statistics.
    assessments : tuple of Assessment
        Sorted by name.
    tags : tuple of str, default=()
        User tags.
    binning : BinningRecommendation or None, default=None
        Bin recommendation, when binning is enabled.
    aliases : tuple of str, default=()
        Earlier names, oldest first (see :meth:`FeatureSchema.rename`).

    See Also
    --------
    FeatureSchema : The collection of specs.
    """

    name: str
    position: int
    role: str
    family: str
    kind: str
    kind_origin: str
    stats: FeatureStats
    assessments: tuple[Assessment, ...] = ()
    tags: tuple[str, ...] = ()
    binning: BinningRecommendation | None = None
    aliases: tuple[str, ...] = ()

    # -- self-awareness -------------------------------------------------------

    @property
    def assessment_names(self) -> tuple[str, ...]:
        """Tuple of str: Names of the recorded assessments."""
        return tuple(a.name for a in self.assessments)

    def assess(self, name: str) -> Assessment:
        """
        Return the recorded answer to assessment ``name``.

        Parameters
        ----------
        name : str
            Assessment name (``"skewed"``, ``"discrete"`` ...).

        Returns
        -------
        Assessment
            State, reasons, evidence and provenance.

        Raises
        ------
        KeyError
            If the schema's policy did not run that assessment; the message
            lists the available names.
        """
        for assessment in self.assessments:
            if assessment.name == name:
                return assessment
        raise KeyError(
            f"feature {self.name!r} has no assessment {name!r}; available:"
            f" {list(self.assessment_names)}"
        )

    def state(self, name: str) -> str:
        """
        Return the state of assessment ``name``.

        Parameters
        ----------
        name : str
            Assessment name.

        Returns
        -------
        str
            ``"true"``, ``"false"``, ``"unknown"`` or ``"not_applicable"``.
        """
        return self.assess(name).state

    def is_(self, name: str) -> bool:
        """
        Answer "am I ``name``?" strictly.

        Parameters
        ----------
        name : str
            Assessment name.

        Returns
        -------
        bool
            True only when the state is ``"true"``; ``unknown`` and
            ``not_applicable`` give False. Use :meth:`state` to tell them apart.
        """
        return self.assess(name).is_true

    @property
    def is_model_input(self) -> bool:
        """bool: The role is ``"feature"``."""
        return self.role == FeatureRole.FEATURE.value

    def explain(self) -> str:
        """
        Return a short, human-readable description of the feature.

        Returns
        -------
        str
            Several lines: role, kind, key statistics, true assessments, and
            the binning recommendation.
        """
        s = self.stats
        lines = [
            (
                f"{self.name}: {self.kind} ({self.kind_origin}), role={self.role},"
                f" dtype={s.dtype}"
            ),
            (
                f"  rows={s.n_rows} missing={s.n_missing} ({s.missing_ratio:.1%})"
                f" unique={s.n_unique}"
            ),
        ]
        if s.is_numeric and s.min is not None:
            lines.append(
                f"  min={s.min:g} median={s.median:g} max={s.max:g} skewness="
                f"{'n/a' if s.skewness is None else format(s.skewness, '.3g')}"
            )
        true = [
            a.name + ("*" if a.origin == "manual" else "")
            for a in self.assessments
            if a.is_true
        ]
        unknown = [
            a.name for a in self.assessments if a.state == AssessmentState.UNKNOWN.value
        ]
        lines.append(f"  is: {', '.join(true) or '-'}")
        if unknown:
            lines.append(f"  unknown: {', '.join(unknown)}")
        if self.tags:
            lines.append(f"  tags: {', '.join(self.tags)}")
        if self.binning is not None and self.binning.is_suitable:
            b = self.binning
            lines.append(f"  binning: {b.n_bins} {b.strategy} bins ({b.rule})")
        if self.aliases:
            lines.append(f"  aliases: {', '.join(self.aliases)}")
        return "\n".join(lines)

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the shared-data-model form."""
        return {
            "name": self.name,
            "position": self.position,
            "role": self.role,
            "family": self.family,
            "kind": self.kind,
            "kind_origin": self.kind_origin,
            "stats": self.stats.to_dict(),
            "assessments": [a.to_dict() for a in self.assessments],
            "tags": list(self.tags),
            "binning": None if self.binning is None else self.binning.to_dict(),
            "aliases": list(self.aliases),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeatureSpec:
        """
        Rebuild a spec from :meth:`to_dict` output, checking every field.

        Parameters
        ----------
        data : mapping
            Stored form.

        Returns
        -------
        FeatureSpec
            The spec.

        Raises
        ------
        ValueError
            For an off-contract record.
        """
        allowed = {f.name for f in dataclasses.fields(cls)}
        required = {
            "name",
            "position",
            "role",
            "family",
            "kind",
            "kind_origin",
            "stats",
        }
        if set(data) - allowed or not required <= set(data):
            raise ValueError(f"off-contract feature record: keys {sorted(data)}")
        if not isinstance(data["name"], str):
            raise ValueError(  # ruff: ignore[type-check-without-type-error]
                "feature name must be a string",
            )
        position = data["position"]
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            raise ValueError("feature position must be a non-negative int")
        FeatureRole(data["role"])
        DtypeFamily(data["family"])
        FeatureKind(data["kind"])
        if data["kind_origin"] not in ("inferred", "declared"):
            raise ValueError("kind_origin must be 'inferred' or 'declared'")
        for key in ("assessments", "tags", "aliases"):
            if not isinstance(data.get(key, []), list):
                raise ValueError(  # ruff: ignore[type-check-without-type-error]
                    f"feature {key} must be a list"
                )
        for key in ("tags", "aliases"):
            if not all(isinstance(t, str) for t in data.get(key, [])):
                raise ValueError(f"feature {key} must be strings")
        binning = data.get("binning")
        if (
            isinstance(data["stats"], Mapping)
            and data["stats"].get("name") != data["name"]
        ):
            raise ValueError("feature stats must carry the feature's name")
        if isinstance(binning, Mapping) and binning.get("feature") != data["name"]:
            raise ValueError("feature binning must carry the feature's name")
        return cls(
            name=data["name"],
            position=position,
            role=data["role"],
            family=data["family"],
            kind=data["kind"],
            kind_origin=data["kind_origin"],
            stats=FeatureStats.from_dict(data["stats"]),
            assessments=tuple(
                Assessment.from_dict(a) for a in data.get("assessments", [])
            ),
            tags=tuple(data.get("tags", [])),
            binning=(
                None if binning is None else BinningRecommendation.from_dict(binning)
            ),
            aliases=tuple(data.get("aliases", [])),
        )


# --------------------------------------------------------------------------- #
# FeatureSchema
# --------------------------------------------------------------------------- #

_GROUP_BY = ("kind", "family", "role", "tag", "assessment")


def _as_set(value: Any, enum_type: type | None, field: str) -> set[str] | None:
    if value is None:
        return None
    values = [value] if isinstance(value, str) else list(value)
    if enum_type is not None:
        return {coerce_enum(v, enum_type, field=field).value for v in values}
    for v in values:
        if not isinstance(v, str):
            raise TypeError(f"{field} must be strings, got {v!r}")
    return set(values)


@dataclasses.dataclass(frozen=True)
class FeatureSchema:
    """
    Immutable, fitted description of a table's columns.

    Parameters
    ----------
    features : tuple of FeatureSpec
        One spec per column, in training column order.
    policy : FeaturePolicy
        The policy the schema was fitted with.
    n_rows : int
        Training rows.

    See Also
    --------
    profile_features : Fit a schema.
    FeatureProfiler : Fit a schema inside a scikit-learn pipeline.

    Notes
    -----
    **User note.** A schema is fitted state: build it from training data once,
    keep it (:meth:`save`), and :meth:`audit` later data against it.

    Examples
    --------
    >>> import pandas as pd
    >>> from scikitplot.feature_selection import profile_features
    >>> X = pd.DataFrame(
    ...     {
    ...         "age": [21, 35, 47, 52, 33, 41] * 5,
    ...         "city": ["a", "b", "a", "c", "b", "a"] * 5,
    ...         "vip": [True, False] * 15,
    ...         "churn": [0, 1] * 15,
    ...     }
    ... )
    >>> schema = profile_features(X, {"roles": {"target": ["churn"]}})
    >>> schema.groups(by="kind")
    {'binary': ['vip'], 'categorical_nominal': ['city'], 'numeric_discrete': ['age']}
    >>> schema["age"].is_("discrete"), schema["churn"].role
    (True, 'target')
    """

    features: tuple[FeatureSpec, ...]
    policy: FeaturePolicy
    n_rows: int
    _index: dict = dataclasses.field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:  # noqa: D105
        features = tuple(self.features)
        object.__setattr__(self, "features", features)
        index = {}
        for i, spec in enumerate(features):
            if not isinstance(spec, FeatureSpec):
                raise TypeError("features must be FeatureSpec objects")
            if spec.name in index:
                raise ValueError(f"duplicate feature name {spec.name!r} in schema")
            index[spec.name] = i
        object.__setattr__(self, "_index", index)
        if not isinstance(self.policy, FeaturePolicy):
            raise TypeError("policy must be a FeaturePolicy")

    def __getstate__(self) -> dict[str, Any]:  # noqa: D105
        return {"features": self.features, "policy": self.policy, "n_rows": self.n_rows}

    def __setstate__(self, state: Mapping[str, Any]) -> None:  # noqa: D105
        for key, value in state.items():
            object.__setattr__(self, key, value)
        self.__post_init__()

    # -- container protocol ---------------------------------------------------

    def __len__(self) -> int:  # noqa: D105
        return len(self.features)

    def __iter__(self) -> Iterator[FeatureSpec]:  # noqa: D105
        return iter(self.features)

    def __contains__(self, name: object) -> bool:  # noqa: D105
        return name in self._index

    def __getitem__(self, key: str | int) -> FeatureSpec:
        """
        Return a feature by name (or position).

        Parameters
        ----------
        key : str or int
            Column name, or position.

        Returns
        -------
        FeatureSpec
            The feature.

        Raises
        ------
        KeyError
            If the name is unknown.
        """
        if isinstance(key, int) and not isinstance(key, bool):
            return self.features[key]
        try:
            return self.features[self._index[key]]
        except KeyError:
            raise KeyError(f"no feature named {key!r} in the schema") from None

    feature = __getitem__

    @property
    def names(self) -> list[str]:
        """List of str: Every column name, in training order."""
        return [spec.name for spec in self.features]

    @property
    def feature_names(self) -> list[str]:
        """List of str: Names of the ``role="feature"`` columns."""
        return [spec.name for spec in self.features if spec.is_model_input]

    @property
    def policy_digest(self) -> str:
        """str: Digest of the policy the schema was fitted with."""
        return self.policy.digest

    # -- queries --------------------------------------------------------------

    def select(
        self,
        *,
        kind: Any = None,
        family: Any = None,
        role: Any = "feature",
        tag: Any = None,
        is_: Iterable[str] = (),
        is_not: Iterable[str] = (),
        names: Iterable[str] | None = None,
    ) -> list[str]:
        """
        Return the names of the features that satisfy every given condition.

        Parameters
        ----------
        kind, family : str or sequence of str or None, default=None
            Accept features of any of these kinds / families.
        role : str or sequence of str or None, default="feature"
            Accept these roles; None accepts every role.
        tag : str or sequence of str or None, default=None
            Accept features carrying at least one of these tags.
        is_ : sequence of str, default=()
            Assessments that must all be ``"true"``.
        is_not : sequence of str, default=()
            Assessments that must not be ``"true"`` (``unknown`` passes).
        names : sequence of str or None, default=None
            Restrict to these names.

        Returns
        -------
        list of str
            Matching names in schema order.

        Raises
        ------
        ValueError
            For an unknown kind, family, role or assessment name.

        Examples
        --------
        >>> import pandas as pd
        >>> from scikitplot.feature_selection import profile_features
        >>> X = pd.DataFrame({"a": [1.5, 2.5, 3.5, 95.5] * 5, "b": list("xyzx") * 5})
        >>> schema = profile_features(X)
        >>> schema.select(kind="numeric_continuous", is_=("skewed",))
        ['a']
        """
        kinds = _as_set(kind, FeatureKind, "kind")
        families = _as_set(family, DtypeFamily, "family")
        roles = _as_set(role, FeatureRole, "role")
        tags = _as_set(tag, None, "tag")
        must = tuple(is_) if not isinstance(is_, str) else (is_,)
        must_not = tuple(is_not) if not isinstance(is_not, str) else (is_not,)
        known = set(self.policy.assessments_map)
        for name in must + must_not:
            if name not in known:
                raise ValueError(f"unknown assessment {name!r}; known: {sorted(known)}")
        wanted = None if names is None else set(names)
        out = []
        for spec in self.features:
            if wanted is not None and spec.name not in wanted:
                continue
            if kinds is not None and spec.kind not in kinds:
                continue
            if families is not None and spec.family not in families:
                continue
            if roles is not None and spec.role not in roles:
                continue
            if tags is not None and not tags.intersection(spec.tags):
                continue
            states = {a.name: a.is_true for a in spec.assessments}
            if any(not states.get(n, False) for n in must):
                continue
            if any(states.get(n, False) for n in must_not):
                continue
            out.append(spec.name)
        return out

    def groups(
        self, by: str = "kind", *, role: Any = "feature"
    ) -> dict[str, list[str]]:
        """
        Group feature names.

        Parameters
        ----------
        by : {"kind", "family", "role", "tag", "assessment"}, default="kind"
            Grouping key. ``"assessment"`` lists, per assessment, the features
            whose state is ``"true"``; ``"tag"`` lists, per tag, the tagged
            features (a feature can be in several groups).
        role : str or sequence of str or None, default="feature"
            Only features with these roles; None for all.

        Returns
        -------
        dict of str to list of str
            Group name (sorted) to member names (schema order); empty groups
            are omitted.
        """
        if by not in _GROUP_BY:
            raise ValueError(f"by={by!r} is not one of {_GROUP_BY}")
        roles = _as_set(role, FeatureRole, "role")
        groups: dict[str, list[str]] = {}
        for spec in self.features:
            if roles is not None and spec.role not in roles:
                continue
            if by in ("kind", "family", "role"):
                keys: Sequence[str] = (getattr(spec, by),)
            elif by == "tag":
                keys = spec.tags
            else:
                keys = [a.name for a in spec.assessments if a.is_true]
            for key in keys:
                groups.setdefault(key, []).append(spec.name)
        return dict(sorted(groups.items()))

    def summary(self) -> pd.DataFrame:
        """
        Return one row per feature with role, kind, key statistics and states.

        Returns
        -------
        pandas.DataFrame
            Indexed by feature name. Assessment columns hold state strings.
        """
        rows = []
        for spec in self.features:
            s = spec.stats
            row: dict[str, Any] = {
                "feature": spec.name,
                "role": spec.role,
                "kind": spec.kind,
                "family": spec.family,
                "dtype": s.dtype,
                "n_missing": s.n_missing,
                "missing_ratio": s.missing_ratio,
                "n_unique": s.n_unique,
                "unique_ratio": s.unique_ratio,
                "top_ratio": s.top_ratio,
                "min": s.min,
                "median": s.median,
                "max": s.max,
                "mean": s.mean,
                "std": s.std,
                "skewness": s.skewness,
                "tags": ",".join(spec.tags),
                "n_bins": None if spec.binning is None else spec.binning.n_bins,
            }
            for a in spec.assessments:
                row[a.name] = a.state
            rows.append(row)
        return pd.DataFrame(rows).set_index("feature") if rows else pd.DataFrame()

    def assessment_table(self) -> pd.DataFrame:
        """
        Return a features-by-assessments table of states.

        Returns
        -------
        pandas.DataFrame
            Index: feature names; columns: assessment names; values: state strings.
        """
        return pd.DataFrame(
            {
                spec.name: {a.name: a.state for a in spec.assessments}
                for spec in self.features
            }
        ).T

    def binning_plan(self) -> pd.DataFrame:
        """
        Return the binning recommendation of every numeric feature.

        Returns
        -------
        pandas.DataFrame
            Columns ``state``, ``rule``, ``n_bins``, ``rule_n_bins``, ``strategy``,
            ``edges``, ``reasons``; indexed by feature.
        """
        rows = [
            {**spec.binning.to_dict()}
            for spec in self.features
            if spec.binning is not None
        ]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("feature")

    # -- later data -----------------------------------------------------------

    def audit(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compare later data's columns and dtypes with the fitted schema.

        Parameters
        ----------
        X : pandas.DataFrame
            Validation, test or production data.

        Returns
        -------
        pandas.DataFrame
            One row per finding with columns ``column``, ``issue``
            (``missing_column``, ``extra_column``, ``order_changed``,
            ``family_changed``, ``dtype_changed``), ``expected``, ``observed``.
            Empty when the structure matches. ``dtype_changed`` with an
            unchanged family (``int64`` to ``Int64``) is informational.

        Notes
        -----
        The schema is not changed. Distribution changes are the job of
        :class:`~scikitplot.feature_selection.ReferenceProfile`.
        """
        from ._stats import dtype_family  # noqa: PLC0415

        X = check_frame(X)
        observed = list(X.columns)
        expected = self.names
        present = set(observed)
        rows = [
            (name, "missing_column", "present", "absent")
            for name in expected
            if name not in present
        ]
        rows.extend(
            (name, "extra_column", "absent", "present")
            for name in observed
            if name not in self._index
        )
        common_expected = [n for n in expected if n in present]
        common_observed = [n for n in observed if n in self._index]
        if common_expected != common_observed:
            rows.append(
                (
                    "*",
                    "order_changed",
                    ",".join(common_expected),
                    ",".join(common_observed),
                )
            )
        for name in common_expected:
            spec = self[name]
            family = dtype_family(X[name]).value
            if family != spec.family:
                rows.append((name, "family_changed", spec.family, family))
            elif str(X[name].dtype) != spec.stats.dtype:
                rows.append(
                    (name, "dtype_changed", spec.stats.dtype, str(X[name].dtype))
                )
        return pd.DataFrame(rows, columns=["column", "issue", "expected", "observed"])

    def check(
        self, X: pd.DataFrame, *, allow: Iterable[str] = ("dtype_changed",)
    ) -> None:
        """
        Raise when :meth:`audit` finds an issue that is not allowed.

        Parameters
        ----------
        X : pandas.DataFrame
            Later data.
        allow : sequence of str, default=("dtype_changed",)
            Issue codes to tolerate.

        Raises
        ------
        SchemaMismatchError
            Listing every disallowed finding.
        """
        findings = self.audit(X)
        allowed = set(allow)
        bad = findings[~findings["issue"].isin(allowed)]
        if len(bad):
            lines = "; ".join(
                f"{r.column}: {r.issue} (expected {r.expected}, observed {r.observed})"
                for r in bad.itertuples()
            )
            raise SchemaMismatchError(f"data does not match the fitted schema: {lines}")

    # -- renaming -------------------------------------------------------------

    def rename(self, mapping: Mapping[str, str]) -> FeatureSchema:
        """
        Return a schema whose features are renamed, keeping old names as aliases.

        Parameters
        ----------
        mapping : mapping of str to str
            Old name to new name; unlisted features keep their names.

        Returns
        -------
        FeatureSchema
            The renamed schema (identity, stats and assessments unchanged).

        Raises
        ------
        KeyError
            If a key is not a feature name.
        ValueError
            If the result has duplicate names.
        """
        unknown = [k for k in mapping if k not in self._index]
        if unknown:
            raise KeyError(f"cannot rename unknown feature(s) {unknown}")
        renamed = []
        for spec in self.features:
            new = mapping.get(spec.name, spec.name)
            if not isinstance(new, str) or not new:
                raise TypeError(
                    f"new name for {spec.name!r} must be a non-empty string"
                )
            if new == spec.name:
                renamed.append(spec)
                continue
            renamed.append(
                dataclasses.replace(
                    spec,
                    name=new,
                    aliases=(*spec.aliases, spec.name),
                    stats=dataclasses.replace(spec.stats, name=new),
                    binning=(
                        None
                        if spec.binning is None
                        else dataclasses.replace(spec.binning, feature=new)
                    ),
                )
            )
        return FeatureSchema(
            features=tuple(renamed), policy=self.policy, n_rows=self.n_rows
        )

    def resolve_alias(self, name: str) -> str:
        """
        Return the current name of a feature known by ``name`` or an alias.

        Parameters
        ----------
        name : str
            Current name or any earlier name.

        Returns
        -------
        str
            The current name.

        Raises
        ------
        KeyError
            If no feature has that name or alias.
        """
        if name in self._index:
            return name
        for spec in self.features:
            if name in spec.aliases:
                return spec.name
        raise KeyError(f"no feature named or aliased {name!r}")

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Return the persisted form (schema version, policy, digest, features).

        Returns
        -------
        dict
            A tree of the shared data model.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact": "feature_schema",
            "n_rows": self.n_rows,
            "policy_digest": self.policy.digest,
            "policy": self.policy.to_dict(),
            "features": [spec.to_dict() for spec in self.features],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeatureSchema:
        """
        Rebuild a schema from :meth:`to_dict` output.

        Parameters
        ----------
        data : mapping
            Stored form.

        Returns
        -------
        FeatureSchema
            The schema.

        Raises
        ------
        IncompatibleSchemaError
            For another ``schema_version`` or artefact type.
        ValueError
            For an off-contract record or a policy digest that does not match
            the stored policy.
        """
        if not isinstance(data, Mapping):
            raise ValueError(  # ruff: ignore[type-check-without-type-error]
                "a schema document must be a mapping",
            )
        version = data.get("schema_version")
        if data.get("artifact") != "feature_schema":
            raise IncompatibleSchemaError(
                f"not a feature schema document (artifact={data.get('artifact')!r})"
            )
        if isinstance(version, bool) or version != SCHEMA_VERSION:
            raise IncompatibleSchemaError(
                f"feature schema version {version!r} is not supported (this version"
                f" reads {SCHEMA_VERSION})"
            )
        expected = {
            "schema_version",
            "artifact",
            "n_rows",
            "policy_digest",
            "policy",
            "features",
        }
        if set(data) != expected:
            raise ValueError(f"off-contract schema document: keys {sorted(data)}")
        policy = FeaturePolicy.from_dict(data["policy"])
        if policy.digest != data["policy_digest"]:
            raise ValueError("the stored policy does not match its recorded digest")
        n_rows = data["n_rows"]
        if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 0:
            raise ValueError("n_rows must be a non-negative int")
        if not isinstance(data["features"], list):
            raise ValueError(  # ruff: ignore[type-check-without-type-error]
                "features must be a list"
            )
        features = tuple(FeatureSpec.from_dict(f) for f in data["features"])
        positions = [f.position for f in features]
        if len(set(positions)) != len(positions):
            raise ValueError("feature positions must be distinct")
        if any(f.stats.n_rows != n_rows for f in features):
            raise ValueError("every feature's n_rows must equal the schema's n_rows")
        return cls(features=features, policy=policy, n_rows=n_rows)

    def dumps(self, format: str = "json") -> str:  # noqa: A002
        """
        Serialise the schema (verified by reading it back).

        Parameters
        ----------
        format : {"json", "yaml", "toml"}, default="json"
            Output format.

        Returns
        -------
        str
            The document.
        """
        text = _serialize.dumps(self.to_dict(), format)
        if FeatureSchema.loads(text, format) != self:
            raise ValueError(f"schema does not round-trip through {format}")
        return text

    @classmethod
    def loads(cls, text: str, format: str = "json") -> FeatureSchema:  # noqa: A002
        """
        Parse a schema document.

        Parameters
        ----------
        text : str
            The document.
        format : {"json", "yaml", "toml"}, default="json"
            Input format.

        Returns
        -------
        FeatureSchema
            The schema.
        """
        return cls.from_dict(_serialize.loads(text, format))

    def save(self, path: Any, format: str | None = None) -> pathlib.Path:  # noqa: A002
        """
        Write the schema to ``path`` atomically.

        Parameters
        ----------
        path : str or os.PathLike
            Target file (``.json``, ``.yaml``, ``.toml``).
        format : str or None, default=None
            Explicit format.

        Returns
        -------
        pathlib.Path
            The written path.
        """
        fmt = _serialize.resolve_format(path, format)
        self.dumps(fmt)
        return _serialize.write(self.to_dict(), path, fmt)

    @classmethod
    def load(cls, path: Any, format: str | None = None) -> FeatureSchema:  # noqa: A002
        """
        Read a schema written by :meth:`save`.

        Parameters
        ----------
        path : str or os.PathLike
            Source file.
        format : str or None, default=None
            Explicit format.

        Returns
        -------
        FeatureSchema
            The schema.
        """
        return cls.from_dict(_serialize.read(path, format))


# --------------------------------------------------------------------------- #
# profile_features
# --------------------------------------------------------------------------- #


def profile_features(
    X: pd.DataFrame,
    policy: Any = None,
    *,
    registry: AssessmentRegistry | None = None,
) -> FeatureSchema:
    """
    Fit a :class:`FeatureSchema` on training data.

    Parameters
    ----------
    X : pandas.DataFrame
        Training data with unique string column labels. Target and other
        non-feature columns may be included and are given their roles by the
        policy.
    policy : FeaturePolicy, mapping, path or None, default=None
        Policy, its config form, or a TOML/YAML/JSON file; None uses defaults.
    registry : AssessmentRegistry or None, default=None
        Provider registry; the shared default when None.

    Returns
    -------
    FeatureSchema
        The fitted schema.

    Raises
    ------
    TypeError, ValueError
        For a non-DataFrame, non-string or duplicated labels.
    PolicyError
        For a policy that names missing columns (unless ignored), an unknown
        provider or invalid provider parameters.

    See Also
    --------
    FeatureProfiler : The same inside a scikit-learn pipeline.

    Notes
    -----
    **User note.** Call it on the training split only. Profiling validation or
    test data separately can classify columns differently and leaks their
    statistics into preprocessing decisions; use
    :meth:`FeatureSchema.audit` and
    :class:`~scikitplot.feature_selection.ReferenceProfile` for later data.

    Examples
    --------
    >>> import pandas as pd
    >>> from scikitplot.feature_selection import profile_features
    >>> X = pd.DataFrame({"id": [f"c{i}" for i in range(30)], "x": [0.5] * 30})
    >>> schema = profile_features(X)
    >>> schema["id"].is_("identifier_like"), schema["x"].is_("constant")
    (True, True)
    """
    X = check_frame(X)
    policy = load_policy(FeaturePolicy() if policy is None else policy)
    registry = default_registry if registry is None else registry
    names = list(X.columns)
    roles = policy.resolve_roles(names)
    tags = policy.resolve_tags(names)
    policy.check_columns(names)
    declared = policy.kinds_map
    overrides = policy.overrides_map
    runners = []
    for name, spec in policy.assessments:
        if not spec.enabled:
            continue
        provider = registry.get(spec.provider)
        params = provider.resolve_params(spec.params_dict, field=f"assessments.{name}")
        runners.append((name, spec, params, provider))
    features = []
    for position, column in enumerate(names):
        series = X[column]
        stats = compute_stats(series, column)
        context = AssessmentContext(column, stats, series, declared.get(column))
        assessments = []
        for name, spec, params, provider in runners:
            assessment = run_assessment(name, spec, params, provider, context)
            answer = overrides.get(column, {}).get(name)
            if answer is not None:
                assessment = _override(assessment, answer)
            assessments.append(assessment)
        for name in overrides.get(column, {}):
            if name not in {a.name for a in assessments}:
                raise ValueError(
                    f"overrides.{column}.{name}: assessment {name!r} is disabled"
                )
        states = {a.name: a.state for a in assessments}
        if column in declared:
            kind, origin = declared[column], "declared"
        else:
            manual = {a.name: a.state for a in assessments if a.origin == "manual"}
            kind, origin = _infer_kind(stats, states, manual), "inferred"
        binning = (
            recommend_bins(series, stats, policy.binning, declared.get(column))
            if policy.binning.enabled
            else None
        )
        features.append(
            FeatureSpec(
                name=column,
                position=position,
                role=roles[column],
                family=stats.family,
                kind=kind,
                kind_origin=origin,
                stats=stats,
                assessments=tuple(assessments),
                tags=tags[column],
                binning=binning,
            )
        )
    return FeatureSchema(features=tuple(features), policy=policy, n_rows=len(X))
