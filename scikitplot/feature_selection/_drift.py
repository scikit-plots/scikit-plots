# scikitplot/feature_selection/_drift.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Per-feature drift against a frozen training reference.

:meth:`ReferenceProfile.fit` freezes, for every feature, the buckets and their
training proportions; :meth:`ReferenceProfile.drift` puts later data into the
*same* buckets and compares. Nothing is re-learned from the later data.

Buckets by kind
---------------
``numeric_continuous``
    Quantile cut points of the finite training values (ties merged); values
    equal to a cut point go to the upper bucket, infinities to the end buckets.
``binary``, ``numeric_discrete``, ``categorical_nominal``, ``categorical_ordinal``
    The ``max_categories`` most frequent training values by name, plus one
    ``other`` bucket.
every kind
    One extra bucket for missing values, so a change in missingness is drift.
``datetime``, ``timedelta``, ``text``, ``other``, ``unknown``, and identifier-like features
    Not compared (status ``not_applicable``): identifiers change by design.

Metrics
-------
``psi``
    Population stability index, ``sum((c - r) * ln(c / r))`` with both
    proportions floored at ``epsilon``.
``js_distance``
    Jensen-Shannon distance (base 2, in ``[0, 1]``) of the unfloored proportions.
``missing_delta``
    Current minus reference missing ratio.
``unseen_ratio``
    Share of current non-missing categorical values never seen in training;
    ``None`` when the reference kept only the top categories (it cannot know).

Notes
-----
**User note.** Fit the reference on the training split (or on the frame a
deployed model was validated on) and keep it with the model
(:meth:`ReferenceProfile.save`). For transformed features (one-hot columns,
PCA components) fit a reference on ``pipeline[:-1].transform(X_train)`` and map
findings back to raw columns with
:meth:`~scikitplot.feature_selection.FeatureLineage.origins`.

**Developer note.** Categories are keyed by text: whole numbers by their
integer form (so ``3`` and ``3.0`` share a key when a column gains missing
values and becomes float), everything else by ``str(value)``. Two training
values of different types with one key (``1`` and ``"1"``) are refused at fit
time rather than silently merged.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import _serialize
from ._errors import IncompatibleSchemaError
from ._policy import DriftPolicy
from ._schema import FeatureSchema, check_frame, profile_features
from ._stats import dtype_family, numeric_values
from ._types import DtypeFamily, FeatureKind

__all__ = ["FeatureReference", "ReferenceProfile"]

DRIFT_VERSION = 1

_NUMERIC = (FeatureKind.NUMERIC_CONTINUOUS.value,)
_CATEGORICAL = (
    FeatureKind.BINARY.value,
    FeatureKind.NUMERIC_DISCRETE.value,
    FeatureKind.CATEGORICAL_NOMINAL.value,
    FeatureKind.CATEGORICAL_ORDINAL.value,
)
_REPORT_COLUMNS = [
    "feature",
    "kind",
    "method",
    "n_reference",
    "n_current",
    "psi",
    "js_distance",
    "missing_reference",
    "missing_current",
    "missing_delta",
    "unseen_ratio",
    "status",
]


def _family_group(family: str) -> str:
    """Families whose values compare in the same buckets share a group."""
    if family in (DtypeFamily.INTEGER.value, DtypeFamily.FLOAT.value):
        return "numeric"
    if family in (DtypeFamily.STRING.value, DtypeFamily.CATEGORICAL.value):
        return "text"
    return family


def _key(value: Any) -> str:
    """Text key of a category; numerically equal integers and whole floats share one."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _equal(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except (TypeError, ValueError):  # incomparable values are different values
        return False


def _key_counts(
    series: pd.Series, *, refuse_collisions: bool = False
) -> dict[str, int]:
    counts: dict[str, int] = {}
    sources: dict[str, Any] = {}
    for value, count in series.value_counts(dropna=True).items():
        key = _key(value)
        if refuse_collisions and key in sources and not _equal(sources[key], value):
            raise ValueError(
                f"{series.name!r}: values {sources[key]!r} and {value!r} share the"
                f" category key {key!r}; cast the column to one type before fitting a"
                " drift reference"
            )
        sources.setdefault(key, value)
        counts[key] = counts.get(key, 0) + int(count)
    return counts


@dataclasses.dataclass(frozen=True)
class FeatureReference:
    """
    Frozen reference buckets of one feature.

    Parameters
    ----------
    name : str
        Column name.
    kind : str
        Feature kind at fit time.
    method : {"numeric", "categorical", "skipped"}
        Bucketing method.
    cuts : tuple of float
        Interior cut points (numeric).
    categories : tuple of str
        Kept category keys (categorical).
    proportions : tuple of float
        Training proportion of each bucket; the last bucket is "missing",
        and for categorical features the one before it is "other".
    truncated : bool
        More training categories existed than were kept.
    n_reference : int
        Training rows.
    note : str
        Why a feature is skipped.
    family : str
        Dtype family of the training column; later data of another family
        is reported as ``type_changed`` (integer and float are one numeric
        family, string and categorical one text family).
    """

    name: str
    kind: str
    method: str
    cuts: tuple[float, ...] = ()
    categories: tuple[str, ...] = ()
    proportions: tuple[float, ...] = ()
    truncated: bool = False
    n_reference: int = 0
    note: str = ""
    family: str = ""

    def bucket_counts(self, series: pd.Series) -> np.ndarray:
        """
        Count the values of ``series`` per reference bucket.

        Parameters
        ----------
        series : pandas.Series
            Values to bucket.

        Returns
        -------
        numpy.ndarray of int
            One count per bucket (missing last).

        Raises
        ------
        TypeError
            If a numeric reference receives values that are not numeric.
        """
        n_missing = int(series.isna().sum())
        if self.method == "numeric":
            try:
                values = numeric_values(series)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{self.name!r}: values are no longer numeric ({exc})"
                ) from None
            index = np.searchsorted(np.asarray(self.cuts), values, side="right")
            counts = np.bincount(index, minlength=len(self.cuts) + 1)
        else:
            position = {c: i for i, c in enumerate(self.categories)}
            counts = np.zeros(len(self.categories) + 1, dtype=int)
            for key, count in _key_counts(series).items():
                counts[position.get(key, len(self.categories))] += count
        return np.append(counts, n_missing)

    def unseen_ratio(self, series: pd.Series) -> float | None:
        """
        Return the share of non-missing values never seen in training.

        Parameters
        ----------
        series : pandas.Series
            Current values.

        Returns
        -------
        float or None
            None for numeric or truncated references, or no current values.
        """
        if self.method != "categorical" or self.truncated:
            return None
        counts = _key_counts(series)
        total = sum(counts.values())
        if not total:
            return None
        known = set(self.categories)
        return float(sum(c for k, c in counts.items() if k not in known) / total)

    def to_dict(self) -> dict[str, Any]:
        """Return the shared-data-model form."""
        data = dataclasses.asdict(self)
        for key in ("cuts", "categories", "proportions"):
            data[key] = list(data[key])
        return data

    @classmethod
    def from_dict(  # ruff: ignore[too-many-branches]
        cls,
        data: Mapping[str, Any],
    ) -> FeatureReference:
        """
        Rebuild from :meth:`to_dict` output, checking every field.

        Parameters
        ----------
        data : mapping
            Stored form.

        Returns
        -------
        FeatureReference
            The reference.
        """
        allowed = {f.name for f in dataclasses.fields(cls)}
        if isinstance(data, Mapping) and "family" not in data:
            data = {**data, "family": ""}  # written before the family was recorded
        if not isinstance(data, Mapping) or set(data) != allowed:
            keys = sorted(data) if isinstance(data, Mapping) else data
            raise ValueError(f"off-contract feature reference: keys {keys}")
        for key in ("name", "kind", "method", "note", "family"):
            if not isinstance(data[key], str):
                raise ValueError(  # ruff: ignore[type-check-without-type-error]
                    f"reference {key} must be a string",
                )
        FeatureKind(data["kind"])
        if data["family"]:
            DtypeFamily(data["family"])
        method = data["method"]
        if method not in ("numeric", "categorical", "skipped"):
            raise ValueError(f"unknown reference method {method!r}")
        for key in ("cuts", "proportions"):
            if not isinstance(data[key], list) or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                for v in data[key]
            ):
                raise ValueError(f"reference {key} must be a list of finite numbers")
        categories = data["categories"]
        if not isinstance(categories, list) or not all(
            isinstance(c, str) for c in categories
        ):
            raise ValueError("reference categories must be a list of strings")
        if len(set(categories)) != len(categories):
            raise ValueError("reference categories must be distinct")
        if not isinstance(data["truncated"], bool):
            raise ValueError(  # ruff: ignore[type-check-without-type-error]
                "reference truncated must be a bool",
            )
        n_reference = data["n_reference"]
        if (
            isinstance(n_reference, bool)
            or not isinstance(n_reference, int)
            or (n_reference < 0)
        ):
            raise ValueError("reference n_reference must be a non-negative int")
        cuts, proportions = data["cuts"], data["proportions"]
        if any(b <= a for a, b in zip(cuts, cuts[1:])):
            raise ValueError("reference cuts must be strictly increasing")
        buckets = {
            "numeric": len(cuts) + 2,
            "categorical": len(categories) + 2,
            "skipped": 0,
        }[method]
        if len(proportions) != buckets:
            raise ValueError(
                f"reference {data['name']!r} has {len(proportions)} proportions for"
                f" {buckets} buckets"
            )
        _eps = 1e-9  # ruff: ignore[magic-value-comparison]
        if proportions and (
            min(proportions) < 0 or abs(math.fsum(proportions) - 1.0) > _eps
        ):
            raise ValueError("reference proportions must be >= 0 and sum to 1")
        return cls(
            name=data["name"],
            kind=data["kind"],
            method=method,
            cuts=tuple(float(v) for v in cuts),
            categories=tuple(categories),
            proportions=tuple(float(v) for v in proportions),
            truncated=data["truncated"],
            n_reference=n_reference,
            note=data["note"],
            family=data["family"],
        )


def _fit_reference(series: pd.Series, name: str, kind: str, policy: DriftPolicy):
    n = len(series)
    if n == 0:
        return FeatureReference(
            name, kind, "skipped", n_reference=0, note="empty_frame"
        )
    missing = int(series.isna().sum())
    if kind in _NUMERIC:
        values = numeric_values(series)
        finite = values[np.isfinite(values)]
        if not finite.size:
            return FeatureReference(
                name, kind, "skipped", n_reference=n, note="no_finite_values"
            )
        quantiles = np.quantile(finite, np.linspace(0.0, 1.0, policy.n_bins + 1)[1:-1])
        cuts = tuple(float(c) for c in np.unique(quantiles))
        ref = FeatureReference(name, kind, "numeric", cuts=cuts, n_reference=n)
    elif kind in _CATEGORICAL:
        counts = _key_counts(series, refuse_collisions=True)
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        kept = tuple(k for k, _ in ordered[: policy.max_categories])
        ref = FeatureReference(
            name,
            kind,
            "categorical",
            categories=kept,
            truncated=len(ordered) > policy.max_categories,
            n_reference=n,
        )
    else:
        return FeatureReference(
            name, kind, "skipped", n_reference=n, note="kind_not_compared"
        )
    counts = ref.bucket_counts(series)
    if int(counts.sum()) != n or int(counts[-1]) != missing:
        raise RuntimeError(f"{name!r}: reference buckets do not cover every row")
    return dataclasses.replace(
        ref,
        proportions=tuple(float(c) / n for c in counts),
        family=dtype_family(series).value,
    )


def _psi(reference: np.ndarray, current: np.ndarray, epsilon: float) -> float:
    r = np.maximum(reference, epsilon)
    c = np.maximum(current, epsilon)
    return float(np.sum((c - r) * np.log(c / r)))


def _js_distance(reference: np.ndarray, current: np.ndarray) -> float:
    m = 0.5 * (reference + current)

    def kl(p: np.ndarray) -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / m[mask])))

    value = 0.5 * kl(reference) + 0.5 * kl(current)
    return math.sqrt(max(value, 0.0))


@dataclasses.dataclass(frozen=True)
class ReferenceProfile:
    """
    Frozen training distribution of every compared feature.

    Parameters
    ----------
    features : tuple of FeatureReference
        One reference per feature.
    policy : DriftPolicy
        Bucketing and judging settings.

    See Also
    --------
    FeatureSchema.audit : Structural (column/dtype) comparison.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from scikitplot.feature_selection import ReferenceProfile
    >>> rng = np.random.default_rng(0)
    >>> train = pd.DataFrame(
    ...     {"x": rng.normal(size=500), "c": rng.choice(list("ab"), 500)}
    ... )
    >>> later = pd.DataFrame(
    ...     {"x": rng.normal(2.0, 1.0, 500), "c": rng.choice(list("ab"), 500)}
    ... )
    >>> report = ReferenceProfile.fit(train).drift(later)
    >>> report.set_index("feature")["status"].to_dict()
    {'x': 'drift', 'c': 'stable'}
    """

    features: tuple[FeatureReference, ...]
    policy: DriftPolicy = dataclasses.field(default_factory=DriftPolicy)

    @classmethod
    def fit(
        cls,
        X: pd.DataFrame,
        schema: FeatureSchema | None = None,
        *,
        policy: Any = None,
        columns: Sequence[str] | None = None,
    ) -> ReferenceProfile:
        """
        Freeze the reference buckets of ``X``.

        Parameters
        ----------
        X : pandas.DataFrame
            Reference (training) data.
        schema : FeatureSchema or None, default=None
            Kinds to use; profiled from ``X`` with ``policy`` when None.
        policy : FeaturePolicy, mapping, path or None, default=None
            Used to profile ``X`` when ``schema`` is None; its ``drift``
            section configures the reference.
        columns : sequence of str or None, default=None
            Columns to compare; defaults to the ``role="feature"`` columns.

        Returns
        -------
        ReferenceProfile
            The reference.
        """
        X = check_frame(X)
        if schema is None:
            schema = profile_features(X, policy)
        drift_policy = schema.policy.drift
        names = schema.feature_names if columns is None else list(columns)
        refs = []
        for name in names:
            if name not in X.columns:
                raise KeyError(f"column {name!r} is not in X")
            # not dict
            spec = (
                schema[name]  # ruff: ignore[if-else-block-instead-of-dict-get]
                if name in schema
                else None
            )
            kind = spec.kind if spec is not None else FeatureKind.UNKNOWN.value
            if (
                spec is not None
                and "identifier_like" in spec.assessment_names
                and spec.is_("identifier_like")
            ):
                refs.append(
                    FeatureReference(
                        name,
                        kind,
                        "skipped",
                        n_reference=len(X),
                        note="identifier_like",
                    )
                )
                continue
            refs.append(_fit_reference(X[name], name, kind, drift_policy))
        return cls(features=tuple(refs), policy=drift_policy)

    @property
    def names(self) -> list[str]:
        """List of str: Compared feature names."""
        return [f.name for f in self.features]

    def drift(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compare ``X`` with the reference, feature by feature.

        Parameters
        ----------
        X : pandas.DataFrame
            Current data.

        Returns
        -------
        pandas.DataFrame
            One row per reference feature with the columns ``feature``,
            ``kind``, ``method``, ``n_reference``, ``n_current``, ``psi``,
            ``js_distance``, ``missing_reference``, ``missing_current``,
            ``missing_delta``, ``unseen_ratio`` and ``status`` (``stable``,
            ``warn``, ``drift``, ``insufficient``, ``missing_column``,
            ``type_changed``, ``not_applicable``).
        """
        X = check_frame(X)
        p = self.policy
        rows = []
        for ref in self.features:
            row: dict[str, Any] = dict.fromkeys(_REPORT_COLUMNS)
            row.update(
                feature=ref.name,
                kind=ref.kind,
                method=ref.method,
                n_reference=ref.n_reference,
            )
            if ref.method == "skipped":
                row["status"] = "not_applicable"
                rows.append(row)
                continue
            if ref.name not in X.columns:
                row["status"] = "missing_column"
                rows.append(row)
                continue
            series = X[ref.name]
            if (
                ref.family
                and series.notna().any()  # an all-missing column has no type to judge
                and _family_group(ref.family)
                != _family_group(dtype_family(series).value)
            ):
                row.update(n_current=len(series), status="type_changed")
                rows.append(row)
                continue
            try:
                counts = ref.bucket_counts(series)
            except TypeError:
                row.update(n_current=len(series), status="type_changed")
                rows.append(row)
                continue
            n = int(counts.sum())
            reference = np.asarray(ref.proportions)
            current = counts / n if n else np.zeros_like(reference)
            row.update(
                n_current=n,
                missing_reference=float(reference[-1]),
                missing_current=float(current[-1]) if n else None,
            )
            if n:
                row["missing_delta"] = row["missing_current"] - row["missing_reference"]
            non_missing = n - int(counts[-1])
            if non_missing < p.min_count:
                row["status"] = "insufficient"
                rows.append(row)
                continue
            psi = _psi(reference, current, p.epsilon)
            row.update(
                psi=psi,
                js_distance=_js_distance(reference, current),
                unseen_ratio=ref.unseen_ratio(series),
            )

            def get_status(psi, p):
                if psi >= p.psi_alert:
                    return "drift"
                return "warn" if psi >= p.psi_warn else "stable"

            row["status"] = get_status(psi, p)
            rows.append(row)
        return pd.DataFrame(rows, columns=_REPORT_COLUMNS)

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the persisted form."""
        return {
            "schema_version": DRIFT_VERSION,
            "artifact": "reference_profile",
            "policy": self.policy.to_dict(),
            "features": [f.to_dict() for f in self.features],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReferenceProfile:
        """
        Rebuild a reference from :meth:`to_dict` output.

        Parameters
        ----------
        data : mapping
            Stored form.

        Returns
        -------
        ReferenceProfile
            The reference.

        Raises
        ------
        IncompatibleSchemaError
            For another generation or artefact type.
        """
        if not isinstance(data, Mapping) or data.get("artifact") != "reference_profile":
            raise IncompatibleSchemaError("not a reference profile document")
        version = data.get("schema_version")
        if isinstance(version, bool) or version != DRIFT_VERSION:
            raise IncompatibleSchemaError(
                f"reference schema_version {version!r} is not supported"
            )
        if set(data) != {"schema_version", "artifact", "policy", "features"}:
            raise ValueError(f"off-contract reference document: keys {sorted(data)}")
        from ._policy import _section  # noqa: PLC0415

        if not isinstance(data["features"], list):
            raise ValueError(  # ruff: ignore[type-check-without-type-error]
                "reference features must be a list",
            )
        features = tuple(FeatureReference.from_dict(f) for f in data["features"])
        names = [f.name for f in features]
        if len(set(names)) != len(names):
            raise ValueError("reference features must have distinct names")
        return cls(
            features=features,
            policy=_section(DriftPolicy, "drift", data["policy"]),
        )

    def dumps(self, format: str = "json") -> str:  # noqa: A002
        """
        Serialise the reference (verified by reading it back).

        Parameters
        ----------
        format : {"json", "yaml", "toml"}, default="json"
            Format.

        Returns
        -------
        str
            The document.
        """
        text = _serialize.dumps(self.to_dict(), format)
        if ReferenceProfile.loads(text, format) != self:
            raise ValueError(f"reference does not round-trip through {format}")
        return text

    @classmethod
    def loads(
        cls,
        text: str,
        format: str = "json",
    ) -> ReferenceProfile:  # noqa: A002
        """
        Parse a reference document.

        Parameters
        ----------
        text : str
            The document.
        format : {"json", "yaml", "toml"}, default="json"
            Format.

        Returns
        -------
        ReferenceProfile
            The reference.
        """
        return cls.from_dict(_serialize.loads(text, format))

    def save(self, path: Any, format: str | None = None) -> pathlib.Path:  # noqa: A002
        """
        Write the reference atomically.

        Parameters
        ----------
        path : str or os.PathLike
            Target file.
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
    def load(
        cls,
        path: Any,
        format: str | None = None,
    ) -> ReferenceProfile:  # noqa: A002
        """
        Read a reference written by :meth:`save`.

        Parameters
        ----------
        path : str or os.PathLike
            Source file.
        format : str or None, default=None
            Explicit format.

        Returns
        -------
        ReferenceProfile
            The reference.
        """
        return cls.from_dict(_serialize.read(path, format))
