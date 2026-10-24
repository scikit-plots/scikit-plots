# scikitplot/feature_selection/_stats.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Per-column statistics computed once and shared by every assessment.

Notes
-----
**User note.** :class:`FeatureStats` describes the *training* data a schema was
fitted on. Ratios are over non-missing values (``unique_ratio``,
``top_ratio``) or over all rows (``missing_ratio``), as each field says.

**Developer note.** :func:`compute_stats` touches each column a bounded number
of times (one ``value_counts`` and, for numeric families, one float64 copy),
so profiling a wide table is linear in its size. Values that JSON cannot hold
(NaN, infinity) are stored as ``None``; a statistic that cannot be computed
(cardinality of unhashable cells) is ``None`` too, which assessments report as
``unknown``. Numeric extension dtypes (nullable ``Int64``, pyarrow) are read
with :meth:`pandas.Series.to_numpy` and an explicit ``na_value`` instead of an
implicit cast.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from ._types import DtypeFamily

__all__ = ["FeatureStats"]

_NUMERIC_FAMILIES = (DtypeFamily.INTEGER, DtypeFamily.FLOAT)


def dtype_family(  # ruff: ignore[too-many-branches, too-many-return-statements]
    series: pd.Series,
) -> DtypeFamily:
    """
    Return the physical family of ``series``.

    Parameters
    ----------
    series : pandas.Series
        The column.

    Returns
    -------
    DtypeFamily
        ``object`` columns are inspected: all-``bool`` values give ``boolean``,
        all-``str`` values give ``string``, all-integer values ``integer``,
        all-real values ``float`` (common after JSON ingestion), anything else
        ``object``. Numeric strings stay ``string``: no implicit casting.

    Notes
    -----
    Order matters: a categorical of integers is ``categorical`` (its
    categories may be ordered), and booleans are tested before integers
    because NumPy treats ``bool`` as an integer kind.
    """
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return DtypeFamily.CATEGORICAL
    if ptypes.is_bool_dtype(dtype):
        return DtypeFamily.BOOLEAN
    if ptypes.is_datetime64_any_dtype(dtype):
        return DtypeFamily.DATETIME
    if ptypes.is_timedelta64_dtype(dtype):
        return DtypeFamily.TIMEDELTA
    if ptypes.is_complex_dtype(dtype):
        return DtypeFamily.OBJECT
    if ptypes.is_integer_dtype(dtype):
        return DtypeFamily.INTEGER
    if ptypes.is_float_dtype(dtype):
        return DtypeFamily.FLOAT
    if ptypes.is_object_dtype(dtype):
        values = series.dropna()
        if len(values) and all(isinstance(v, (bool, np.bool_)) for v in values):
            return DtypeFamily.BOOLEAN
        if all(isinstance(v, str) for v in values):
            return DtypeFamily.STRING
        if all(
            isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
            for v in values
        ):
            return DtypeFamily.INTEGER
        if all(
            isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, (bool, np.bool_))
            for v in values
        ):
            return DtypeFamily.FLOAT
        return DtypeFamily.OBJECT
    if ptypes.is_string_dtype(dtype):
        return DtypeFamily.STRING
    return DtypeFamily.OBJECT


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@dataclasses.dataclass(frozen=True)
class FeatureStats:
    """
    Training statistics of one column.

    Parameters
    ----------
    name : str
        Column name.
    dtype : str
        ``str(series.dtype)`` at fit time.
    family : str
        :class:`~scikitplot.feature_selection.DtypeFamily` value.
    n_rows, n_missing, n_non_null : int
        Row count, missing values (``NaN``, ``None``, ``NaT``, ``pd.NA``) and
        the rest.
    n_unique : int or None
        Distinct non-missing values; None when the cells are unhashable.
    top_count : int or None
        Count of the most frequent non-missing value.
    ordered : bool or None
        Whether a categorical dtype is ordered; None for other families.
    n_finite, n_inf : int or None
        Finite and infinite values of a numeric column; None otherwise.
    min, max, mean, std, q1, median, q3 : float or None
        Over finite values; ``std`` uses ``ddof=1`` and needs two values.
    skewness, kurtosis : float or None
        Bias-corrected sample skewness (needs 3 finite values) and excess
        kurtosis (needs 4), as :meth:`pandas.Series.skew` /
        :meth:`pandas.Series.kurt`; None for constant data.
    integer_valued : bool or None
        Every finite value is a whole number; None when there is none.

    See Also
    --------
    FeatureSpec : Holds the stats of a fitted feature.
    """

    name: str
    dtype: str
    family: str
    n_rows: int
    n_missing: int
    n_non_null: int
    n_unique: int | None = None
    top_count: int | None = None
    ordered: bool | None = None
    n_finite: int | None = None
    n_inf: int | None = None
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    q1: float | None = None
    median: float | None = None
    q3: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    integer_valued: bool | None = None

    @property
    def is_numeric(self) -> bool:
        """bool: The family is ``integer`` or ``float``."""
        return self.family in (DtypeFamily.INTEGER.value, DtypeFamily.FLOAT.value)

    @property
    def missing_ratio(self) -> float:
        """float: ``n_missing / n_rows`` (0.0 for an empty frame)."""
        return self.n_missing / self.n_rows if self.n_rows else 0.0

    @property
    def unique_ratio(self) -> float | None:
        """Float or None: ``n_unique / n_non_null``; None when undefined."""
        if self.n_unique is None or not self.n_non_null:
            return None
        return self.n_unique / self.n_non_null

    @property
    def top_ratio(self) -> float | None:
        """Float or None: share of the most frequent value among non-missing values."""
        if self.top_count is None or not self.n_non_null:
            return None
        return self.top_count / self.n_non_null

    def to_dict(self) -> dict[str, Any]:
        """Return the fields as a mapping of the shared data model."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(  # ruff: ignore[too-many-branches]
        cls,
        data: Mapping[str, Any],
    ) -> FeatureStats:
        """
        Rebuild stats from :meth:`to_dict` output, checking field names and types.

        Parameters
        ----------
        data : mapping
            The stored fields.

        Returns
        -------
        FeatureStats
            The stats.

        Raises
        ------
        ValueError
            For unknown or missing fields or wrongly typed values.
        """
        fields = {f.name: f for f in dataclasses.fields(cls)}
        unknown = set(data) - set(fields)
        if unknown:
            raise ValueError(f"unknown FeatureStats field(s) {sorted(unknown)}")
        required = ("name", "dtype", "family", "n_rows", "n_missing", "n_non_null")
        for name in required:
            if name not in data:
                raise ValueError(f"FeatureStats field {name!r} is missing")
        for name in ("name", "dtype", "family"):
            if not isinstance(data[name], str):
                raise ValueError(  # ruff: ignore[type-check-without-type-error]
                    f"FeatureStats.{name} must be a string"
                )
        DtypeFamily(data["family"])
        for name in (
            "n_rows",
            "n_missing",
            "n_non_null",
            "n_unique",
            "top_count",
            "n_finite",
            "n_inf",
        ):
            value = data.get(name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise ValueError(f"FeatureStats.{name} must be an int or None")
        for name in ("ordered", "integer_valued"):
            value = data.get(name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"FeatureStats.{name} must be a bool or None")
        for name in (
            "min",
            "max",
            "mean",
            "std",
            "q1",
            "median",
            "q3",
            "skewness",
            "kurtosis",
        ):
            value = data.get(name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise ValueError(f"FeatureStats.{name} must be a number or None")
        for name in (
            "n_rows",
            "n_missing",
            "n_non_null",
            "n_unique",
            "top_count",
            "n_finite",
            "n_inf",
        ):
            if data.get(name) is not None and data[name] < 0:
                raise ValueError(f"FeatureStats.{name} must be >= 0")
        if data["n_missing"] + data["n_non_null"] != data["n_rows"]:
            raise ValueError("FeatureStats.n_missing + n_non_null must equal n_rows")
        values = dict(data)
        for name in (
            "min",
            "max",
            "mean",
            "std",
            "q1",
            "median",
            "q3",
            "skewness",
            "kurtosis",
        ):
            if values.get(name) is not None:
                values[name] = float(values[name])
        return cls(**values)


def numeric_values(series: pd.Series) -> np.ndarray:
    """
    Return the non-missing values of a numeric column as ``float64``.

    Parameters
    ----------
    series : pandas.Series
        An ``integer`` or ``float`` family column (NumPy, nullable or pyarrow).

    Returns
    -------
    numpy.ndarray
        One-dimensional ``float64`` array, infinities kept, missing removed.
    """
    values = series.to_numpy(dtype="float64", na_value=np.nan)
    return values[~np.isnan(values)]


def compute_stats(series: pd.Series, name: str) -> FeatureStats:
    """
    Compute :class:`FeatureStats` for one column.

    Parameters
    ----------
    series : pandas.Series
        The column.
    name : str
        Its name.

    Returns
    -------
    FeatureStats
        The statistics.
    """
    family = dtype_family(series)
    n_rows = len(series)
    n_missing = int(series.isna().sum())
    n_non_null = n_rows - n_missing
    try:
        counts = series.value_counts(dropna=True)
        counts = counts[counts > 0]  # a categorical lists unused categories with 0
        n_unique: int | None = len(counts)
        top_count: int | None = int(counts.iloc[0]) if len(counts) else 0
    except TypeError:  # unhashable cells (lists, dicts)
        n_unique = top_count = None
    ordered = bool(series.dtype.ordered) if family is DtypeFamily.CATEGORICAL else None
    extra: dict[str, Any] = {}
    if family in _NUMERIC_FAMILIES:
        values = numeric_values(series)
        finite = values[np.isfinite(values)]
        n_finite = int(finite.size)
        extra.update(n_finite=n_finite, n_inf=int(values.size - n_finite))
        n_skew = 2
        n_kurt = 2
        if n_finite:
            q1, median, q3 = np.quantile(finite, [0.25, 0.5, 0.75])
            frame = pd.Series(finite)
            with np.errstate(all="ignore"):
                extra.update(
                    min=_finite_or_none(finite.min()),
                    max=_finite_or_none(finite.max()),
                    mean=_finite_or_none(finite.mean()),
                    std=_finite_or_none(finite.std(ddof=1)) if n_finite > 1 else None,
                    q1=_finite_or_none(q1),
                    median=_finite_or_none(median),
                    q3=_finite_or_none(q3),
                )
                if (
                    finite.min() != finite.max()
                ):  # moments of constant data are undefined
                    extra.update(
                        skewness=(
                            _finite_or_none(frame.skew()) if n_finite > n_skew else None
                        ),
                        kurtosis=(
                            _finite_or_none(frame.kurt()) if n_finite > n_kurt else None
                        ),
                    )
            if family is DtypeFamily.INTEGER:
                extra["integer_valued"] = True
            else:
                extra["integer_valued"] = bool(np.all(finite == np.floor(finite)))
    return FeatureStats(
        name=name,
        dtype=str(series.dtype),
        family=family.value,
        n_rows=n_rows,
        n_missing=n_missing,
        n_non_null=n_non_null,
        n_unique=n_unique,
        top_count=top_count,
        ordered=ordered,
        **extra,
    )
