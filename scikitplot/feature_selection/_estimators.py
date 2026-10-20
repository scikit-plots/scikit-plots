# scikitplot/feature_selection/_estimators.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikit-learn building blocks: profile, filter, select and rename inside a pipeline.

``FeatureProfiler``
    Fits a :class:`~scikitplot.feature_selection.FeatureSchema` on the training
    fold and passes data through, checking later frames against it.
``FeatureFilter``
    Drops features whose assessments are true (constant, identifier-like ...)
    and non-feature roles (target, identifier ...), with ``get_support``.
``FeatureColumnSelector``
    A callable for ``ColumnTransformer`` that picks columns by kind,
    assessment or tag, resolved on each training fold.
``FeatureRenamer``
    Applies a frozen :class:`~scikitplot.feature_selection.RenamePlan`.

Notes
-----
**User note.** All four take and return pandas DataFrames, fit only on the
data passed to ``fit`` (so ``cross_val_score`` and ``GridSearchCV`` never see
validation statistics), and implement ``get_feature_names_out`` and
``get_feature_lineage`` so :func:`~scikitplot.feature_selection.trace_pipeline`
traces them exactly.

**Developer note.** Constructor arguments are stored verbatim and resolved in
``fit`` (scikit-learn's ``clone`` contract). The assessment registry is not an
estimator parameter: estimators use the shared default registry, so a custom
provider registered once is visible to every clone.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._errors import SchemaMismatchError, SchemaMismatchWarning
from ._lineage import FeatureMapping
from ._rename import RenamePlan
from ._schema import FeatureSchema, check_frame, profile_features
from ._types import FeatureRole, coerce_enum

__all__ = [
    "FeatureColumnSelector",
    "FeatureFilter",
    "FeatureProfiler",
    "FeatureRenamer",
]

_ON_MISMATCH = ("raise", "warn", "ignore")


class _PandasTransformerMixin:
    """Shared tags and checks for transformers that take labelled DataFrames."""

    def _more_tags(self) -> dict[str, Any]:  # scikit-learn < 1.6
        return {
            "X_types": ["2darray", "string", "categorical"],
            "allow_nan": True,
            "requires_fit": True,
        }

    def __sklearn_tags__(self) -> Any:  # scikit-learn >= 1.6
        tags = super().__sklearn_tags__()  # type: ignore[misc]
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        tags.input_tags.categorical = True
        return tags

    def _check_on_mismatch(self) -> str:
        if self.on_mismatch not in _ON_MISMATCH:
            raise ValueError(
                f"on_mismatch={self.on_mismatch!r} is not one of {_ON_MISMATCH}"
            )
        return self.on_mismatch

    def _check_later_frame(self, X: Any) -> pd.DataFrame:
        """Check a transform-time frame against the fitted schema."""
        check_is_fitted(self)
        X = check_frame(X)
        findings = self.schema_.audit(X)
        missing = findings[findings["issue"] == "missing_column"]
        if len(missing):
            raise SchemaMismatchError(
                f"columns {missing['column'].tolist()} seen in fit are missing"
            )
        serious = findings[~findings["issue"].isin(["dtype_changed"])]
        if len(serious):
            message = "; ".join(
                f"{r.column}: {r.issue} (expected {r.expected}, observed {r.observed})"
                for r in serious.itertuples()
            )
            mode = self._check_on_mismatch()
            if mode == "raise":
                raise SchemaMismatchError(
                    f"data does not match the fitted schema: {message}; pass"
                    " on_mismatch='warn' or 'ignore' to proceed with the fitted columns"
                )
            if mode == "warn":
                warnings.warn(
                    f"data does not match the fitted schema: {message}",
                    SchemaMismatchWarning,
                    stacklevel=3,
                )
        return X

    def _check_input_features(self, input_features: Any) -> list[str]:
        check_is_fitted(self)
        names = list(self.feature_names_in_)
        if input_features is not None and [str(n) for n in input_features] != names:
            raise ValueError("input_features is not equal to feature_names_in_")
        return names


class FeatureProfiler(_PandasTransformerMixin, TransformerMixin, BaseEstimator):
    """
    Fit a feature schema on the training data; pass data through unchanged.

    Parameters
    ----------
    policy : FeaturePolicy, mapping, path or None, default=None
        Profiling policy (see :class:`~scikitplot.feature_selection.FeaturePolicy`).
    output : {"all", "features"}, default="all"
        ``"all"`` returns every fitted column; ``"features"`` only the columns
        whose role is ``"feature"`` (target, identifier, drop ... removed).
    on_mismatch : {"raise", "warn", "ignore"}, default="raise"
        What ``transform`` does with extra, reordered or retyped columns.
        Missing columns always raise. Output always has the fitted columns in
        the fitted order.

    Attributes
    ----------
    schema_ : FeatureSchema
        Schema fitted on the training data.
    feature_names_in_ : ndarray of str
        Columns seen in ``fit``.
    n_features_in_ : int
        Their count.

    See Also
    --------
    profile_features : The same without scikit-learn.
    FeatureFilter : Drops features by assessment.

    Notes
    -----
    **User note.** Put it first in a pipeline to keep the training schema next
    to the model: ``pipe.named_steps["profile"].schema_``. Use
    :meth:`audit` on validation or test data to see structural differences
    without refitting.

    Examples
    --------
    >>> import pandas as pd
    >>> from scikitplot.feature_selection import FeatureProfiler
    >>> X = pd.DataFrame({"a": [1.0, 2.0, 3.0] * 4, "id": [str(i) for i in range(12)]})
    >>> prof = FeatureProfiler(
    ...     policy={"roles": {"identifier": ["id"]}}, output="features"
    ... )
    >>> prof.fit_transform(X).columns.tolist()
    ['a']
    >>> prof.schema_["id"].role
    'identifier'
    """

    def __init__(
        self, policy: Any = None, *, output: str = "all", on_mismatch: str = "raise"
    ) -> None:  # noqa: D107
        self.policy = policy
        self.output = output
        self.on_mismatch = on_mismatch

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureProfiler:
        """
        Fit the schema on ``X``.

        Parameters
        ----------
        X : pandas.DataFrame
            Training data.
        y : ignored
            Present for API consistency.

        Returns
        -------
        self : FeatureProfiler
            The fitted profiler.
        """
        if self.output not in ("all", "features"):
            raise ValueError(f"output={self.output!r} is not 'all' or 'features'")
        self._check_on_mismatch()
        X = check_frame(X)
        self.schema_ = profile_features(X, self.policy)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self._names_out = (
            self.schema_.names if self.output == "all" else self.schema_.feature_names
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Check ``X`` against the schema and return the fitted columns.

        Parameters
        ----------
        X : pandas.DataFrame
            Data to pass through.

        Returns
        -------
        pandas.DataFrame
            ``X`` itself when it already has exactly the output columns,
            otherwise ``X[output columns]``.

        Raises
        ------
        SchemaMismatchError
            For missing columns, or other differences with ``on_mismatch="raise"``.
        """
        X = self._check_later_frame(X)
        if list(X.columns) == self._names_out:
            return X
        return X[self._names_out]

    def audit(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return :meth:`FeatureSchema.audit` of ``X``.

        Parameters
        ----------
        X : pandas.DataFrame
            Later data.

        Returns
        -------
        pandas.DataFrame
            Structural findings.
        """
        check_is_fitted(self)
        return self.schema_.audit(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """
        Return the output column names.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Must equal ``feature_names_in_`` when given.

        Returns
        -------
        ndarray of str
            Output names.
        """
        self._check_input_features(input_features)
        return np.asarray(self._names_out, dtype=object)

    def get_feature_lineage(self, input_features: Any = None) -> FeatureMapping:
        """
        Return the exact lineage of this step (a subset or the identity).

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Must equal ``feature_names_in_`` when given.

        Returns
        -------
        FeatureMapping
            The mapping.
        """
        names = self._check_input_features(input_features)
        index = {n: i for i, n in enumerate(names)}
        if self._names_out == names:
            return FeatureMapping.identity(len(names))
        return FeatureMapping.subset(index[n] for n in self._names_out)


class FeatureFilter(_PandasTransformerMixin, TransformerMixin, BaseEstimator):
    """
    Drop features whose assessments are true, and columns with non-feature roles.

    Parameters
    ----------
    policy : FeaturePolicy, mapping, path or None, default=None
        Profiling policy.
    drop : sequence of str, default=("constant", "identifier_like")
        A feature is dropped when *any* of these assessments is ``"true"``.
        ``unknown`` never drops.
    keep_roles : sequence of str, default=("feature",)
        Roles that survive; every other role (target, identifier, drop ...) is
        removed so it cannot leak into the model.
    keep : sequence of str, default=()
        Columns kept regardless of their assessments (not of their role).
    on_mismatch : {"raise", "warn", "ignore"}, default="raise"
        As in :class:`FeatureProfiler`.

    Attributes
    ----------
    schema_ : FeatureSchema
        Fitted schema.
    support_ : ndarray of bool
        Kept columns.
    report_ : pandas.DataFrame
        One row per column: ``kept`` and the ``reasons`` for dropping.
    feature_names_in_, n_features_in_
        As in scikit-learn.

    Examples
    --------
    >>> import pandas as pd
    >>> from scikitplot.feature_selection import FeatureFilter
    >>> X = pd.DataFrame(
    ...     {"id": [f"r{i}" for i in range(30)], "c": [1] * 30, "x": list(range(30))}
    ... )
    >>> f = FeatureFilter().fit(X)
    >>> f.get_feature_names_out().tolist()
    ['x']
    >>> f.report_.loc["id", "reasons"]
    ('identifier_like',)
    """

    def __init__(
        self,
        policy: Any = None,
        *,
        drop: Sequence[str] = ("constant", "identifier_like"),
        keep_roles: Sequence[str] = ("feature",),
        keep: Sequence[str] = (),
        on_mismatch: str = "raise",
    ) -> None:  # noqa: D107
        self.policy = policy
        self.drop = drop
        self.keep_roles = keep_roles
        self.keep = keep
        self.on_mismatch = on_mismatch

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureFilter:
        """
        Profile ``X`` and decide which columns to keep.

        Parameters
        ----------
        X : pandas.DataFrame
            Training data.
        y : ignored
            Present for API consistency.

        Returns
        -------
        self : FeatureFilter
            The fitted filter.

        Raises
        ------
        ValueError
            For unknown assessment or role names, ``keep`` columns absent from
            ``X``, or when every column would be dropped.
        """
        self._check_on_mismatch()
        X = check_frame(X)
        drop = [self.drop] if isinstance(self.drop, str) else list(self.drop)
        roles = (
            [self.keep_roles]
            if isinstance(self.keep_roles, str)
            else list(self.keep_roles)
        )
        keep = [self.keep] if isinstance(self.keep, str) else list(self.keep)
        roles = [coerce_enum(r, FeatureRole, field="keep_roles").value for r in roles]
        schema = profile_features(X, self.policy)
        known = set(schema.policy.assessments_map)
        unknown = sorted(set(drop) - known)
        if unknown:
            raise ValueError(
                f"drop names unknown assessment(s) {unknown}; known: {sorted(known)}"
            )
        missing = sorted(set(keep) - set(schema.names))
        if missing:
            raise ValueError(f"keep names column(s) {missing} that are not in X")
        support, rows = [], []
        for spec in schema:
            reasons = []
            if spec.role not in roles:
                reasons.append(f"role={spec.role}")
            if spec.name not in keep:
                reasons.extend(n for n in drop if spec.is_(n))
            support.append(not reasons)
            rows.append(
                {"feature": spec.name, "kept": not reasons, "reasons": tuple(reasons)}
            )
        if not any(support):
            raise ValueError(
                "FeatureFilter would drop every column; relax drop or keep_roles"
            )
        self.schema_ = schema
        self.support_ = np.asarray(support, dtype=bool)
        self.report_ = pd.DataFrame(rows).set_index("feature")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return the kept columns of ``X``.

        Parameters
        ----------
        X : pandas.DataFrame
            Data with the fitted columns.

        Returns
        -------
        pandas.DataFrame
            Kept columns in fitted order.
        """
        X = self._check_later_frame(X)
        return X[list(self.feature_names_in_[self.support_])]

    def get_support(self, indices: bool = False) -> np.ndarray:
        """
        Return the mask (or indices) of kept input columns.

        Parameters
        ----------
        indices : bool, default=False
            Return integer positions instead of a mask.

        Returns
        -------
        ndarray
            Boolean mask or integer positions.
        """
        check_is_fitted(self)
        return np.flatnonzero(self.support_) if indices else self.support_.copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """
        Return the kept column names.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Must equal ``feature_names_in_`` when given.

        Returns
        -------
        ndarray of str
            Kept names.
        """
        names = self._check_input_features(input_features)
        return np.asarray([n for n, k in zip(names, self.support_) if k], dtype=object)

    def get_feature_lineage(self, input_features: Any = None) -> FeatureMapping:
        """
        Return the exact subset lineage of this step.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Must equal ``feature_names_in_`` when given.

        Returns
        -------
        FeatureMapping
            The mapping.
        """
        self._check_input_features(input_features)
        return FeatureMapping.subset(np.flatnonzero(self.support_))


@dataclasses.dataclass(frozen=True)
class FeatureColumnSelector:
    """
    Column selector for ``ColumnTransformer`` driven by a feature schema.

    Called with the training frame when the ``ColumnTransformer`` is fitted, it
    profiles that frame (or uses a fixed ``schema``) and returns the matching
    column names; ``ColumnTransformer`` stores them, so validation and test
    data are transformed with the training selection.

    Parameters
    ----------
    kind, family, role, tag, is_, is_not, names
        Conditions, as in :meth:`FeatureSchema.select`. Sequences are stored
        as tuples.
    policy : FeaturePolicy, mapping, path or None, default=None
        Policy used to profile each training frame.
    schema : FeatureSchema or None, default=None
        Use this fitted schema instead of profiling; its selected columns
        must all be present in the frame.

    See Also
    --------
    sklearn.compose.make_column_selector : The dtype-based counterpart.

    Notes
    -----
    **User note.** With ``schema=None`` the selection is learned per training
    fold, which is the correct behaviour under cross-validation. Each selector
    profiles the frame it is given; with many selectors on a very wide frame,
    profile once and pass ``schema=`` for speed (accepting that the schema was
    fitted outside the folds).

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.compose import ColumnTransformer
    >>> from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
    >>> from scikitplot.feature_selection import FeatureColumnSelector
    >>> X = pd.DataFrame(
    ...     {
    ...         "income": [1.0, 2.0, 2.5, 3.0, 400.0] * 4,
    ...         "height": [1.6, 1.7, 1.8, 1.75, 1.65] * 4,
    ...         "city": list("abcab") * 4,
    ...     }
    ... )
    >>> ct = ColumnTransformer(
    ...     [
    ...         (
    ...             "skewed",
    ...             RobustScaler(),
    ...             FeatureColumnSelector(kind="numeric_continuous", is_=("skewed",)),
    ...         ),
    ...         (
    ...             "normal",
    ...             StandardScaler(),
    ...             FeatureColumnSelector(
    ...                 kind="numeric_continuous", is_not=("skewed",)
    ...             ),
    ...         ),
    ...         (
    ...             "cat",
    ...             OneHotEncoder(),
    ...             FeatureColumnSelector(kind="categorical_nominal"),
    ...         ),
    ...     ]
    ... ).fit(X)
    >>> [(name, cols) for name, _, cols in ct.transformers_]
    [('skewed', ['income']), ('normal', ['height']), ('cat', ['city'])]
    """

    kind: Any = None
    family: Any = None
    role: Any = "feature"
    tag: Any = None
    is_: Sequence[str] = ()
    is_not: Sequence[str] = ()
    names: Sequence[str] | None = None
    policy: Any = None
    schema: FeatureSchema | None = None

    def __post_init__(self) -> None:  # noqa: D105
        for field in ("kind", "family", "role", "tag", "is_", "is_not", "names"):
            value = getattr(self, field)
            if value is not None and not isinstance(value, str):
                object.__setattr__(self, field, tuple(value))
        if self.schema is not None and not isinstance(self.schema, FeatureSchema):
            raise TypeError("schema must be a FeatureSchema or None")

    def __call__(self, X: pd.DataFrame) -> list[str]:
        """
        Return the matching columns of ``X``.

        Parameters
        ----------
        X : pandas.DataFrame
            The frame given to ``ColumnTransformer.fit``.

        Returns
        -------
        list of str
            Matching column names in frame order.
        """
        X = check_frame(X)
        schema = (
            self.schema if self.schema is not None else profile_features(X, self.policy)
        )
        selected = schema.select(
            kind=self.kind,
            family=self.family,
            role=self.role,
            tag=self.tag,
            is_=self.is_,
            is_not=self.is_not,
            names=self.names,
        )
        missing = [c for c in selected if c not in X.columns]
        if missing:
            raise SchemaMismatchError(
                f"selected column(s) {missing} are not in the frame"
            )
        order = {c: i for i, c in enumerate(X.columns)}
        return sorted(selected, key=order.__getitem__)


class FeatureRenamer(_PandasTransformerMixin, TransformerMixin, BaseEstimator):
    """
    Rename columns with a plan frozen at fit time.

    Parameters
    ----------
    rule : mapping, sequence of str, callable, str or None, default=None
        Naming rule (see :class:`~scikitplot.feature_selection.RenamePlan`):
        ``{"old": "new"}``, a list of new names, ``callable(name)``, or a
        template such as ``"f{index}"``. None keeps every name.
    start : int, default=0
        First ``{index}`` of a template.
    on_collision : {"raise", "suffix"}, default="raise"
        Handling of duplicate new names.
    on_missing : {"raise", "ignore"}, default="raise"
        Handling of mapping keys that are not columns.

    Attributes
    ----------
    plan_ : RenamePlan
        The frozen plan.
    feature_names_in_ : ndarray of str
        Set only when fitted on a DataFrame, as in scikit-learn.
    n_features_in_ : int
        Input count. A NumPy input is named ``x0, x1, ...`` for the rule.

    Notes
    -----
    **User note.** Output is always a DataFrame. A callable rule runs once in
    ``fit``; ``transform`` only applies the stored plan, so test data can never
    be renamed differently. Define callables with ``def`` (not ``lambda``) if
    the pipeline must be pickled.

    Examples
    --------
    >>> import pandas as pd
    >>> from scikitplot.feature_selection import FeatureRenamer
    >>> X = pd.DataFrame({"num__age": [1, 2], "cat__city_a": [0, 1]})
    >>> FeatureRenamer(lambda n: n.split("__", 1)[1]).fit_transform(X).columns.tolist()
    ['age', 'city_a']
    """

    def __init__(
        self,
        rule: Any = None,
        *,
        start: int = 0,
        on_collision: str = "raise",
        on_missing: str = "raise",
    ) -> None:  # noqa: D107
        self.rule = rule
        self.start = start
        self.on_collision = on_collision
        self.on_missing = on_missing

    def _frame(self, X: Any, *, fitting: bool) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return check_frame(X)
        array = np.asarray(X)
        ndim = 2
        if array.ndim != ndim:
            raise ValueError(f"X must be 2-D, got {array.ndim}-D")
        names = (
            [f"x{i}" for i in range(array.shape[1])]
            if fitting
            else list(self._names_in)
        )
        if not fitting and array.shape[1] != len(names):
            raise SchemaMismatchError(
                f"X has {array.shape[1]} columns, the renamer was fitted on {len(names)}"
            )
        return pd.DataFrame(array, columns=names)

    def fit(self, X: Any, y: Any = None) -> FeatureRenamer:
        """
        Build and freeze the rename plan from the columns of ``X``.

        Parameters
        ----------
        X : pandas.DataFrame or array-like
            Training data.
        y : ignored
            Present for API consistency.

        Returns
        -------
        self : FeatureRenamer
            The fitted renamer.
        """
        frame = self._frame(X, fitting=True)
        self.plan_ = RenamePlan.build(
            list(frame.columns),
            self.rule,
            start=self.start,
            on_collision=self.on_collision,
            on_missing=self.on_missing,
        )
        self._names_in = list(frame.columns)
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        self.n_features_in_ = frame.shape[1]
        return self

    def _check_input_features(self, input_features: Any) -> list[str]:
        check_is_fitted(self, "plan_")
        if input_features is None:
            return list(self._names_in)
        names = [str(n) for n in input_features]
        if len(names) != self.n_features_in_:
            raise ValueError(
                f"input_features has {len(names)} names, expected {self.n_features_in_}"
            )
        if hasattr(self, "feature_names_in_") and names != list(self.feature_names_in_):
            raise ValueError("input_features is not equal to feature_names_in_")
        return names

    def transform(self, X: Any) -> pd.DataFrame:
        """
        Apply the frozen plan.

        Parameters
        ----------
        X : pandas.DataFrame or array-like
            Data with the fitted columns, in the fitted order.

        Returns
        -------
        pandas.DataFrame
            Renamed frame.

        Raises
        ------
        SchemaMismatchError
            If the columns differ from those seen in ``fit``.
        """
        check_is_fitted(self, "plan_")
        frame = self._frame(X, fitting=False)
        if list(frame.columns) != list(self._names_in):
            raise SchemaMismatchError(
                "FeatureRenamer.transform needs exactly the fitted columns in the fitted"
                " order"
            )
        return self.plan_.apply(frame)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """
        Return the new names.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Must equal ``feature_names_in_`` when given.

        Returns
        -------
        ndarray of str
            New names.
        """
        self._check_input_features(input_features)
        return np.asarray(self.plan_.names_out, dtype=object)

    def get_feature_lineage(self, input_features: Any = None) -> FeatureMapping:
        """
        Return the one-to-one lineage of this step.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Checked as in :meth:`get_feature_names_out`.

        Returns
        -------
        FeatureMapping
            The identity mapping.
        """
        self._check_input_features(input_features)
        return FeatureMapping.identity(self.n_features_in_)
