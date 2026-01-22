# scikitplot/decile/_decile_modelplotpy.py

# pylint: disable=import-error
# pylint: disable=broad-exception-caught
# pylint: disable=logging-fstring-interpolation
# pylint: disable=consider-using-f-string

# flake8: noqa: D213
# ruff: noqa: PLR2004

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The :mod:`~scikitplot.decile` and (⚠️ alternative legacy :mod:`scikitplot.decile.modelplotpy`) module.

Includes plots for machine learning evaluation decile / ntile analysis
(e.g., Response, Lift, Gain and related financial charts).

References
----------
* https://github.com/modelplot/modelplotpy/blob/master/modelplotpy/functions.py
* https://modelplot.github.io/intro_modelplotpy.html
"""

from __future__ import annotations

import os  # noqa: F401
import re  # noqa: F401
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .._docstrings import _docstring
from ..utils._matplotlib import save_plot_decorator

if TYPE_CHECKING:
    # Only imported during type checking
    from typing import (  # noqa: F401
        Any,
        Iterable,
        Mapping,
        Optional,
        # Sequence,
        # TypeVar,
        Union,
    )

# import logging
# logger = logging.getLogger(__name__)
from .. import logger

T = TypeVar("T")

__all__ = [  # noqa: RUF022
    "ModelPlotPy",
    # (cumulative) Response, Lift, Gains and plots
    "plot_response",
    "plot_cumresponse",
    "plot_cumlift",
    "plot_cumgains",
    "plot_all",
    # financial
    "plot_costsrevs",
    "plot_profit",
    "plot_roi",
    # helper
    "summarize_selection",
]

##########################################################################
## helper func
##########################################################################


def _range01(x: Any) -> np.ndarray:
    """
    Normalize numeric input into the closed interval [0, 1].

    Parameters
    ----------
    x : Any
        Numeric array-like. Supported inputs include numpy arrays, pandas Series,
        and array-like objects accepted by :func:`numpy.asarray`.

    Returns
    -------
    numpy.ndarray
        Normalized values in [0, 1] with the same shape as the input.

    Raises
    ------
    ValueError
        If the input contains NaN or infinite values.

    See Also
    --------
    numpy.asarray

    Notes
    -----
    Rule:

    - If max(x) == min(x), returns an array of zeros.

    Examples
    --------
    >>> _range01([2.0, 4.0]).tolist()
    [0.0, 1.0]
    """
    arr = np.asarray(x, dtype=float)

    # Dev note: do not silently accept NaN/inf; downstream ranking depends on finite values.
    if not np.isfinite(arr).all():
        raise ValueError("x must contain only finite values (no NaN/inf).")

    xmin = float(arr.min())
    xmax = float(arr.max())
    if xmax == xmin:
        return np.zeros_like(arr, dtype=float)

    return (arr - xmin) / (xmax - xmin)


def _check_input(
    input_list: Iterable[T] | None,
    check_list: Sequence[T],
    check: str = "",
) -> list[T]:
    """
    Validate that all selected values are contained in an allowed list.

    Parameters
    ----------
    input_list : Iterable[T] or None
        User selection. If None or empty, an empty list is returned.
    check_list : Sequence[T]
        Allowed values.
    check : str, default=''
        Parameter name for error messages.

    Returns
    -------
    list[T]
        Materialized list of values from ``input_list``.

    Raises
    ------
    ValueError
        If any provided value is not present in ``check_list``.

    See Also
    --------
    set.issubset

    Notes
    -----
    The original implementation only required *any* overlap between ``input_list``
    and ``check_list``, which allows invalid values to pass silently. In strict mode,
    **all** selected values must be valid.

    Examples
    --------
    >>> _check_input(["a"], ["a", "b"], check="x")
    ['a']
    """
    if input_list is None:
        return []

    selected = list(input_list)
    if not selected:
        return []

    allowed = set(check_list)
    invalid = [v for v in selected if v not in allowed]
    if invalid:
        raise ValueError(
            f"Invalid input for parameter {check}. Invalid values={invalid}. Allowed={list(check_list)}."
        )

    return selected


def _as_dataframe(X: Any) -> pd.DataFrame:
    """
    Coerce features into a pandas DataFrame with a deterministic index.

    Parameters
    ----------
    X : Any
        Feature matrix. Supported: pandas.DataFrame, numpy.ndarray.

    Returns
    -------
    pandas.DataFrame
        Feature DataFrame.

    Raises
    ------
    TypeError
        If X is not a pandas DataFrame or numpy.ndarray.

    See Also
    --------
    pandas.DataFrame

    Notes
    -----
    - If ``X`` is an ndarray, a RangeIndex is created deterministically.

    Examples
    --------
    >>> _as_dataframe(np.array([[1, 2], [3, 4]])).shape
    (2, 2)
    """
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, np.ndarray):
        return pd.DataFrame(X)
    raise TypeError("feature_data elements must be pandas.DataFrame or numpy.ndarray.")


def _as_series(y: Any, *, index: pd.Index) -> pd.Series:
    """
    Coerce labels into a pandas Series aligned strictly by position.

    Parameters
    ----------
    y : Any
        Label vector. Supported: pandas.Series, numpy.ndarray, list, tuple.
    index : pandas.Index
        Index to enforce.

    Returns
    -------
    pandas.Series
        Series with name ``'target_class'`` and the provided ``index``.

    Raises
    ------
    ValueError
        If the length of ``y`` does not match ``index``.
    TypeError
        If ``y`` cannot be coerced.

    See Also
    --------
    pandas.Series

    Notes
    -----
    Dev note: we do *not* reindex by label, because that can silently reorder rows.
    We enforce strict positional alignment.

    Examples
    --------
    >>> _as_series([0, 1], index=pd.RangeIndex(2)).tolist()
    [0, 1]
    """
    if isinstance(y, pd.Series):
        if len(y) != len(index):
            raise ValueError("label_data length must match feature_data length.")
        return pd.Series(y.to_numpy(copy=False), index=index, name="target_class")

    if isinstance(y, (np.ndarray, list, tuple)):
        if len(y) != len(index):
            raise ValueError("label_data length must match feature_data length.")
        return pd.Series(y, index=index, name="target_class")

    raise TypeError(
        "label_data elements must be pandas.Series, numpy.ndarray, list, or tuple."
    )


def _assign_descending_ntiles(scores: pd.Series, *, ntiles: int) -> pd.Series:
    """
    Assign deterministic ntiles (1 = highest score) without randomness.

    Parameters
    ----------
    scores : pandas.Series
        Scores/probabilities for a single class. Must be finite and non-null.
    ntiles : int
        Number of bins. Must satisfy 2 <= ntiles <= n_samples.

    Returns
    -------
    pandas.Series
        Integer ntile labels in [1, ntiles], aligned to ``scores.index``.

    Raises
    ------
    ValueError
        If ``ntiles`` is invalid or scores contain NaN/inf.

    See Also
    --------
    numpy.lexsort

    Notes
    -----
    The legacy implementation added random noise and used :func:`pandas.qcut` to
    break ties. This is not necessary and contaminates RNG state.

    Deterministic rule:

    - Sort by score descending.
    - Break ties by original row position (stable).
    - Assign ntile = floor(rank * ntiles / n) + 1.

    Examples
    --------
    >>> s = pd.Series([0.9, 0.1, 0.9, 0.5], index=list("abcd"))
    >>> _assign_descending_ntiles(s, ntiles=2).loc[list("abcd")].tolist()
    [1, 2, 1, 2]
    """
    if not isinstance(ntiles, int) or ntiles < 2:  # noqa: PLR2004
        raise ValueError("ntiles must be an int >= 2.")

    if scores.isna().any():
        raise ValueError("scores must not contain NaN.")

    arr = scores.to_numpy(dtype=float, copy=False)
    if not np.isfinite(arr).all():
        raise ValueError("scores must contain only finite values.")

    n = int(arr.shape[0])
    if n < ntiles:
        raise ValueError(f"ntiles={ntiles} cannot exceed n_samples={n}.")

    # Dev note: stable ordering via (score desc, original position asc).
    pos = np.arange(n, dtype=np.int64)
    order = np.lexsort((pos, -arr))

    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n, dtype=np.int64)

    nt = (rank * ntiles // n) + 1
    return pd.Series(nt.astype(np.int64), index=scores.index, name="ntile")


@dataclass(frozen=True)
class _EvalKey:
    """Internal composite key for model/dataset grouping."""

    model_label: str
    dataset_label: str


##########################################################################
## ModelPlotPy class
##########################################################################


# class ModelPlotPy(ABC):
class ModelPlotPy:
    """
    Decile/ntile analysis for sklearn classifiers.

    Parameters
    ----------
    feature_data : Sequence[Any] or None, default=None
        Sequence of feature matrices (DataFrame or ndarray). One per dataset.
    label_data : Sequence[Any] or None, default=None
        Sequence of label vectors (Series/ndarray/list). One per dataset.
    dataset_labels : Sequence[str] or None, default=None
        Names for datasets; must match length of `feature_data` and `label_data`.
    models : Sequence[ClassifierMixin] or None, default=None
        Fitted sklearn-like classifiers that implement `predict_proba` and `classes_`.
    model_labels : Sequence[str] or None, default=None
        Names for models; must match length of `models`.
    ntiles : int, default=10
        Number of ntiles. Must satisfy 2 <= ntiles <= n_samples for each dataset.
    seed : int, default=0
        Reserved for backward compatibility. Not used (ntiles are deterministic).

    Returns
    -------
    ModelPlotPy
        Instance.

    Raises
    ------
    ValueError
        If list lengths are inconsistent or ntiles is invalid.
    TypeError
        If models are not sklearn classifiers.

    See Also
    --------
    sklearn.base.ClassifierMixin

    Notes
    -----
    Key design rules:

    - No mutable defaults in ``__init__``.
    - No randomness in ntile assignment (random noise added for qcut).

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> X = pd.DataFrame({"x": [0, 1, 2, 3]})
    >>> y = pd.Series([0, 0, 1, 1])
    >>> m = LogisticRegression().fit(X, y)
    >>> mp = ModelPlotPy([X], [y], ["train"], [m], ["lr"], ntiles=2)
    >>> scores = mp.prepare_scores_and_ntiles()
    >>> set(scores.columns) >= {
    ...     "dataset_label",
    ...     "model_label",
    ...     "target_class",
    ...     "prob_0",
    ...     "prob_1",
    ...     "dec_0",
    ...     "dec_1",
    ... }
    True
    """

    def __init__(
        self,
        feature_data: Sequence[Any] | None = None,
        label_data: Sequence[Any] | None = None,
        dataset_labels: Sequence[str] | None = None,
        models: Sequence[ClassifierMixin] | None = None,
        model_labels: Sequence[str] | None = None,
        ntiles: int = 10,
        seed: int = 0,
    ) -> None:
        # User note: we copy inputs into lists to avoid surprising aliasing.
        self.feature_data = list(feature_data) if feature_data is not None else []
        self.label_data = list(label_data) if label_data is not None else []
        self.dataset_labels = list(dataset_labels) if dataset_labels is not None else []
        self.models = list(models) if models is not None else []
        self.model_labels = list(model_labels) if model_labels is not None else []
        self.ntiles = ntiles
        self.seed = seed

        self._validate_state()

    def _validate_state(self) -> None:
        """
        Validate internal configuration and model readiness.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If list lengths mismatch or ``ntiles`` is invalid.
        TypeError
            If any model does not implement ``predict_proba`` or lacks ``classes_``.

        See Also
        --------
        sklearn.utils.validation.check_is_fitted

        Notes
        -----
        This method is intentionally strict and fails fast.

        Examples
        --------
        >>> # Called internally.
        """
        if not isinstance(self.ntiles, int) or self.ntiles < 2:  # noqa: PLR2004
            raise ValueError("ntiles must be an int >= 2.")

        if len(self.models) != len(self.model_labels):
            raise ValueError(
                "The number of models and model_labels must be equal. "
                f"models={len(self.models)} model_labels={len(self.model_labels)}."
            )

        if not (
            len(self.feature_data) == len(self.label_data) == len(self.dataset_labels)
        ):
            raise ValueError(
                "feature_data, label_data, dataset_labels must have the same length. "
                f"feature_data={len(self.feature_data)} label_data={len(self.label_data)} "
                f"dataset_labels={len(self.dataset_labels)}."
            )

        if not self.models:
            return

        # Dev note: require consistent class sets across models to keep columns stable.
        ref_classes = None
        for m in self.models:
            if isinstance(m, ClassifierMixin):
                check_is_fitted(m)
            if not hasattr(m, "predict_proba"):
                raise TypeError("All models must implement predict_proba().")
            if not hasattr(m, "classes_"):
                raise TypeError("All models must expose fitted attribute 'classes_'.")

            cls = tuple(getattr(m, "classes_"))  # noqa: B009
            if ref_classes is None:
                ref_classes = cls
            elif cls != ref_classes:
                raise ValueError("All models must have identical classes_.")

        # Validate dataset element shapes deterministically.
        for X_raw, y_raw, ds_label in zip(
            self.feature_data, self.label_data, self.dataset_labels
        ):
            X = _as_dataframe(X_raw)
            y = _as_series(y_raw, index=X.index)
            n = len(X)
            if n < self.ntiles:
                raise ValueError(
                    f"Dataset '{ds_label}' has n_samples={n} < ntiles={self.ntiles}."
                )

    def get_params(self) -> dict[str, Any]:
        """
        Get parameters (sklearn-style API).

        Parameters
        ----------
        None

        Returns
        -------
        dict[str, Any]
            Parameter dictionary.

        Raises
        ------
        None

        See Also
        --------
        set_params

        Notes
        -----
        The returned objects are the current attributes; callers should not mutate
        them in-place if they want stable behavior.

        Examples
        --------
        >>> # mp.get_params()
        """
        return {
            "feature_data": self.feature_data,
            "label_data": self.label_data,
            "dataset_labels": self.dataset_labels,
            "models": self.models,
            "model_labels": self.model_labels,
            "ntiles": self.ntiles,
            "seed": self.seed,
        }

    def set_params(self, **params: Any) -> None:
        """
        Set parameters (sklearn-style API) and re-validate.

        Parameters
        ----------
        **params : Any
            Attributes to set on the object.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If an invalid parameter is provided.

        See Also
        --------
        get_params

        Notes
        -----
        After updating attributes, the internal state is validated.

        Examples
        --------
        >>> # mp.set_params(ntiles=20)
        """
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter '{key}' for ModelPlotPy.")
            setattr(self, key, value)
        self._validate_state()

    def reset_params(self) -> None:
        """
        Reset all parameters to a default empty state.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        None

        See Also
        --------
        set_params, get_params

        Notes
        -----
        The object remains usable after repopulating fields.

        Examples
        --------
        >>> # mp.reset_params()
        """
        self.feature_data = []
        self.label_data = []
        self.dataset_labels = []
        self.models = []
        self.model_labels = []
        self.ntiles = 10
        self.seed = 0

    def prepare_scores_and_ntiles(self) -> pd.DataFrame:
        """
        Compute per-row class probabilities and deterministic ntiles.

        Parameters
        ----------
        None

        Returns
        -------
        pandas.DataFrame
            DataFrame containing:

            - ``dataset_label`` and ``model_label``
            - ``target_class`` (true label)
            - ``prob_<class>`` columns
            - ``dec_<class>`` columns (1..ntiles; 1 = highest probability)

        Raises
        ------
        ValueError
            If there are no models/datasets, ntiles is invalid, or any dataset has
            fewer rows than ``ntiles``.

        See Also
        --------
        aggregate_over_ntiles

        Notes
        -----
        This replaces the legacy approach that added random noise and used
        :func:`pandas.qcut` for binning.

        Examples
        --------
        >>> # scores = mp.prepare_scores_and_ntiles()
        """
        self._validate_state()

        if not self.models:
            raise ValueError("At least one model must be provided.")
        if not self.feature_data:
            raise ValueError("At least one dataset must be provided.")

        frames: list[pd.DataFrame] = []
        for model, model_label in zip(self.models, self.model_labels):
            classes = [str(c) for c in model.classes_]

            for X_raw, y_raw, ds_label in zip(
                self.feature_data, self.label_data, self.dataset_labels
            ):
                X = _as_dataframe(X_raw)
                y = _as_series(y_raw, index=X.index)

                proba = model.predict_proba(X)
                proba_df = pd.DataFrame(
                    proba,
                    index=X.index,
                    columns=[f"prob_{c}" for c in classes],
                )

                out = pd.DataFrame(
                    {
                        "dataset_label": ds_label,
                        "model_label": model_label,
                        "target_class": y,
                    },
                    index=X.index,
                )

                # Dev note: keep output free of feature columns; plotting uses only probs + labels.
                out = pd.concat([out, proba_df], axis=1)

                for c in classes:
                    # 1 = best/highest score bin.
                    out[f"dec_{c}"] = _assign_descending_ntiles(
                        out[f"prob_{c}"], ntiles=self.ntiles
                    )

                frames.append(out)

        return pd.concat(frames, axis=0, ignore_index=False)

    def aggregate_over_ntiles(self) -> pd.DataFrame:
        """
        Aggregate counts and lift/gain metrics per ntile.

        Parameters
        ----------
        None

        Returns
        -------
        pandas.DataFrame
            Aggregated metrics per (model_label, dataset_label, target_class, ntile).
            The output schema matches the legacy implementation so it can be consumed
            by the existing plot functions.

        Raises
        ------
        ValueError
            If any group has zero positives for a requested target class.

        See Also
        --------
        prepare_scores_and_ntiles

        Notes
        -----
        Dev note: this implementation avoids mutating shared columns in the scores
        dataframe during loops (the legacy code writes ``pos``/``neg`` repeatedly).

        Examples
        --------
        >>> # agg = mp.aggregate_over_ntiles()
        """
        scores = self.prepare_scores_and_ntiles()

        rows: list[pd.DataFrame] = []
        for model, model_label in zip(self.models, self.model_labels):
            classes = list(model.classes_)
            class_str = [str(c) for c in classes]

            for ds_label in self.dataset_labels:
                subset = scores[
                    (scores["model_label"] == model_label)
                    & (scores["dataset_label"] == ds_label)
                ]
                if subset.empty:
                    continue

                for cls, cls_s in zip(classes, class_str):
                    ntile_col = f"dec_{cls_s}"
                    if ntile_col not in subset.columns:
                        raise ValueError(f"Missing ntile column '{ntile_col}'.")

                    tmp = subset[[ntile_col, "target_class"]].copy()
                    tmp["pos"] = tmp["target_class"] == cls
                    tmp["neg"] = ~tmp["pos"]

                    agg = tmp.groupby(ntile_col, sort=True).agg(
                        tot=("target_class", "size"),
                        pos=("pos", "sum"),
                        neg=("neg", "sum"),
                    )

                    # Ensure a full ntile index for plotting (1..ntiles).
                    agg = agg.reindex(range(1, self.ntiles + 1), fill_value=0)
                    agg.index.name = "ntile"
                    agg = agg.reset_index()

                    # Totals (repeat constants per row; plotting functions rely on these).
                    postot = int(agg["pos"].sum())
                    negtot = int(agg["neg"].sum())
                    tottot = int(agg["tot"].sum())
                    if tottot <= 0:
                        raise ValueError("Empty group encountered during aggregation.")
                    if postot == 0:
                        raise ValueError(
                            f"No positive examples for target_class={cls} in (model={model_label}, dataset={ds_label})."
                        )

                    agg["pct"] = agg["pos"] / agg["tot"]
                    agg["postot"] = postot
                    agg["negtot"] = negtot
                    agg["tottot"] = tottot
                    agg["pcttot"] = float(agg["pct"].sum())

                    agg["cumpos"] = agg["pos"].cumsum()
                    agg["cumneg"] = agg["neg"].cumsum()
                    agg["cumtot"] = agg["tot"].cumsum()
                    agg["cumpct"] = agg["cumpos"] / agg["cumtot"]

                    agg["gain"] = agg["pos"] / postot
                    agg["cumgain"] = agg["cumpos"] / postot
                    agg["gain_ref"] = agg["ntile"] / self.ntiles

                    base_rate = postot / tottot
                    agg["pct_ref"] = base_rate

                    # Optimal gain curve: saturates at 1 when cumtot >= postot.
                    agg["gain_opt"] = np.minimum(agg["cumtot"] / postot, 1.0)

                    agg["lift"] = agg["pct"] / base_rate
                    agg["cumlift"] = agg["cumpct"] / base_rate
                    agg["cumlift_ref"] = 1.0

                    # Add metadata columns.
                    agg.insert(0, "target_class", cls)
                    agg.insert(0, "dataset_label", ds_label)
                    agg.insert(0, "model_label", model_label)

                    # Origin row (ntile=0) for plot continuity.
                    origin = agg.iloc[[0]].copy()
                    origin["ntile"] = 0
                    origin[["tot", "pos", "neg"]] = 0
                    origin[
                        [
                            "pct",
                            "pcttot",
                            "cumpos",
                            "cumneg",
                            "cumtot",
                            "cumpct",
                            "gain",
                            "cumgain",
                            "gain_ref",
                            "gain_opt",
                            "lift",
                            "cumlift",
                        ]
                    ] = 0.0
                    origin["cumlift_ref"] = 1.0

                    out = pd.concat([origin, agg], axis=0, ignore_index=True)
                    rows.append(out)

        result = pd.concat(rows, axis=0, ignore_index=True)
        return result.sort_values(
            ["model_label", "dataset_label", "target_class", "ntile"]
        ).reset_index(drop=True)

    def plotting_scope(
        self,
        scope: str = "auto",
        select_model_label: Sequence[str] | None = None,
        select_dataset_label: Sequence[str] | None = None,
        select_targetclass: Sequence[Any] | None = None,
        select_smallest_targetclass: bool = True,
    ) -> pd.DataFrame:
        """
        Build ``plot_input`` subset according to a strict scope contract.

        Parameters
        ----------
        scope : {'auto', 'no_comparison', 'compare_models', 'compare_datasets', 'compare_targetclasses'}, default='auto'
            Evaluation perspective.

            If ``scope='auto'``, the scope is inferred deterministically from the provided
            selectors and the available options.
        select_model_label : Sequence[str] or None, default=None
            Model labels to include.
        select_dataset_label : Sequence[str] or None, default=None
            Dataset labels to include.
        select_targetclass : Sequence[Any] or None, default=None
            Target classes to include.
        select_smallest_targetclass : bool, default=True
            Should the plot only contain the results of the smallest targetclass.
            If True, the specific target is defined from the first dataset.
            If False and select_targetclass is None try to uses
            ``list(self.models[0].classes_)``

        Returns
        -------
        pandas.DataFrame
            Subset dataframe ready for plotting functions.

        Raises
        ------
        ValueError
            If the scope is invalid, selector values are invalid, or the selection
            is ambiguous under the strict contract.

        See Also
        --------
        aggregate_over_ntiles

        Notes
        -----
        **Inference rules for ``scope='auto'``**

        Let the universes be:

        - ``M`` = all ``model_labels``
        - ``D`` = all ``dataset_labels``
        - ``T`` = all fitted ``classes_``

        After validating selectors (membership is strict):

        1. If exactly one selector among (models, datasets, targetclasses) contains
           **two or more** values, then ``auto`` selects the corresponding comparison
           scope.

           - ``len(select_model_label) >= 2``  -> ``compare_models``
           - ``len(select_dataset_label) >= 2`` -> ``compare_datasets``
           - ``len(select_targetclass) >= 2``   -> ``compare_targetclasses``

           If **more than one** selector has length >= 2, the request is ambiguous
           and a ValueError is raised.

        2. If no selector has length >= 2:

           - If all dimensions are fixed (either explicitly selected with length 1,
             or the universe size is 1), ``auto`` selects ``no_comparison``.

           - Otherwise, if exactly one dimension is unfixed (universe size > 1) while
             the other two are fixed, ``auto`` selects the corresponding comparison
             scope comparing **all** values in that unfixed dimension.

           - If the remaining degrees of freedom are not unique, the request is
             ambiguous and a ValueError is raised.

        Examples
        --------
        >>> # Compare all models on a fixed dataset and target class (scope inferred):
        >>> # plot_input = mp.plotting_scope(select_dataset_label=['test'], select_targetclass=[1])
        >>>
        >>> # Compare two datasets for a fixed model and target class (scope inferred):
        >>> # plot_input = mp.plotting_scope(select_model_label=['lr'], select_targetclass=[1])
        """
        allowed_scopes = {
            "auto",
            "no_comparison",
            "compare_models",
            "compare_datasets",
            "compare_targetclasses",
        }
        if scope not in allowed_scopes:
            raise ValueError(
                f"Invalid scope={scope}. Allowed={sorted(allowed_scopes)}."
            )

        # Materialize selections and validate membership.
        sm = _check_input(select_model_label, self.model_labels, "select_model_label")
        sd = _check_input(
            select_dataset_label, self.dataset_labels, "select_dataset_label"
        )
        st = _check_input(
            (
                select_targetclass
                if select_targetclass is not None
                else (
                    [self.label_data[0].value_counts(ascending=True).idxmin()]
                    if select_smallest_targetclass
                    else (list(self.models[0].classes_) if self.models else [])
                )
            ),
            list(self.models[0].classes_) if self.models else [],
            "select_targetclass",
        )

        models_universe = self.model_labels
        datasets_universe = self.dataset_labels
        classes_universe = list(self.models[0].classes_) if self.models else []

        def _infer_scope(*, scope_in: str) -> str:
            if scope_in != "auto":
                return scope_in

            # Case 1: explicit multi-select determines the comparison dimension.
            dims_multi = {
                "compare_models": len(sm) >= 2,
                "compare_datasets": len(sd) >= 2,
                "compare_targetclasses": len(st) >= 2,
            }
            chosen = [k for k, v in dims_multi.items() if v]
            if len(chosen) == 1:
                return chosen[0]
            if len(chosen) > 1:
                raise ValueError(
                    "scope='auto' is ambiguous: multiple selectors request comparisons "
                    f"simultaneously (models={len(sm)}, datasets={len(sd)}, targetclasses={len(st)}). "
                    "Provide an explicit scope or reduce selectors to one comparison dimension."
                )

            # Case 2: no explicit multi-select; infer only when degrees of freedom are unique.
            fixed_model = (len(sm) == 1) or (len(models_universe) == 1)
            fixed_dataset = (len(sd) == 1) or (len(datasets_universe) == 1)
            fixed_target = (len(st) == 1) or (len(classes_universe) == 1)

            if fixed_model and fixed_dataset and fixed_target:
                return "no_comparison"

            candidates: list[str] = []
            if (not fixed_model) and fixed_dataset and fixed_target:
                candidates.append("compare_models")
            if fixed_model and (not fixed_dataset) and fixed_target:
                candidates.append("compare_datasets")
            if fixed_model and fixed_dataset and (not fixed_target):
                candidates.append("compare_targetclasses")

            if len(candidates) == 1:
                return candidates[0]

            raise ValueError(
                "scope='auto' cannot be inferred unambiguously. "
                "Provide an explicit scope or specify selectors so that exactly one dimension varies. "
                f"Available counts: models={len(models_universe)}, datasets={len(datasets_universe)}, "
                f"targetclasses={len(classes_universe)}."
            )

        scope = _infer_scope(scope_in=scope)

        def _one_or_error(
            values: list[Any],
            universe: Sequence[Any],
            *,
            what: str,
            scope_name: str = scope,
        ) -> Any:
            """Return the unique selected value or a unique universe value; else error."""
            if values:
                if len(values) != 1:
                    raise ValueError(
                        f"{what} must contain exactly 1 value for scope='{scope_name}'."
                    )
                return values[0]
            if len(universe) == 1:
                return universe[0]
            raise ValueError(
                f"{what} is required for scope='{scope_name}' when multiple options exist."
            )

        # Aggregate once; selection is a pure filter.
        agg = self.aggregate_over_ntiles().copy()
        agg["scope"] = scope

        if scope == "no_comparison":
            m = _one_or_error(sm, self.model_labels, what="select_model_label")
            d = _one_or_error(sd, self.dataset_labels, what="select_dataset_label")
            t = _one_or_error(
                st,
                list(self.models[0].classes_) if self.models else [],
                what="select_targetclass",
            )
            return agg[
                (agg.model_label == m)
                & (agg.dataset_label == d)
                & (agg.target_class == t)
            ]

        if scope == "compare_models":
            models_sel = sm if sm else self.model_labels
            if len(models_sel) < 2:
                raise ValueError("compare_models requires at least 2 models.")
            d = _one_or_error(sd, self.dataset_labels, what="select_dataset_label")
            t = _one_or_error(
                st, list(self.models[0].classes_), what="select_targetclass"
            )
            return agg[
                (agg.model_label.isin(models_sel))
                & (agg.dataset_label == d)
                & (agg.target_class == t)
            ]

        if scope == "compare_datasets":
            m = _one_or_error(sm, self.model_labels, what="select_model_label")
            datasets_sel = sd if sd else self.dataset_labels
            if len(datasets_sel) < 2:
                raise ValueError("compare_datasets requires at least 2 datasets.")
            t = _one_or_error(
                st, list(self.models[0].classes_), what="select_targetclass"
            )
            return agg[
                (agg.model_label == m)
                & (agg.dataset_label.isin(datasets_sel))
                & (agg.target_class == t)
            ]

        # scope == 'compare_targetclasses'
        m = _one_or_error(sm, self.model_labels, what="select_model_label")
        d = _one_or_error(sd, self.dataset_labels, what="select_dataset_label")
        classes_sel = st if st else list(self.models[0].classes_)
        if len(classes_sel) < 2:
            raise ValueError(
                "compare_targetclasses requires at least 2 target classes."
            )
        return agg[
            (agg.model_label == m)
            & (agg.dataset_label == d)
            & (agg.target_class.isin(classes_sel))
        ]


##########################################################################
## (cumulative) Response, Lift, Gains and Financial plots
##########################################################################

# Dev note:
# The plot_* functions below are intentionally thin wrappers around deterministic,
# unit-testable internal helpers. This prevents repetition and makes behavior
# consistent across plots.
#
# User note:
# - All highlight text follows a single standardized template.
# - All formatting is done via format strings/callables (no explicit round()).

import numbers
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence


class _PlotInputError(ValueError):
    """Raised when plot_input does not conform to the required schema."""


# ---------------------------
# Schema + validation helpers
# ---------------------------


def _require_columns(df: pd.DataFrame, *, required: Sequence[str], where: str) -> None:
    """Validate that required columns exist.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    required : Sequence[str]
        Column names that must exist.
    where : str
        Caller name for error messages.

    Returns
    -------
    None

    Raises
    ------
    _PlotInputError
        If any required column is missing.

    See Also
    --------
    pandas.DataFrame.columns

    Notes
    -----
    Dev note: plotting functions should fail fast with a clear message when the
    input schema is incorrect, instead of raising a KeyError deep inside.

    Examples
    --------
    >>> import pandas as pd
    >>> _require_columns(pd.DataFrame({"a": [1]}), required=["a"], where="x")
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise _PlotInputError(
            f"{where}: plot_input is missing required columns: {missing}."
        )


def _unique_one(values: pd.Series, *, name: str, where: str) -> Any:
    """Return the single unique value in a Series, or raise.

    Parameters
    ----------
    values : pandas.Series
        Series to inspect.
    name : str
        Field name.
    where : str
        Caller name for error messages.

    Returns
    -------
    Any
        The single unique value.

    Raises
    ------
    _PlotInputError
        If the series contains zero or multiple unique values.

    See Also
    --------
    pandas.Series.unique

    Notes
    -----
    Plotting functions in this module assume a single scope per `plot_input`.

    Examples
    --------
    >>> import pandas as pd
    >>> _unique_one(pd.Series(["a", "a"]), name="scope", where="x")
    'a'
    """
    uniq = pd.unique(values.dropna())
    if len(uniq) != 1:
        raise _PlotInputError(
            f"{where}: expected exactly 1 unique '{name}', found {list(uniq)}."
        )
    return uniq[0]


def _ntile_label(ntiles: int) -> str:
    """Map ntile count to a descriptive label.

    Parameters
    ----------
    ntiles : int
        Number of bins.

    Returns
    -------
    str
        One of {'decile', 'percentile', 'ntile'}.

    Raises
    ------
    ValueError
        If ntiles is not a positive integer.

    See Also
    --------
    ModelPlotPy.ntiles

    Notes
    -----
    Rule:

    - 10 -> 'decile'
    - 100 -> 'percentile'
    - otherwise -> 'ntile'

    Examples
    --------
    >>> _ntile_label(10)
    'decile'
    """
    if not isinstance(ntiles, int) or ntiles <= 0:
        raise ValueError("ntiles must be a positive int.")
    if ntiles == 10:
        return "decile"
    if ntiles == 100:
        return "percentile"
    return "ntile"


def _get_ntiles_from_plot_input(df: pd.DataFrame, *, where: str) -> int:
    """Derive ntiles from plot_input deterministically.

    Parameters
    ----------
    df : pandas.DataFrame
        plot_input containing an integer-like 'ntile' column.
    where : str
        Caller name for error messages.

    Returns
    -------
    int
        Maximum ntile value.

    Raises
    ------
    _PlotInputError
        If 'ntile' contains non-integer-like values or is empty.

    See Also
    --------
    ModelPlotPy.aggregate_over_ntiles

    Notes
    -----
    We use max(ntile) instead of nunique()-1. The latter silently breaks when:

    - origin row ntile=0 is absent
    - plot_input is filtered

    Examples
    --------
    >>> import pandas as pd
    >>> _get_ntiles_from_plot_input(pd.DataFrame({"ntile": [0, 1, 2]}), where="x")
    2
    """
    if df.empty:
        raise _PlotInputError(f"{where}: plot_input is empty.")
    nt = df["ntile"]
    if nt.isna().any():
        raise _PlotInputError(f"{where}: plot_input.ntile contains NaN.")

    arr = nt.to_numpy(dtype=float)
    if not np.all(np.equal(arr, np.floor(arr))):
        raise _PlotInputError(f"{where}: plot_input.ntile must be integer-like.")

    ntiles = int(np.max(arr))
    if ntiles < 1:
        raise _PlotInputError(f"{where}: invalid max(ntile)={ntiles}. Expected >= 1.")
    return ntiles


def _scope_grouping(scope: str) -> tuple[str | None, str]:
    """Determine grouping column and label description from scope.

    Parameters
    ----------
    scope : str
        Scope value.

    Returns
    -------
    tuple[Optional[str], str]
        (group_column, label_kind)

    Raises
    ------
    ValueError
        If scope is unknown.

    See Also
    --------
    ModelPlotPy.plotting_scope

    Notes
    -----
    'no_comparison' produces a single line, so group_column is None.

    Examples
    --------
    >>> _scope_grouping("compare_models")
    ('model_label', 'model')
    """
    if scope == "no_comparison":
        return None, "item"
    if scope == "compare_models":
        return "model_label", "model"
    if scope == "compare_datasets":
        return "dataset_label", "dataset"
    if scope == "compare_targetclasses":
        return "target_class", "target"
    raise ValueError(f"Unknown scope='{scope}'.")


def _iter_groups(df: pd.DataFrame, *, scope: str) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield (label, dataframe) groups according to scope.

    Parameters
    ----------
    df : pandas.DataFrame
        plot_input.
    scope : str
        Scope value.

    Yields
    ------
    tuple[str, pandas.DataFrame]
        Label and per-group dataframe.

    Raises
    ------
    ValueError
        If scope is unknown.

    See Also
    --------
    _scope_grouping

    Notes
    -----
    Dev note: Groups are yielded in stable first-seen order using `pd.unique`.

    Examples
    --------
    >>> # internal
    """
    group_col, _ = _scope_grouping(scope)
    if group_col is None:
        yield "single", df
        return

    labels = [str(v) for v in pd.unique(df[group_col])]
    for lab in labels:
        if group_col == "target_class":
            mask = df[group_col].astype(str) == lab
        else:
            mask = df[group_col] == lab
        yield lab, df.loc[mask].copy()


# ---------------------------
# Formatting helpers (no round())
# ---------------------------


def _autopct(frac: float, autopct: str | Callable[[float], str] | None = None) -> str:
    """Format a fraction into a percentage string.

    Parameters
    ----------
    frac : float
        Fraction in [0, 1] (e.g., 0.031).
    autopct : None or str or callable, default=None
        If not *None*, a format string or callable applied to percent values.

        - If a string, it is used as ``fmt % pct``.
        - If a callable, it is called as ``autopct(pct)``.

    Returns
    -------
    str
        Formatted percentage string.

    Raises
    ------
    TypeError
        If `autopct` is neither a string nor callable.

    See Also
    --------
    matplotlib.axes.Axes.annotate

    Notes
    -----
    User note: This helper exists to avoid explicit ``round()`` and to keep a
    consistent formatting policy across plots.

    Examples
    --------
    >>> _autopct(0.031, "%.2f%%")
    '3.10%'
    """
    pct = 100.0 * float(frac)
    if autopct is None:
        return f"{pct:.2f}%"
    if isinstance(autopct, str):
        s = autopct % pct
    elif callable(autopct):
        s = autopct(pct)
    else:
        raise TypeError("autopct must be callable or a format string")

    # Escape percent for matplotlib text backends when needed.
    # re.sub(r"([^\\])%", r"\\1\\%", s)
    return s


def _as_int_like(x: Any, *, name: str, where: str) -> int:
    """Convert an integer-like scalar to int.

    Parameters
    ----------
    x : Any
        Value to convert.
    name : str
        Field name for error messages.
    where : str
        Caller.

    Returns
    -------
    int
        Integer value.

    Raises
    ------
    TypeError
        If the value is not integer-like.
    ValueError
        If the value is not finite.

    See Also
    --------
    numpy.floor

    Notes
    -----
    Strict rule: floats are accepted only if they represent an integer exactly
    (e.g., 10.0). No rounding is performed.

    Examples
    --------
    >>> _as_int_like(10.0, name="n", where="x")
    10
    """
    if isinstance(x, bool):
        raise TypeError(f"{where}: {name} must be an integer-like number, got bool.")
    if isinstance(x, numbers.Integral):
        return int(x)
    if isinstance(x, numbers.Real):
        v = float(x)
        if not np.isfinite(v):
            raise ValueError(f"{where}: {name} must be finite.")
        if v != float(np.floor(v)):
            raise TypeError(f"{where}: {name} must be integer-like, got {x}.")
        return int(v)
    raise TypeError(
        f"{where}: {name} must be an integer-like number, got {type(x).__name__}."
    )


def _currency_fmt(v: float, currency: str = "€") -> str:
    """Format a value as EUR currency string.

    Parameters
    ----------
    v : float
        Amount.
    currency : str
        such as the dollar sign ($), euro sign (€), or pound sign (£).
        These symbols indicate monetary values in financial contexts.

    Returns
    -------
    str
        Currency label (EUR).

    Raises
    ------
    ValueError
        If value is not finite.

    See Also
    --------
    str.format

    Notes
    -----
    User note: This uses numeric formatting (no explicit ``round()`` call).

    Examples
    --------
    >>> _currency_fmt(12.3)
    '€12'
    """
    v = float(v)
    if not np.isfinite(v):
        raise ValueError("currency amount must be finite")
    return f"{currency}{v:,.0f}"


# ---------------------------
# Highlight + annotation helpers
# ---------------------------


def _normalize_highlight_ntiles(
    highlight_ntile: Any, *, ntiles: int, where: str
) -> list[int]:
    """Normalize highlight_ntile into a sorted unique list of ints.

    Parameters
    ----------
    highlight_ntile : Any
        None, an int, or a sequence of ints.
    ntiles : int
        Maximum ntile.
    where : str
        Caller.

    Returns
    -------
    list[int]
        Sorted unique ntile indices in [1, ntiles].

    Raises
    ------
    TypeError
        If `highlight_ntile` is not int-like.
    ValueError
        If any value is outside [1, ntiles].

    See Also
    --------
    ModelPlotPy.ntiles

    Notes
    -----
    Strict rules:

    - Values must be integer-like (no floats/decimals).
    - All values must be within [1, ntiles].
    - Duplicates are removed.

    Examples
    --------
    >>> _normalize_highlight_ntiles([1, 2, 2], ntiles=10, where="x")
    [1, 2]
    """
    if highlight_ntile is None:
        return []

    if isinstance(highlight_ntile, numbers.Integral) and not isinstance(
        highlight_ntile, bool
    ):
        candidates = [int(highlight_ntile)]
    elif isinstance(highlight_ntile, (list, tuple, np.ndarray, pd.Series)):
        candidates = list(highlight_ntile)
    else:
        raise TypeError(
            f"{where}: highlight_ntile must be None, an int, or a sequence of ints within [1, {ntiles}]."
        )

    out: list[int] = []
    for v in candidates:
        if isinstance(v, bool) or not isinstance(v, numbers.Integral):
            raise TypeError(
                f"{where}: highlight_ntile values must be ints. Got {type(v).__name__}."
            )
        h = int(v)
        if h < 1 or h > ntiles:
            raise ValueError(
                f"{where}: highlight_ntile must be in [1, {ntiles}], got {h}."
            )
        out.append(h)
    return sorted(set(out))


def _validate_highlight_how(highlight_how: str, *, where: str) -> None:
    """Validate highlight_how.

    Parameters
    ----------
    highlight_how : str
        One of {'plot', 'text', 'plot_text'}.
    where : str
        Caller.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If highlight_how is invalid.

    See Also
    --------
    plot_response

    Notes
    -----
    Strict membership check.

    Examples
    --------
    >>> _validate_highlight_how("plot", where="x")
    """
    if highlight_how not in ("plot", "text", "plot_text"):
        raise ValueError(
            f"{where}: invalid highlight_how='{highlight_how}'. Allowed={'plot', 'text', 'plot_text'}."
        )


def _annotation_xytext(
    annotation_index: int,
    *,
    layout_cols: int = 5,
    x_offset: int = -24,
    x_step: int = 24,
    base: int = 24,
    gap: int = 12,
) -> tuple[int, int]:
    """Compute a deterministic annotation offset to reduce overlaps.

    Parameters
    ----------
    annotation_index : int
        0-based index of the annotation.
    layout_cols : int, default=5
        Number of horizontal slots used before moving to the next row.
    x_offset : int, default=-40
        Starting x offset (points).
    x_step : int, default=20
        Horizontal step between slots (points).
    base : int, default=30
        Base absolute y offset (points).
    gap : int, default=12
        Extra y gap added per row.

    Returns
    -------
    tuple[int, int]
        (x_offset, y_offset) in points.

    Raises
    ------
    ValueError
        If `annotation_index` is negative or `layout_cols` is invalid.

    See Also
    --------
    matplotlib.axes.Axes.annotate

    Notes
    -----
    Deterministic rule (no heuristics):

    - Place annotations on a fixed grid of offsets.
    - Alternate above/below each row to reduce collisions.

    Examples
    --------
    >>> _annotation_xytext(0)
    (-40, -30)
    """
    if annotation_index < 0:
        raise ValueError("annotation_index must be >= 0.")
    if layout_cols < 1:
        raise ValueError("layout_cols must be >= 1.")

    col = annotation_index % layout_cols
    row = annotation_index // layout_cols

    xo = x_offset + x_step * col
    mag = base + gap * row
    yo = -mag if (annotation_index % 2) == 0 else mag
    return (int(xo), int(yo))


def _value_at_ntile(df: pd.DataFrame, *, ntile: int, col: str, where: str) -> float:
    """Select a single value at a given ntile.

    Parameters
    ----------
    df : pandas.DataFrame
        Group dataframe.
    ntile : int
        Ntile to select.
    col : str
        Column to retrieve.
    where : str
        Caller.

    Returns
    -------
    float
        Selected value.

    Raises
    ------
    _PlotInputError
        If there is not exactly one row for that ntile.

    See Also
    --------
    pandas.DataFrame.loc

    Notes
    -----
    Dev note: We require a single row at a given ntile to avoid silent aggregation.

    Examples
    --------
    >>> import pandas as pd
    >>> _value_at_ntile(
    ...     pd.DataFrame({"ntile": [1], "x": [0.1]}), ntile=1, col="x", where="x"
    ... )
    0.1
    """
    vals = df.loc[df["ntile"] == ntile, col].to_numpy(dtype=float)
    if vals.size != 1:
        raise _PlotInputError(
            f"{where}: expected exactly 1 value for {col} at ntile={ntile}, found {vals.size}."
        )
    v = float(vals[0])
    if not np.isfinite(v):
        raise _PlotInputError(f"{where}: {col} at ntile={ntile} is not finite.")
    return v


def _annotate_highlight(
    ax: plt.Axes,
    *,
    x: int,
    y: float,
    x0: int,
    y0: float,
    color: str,
    text: str,
    xytext: tuple[int, int],
    annotation_kws: Mapping[str, Any] | None = None,
) -> None:
    """Draw guide lines and a callout annotation.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    x, y : int, float
        Highlight point.
    x0, y0 : int, float
        Origin for guide lines.
    color : str
        Color for guides/marker.
    text : str
        Annotation text.
    xytext : tuple[int, int]
        Offset (points) for the callout.
    annotation_kws : Mapping[str, Any] or None, default=None
        Extra kwargs for :meth:`matplotlib.axes.Axes.annotate`.

    Returns
    -------
    None

    Raises
    ------
    None

    See Also
    --------
    matplotlib.axes.Axes.annotate

    Notes
    -----
    Dev note: We use ax.plot (not plt.plot) to avoid leaking state across figures.

    Examples
    --------
    >>> # internal
    """
    ak = dict(annotation_kws or {})

    marker_size = float(ak.pop("marker_size", 6.0))

    ax.plot([x0, x], [y] * 2, linestyle="-.", color=color, lw=1.5)
    ax.plot([x] * 2, [y0, y], linestyle="-.", color=color, lw=1.5)
    ax.plot(x, y, marker="o", ms=marker_size, color=color)

    mode = ak.pop("mode", "callout")
    if mode == "marker":
        return

    # bbox = ak.pop("bbox", {"boxstyle": "round,pad=0.4", "fc": color, "alpha": 0.85})
    # arrowprops = ak.pop("arrowprops", {"arrowstyle": "->", "color": "black"})
    bbox = {"boxstyle": "round,pad=0.4", "fc": color, "alpha": 0.85}
    bbox.update(ak.pop("bbox", {}) if isinstance(ak.get("bbox", {}), dict) else {})
    arrowprops = {"arrowstyle": "->", "color": "black"}
    arrowprops.update(
        ak.pop("arrowprops", {}) if isinstance(ak.get("arrowprops", {}), dict) else {}
    )

    va = "bottom" if xytext[1] >= 0 else "top"
    ax.annotate(
        text,
        xy=(x, y),
        xytext=xytext,
        textcoords="offset points",
        ha=ak.pop("ha", "center"),
        va=ak.pop("va", va),
        color=ak.pop("color", "black"),
        bbox=bbox,
        arrowprops=arrowprops,
        **ak,
    )


# ---------------------------
# Plot kwarg parsing
# ---------------------------


@dataclass(frozen=True)
class _PlotKws:
    """Container for per-component plot kwargs.

    Notes
    -----
    User note: This prevents accidental mixing of kwargs meant for different
    Matplotlib calls (line vs legend vs footer).
    """

    line_kws: Mapping[str, Any]
    ref_line_kws: Mapping[str, Any]
    legend_kws: Mapping[str, Any]
    grid_kws: Mapping[str, Any]
    axes_kws: Mapping[str, Any]
    annotation_kws: Mapping[str, Any]
    footer_kws: Mapping[str, Any]


def _parse_plot_kws(
    *,
    line_kws: Mapping[str, Any] | None,
    ref_line_kws: Mapping[str, Any] | None,
    legend_kws: Mapping[str, Any] | None,
    grid_kws: Mapping[str, Any] | None,
    axes_kws: Mapping[str, Any] | None,
    annotation_kws: Mapping[str, Any] | None,
    footer_kws: Mapping[str, Any] | None,
    legacy_line_kwargs: Mapping[str, Any],
    where: str,
) -> _PlotKws:
    """Parse and validate plot kwargs.

    Parameters
    ----------
    line_kws, ref_line_kws, legend_kws, grid_kws, axes_kws, annotation_kws, footer_kws : Mapping[str, Any] or None
        Per-component keyword arguments.
    legacy_line_kwargs : Mapping[str, Any]
        Backward-compatible `**kwargs` forwarded to line plotting.
    where : str
        Caller.

    Returns
    -------
    _PlotKws
        Parsed kwargs.

    Raises
    ------
    ValueError
        If both `line_kws` and legacy `**kwargs` are provided.

    Notes
    -----
    Strict rule: do not merge two sources of line kwargs silently.

    Examples
    --------
    >>> # internal
    """
    if line_kws is not None and legacy_line_kwargs:
        raise ValueError(
            f"{where}: do not pass both line_kws and **kwargs. Use line_kws only."
        )

    lk = dict(line_kws or legacy_line_kwargs or {})
    rlk = dict(ref_line_kws or {})
    rlk.setdefault("linestyle", "dashed")

    legk = dict(legend_kws or {})
    legk.setdefault("shadow", False)
    legk.setdefault("frameon", False)

    gk = dict(grid_kws or {})
    gk.setdefault("visible", True)

    axk = dict(axes_kws or {})

    ank = dict(annotation_kws or {})

    fk = dict(footer_kws or {})
    fk.setdefault("x", 0.00)
    fk.setdefault("y", 0.00)
    fk.setdefault("ha", "left")
    fk.setdefault("va", "top")
    fk.setdefault("fontsize", 10)
    fk.setdefault("base_pad", 0.10)
    fk.setdefault("line_pad", 0.028)
    fk.setdefault("y_margin", 0.01)

    return _PlotKws(
        line_kws=lk,
        ref_line_kws=rlk,
        legend_kws=legk,
        grid_kws=gk,
        axes_kws=axk,
        annotation_kws=ank,
        footer_kws=fk,
    )


def _setup_axis(
    ax: plt.Axes,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    ntiles: int,
    xlim: tuple[int, int],
    percent_y: bool,
    grid: bool,
) -> None:
    """Apply consistent axis styling.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to configure.
    title : str
        Axes title.
    xlabel : str
        X label.
    ylabel : str
        Y label.
    ntiles : int
        Number of ntiles.
    xlim : tuple[int, int]
        X axis limits.
    percent_y : bool
        Whether to format y axis as percent in [0, 1].
    grid : bool
        Whether to enable grid.

    Returns
    -------
    None

    Notes
    -----
    Strict rule: show every integer ntile tick (no downsampling).

    Examples
    --------
    >>> # internal
    """
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if percent_y:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax.set_xticks(np.arange(0, ntiles + 1, 1))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    ax.set_xlim([int(xlim[0]), int(xlim[1])])
    if grid:
        ax.grid(True)


def _scope_triplet(
    *,
    scope: str,
    group_label: str,
    models: Sequence[str],
    datasets: Sequence[str],
    targets: Sequence[str],
) -> tuple[str, str, str]:
    """Resolve (model, dataset, target) for a given group label.

    Parameters
    ----------
    scope : str
        Scope name.
    group_label : str
        Group label as produced by `_iter_groups`.
    models : Sequence[str]
        Unique models.
    datasets : Sequence[str]
        Unique datasets.
    targets : Sequence[str]
        Unique targets.

    Returns
    -------
    tuple[str, str, str]
        (model, dataset, target)

    Raises
    ------
    ValueError
        If scope is invalid.

    Notes
    -----
    This keeps highlight text consistent across plots.

    Examples
    --------
    >>> _scope_triplet(
    ...     scope="compare_models",
    ...     group_label="M1",
    ...     models=["M1"],
    ...     datasets=["D"],
    ...     targets=["1"],
    ... )
    ('M1', 'D', '1')
    """
    if scope == "compare_models":
        return (group_label, datasets[0], targets[0])
    if scope == "compare_datasets":
        return (models[0], group_label, targets[0])
    if scope == "compare_targetclasses":
        return (models[0], datasets[0], group_label)
    if scope == "no_comparison":
        return (models[0], datasets[0], targets[0])
    raise ValueError(f"Unknown scope='{scope}'.")


def _format_highlight_line(
    *,
    metric: str,
    range_kind: Literal["at", "cum"],
    ntile_label: str,
    ntile: int,
    model: str,
    dataset: str,
    target: str,
    value: str,
    numerator: int | None = None,
    denominator: int | None = None,
) -> str:
    """Build a standardized highlight line.

    Parameters
    ----------
    metric : str
        Metric name shown in the highlight prefix.
    range_kind : {'at', 'cum'}
        Whether the metric is evaluated at a single ntile ('at') or from
        ntile 1 up to the selected ntile ('cum').
    ntile_label : str
        One of {'decile','percentile','ntile'}.
    ntile : int
        Selected ntile.
    model, dataset, target : str
        Identifiers included in the standardized template.
    value : str
        Formatted metric value.
    numerator, denominator : int or None
        Optional count fields, displayed as "num / den".

    Returns
    -------
    str
        Standardized highlight line.

    Notes
    -----
    Template:

    - Response @ decile 3 | model=... | dataset=... | target=... | value=3.00% — 302 / 10000

    Examples
    --------
    >>> _format_highlight_line(
    ...     metric="Response",
    ...     range_kind="at",
    ...     ntile_label="decile",
    ...     ntile=3,
    ...     model="M",
    ...     dataset="D",
    ...     target="1",
    ...     value="3.00%",
    ... )
    'Response @ decile 3 | model=M | dataset=D | target=1 | value=3.00%'
    """
    prefix = (
        f"{metric} @ {ntile_label} {ntile}"
        if range_kind == "at"
        else f"{metric} 1..{ntile_label} {ntile}"
    )
    parts = [
        prefix,
        f"model={model}",
        f"dataset={dataset}",
        f"target={target}",
        f"value={value}",
    ]
    s = " | ".join(parts)
    if numerator is not None and denominator is not None:
        s += f" — pos/tot={numerator:,} / {denominator:,}"
    return s


def _render_footer_text(
    fig: plt.Figure,
    *,
    lines: Sequence[str],
    footer_kws: Mapping[str, Any],
) -> None:
    """Render footer text without overlapping the plot area.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to write into.
    lines : Sequence[str]
        Lines to render.
    footer_kws : Mapping[str, Any]
        Footer configuration.

        Recognized keys:

        - base_pad : float
        - line_pad : float
        - y_margin : float
        - x, y, ha, va, fontsize : forwarded to text call

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If computed padding exceeds a safe fraction of the figure.

    Notes
    -----
    Deterministic layout:

    - bottom_pad = base_pad + line_pad * (n_lines - 1)
    - reserve [0, bottom_pad] and draw the footer inside that region

    Examples
    --------
    >>> # internal
    """
    if not lines:
        return

    n_lines = len(lines)
    base_pad = float(footer_kws.get("base_pad", 0.10))
    line_pad = float(footer_kws.get("line_pad", 0.028))
    y_margin = float(footer_kws.get("y_margin", 0.01))

    bottom_pad = base_pad + line_pad * max(0, n_lines - 1)
    if bottom_pad >= 1.0:
        logger.info(
            f"footer would reserve too much vertical space (bottom_pad={bottom_pad:.2f}). "
            "Reduce highlighted items or override footer_kws(base_pad/line_pad)."
        )
        # bottom_pad=min(0.6, bottom_pad)

    # Use a dedicated axes for the footer to avoid overlap across backends.
    footer_ax = fig.add_axes([0.0, 0.0, 1.0, bottom_pad], frameon=False)
    footer_ax.set_axis_off()

    x = float(footer_kws.get("x", 0.00))
    y = float(footer_kws.get("y", 0.00))

    text = "\n".join(lines)
    footer_ax.text(
        x,
        min(1.0 - y_margin, y),
        text,
        ha=str(footer_kws.get("ha", "left")),
        va=str(footer_kws.get("va", "top")),
        fontsize=float(footer_kws.get("fontsize", 10)),
        transform=footer_ax.transAxes,
    )


# ---------------------------
# Core plotting helper
# ---------------------------


def _plot_metric_with_reference(  # noqa: PLR0912
    plot_input: pd.DataFrame,
    *,
    metric_col: str,
    ref_col: str | None,
    metric_name: str,
    title: str,
    ylabel: str,
    percent_y: bool,
    xlim_start: int,
    legend_loc: str,
    highlight_ntiles: Sequence[int],
    highlight_how: Literal["plot", "text", "plot_text"],
    highlight_kind: Literal["at", "cum"],
    value_formatter: Callable[[float], str],
    counts_at_ntile: Callable[[pd.DataFrame, int], tuple[int | None, int | None]],
    x0_for_highlight: int,
    y0_for_highlight: float,
    autopct: str | Callable[[float], str] | None,
    ax: plt.Axes | None,
    figsize: tuple[int, int],
    plot_kws: _PlotKws,
    collect_highlight_lines: list[str] | None,
    render_footer: bool,
) -> plt.Axes:
    """Core implementation for single-metric plots.

    Parameters
    ----------
    plot_input : pandas.DataFrame
        Data returned by :meth:`ModelPlotPy.plotting_scope`.
    metric_col : str
        Column name for the main metric.
    ref_col : str or None
        Optional reference column.
    metric_name : str
        Metric name used in standardized highlight lines.
    title : str
        Plot title (figure title).
    ylabel : str
        Y-axis label.
    percent_y : bool
        Whether to format y axis as percent.
    xlim_start : int
        Left x-limit.
    legend_loc : str
        Legend location.
    highlight_ntiles : Sequence[int]
        Ntile indices to highlight.
    highlight_how : {'plot','text','plot_text'}
        Where to render highlight information.
    highlight_kind : {'at','cum'}
        Whether the highlighted metric is at-N or cumulative 1..N.
    value_formatter : callable
        Function converting raw metric value into string.
    counts_at_ntile : callable
        Function returning (numerator, denominator) for the standardized line.
    x0_for_highlight, y0_for_highlight : int, float
        Origins for guide lines.
    autopct : str | callable | None
        Percentage formatter for metrics in [0, 1].
    ax : matplotlib.axes.Axes or None
        If provided, draw into this axes.
    figsize : tuple[int, int]
        Figure size when `ax` is None.
    plot_kws : _PlotKws
        Parsed per-component kwargs.
    collect_highlight_lines : list[str] or None
        If provided, highlight lines are appended here (used by plot_all).
    render_footer : bool
        Whether to render footer text in this call.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.

    Raises
    ------
    _PlotInputError
        If required columns are missing.

    Notes
    -----
    - Grouping is controlled by plot_input['scope'].
    - All groups are sorted by ntile before plotting.

    Examples
    --------
    >>> # internal
    """
    where = f"plot_{metric_col.lower()}"

    required = [
        "model_label",
        "dataset_label",
        "target_class",
        "scope",
        "ntile",
        metric_col,
    ]
    if ref_col is not None:
        required.append(ref_col)
    _require_columns(plot_input, required=required, where=where)

    scope = str(_unique_one(plot_input["scope"], name="scope", where=where))
    ntiles = _get_ntiles_from_plot_input(plot_input, where=where)
    nlabel = _ntile_label(ntiles)

    # Create figure/axes.
    multi_panel = ax is not None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(title, fontsize=16)
    else:
        fig = ax.figure

    models = [str(v) for v in pd.unique(plot_input["model_label"])]
    datasets = [str(v) for v in pd.unique(plot_input["dataset_label"])]
    targets = [str(v) for v in pd.unique(plot_input["target_class"].astype(str))]

    # Details title by scope.
    if scope == "no_comparison":
        details_title = (
            f"model: {models[0]} & dataset: {datasets[0]} & target class: {targets[0]}"
        )
    elif scope == "compare_datasets":
        details_title = f"scope: comparing datasets & model: {models[0]} & target class: {targets[0]}"
    elif scope == "compare_models":
        details_title = f"scope: comparing models & dataset: {datasets[0]} & target class: {targets[0]}"
    else:
        details_title = f"scope: comparing target classes & dataset: {datasets[0]} & model: {models[0]}"

    ax_title = f"{title}\n{details_title}" if multi_panel else details_title

    grid_visible = bool(plot_kws.grid_kws.get("visible", True))

    _setup_axis(
        ax,
        title=ax_title,
        xlabel=nlabel,
        ylabel=ylabel,
        ntiles=ntiles,
        xlim=(int(xlim_start), int(ntiles)),
        percent_y=percent_y,
        grid=grid_visible,
    )

    # Plot lines and capture per-group colors.
    text_lines: list[str] = []
    label_to_color: dict[str, str] = {}

    for lab, gdf in _iter_groups(plot_input, scope=scope):
        gdf = gdf.sort_values("ntile")  # noqa: PLW2901

        # Label on main metric curve.
        line_label = targets[0] if scope == "no_comparison" else lab
        line = ax.plot(
            gdf["ntile"], gdf[metric_col], label=line_label, **dict(plot_kws.line_kws)
        )
        color = str(line[0].get_color())
        label_to_color[str(lab)] = color

        # Reference curve.
        if ref_col is not None:
            rk = dict(plot_kws.ref_line_kws)
            rk.setdefault("color", color)
            rk.setdefault("linestyle", "dashed")
            ax.plot(gdf["ntile"], gdf[ref_col], label=f"reference ({line_label})", **rk)

    ax.legend(loc=legend_loc, **dict(plot_kws.legend_kws))

    # Highlight.
    if highlight_ntiles:
        _validate_highlight_how(highlight_how, where=where)

        groups = list(_iter_groups(plot_input, scope=scope))
        n_groups = len(groups)

        for j, nt in enumerate(list(highlight_ntiles)):
            for i, (lab, gdf) in enumerate(groups):
                gdf = gdf.sort_values("ntile")  # noqa: PLW2901
                y = _value_at_ntile(gdf, ntile=int(nt), col=metric_col, where=where)
                color = label_to_color.get(str(lab), "C0")

                # Callout text value.
                val_text = value_formatter(float(y))

                # Plot annotation.
                ann_idx = i + j * n_groups
                xytext = _annotation_xytext(
                    ann_idx,
                    layout_cols=_as_int_like(
                        plot_kws.annotation_kws.get("layout_cols", 5),
                        name="layout_cols",
                        where=where,
                    ),
                    x_offset=_as_int_like(
                        plot_kws.annotation_kws.get("x_offset", -40),
                        name="x_offset",
                        where=where,
                    ),
                    x_step=_as_int_like(
                        plot_kws.annotation_kws.get("x_step", 20),
                        name="x_step",
                        where=where,
                    ),
                    base=_as_int_like(
                        plot_kws.annotation_kws.get("base", 30),
                        name="base",
                        where=where,
                    ),
                    gap=_as_int_like(
                        plot_kws.annotation_kws.get("gap", 12), name="gap", where=where
                    ),
                )

                _annotate_highlight(
                    ax,
                    x=int(nt),
                    y=float(y),
                    x0=int(x0_for_highlight),
                    y0=float(y0_for_highlight),
                    color=color,
                    text=val_text,
                    xytext=xytext,
                    annotation_kws=plot_kws.annotation_kws,
                )

                # Standardized line.
                model, dataset, target = _scope_triplet(
                    scope=scope,
                    group_label=str(lab),
                    models=models,
                    datasets=datasets,
                    targets=targets,
                )

                num, den = counts_at_ntile(gdf, int(nt))
                line_txt = _format_highlight_line(
                    metric=metric_name,
                    range_kind=highlight_kind,
                    ntile_label=nlabel,
                    ntile=int(nt),
                    model=model,
                    dataset=dataset,
                    target=target,
                    value=val_text,
                    numerator=num,
                    denominator=den,
                )

                # Append to local and optional collector.
                text_lines.append(line_txt)
                if collect_highlight_lines is not None:
                    collect_highlight_lines.append(line_txt)

        # Render highlight text.
        if highlight_how in ("text", "plot_text"):
            print("\n".join(text_lines))  # noqa: T201

        if render_footer and highlight_how in ("plot", "plot_text"):
            _render_footer_text(fig, lines=text_lines, footer_kws=plot_kws.footer_kws)

    return ax


# ---------------------------
# Public plotting functions
# ---------------------------


@save_plot_decorator
@_docstring.interpd
def plot_response(
    plot_input: pd.DataFrame,
    *,
    highlight_ntile: int | Sequence[int] | None = None,
    highlight_how: Literal["plot", "text", "plot_text"] = "plot_text",
    autopct: str | Callable[[float], str] | None = "%.2f%%",
    line_kws: Mapping[str, Any] | None = None,
    ref_line_kws: Mapping[str, Any] | None = None,
    legend_kws: Mapping[str, Any] | None = None,
    grid_kws: Mapping[str, Any] | None = None,
    axes_kws: Mapping[str, Any] | None = None,
    annotation_kws: Mapping[str, Any] | None = None,
    footer_kws: Mapping[str, Any] | None = None,
    save_fig: bool = True,
    save_fig_filename: str = "",
    **kwargs: Any,
) -> plt.Axes:
    """Plot response curve.

    Parameters
    ----------
    plot_input : pandas.DataFrame
        Output of :meth:`ModelPlotPy.plotting_scope`.
    highlight_ntile : int, Sequence[int], or None, default=None
        Ntile index/indices to highlight (each must be in ``1..ntiles``).
    highlight_how : {'plot', 'text', 'plot_text'}, default='plot_text'
        Where to render highlight information.
    autopct : None or str or callable, default='%.2f%%'
        Percentage formatter for values in [0, 1].

        - If a string, it is used as ``fmt % pct``.
        - If a callable, it is called as ``autopct(pct)``.

        If None, values are formatted with two decimals.
    line_kws, ref_line_kws, legend_kws, grid_kws, axes_kws, annotation_kws, footer_kws : Mapping[str, Any] or None
        Per-component styling kwargs.
    save_fig : bool, default=True
        Used by :func:`~scikitplot.utils._matplotlib.save_plot_decorator`.
    save_fig_filename : str, default=''
        Used by :func:`~scikitplot.utils._matplotlib.save_plot_decorator`.
    **kwargs : Any
        Legacy alias for ``line_kws``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.

    Raises
    ------
    _PlotInputError
        If plot_input lacks required columns.

    See Also
    --------
    plot_cumresponse, plot_cumlift, plot_cumgains

    Notes
    -----
    Response is the positive rate within each ntile bucket:

    - response = pos / tot (at ntile N)

    The highlight line uses:

    - Response @ decile N | ... | value=..% — pos / tot

    Examples
    --------
    >>> # ax = plot_response(plot_input)
    """
    where = "plot_response"
    _require_columns(
        plot_input,
        required=[
            "scope",
            "ntile",
            "pct",
            "pct_ref",
            "pos",
            "tot",
            "model_label",
            "dataset_label",
            "target_class",
        ],
        where=where,
    )

    ntiles = _get_ntiles_from_plot_input(plot_input, where=where)
    hl = _normalize_highlight_ntiles(highlight_ntile, ntiles=ntiles, where=where)

    pk = _parse_plot_kws(
        line_kws=line_kws,
        ref_line_kws=ref_line_kws,
        legend_kws=legend_kws,
        grid_kws=grid_kws,
        axes_kws=axes_kws,
        annotation_kws=annotation_kws,
        footer_kws=footer_kws,
        legacy_line_kwargs=kwargs,
        where=where,
    )

    def _val_fmt(v: float) -> str:
        return _autopct(v, autopct)

    def _counts(gdf: pd.DataFrame, nt: int) -> tuple[int | None, int | None]:
        pos = _as_int_like(
            _value_at_ntile(gdf, ntile=nt, col="pos", where=where),
            name="pos",
            where=where,
        )
        tot = _as_int_like(
            _value_at_ntile(gdf, ntile=nt, col="tot", where=where),
            name="tot",
            where=where,
        )
        return pos, tot

    return _plot_metric_with_reference(
        plot_input,
        metric_col="pct",
        ref_col="pct_ref",
        metric_name="Response",
        title="Response",
        ylabel="response",
        percent_y=True,
        xlim_start=1,
        legend_loc="upper right",
        highlight_ntiles=hl,
        highlight_how=highlight_how,
        highlight_kind="at",
        value_formatter=_val_fmt,
        counts_at_ntile=_counts,
        x0_for_highlight=1,
        y0_for_highlight=0.0,
        autopct=autopct,
        ax=None,
        figsize=(10, 5),
        plot_kws=pk,
        collect_highlight_lines=None,
        render_footer=True,
    )


@save_plot_decorator
@_docstring.interpd
def plot_cumresponse(
    plot_input: pd.DataFrame,
    *,
    highlight_ntile: int | Sequence[int] | None = None,
    highlight_how: Literal["plot", "text", "plot_text"] = "plot_text",
    autopct: str | Callable[[float], str] | None = "%.2f%%",
    line_kws: Mapping[str, Any] | None = None,
    ref_line_kws: Mapping[str, Any] | None = None,
    legend_kws: Mapping[str, Any] | None = None,
    grid_kws: Mapping[str, Any] | None = None,
    axes_kws: Mapping[str, Any] | None = None,
    annotation_kws: Mapping[str, Any] | None = None,
    footer_kws: Mapping[str, Any] | None = None,
    save_fig: bool = True,
    save_fig_filename: str = "",
    **kwargs: Any,
) -> plt.Axes:
    """Plot cumulative response curve.

    Parameters
    ----------
    plot_input : pandas.DataFrame
        Output of :meth:`ModelPlotPy.plotting_scope`.
    highlight_ntile : int, Sequence[int], or None, default=None
        Ntile index/indices to highlight (each must be in ``1..ntiles``).
    highlight_how : {'plot', 'text', 'plot_text'}, default='plot_text'
        Where to render highlight information.
    autopct : None or str or callable, default='%.2f%%'
        Percentage formatter for values in [0, 1].
    line_kws, ref_line_kws, legend_kws, grid_kws, axes_kws, annotation_kws, footer_kws : Mapping[str, Any] or None
        Per-component styling kwargs.
    save_fig : bool, default=True
        Used by :func:`~scikitplot.utils._matplotlib.save_plot_decorator`.
    save_fig_filename : str, default=''
        Used by :func:`~scikitplot.utils._matplotlib.save_plot_decorator`.
    **kwargs : Any
        Legacy alias for ``line_kws``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.

    Raises
    ------
    _PlotInputError
        If plot_input lacks required columns.

    See Also
    --------
    plot_response, plot_cumlift, plot_cumgains

    Notes
    -----
    Cumulative response is the positive rate from ntile 1 up to N:

    - cumresponse = cumpos / cumtot (1..N)

    Highlight line uses:

    - CumResponse 1..decile N | ... | value=..% — cumpos / cumtot

    Examples
    --------
    >>> # ax = plot_cumresponse(plot_input)
    """
    where = "plot_cumresponse"
    _require_columns(
        plot_input,
        required=[
            "scope",
            "ntile",
            "cumpct",
            "pct_ref",
            "cumpos",
            "cumtot",
            "model_label",
            "dataset_label",
            "target_class",
        ],
        where=where,
    )

    ntiles = _get_ntiles_from_plot_input(plot_input, where=where)
    hl = _normalize_highlight_ntiles(highlight_ntile, ntiles=ntiles, where=where)

    pk = _parse_plot_kws(
        line_kws=line_kws,
        ref_line_kws=ref_line_kws,
        legend_kws=legend_kws,
        grid_kws=grid_kws,
        axes_kws=axes_kws,
        annotation_kws=annotation_kws,
        footer_kws=footer_kws,
        legacy_line_kwargs=kwargs,
        where=where,
    )

    def _val_fmt(v: float) -> str:
        return _autopct(v, autopct)

    def _counts(gdf: pd.DataFrame, nt: int) -> tuple[int | None, int | None]:
        num = _as_int_like(
            _value_at_ntile(gdf, ntile=nt, col="cumpos", where=where),
            name="cumpos",
            where=where,
        )
        den = _as_int_like(
            _value_at_ntile(gdf, ntile=nt, col="cumtot", where=where),
            name="cumtot",
            where=where,
        )
        return num, den

    return _plot_metric_with_reference(
        plot_input,
        metric_col="cumpct",
        ref_col="pct_ref",
        metric_name="CumResponse",
        title="Cumulative Response",
        ylabel="cumulative response",
        percent_y=True,
        xlim_start=1,
        legend_loc="upper right",
        highlight_ntiles=hl,
        highlight_how=highlight_how,
        highlight_kind="cum",
        value_formatter=_val_fmt,
        counts_at_ntile=_counts,
        x0_for_highlight=1,
        y0_for_highlight=0.0,
        autopct=autopct,
        ax=None,
        figsize=(10, 5),
        plot_kws=pk,
        collect_highlight_lines=None,
        render_footer=True,
    )


@save_plot_decorator
@_docstring.interpd
def plot_cumlift(
    plot_input: pd.DataFrame,
    *,
    highlight_ntile: int | Sequence[int] | None = None,
    highlight_how: Literal["plot", "text", "plot_text"] = "plot_text",
    line_kws: Mapping[str, Any] | None = None,
    ref_line_kws: Mapping[str, Any] | None = None,
    legend_kws: Mapping[str, Any] | None = None,
    grid_kws: Mapping[str, Any] | None = None,
    axes_kws: Mapping[str, Any] | None = None,
    annotation_kws: Mapping[str, Any] | None = None,
    footer_kws: Mapping[str, Any] | None = None,
    save_fig: bool = True,
    save_fig_filename: str = "",
    **kwargs: Any,
) -> plt.Axes:
    """Plot cumulative lift curve.

    Parameters
    ----------
    plot_input : pandas.DataFrame
        Output of :meth:`ModelPlotPy.plotting_scope`.
    highlight_ntile : int, Sequence[int], or None, default=None
        Ntile index/indices to highlight (each must be in ``1..ntiles``).
    highlight_how : {'plot', 'text', 'plot_text'}, default='plot_text'
        Where to render highlight information.
    line_kws, ref_line_kws, legend_kws, grid_kws, axes_kws, annotation_kws, footer_kws : Mapping[str, Any] or None
        Per-component styling kwargs.
    save_fig : bool, default=True
        Used by :func:`~scikitplot.utils._matplotlib.save_plot_decorator`.
    save_fig_filename : str, default=''
        Used by :func:`~scikitplot.utils._matplotlib.save_plot_decorator`.
    **kwargs : Any
        Legacy alias for ``line_kws``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.

    Raises
    ------
    _PlotInputError
        If plot_input lacks required columns.

    See Also
    --------
    plot_response, plot_cumresponse, plot_cumgains

    Notes
    -----
    Lift is a ratio (baseline = 1.0). It is highlighted as cumulative 1..N.

    Examples
    --------
    >>> # ax = plot_cumlift(plot_input)
    """
    where = "plot_cumlift"
    _require_columns(
        plot_input,
        required=[
            "scope",
            "ntile",
            "cumlift",
            "cumpos",
            "cumtot",
            "model_label",
            "dataset_label",
            "target_class",
        ],
        where=where,
    )

    ntiles = _get_ntiles_from_plot_input(plot_input, where=where)
    hl = _normalize_highlight_ntiles(highlight_ntile, ntiles=ntiles, where=where)

    pk = _parse_plot_kws(
        line_kws=line_kws,
        ref_line_kws=ref_line_kws,
        legend_kws=legend_kws,
        grid_kws=grid_kws,
        axes_kws=axes_kws,
        annotation_kws=annotation_kws,
        footer_kws=footer_kws,
        legacy_line_kwargs=kwargs,
        where=where,
    )

    def _val_fmt(v: float) -> str:
        return f"{float(v):.2f}x"

    def _counts(gdf: pd.DataFrame, nt: int) -> tuple[int | None, int | None]:
        num = _as_int_like(
            _value_at_ntile(gdf, ntile=nt, col="cumpos", where=where),
            name="cumpos",
            where=where,
        )
        den = _as_int_like(
            _value_at_ntile(gdf, ntile=nt, col="cumtot", where=where),
            name="cumtot",
            where=where,
        )
        return num, den

    ax = _plot_metric_with_reference(
        plot_input,
        metric_col="cumlift",
        ref_col=None,
        metric_name="CumLift",
        title="Cumulative Lift",
        ylabel="cumulative lift (x)",
        percent_y=False,
        xlim_start=1,
        legend_loc="upper right",
        highlight_ntiles=hl,
        highlight_how=highlight_how,
        highlight_kind="cum",
        value_formatter=_val_fmt,
        counts_at_ntile=_counts,
        x0_for_highlight=1,
        y0_for_highlight=0.0,
        autopct=None,
        ax=None,
        figsize=(10, 5),
        plot_kws=pk,
        collect_highlight_lines=None,
        render_footer=True,
    )

    # Baseline.
    ax.axhline(1.0, linestyle="dashed", color="grey", lw=1.0, label="baseline (1.0x)")
    ax.legend(loc="upper right", **dict(pk.legend_kws))
    return ax


@save_plot_decorator
@_docstring.interpd
def plot_cumgains(
    plot_input: pd.DataFrame,
    *,
    highlight_ntile: int | Sequence[int] | None = None,
    highlight_how: Literal["plot", "text", "plot_text"] = "plot_text",
    autopct: str | Callable[[float], str] | None = "%.2f%%",
    line_kws: Mapping[str, Any] | None = None,
    ref_line_kws: Mapping[str, Any] | None = None,
    legend_kws: Mapping[str, Any] | None = None,
    grid_kws: Mapping[str, Any] | None = None,
    axes_kws: Mapping[str, Any] | None = None,
    annotation_kws: Mapping[str, Any] | None = None,
    footer_kws: Mapping[str, Any] | None = None,
    save_fig: bool = True,
    save_fig_filename: str = "",
    **kwargs: Any,
) -> plt.Axes:
    """Plot cumulative gains curve.

    Parameters
    ----------
    plot_input : pandas.DataFrame
        Output of :meth:`ModelPlotPy.plotting_scope`.
    highlight_ntile : int, Sequence[int], or None, default=None
        Ntile index/indices to highlight (each must be in ``1..ntiles``).
    highlight_how : {'plot', 'text', 'plot_text'}, default='plot_text'
        Where to render highlight information.
    autopct : None or str or callable, default='%.2f%%'
        Percentage formatter for values in [0, 1].
    line_kws, ref_line_kws, legend_kws, grid_kws, axes_kws, annotation_kws, footer_kws : Mapping[str, Any] or None
        Per-component styling kwargs.
    save_fig : bool, default=True
        Used by :func:`~scikitplot.utils._matplotlib.save_plot_decorator`.
    save_fig_filename : str, default=''
        Used by :func:`~scikitplot.utils._matplotlib.save_plot_decorator`.
    **kwargs : Any
        Legacy alias for ``line_kws``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.

    Raises
    ------
    _PlotInputError
        If plot_input lacks required columns.

    See Also
    --------
    plot_response, plot_cumresponse, plot_cumlift

    Notes
    -----
    Cumulative gains is highlighted as 1..N.

    - value = cumgain
    - counts use cumpos / total_pos

    Examples
    --------
    >>> # ax = plot_cumgains(plot_input)
    """
    where = "plot_cumgains"
    _require_columns(
        plot_input,
        required=[
            "scope",
            "ntile",
            "cumgain",
            "gain_opt",
            "cumpos",
            "model_label",
            "dataset_label",
            "target_class",
        ],
        where=where,
    )

    ntiles = _get_ntiles_from_plot_input(plot_input, where=where)
    hl = _normalize_highlight_ntiles(highlight_ntile, ntiles=ntiles, where=where)

    pk = _parse_plot_kws(
        line_kws=line_kws,
        ref_line_kws=ref_line_kws,
        legend_kws=legend_kws,
        grid_kws=grid_kws,
        axes_kws=axes_kws,
        annotation_kws=annotation_kws,
        footer_kws=footer_kws,
        legacy_line_kwargs=kwargs,
        where=where,
    )

    def _val_fmt(v: float) -> str:
        return _autopct(v, autopct)

    def _counts(gdf: pd.DataFrame, nt: int) -> tuple[int | None, int | None]:
        # numerator is cumpos at nt; denominator is total positives in the group.
        num = _as_int_like(
            _value_at_ntile(gdf, ntile=nt, col="cumpos", where=where),
            name="cumpos",
            where=where,
        )
        den = _as_int_like(
            _value_at_ntile(gdf, ntile=ntiles, col="cumpos", where=where),
            name="total_pos",
            where=where,
        )
        return num, den

    ax = _plot_metric_with_reference(
        plot_input,
        metric_col="cumgain",
        ref_col="gain_opt",
        metric_name="CumGains",
        title="Cumulative Gains",
        ylabel="cumulative gain",
        percent_y=True,
        xlim_start=0,
        legend_loc="lower right",
        highlight_ntiles=hl,
        highlight_how=highlight_how,
        highlight_kind="cum",
        value_formatter=_val_fmt,
        counts_at_ntile=_counts,
        x0_for_highlight=0,
        y0_for_highlight=0.0,
        autopct=autopct,
        ax=None,
        figsize=(10, 5),
        plot_kws=pk,
        collect_highlight_lines=None,
        render_footer=True,
    )

    # Minimal gains (random selection) baseline.
    ax.plot(
        list(range(ntiles + 1)),
        np.linspace(0.0, 1.0, num=ntiles + 1).tolist(),
        linestyle="dashed",
        color="grey",
        label="minimal gains",
    )
    ax.set_xlim(0, ntiles)
    ax.set_ylim(0.0, 1.0)
    # ax.legend(loc="lower right", **dict(pk.legend_kws))
    ax.legend(
        **{
            "loc": "lower right",
            "shadow": False,
            "frameon": False,
            **dict(pk.legend_kws),
        }
    )
    return ax


@save_plot_decorator
@_docstring.interpd
def plot_all(
    plot_input: pd.DataFrame,
    *,
    highlight_ntile: int | Sequence[int] | None = None,
    highlight_how: Literal["plot", "text", "plot_text"] = "plot_text",
    autopct: str | Callable[[float], str] | None = "%.2f%%",
    figsize: tuple[int, int] = (15, 10),
    line_kws: Mapping[str, Any] | None = None,
    ref_line_kws: Mapping[str, Any] | None = None,
    legend_kws: Mapping[str, Any] | None = None,
    grid_kws: Mapping[str, Any] | None = None,
    axes_kws: Mapping[str, Any] | None = None,
    annotation_kws: Mapping[str, Any] | None = None,
    footer_kws: Mapping[str, Any] | None = None,
    save_fig: bool = True,
    save_fig_filename: str = "",
    **kwargs: Any,
) -> plt.Axes:
    """Plot response, cumulative response, cumulative lift, and cumulative gains as a 2x2 panel.

    Parameters
    ----------
    plot_input : pandas.DataFrame
        Output of :meth:`ModelPlotPy.plotting_scope`.
    highlight_ntile : int, Sequence[int], or None, default=None
        Ntile(s) to highlight across all subplots.
    highlight_how : {'plot', 'text', 'plot_text'}, default='plot_text'
        Where to render highlight information.

        - 'text' prints standardized lines to stdout
        - 'plot' renders them in a footer area
        - 'plot_text' does both
    autopct : None or str or callable, default='%.2f%%'
        Percentage formatter.
    figsize : tuple[int, int], default=(15, 10)
        Figure size.
    line_kws, ref_line_kws, legend_kws, grid_kws, axes_kws, annotation_kws, footer_kws : Mapping[str, Any] or None
        Per-component styling kwargs.
    save_fig : bool, default=True
        Used by :func:`~scikitplot.utils._matplotlib.save_plot_decorator`.
    save_fig_filename : str, default=''
        Used by :func:`~scikitplot.utils._matplotlib.save_plot_decorator`.
    **kwargs : Any
        Legacy alias for ``line_kws``.

    Returns
    -------
    matplotlib.axes.Axes
        The top-left subplot axis (cumulative gains), matching legacy behavior.

    Raises
    ------
    _PlotInputError
        If required columns are missing.

    See Also
    --------
    plot_response, plot_cumresponse, plot_cumlift, plot_cumgains

    Notes
    -----
    Dev note: plot_all must not call other decorated plot_* functions.
    Nesting save_plot_decorator calls can clear/close figures unexpectedly.

    Examples
    --------
    >>> # ax = plot_all(plot_input)
    """
    where = "plot_all"
    _require_columns(
        plot_input,
        required=[
            "model_label",
            "dataset_label",
            "target_class",
            "scope",
            "ntile",
            "pct",
            "cumpct",
            "cumlift",
            "cumgain",
            "pct_ref",
            "gain_opt",
            "pos",
            "tot",
            "cumpos",
            "cumtot",
        ],
        where=where,
    )

    ntiles = _get_ntiles_from_plot_input(plot_input, where=where)
    hl = _normalize_highlight_ntiles(highlight_ntile, ntiles=ntiles, where=where)
    _validate_highlight_how(highlight_how, where=where)

    pk = _parse_plot_kws(
        line_kws=line_kws,
        ref_line_kws=ref_line_kws,
        legend_kws=legend_kws,
        grid_kws=grid_kws,
        axes_kws=axes_kws,
        annotation_kws=annotation_kws,
        footer_kws=footer_kws,
        legacy_line_kwargs=kwargs,
        where=where,
    )

    scope = str(_unique_one(plot_input["scope"], name="scope", where=where))

    models = [str(x) for x in pd.unique(plot_input["model_label"])]
    datasets = [str(x) for x in pd.unique(plot_input["dataset_label"])]
    targets = [str(x) for x in pd.unique(plot_input["target_class"].astype(str))]

    if scope == "no_comparison":
        global_title = (
            f"model: {models[0]} & dataset: {datasets[0]} & target class: {targets[0]}"
        )
    elif scope == "compare_models":
        global_title = f"scope: comparing models & dataset: {datasets[0]} & target class: {targets[0]}"
    elif scope == "compare_datasets":
        global_title = f"scope: comparing datasets & model: {models[0]} & target class: {targets[0]}"
    else:
        global_title = f"scope: comparing target classes & dataset: {datasets[0]} & model: {models[0]}"

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(global_title, fontsize=16)

    # Collect all highlight lines once and render a single footer.
    collected: list[str] = []

    # --- Cumulative gains (top-left) ---
    ax1.plot(
        list(range(0, ntiles + 1, 1)),
        np.linspace(0.0, 1.0, num=ntiles + 1).tolist(),
        linestyle="dashed",
        color="grey",
        label="minimal gains",
    )

    def _gains_fmt(v: float) -> str:
        return _autopct(v, autopct)

    def _gains_counts(gdf: pd.DataFrame, nt: int) -> tuple[int | None, int | None]:
        num = _as_int_like(
            _value_at_ntile(gdf, ntile=nt, col="cumpos", where=where),
            name="cumpos",
            where=where,
        )
        den = _as_int_like(
            _value_at_ntile(gdf, ntile=ntiles, col="cumpos", where=where),
            name="total_pos",
            where=where,
        )
        return num, den

    _plot_metric_with_reference(
        plot_input,
        metric_col="cumgain",
        ref_col="gain_opt",
        metric_name="CumGains",
        title="Cumulative Gains",
        ylabel="cumulative gain",
        percent_y=True,
        xlim_start=0,
        legend_loc="lower right",
        highlight_ntiles=hl,
        highlight_how=highlight_how,
        highlight_kind="cum",
        value_formatter=_gains_fmt,
        counts_at_ntile=_gains_counts,
        x0_for_highlight=0,
        y0_for_highlight=0.0,
        autopct=autopct,
        ax=ax1,
        figsize=figsize,
        plot_kws=pk,
        collect_highlight_lines=collected,
        render_footer=False,
    )
    ax1.set_ylim(0.0, 1.0)

    # --- Cumulative lift (top-right) ---
    def _lift_fmt(v: float) -> str:
        return f"{float(v):.2f}x"

    def _lift_counts(gdf: pd.DataFrame, nt: int) -> tuple[int | None, int | None]:
        num = _as_int_like(
            _value_at_ntile(gdf, ntile=nt, col="cumpos", where=where),
            name="cumpos",
            where=where,
        )
        den = _as_int_like(
            _value_at_ntile(gdf, ntile=nt, col="cumtot", where=where),
            name="cumtot",
            where=where,
        )
        return num, den

    _plot_metric_with_reference(
        plot_input,
        metric_col="cumlift",
        ref_col=None,
        metric_name="CumLift",
        title="Cumulative Lift",
        ylabel="cumulative lift (x)",
        percent_y=False,
        xlim_start=1,
        legend_loc="upper right",
        highlight_ntiles=hl,
        highlight_how=highlight_how,
        highlight_kind="cum",
        value_formatter=_lift_fmt,
        counts_at_ntile=_lift_counts,
        x0_for_highlight=1,
        y0_for_highlight=0.0,
        autopct=None,
        ax=ax2,
        figsize=figsize,
        plot_kws=pk,
        collect_highlight_lines=collected,
        render_footer=False,
    )
    ax2.axhline(1.0, linestyle="dashed", color="grey", lw=1.0, label="baseline (1.0x)")
    ax2.legend(loc="upper right", **dict(pk.legend_kws))

    # --- Response (bottom-left) ---
    def _resp_fmt(v: float) -> str:
        return _autopct(v, autopct)

    def _resp_counts(gdf: pd.DataFrame, nt: int) -> tuple[int | None, int | None]:
        pos = _as_int_like(
            _value_at_ntile(gdf, ntile=nt, col="pos", where=where),
            name="pos",
            where=where,
        )
        tot = _as_int_like(
            _value_at_ntile(gdf, ntile=nt, col="tot", where=where),
            name="tot",
            where=where,
        )
        return pos, tot

    _plot_metric_with_reference(
        plot_input,
        metric_col="pct",
        ref_col="pct_ref",
        metric_name="Response",
        title="Response",
        ylabel="response",
        percent_y=True,
        xlim_start=1,
        legend_loc="upper right",
        highlight_ntiles=hl,
        highlight_how=highlight_how,
        highlight_kind="at",
        value_formatter=_resp_fmt,
        counts_at_ntile=_resp_counts,
        x0_for_highlight=1,
        y0_for_highlight=0.0,
        autopct=autopct,
        ax=ax3,
        figsize=figsize,
        plot_kws=pk,
        collect_highlight_lines=collected,
        render_footer=False,
    )

    # --- Cumulative response (bottom-right) ---
    def _cumresp_fmt(v: float) -> str:
        return _autopct(v, autopct)

    def _cumresp_counts(gdf: pd.DataFrame, nt: int) -> tuple[int | None, int | None]:
        num = _as_int_like(
            _value_at_ntile(gdf, ntile=nt, col="cumpos", where=where),
            name="cumpos",
            where=where,
        )
        den = _as_int_like(
            _value_at_ntile(gdf, ntile=nt, col="cumtot", where=where),
            name="cumtot",
            where=where,
        )
        return num, den

    _plot_metric_with_reference(
        plot_input,
        metric_col="cumpct",
        ref_col="pct_ref",
        metric_name="CumResponse",
        title="Cumulative Response",
        ylabel="cumulative response",
        percent_y=True,
        xlim_start=1,
        legend_loc="upper right",
        highlight_ntiles=hl,
        highlight_how=highlight_how,
        highlight_kind="cum",
        value_formatter=_cumresp_fmt,
        counts_at_ntile=_cumresp_counts,
        x0_for_highlight=1,
        y0_for_highlight=0.0,
        autopct=autopct,
        ax=ax4,
        figsize=figsize,
        plot_kws=pk,
        collect_highlight_lines=collected,
        render_footer=False,
    )

    # Footer once.
    if hl:
        if highlight_how in ("text", "plot_text"):
            print("\n".join(collected))  # noqa: T201
        if highlight_how in ("plot", "plot_text"):
            _render_footer_text(fig, lines=collected, footer_kws=pk.footer_kws)

    # Tight layout (keep suptitle visible). If footer reserved space, it was
    # already applied via subplots_adjust in _render_footer_text.
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])

    return ax1


# ---------------------------
# Financial plots
# ---------------------------


def _validate_number(x: Any, *, name: str, where: str) -> float:
    """Validate a numeric scalar.

    Parameters
    ----------
    x : Any
        Value to validate.
    name : str
        Name for error messages.
    where : str
        Caller.

    Returns
    -------
    float
        Validated float.

    Raises
    ------
    TypeError
        If x is not a real number.
    ValueError
        If x is NaN/inf.

    See Also
    --------
    numpy.isfinite

    Notes
    -----
    Strict scalar validation for financial parameters.

    Examples
    --------
    >>> _validate_number(1, name="x", where="t")
    1.0
    """
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        raise TypeError(f"{where}: {name} must be a real number.")
    v = float(x)
    if not np.isfinite(v):
        raise ValueError(f"{where}: {name} must be finite.")
    return v


@save_plot_decorator
@_docstring.interpd
def plot_costsrevs(  # noqa: PLR0912
    plot_input: pd.DataFrame,
    *,
    fixed_costs: float,
    currency: str = "€",
    variable_costs_per_unit: float,
    profit_per_unit: float,
    highlight_ntile: int | Sequence[int] | None = None,
    highlight_how: Literal["plot", "text", "plot_text"] = "plot_text",
    line_kws: Mapping[str, Any] | None = None,
    ref_line_kws: Mapping[str, Any] | None = None,
    legend_kws: Mapping[str, Any] | None = None,
    grid_kws: Mapping[str, Any] | None = None,
    axes_kws: Mapping[str, Any] | None = None,
    annotation_kws: Mapping[str, Any] | None = None,
    footer_kws: Mapping[str, Any] | None = None,
    save_fig: bool = True,
    save_fig_filename: str = "",
    **kwargs: Any,
) -> plt.Axes:
    """Plot costs and revenues curves.

    Parameters
    ----------
    plot_input : pandas.DataFrame
        Output of :meth:`ModelPlotPy.plotting_scope`.
    fixed_costs : float
        Fixed costs independent of selection size.
    currency : str
        such as the dollar sign ($), euro sign (€), or pound sign (£).
        These symbols indicate monetary values in financial contexts.
    variable_costs_per_unit : float
        Variable cost per selected unit.
    profit_per_unit : float
        Revenue per positive unit.
    highlight_ntile : int or Sequence[int] or None, default=None
        Ntile index/indices to highlight (each in 1..ntiles).
    highlight_how : {'plot', 'text', 'plot_text'}, default='plot_text'
        Where to render highlight information.
    line_kws, ref_line_kws, legend_kws, grid_kws, axes_kws, annotation_kws, footer_kws : Mapping[str, Any] or None
        Per-component styling kwargs.
    save_fig : bool, default=True
        Used by :func:`~scikitplot.utils._matplotlib.save_plot_decorator`.
    save_fig_filename : str, default=''
        Used by :func:`~scikitplot.utils._matplotlib.save_plot_decorator`.
    **kwargs : Any
        Legacy alias for ``line_kws``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.

    Raises
    ------
    TypeError
        If any financial parameter is not numeric.
    _PlotInputError
        If plot_input lacks required columns.

    See Also
    --------
    plot_profit, plot_roi

    Notes
    -----
    - Revenue is computed as ``profit_per_unit * cumpos``.
    - Total costs are computed as ``fixed_costs + variable_costs_per_unit * cumtot``.

    Highlight lines are standardized and use cumulative 1..N semantics.

    Examples
    --------
    >>> # ax = plot_costsrevs(plot_input, fixed_costs=100, variable_costs_per_unit=1, profit_per_unit=10)
    """
    where = "plot_costsrevs"

    fixed_costs = _validate_number(fixed_costs, name="fixed_costs", where=where)
    variable_costs_per_unit = _validate_number(
        variable_costs_per_unit, name="variable_costs_per_unit", where=where
    )
    profit_per_unit = _validate_number(
        profit_per_unit, name="profit_per_unit", where=where
    )

    _require_columns(
        plot_input,
        required=[
            "scope",
            "ntile",
            "cumtot",
            "cumpos",
            "model_label",
            "dataset_label",
            "target_class",
        ],
        where=where,
    )

    ntiles = _get_ntiles_from_plot_input(plot_input, where=where)
    nlabel = _ntile_label(ntiles)
    hl = _normalize_highlight_ntiles(highlight_ntile, ntiles=ntiles, where=where)
    _validate_highlight_how(highlight_how, where=where)

    pk = _parse_plot_kws(
        line_kws=line_kws,
        ref_line_kws=ref_line_kws,
        legend_kws=legend_kws,
        grid_kws=grid_kws,
        axes_kws=axes_kws,
        annotation_kws=annotation_kws,
        footer_kws=footer_kws,
        legacy_line_kwargs=kwargs,
        where=where,
    )

    df = plot_input.copy()

    df["variable_costs"] = variable_costs_per_unit * df["cumtot"]
    df["investments"] = fixed_costs + df["variable_costs"]
    df["revenues"] = profit_per_unit * df["cumpos"]

    scope = str(_unique_one(df["scope"], name="scope", where=where))

    models = [str(v) for v in pd.unique(df["model_label"])]
    datasets = [str(v) for v in pd.unique(df["dataset_label"])]
    targets = [str(v) for v in pd.unique(df["target_class"].astype(str))]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Costs & Revenues", fontsize=16)

    # Axis styling.
    _setup_axis(
        ax,
        title="",  # set below
        xlabel=nlabel,
        ylabel="€",
        ntiles=ntiles,
        xlim=(1, ntiles),
        percent_y=False,
        grid=bool(pk.grid_kws.get("visible", True)),
    )

    # Title details.
    if scope == "no_comparison":
        ax.set_title(
            f"model: {models[0]} & dataset: {datasets[0]} & target class: {targets[0]}",
            fontweight="bold",
        )
    elif scope == "compare_datasets":
        ax.set_title(
            f"scope: comparing datasets & model: {models[0]} & target class: {targets[0]}",
            fontweight="bold",
        )
    elif scope == "compare_models":
        ax.set_title(
            f"scope: comparing models & dataset: {datasets[0]} & target class: {targets[0]}",
            fontweight="bold",
        )
    else:
        ax.set_title(
            f"scope: comparing target classes & dataset: {datasets[0]} & model: {models[0]}",
            fontweight="bold",
        )

    # Plot: solid revenues, dashed investments.
    text_lines: list[str] = []
    label_to_color: dict[str, str] = {}

    groups = list(_iter_groups(df, scope=scope))

    # Special case: compare_models must have identical cumtot per ntile for a single investments curve.
    if scope == "compare_models":
        chk = df.groupby("ntile", sort=False)["cumtot"].nunique(dropna=False)
        if (chk > 1).any():
            bad = chk[chk > 1].index.to_list()
            raise _PlotInputError(
                f"{where}: compare_models expects cumtot to be identical across models for each ntile. "
                f"Non-identical cumtot found at ntile(s): {bad}."
            )

        base = df.loc[df["model_label"] == models[0]].sort_values("ntile")
        inv_curve = fixed_costs + variable_costs_per_unit * base["cumtot"]
        ax.plot(
            base["ntile"],
            inv_curve,
            linestyle="dashed",
            label="total costs",
            color="grey",
        )

    for g_idx, (lab, gdf) in enumerate(groups):
        gdf = gdf.sort_values("ntile")  # noqa: PLW2901

        # Label selection.
        if scope == "no_comparison":
            line_label = targets[0]
            ref_label = "total costs"
        else:
            line_label = f"revenues ({lab})"
            ref_label = f"total costs ({lab})"

        rev_line = ax.plot(
            gdf["ntile"], gdf["revenues"], label=line_label, **dict(pk.line_kws)
        )
        color = str(rev_line[0].get_color())
        label_to_color[str(lab)] = color

        if scope != "compare_models":
            ax.plot(
                gdf["ntile"],
                gdf["investments"],
                label=ref_label,
                linestyle="dashed",
                color=color,
            )

        if hl:
            for j, nt in enumerate(hl):
                rev = _value_at_ntile(gdf, ntile=int(nt), col="revenues", where=where)

                model, dataset, target = _scope_triplet(
                    scope=scope,
                    group_label=str(lab),
                    models=models,
                    datasets=datasets,
                    targets=targets,
                )

                # Counts: selection size and positives.
                pos = _as_int_like(
                    _value_at_ntile(gdf, ntile=int(nt), col="cumpos", where=where),
                    name="cumpos",
                    where=where,
                )
                tot = _as_int_like(
                    _value_at_ntile(gdf, ntile=int(nt), col="cumtot", where=where),
                    name="cumtot",
                    where=where,
                )

                line_txt = _format_highlight_line(
                    metric="Revenues",
                    range_kind="cum",
                    ntile_label=nlabel,
                    ntile=int(nt),
                    model=model,
                    dataset=dataset,
                    target=target,
                    value=_currency_fmt(float(rev), currency),
                    numerator=pos,
                    denominator=tot,
                )
                text_lines.append(line_txt)

                ann_idx = g_idx + j * len(groups)
                xytext = _annotation_xytext(
                    ann_idx,
                    layout_cols=_as_int_like(
                        pk.annotation_kws.get("layout_cols", 5),
                        name="layout_cols",
                        where=where,
                    ),
                    x_offset=_as_int_like(
                        pk.annotation_kws.get("x_offset", -40),
                        name="x_offset",
                        where=where,
                    ),
                    x_step=_as_int_like(
                        pk.annotation_kws.get("x_step", 20), name="x_step", where=where
                    ),
                    base=_as_int_like(
                        pk.annotation_kws.get("base", 30), name="base", where=where
                    ),
                    gap=_as_int_like(
                        pk.annotation_kws.get("gap", 12), name="gap", where=where
                    ),
                )

                _annotate_highlight(
                    ax,
                    x=int(nt),
                    y=float(rev),
                    x0=1,
                    y0=0.0,
                    color=color,
                    text=_currency_fmt(float(rev), currency),
                    xytext=xytext,
                    annotation_kws=pk.annotation_kws,
                )

    ax.legend(
        loc=(
            "lower right"
            if scope in ("no_comparison", "compare_models", "compare_targetclasses")
            else "upper right"
        ),
        **dict(pk.legend_kws),
    )

    if hl:
        if highlight_how in ("text", "plot_text"):
            print("\n".join(text_lines))  # noqa: T201
        if highlight_how in ("plot", "plot_text"):
            _render_footer_text(fig, lines=text_lines, footer_kws=pk.footer_kws)

    return ax


@save_plot_decorator
@_docstring.interpd
def plot_profit(
    plot_input: pd.DataFrame,
    *,
    fixed_costs: float,
    currency: str = "€",
    variable_costs_per_unit: float,
    profit_per_unit: float,
    highlight_ntile: int | Sequence[int] | None = None,
    highlight_how: Literal["plot", "text", "plot_text"] = "plot_text",
    line_kws: Mapping[str, Any] | None = None,
    ref_line_kws: Mapping[str, Any] | None = None,
    legend_kws: Mapping[str, Any] | None = None,
    grid_kws: Mapping[str, Any] | None = None,
    axes_kws: Mapping[str, Any] | None = None,
    annotation_kws: Mapping[str, Any] | None = None,
    footer_kws: Mapping[str, Any] | None = None,
    save_fig: bool = True,
    save_fig_filename: str = "",
    **kwargs: Any,
) -> plt.Axes:
    """Plot profit curve.

    Parameters
    ----------
    plot_input : pandas.DataFrame
        Output of :meth:`ModelPlotPy.plotting_scope`.
    fixed_costs : float
        Fixed costs independent of selection size.
    currency : str
        such as the dollar sign ($), euro sign (€), or pound sign (£).
        These symbols indicate monetary values in financial contexts.
    variable_costs_per_unit : float
        Variable cost per selected unit.
    profit_per_unit : float
        Revenue per positive unit.
    highlight_ntile : int or Sequence[int] or None, default=None
        Ntile index/indices to highlight (each in 1..ntiles).
    highlight_how : {'plot', 'text', 'plot_text'}, default='plot_text'
        Where to render highlight information.
    line_kws, ref_line_kws, legend_kws, grid_kws, axes_kws, annotation_kws, footer_kws : Mapping[str, Any] or None
        Per-component styling kwargs.
    save_fig : bool, default=True
        Used by :func:`~scikitplot.utils._matplotlib.save_plot_decorator`.
    save_fig_filename : str, default=''
        Used by :func:`~scikitplot.utils._matplotlib.save_plot_decorator`.
    **kwargs : Any
        Legacy alias for ``line_kws``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.

    Raises
    ------
    _PlotInputError
        If plot_input lacks required columns.

    See Also
    --------
    plot_costsrevs, plot_roi

    Notes
    -----
    Profit = revenues - investments.

    Highlight lines use cumulative 1..N semantics and include cumpos / cumtot.

    Examples
    --------
    >>> # ax = plot_profit(plot_input, fixed_costs=100, variable_costs_per_unit=1, profit_per_unit=10)
    """
    where = "plot_profit"

    fixed_costs = _validate_number(fixed_costs, name="fixed_costs", where=where)
    variable_costs_per_unit = _validate_number(
        variable_costs_per_unit, name="variable_costs_per_unit", where=where
    )
    profit_per_unit = _validate_number(
        profit_per_unit, name="profit_per_unit", where=where
    )

    _require_columns(
        plot_input,
        required=[
            "scope",
            "ntile",
            "cumtot",
            "cumpos",
            "model_label",
            "dataset_label",
            "target_class",
        ],
        where=where,
    )

    ntiles = _get_ntiles_from_plot_input(plot_input, where=where)
    hl = _normalize_highlight_ntiles(highlight_ntile, ntiles=ntiles, where=where)

    pk = _parse_plot_kws(
        line_kws=line_kws,
        ref_line_kws=ref_line_kws,
        legend_kws=legend_kws,
        grid_kws=grid_kws,
        axes_kws=axes_kws,
        annotation_kws=annotation_kws,
        footer_kws=footer_kws,
        legacy_line_kwargs=kwargs,
        where=where,
    )

    df = plot_input.copy()
    df["variable_costs"] = variable_costs_per_unit * df["cumtot"]
    df["investments"] = fixed_costs + df["variable_costs"]
    df["revenues"] = profit_per_unit * df["cumpos"]
    df["profit"] = df["revenues"] - df["investments"]

    def _val_fmt(v: float) -> str:
        return _currency_fmt(v, currency)

    def _counts(gdf: pd.DataFrame, nt: int) -> tuple[int | None, int | None]:
        num = _as_int_like(
            _value_at_ntile(gdf, ntile=nt, col="cumpos", where=where),
            name="cumpos",
            where=where,
        )
        den = _as_int_like(
            _value_at_ntile(gdf, ntile=nt, col="cumtot", where=where),
            name="cumtot",
            where=where,
        )
        return num, den

    return _plot_metric_with_reference(
        df,
        metric_col="profit",
        ref_col=None,
        metric_name="Profit",
        title="Profit",
        ylabel="€",
        percent_y=False,
        xlim_start=1,
        legend_loc="upper right",
        highlight_ntiles=hl,
        highlight_how=highlight_how,
        highlight_kind="cum",
        value_formatter=_val_fmt,
        counts_at_ntile=_counts,
        x0_for_highlight=1,
        y0_for_highlight=0.0,
        autopct=None,
        ax=None,
        figsize=(10, 5),
        plot_kws=pk,
        collect_highlight_lines=None,
        render_footer=True,
    )


@save_plot_decorator
@_docstring.interpd
def plot_roi(
    plot_input: pd.DataFrame,
    *,
    fixed_costs: float,
    variable_costs_per_unit: float,
    profit_per_unit: float,
    highlight_ntile: int | Sequence[int] | None = None,
    highlight_how: Literal["plot", "text", "plot_text"] = "plot_text",
    autopct: str | Callable[[float], str] | None = "%.2f%%",
    line_kws: Mapping[str, Any] | None = None,
    ref_line_kws: Mapping[str, Any] | None = None,
    legend_kws: Mapping[str, Any] | None = None,
    grid_kws: Mapping[str, Any] | None = None,
    axes_kws: Mapping[str, Any] | None = None,
    annotation_kws: Mapping[str, Any] | None = None,
    footer_kws: Mapping[str, Any] | None = None,
    save_fig: bool = True,
    save_fig_filename: str = "",
    **kwargs: Any,
) -> plt.Axes:
    """Plot ROI (return on investment) curve.

    Parameters
    ----------
    plot_input : pandas.DataFrame
        Output of :meth:`ModelPlotPy.plotting_scope`.
    fixed_costs : float
        Fixed costs independent of selection size.
    variable_costs_per_unit : float
        Variable cost per selected unit.
    profit_per_unit : float
        Revenue per positive unit.
    highlight_ntile : int or Sequence[int] or None, default=None
        Ntile index/indices to highlight (each in 1..ntiles).
    highlight_how : {'plot', 'text', 'plot_text'}, default='plot_text'
        Where to render highlight information.
    autopct : None or str or callable, default='%.2f%%'
        Percentage formatter for ROI (expressed as a fraction).
    line_kws, ref_line_kws, legend_kws, grid_kws, axes_kws, annotation_kws, footer_kws : Mapping[str, Any] or None
        Per-component styling kwargs.
    save_fig : bool, default=True
        Used by :func:`~scikitplot.utils._matplotlib.save_plot_decorator`.
    save_fig_filename : str, default=''
        Used by :func:`~scikitplot.utils._matplotlib.save_plot_decorator`.
    **kwargs : Any
        Legacy alias for ``line_kws``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.

    Raises
    ------
    ValueError
        If investments are zero for any ntile.
    _PlotInputError
        If plot_input lacks required columns.

    See Also
    --------
    plot_profit, plot_costsrevs

    Notes
    -----
    ROI = profit / investments.

    - profit = revenues - investments
    - revenues = profit_per_unit * cumpos
    - investments = fixed_costs + variable_costs_per_unit * cumtot

    Highlight lines use cumulative 1..N semantics and include cumpos / cumtot.

    Examples
    --------
    >>> # ax = plot_roi(plot_input, fixed_costs=100, variable_costs_per_unit=1, profit_per_unit=10)
    """
    where = "plot_roi"

    fixed_costs = _validate_number(fixed_costs, name="fixed_costs", where=where)
    variable_costs_per_unit = _validate_number(
        variable_costs_per_unit, name="variable_costs_per_unit", where=where
    )
    profit_per_unit = _validate_number(
        profit_per_unit, name="profit_per_unit", where=where
    )

    _require_columns(
        plot_input,
        required=[
            "scope",
            "ntile",
            "cumtot",
            "cumpos",
            "model_label",
            "dataset_label",
            "target_class",
        ],
        where=where,
    )

    ntiles = _get_ntiles_from_plot_input(plot_input, where=where)
    hl = _normalize_highlight_ntiles(highlight_ntile, ntiles=ntiles, where=where)

    pk = _parse_plot_kws(
        line_kws=line_kws,
        ref_line_kws=ref_line_kws,
        legend_kws=legend_kws,
        grid_kws=grid_kws,
        axes_kws=axes_kws,
        annotation_kws=annotation_kws,
        footer_kws=footer_kws,
        legacy_line_kwargs=kwargs,
        where=where,
    )

    df = plot_input.copy()
    df["variable_costs"] = variable_costs_per_unit * df["cumtot"]
    df["investments"] = fixed_costs + df["variable_costs"]
    df["revenues"] = profit_per_unit * df["cumpos"]
    df["profit"] = df["revenues"] - df["investments"]

    inv = df["investments"].to_numpy(dtype=float)
    if np.any(inv == 0.0):
        raise ValueError("ROI undefined when investments are zero for any ntile.")

    df["roi"] = df["profit"] / df["investments"]

    def _val_fmt(v: float) -> str:
        return _autopct(v, autopct)

    def _counts(gdf: pd.DataFrame, nt: int) -> tuple[int | None, int | None]:
        num = _as_int_like(
            _value_at_ntile(gdf, ntile=nt, col="cumpos", where=where),
            name="cumpos",
            where=where,
        )
        den = _as_int_like(
            _value_at_ntile(gdf, ntile=nt, col="cumtot", where=where),
            name="cumtot",
            where=where,
        )
        return num, den

    ax = _plot_metric_with_reference(
        df,
        metric_col="roi",
        ref_col=None,
        metric_name="ROI",
        title="Return on Investment (ROI)",
        ylabel="ROI",
        percent_y=True,
        xlim_start=1,
        legend_loc="lower right",
        highlight_ntiles=hl,
        highlight_how=highlight_how,
        highlight_kind="cum",
        value_formatter=_val_fmt,
        counts_at_ntile=_counts,
        x0_for_highlight=1,
        y0_for_highlight=0.0,
        autopct=autopct,
        ax=None,
        figsize=(10, 5),
        plot_kws=pk,
        collect_highlight_lines=None,
        render_footer=True,
    )

    ax.axhline(0.0, linestyle="dashed", color="grey", lw=1.0, label="break even")
    ax.legend(loc="lower right", **dict(pk.legend_kws))
    return ax


# ---------------------------
# Advanced decile helpers (data-science oriented)
# ---------------------------


def summarize_selection(
    plot_input: pd.DataFrame,
    *,
    ntile: int = 10,
) -> pd.DataFrame:
    """Return a standardized summary row per group at a selected ntile.

    Parameters
    ----------
    plot_input : pandas.DataFrame
        Output of :meth:`ModelPlotPy.plotting_scope`.
    ntile : int
        Ntile to summarize (must exist in plot_input).

    Returns
    -------
    pandas.DataFrame
        One row per group with standardized fields.

        Columns include:

        - model_label, dataset_label, target_class
        - ntile
        - tot, pos, pct
        - cumtot, cumpos, cumpct
        - cumlift, cumgain

    Raises
    ------
    _PlotInputError
        If required columns are missing.
    ValueError
        If ntile is outside valid range.

    See Also
    --------
    ModelPlotPy.plotting_scope

    Notes
    -----
    This helper is intended for DS/MLOps workflows where you want a stable,
    audit-friendly view of a chosen operating point (e.g., decile 3).

    Examples
    --------
    >>> # df = summarize_selection(plot_input, ntile=3)
    """
    where = "summarize_selection"
    _require_columns(
        plot_input,
        required=[
            "scope",
            "ntile",
            "model_label",
            "dataset_label",
            "target_class",
            "tot",
            "pos",
            "pct",
            "cumtot",
            "cumpos",
            "cumpct",
            "cumlift",
            "cumgain",
        ],
        where=where,
    )

    ntiles = _get_ntiles_from_plot_input(plot_input, where=where)
    nt = _as_int_like(ntile, name="ntile", where=where)
    if nt < 1 or nt > ntiles:
        raise ValueError(f"{where}: ntile must be in [1, {ntiles}], got {nt}.")

    scope = str(_unique_one(plot_input["scope"], name="scope", where=where))

    rows = []
    for lab, gdf in _iter_groups(plot_input, scope=scope):
        gdf = gdf.sort_values("ntile")  # noqa: PLW2901
        row = gdf.loc[gdf["ntile"] == nt]
        if row.shape[0] != 1:
            raise _PlotInputError(
                f"{where}: expected 1 row at ntile={nt} for group '{lab}', found {row.shape[0]}."
            )
        rows.append(row)

    out = pd.concat(rows, axis=0).reset_index(drop=True)
    return out[
        [
            "model_label",
            "dataset_label",
            "target_class",
            "ntile",
            "tot",
            "pos",
            "pct",
            "cumtot",
            "cumpos",
            "cumpct",
            "cumlift",
            "cumgain",
        ]
    ]
