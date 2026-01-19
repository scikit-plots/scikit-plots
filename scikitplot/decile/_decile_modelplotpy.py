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
The :mod:`~scikitplot.decile` module.

Includes plots for machine learning evaluation decile / ntile analysis
(e.g., Response, Lift, Gain and related financial charts).

References
----------
* https://github.com/modelplot/modelplotpy/blob/master/modelplotpy/functions.py
* https://modelplot.github.io/intro_modelplotpy.html
"""

from __future__ import annotations

import logging
import os  # noqa: F401
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .._docstrings import _docstring
from ..utils.utils_plot_mpl import save_plot_decorator

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

logger = logging.getLogger(__name__)

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
            if not isinstance(m, ClassifierMixin):
                raise TypeError("All models must be sklearn ClassifierMixin instances.")
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

# Dev note: The plot_* functions below are intentionally thin wrappers around
# deterministic, unit-testable internal helpers. This prevents repetition and
# makes the behavior consistent across plots.

import numbers
from typing import Callable, Iterator, Literal, Sequence


class _PlotInputError(ValueError):
    """Raised when plot_input does not conform to the required schema."""


def _require_columns(df: pd.DataFrame, *, required: Sequence[str], where: str) -> None:
    """Validate that the required columns exist.

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
    """Map ntile count to a stable descriptive label.

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
    # Accept numpy/pandas integer dtypes and python ints; reject floats with decimals.
    arr = nt.to_numpy()
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
        return "target_class", "target class"
    raise ValueError(f"Unknown scope='{scope}'.")


def _setup_axis(
    ax: plt.Axes,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    ntiles: int,
    xlim: tuple[int, int] = (1, 1),
    percent_y: bool = False,
    grid: bool = True,
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
    xlim : tuple[int, int], default=(1, 1)
        X axis limits.
    percent_y : bool, default=False
        Whether to format y axis as percent in [0, 1].
    grid : bool, default=True
        Whether to enable grid.

    Returns
    -------
    None

    Raises
    ------
    None

    See Also
    --------
    matplotlib.ticker.PercentFormatter

    Notes
    -----
    User note: We intentionally do not apply heuristic y-limits. Matplotlib auto-
    scales the y-range; we only ensure a clean baseline for metrics that are
    naturally non-negative.

    Examples
    --------
    >>> # internal
    """
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if percent_y:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # Strict rule: show every integer ntile tick (no heuristic downsampling).
    ax.set_xticks(np.arange(0, ntiles + 1, 1))

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    ax.set_xlim(list(xlim))
    if grid:
        ax.grid(True)


def _iter_groups(
    df: pd.DataFrame,
    *,
    scope: str,
) -> Iterator[tuple[str, pd.DataFrame]]:
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

    # Stable order by first appearance.
    labels = [str(v) for v in pd.unique(df[group_col])]
    for lab in labels:
        # For target_class we stringify consistently.
        if group_col == "target_class":
            mask = df[group_col].astype(str) == lab
        else:
            mask = df[group_col] == lab
        yield lab, df.loc[mask].copy()


def _normalize_highlight_ntiles(
    highlight_ntile: Any,
    *,
    ntiles: int,
    where: str,
) -> list[int]:
    """Normalize and validate ``highlight_ntile`` into a sorted unique list.

    Parameters
    ----------
    highlight_ntile : Any
        Requested ntile(s). Supported inputs:

        - ``None``
        - int
        - Sequence[int]
    ntiles : int
        Maximum ntile.
    where : str
        Caller name.

    Returns
    -------
    list[int]
        Sorted unique ntile indices in ``[1, ntiles]``.

    Raises
    ------
    TypeError
        If ``highlight_ntile`` is neither an integer nor a sequence of integers.
    ValueError
        If any ntile is outside ``[1, ntiles]``.

    See Also
    --------
    ModelPlotPy.ntiles

    Notes
    -----
    Strict rules:

    - Values must be integer-like (no floats/decimals).
    - All values must be within ``[1, ntiles]``.
    - Duplicates are removed.

    Examples
    --------
    >>> _normalize_highlight_ntiles(None, ntiles=10, where="x")
    []
    >>> _normalize_highlight_ntiles([1, 2, 2], ntiles=10, where="x")
    [1, 2]
    """
    if highlight_ntile is None:
        return []

    # Single integer.
    if isinstance(highlight_ntile, numbers.Integral) and not isinstance(
        highlight_ntile, bool
    ):
        candidates = [int(highlight_ntile)]
    # Sequence of integers.
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

    # Deterministic ordering: sorted unique.
    return sorted(set(out))


def _annotation_xytext(
    annotation_index: int, *, x_offset: int = -30, base: int = 30, gap: int = 12
) -> tuple[int, int]:
    """Compute a deterministic annotation offset to reduce overlaps.

    Parameters
    ----------
    annotation_index : int
        0-based index of the annotation.
    x_offset : int, default=-30
        X offset in points.
    base : int, default=30
        Base absolute Y offset in points.
    gap : int, default=12
        Additional vertical gap added every two annotations.

    Returns
    -------
    tuple[int, int]
        ``(x_offset, y_offset)`` in display points.

    Raises
    ------
    ValueError
        If ``annotation_index`` is negative.

    See Also
    --------
    matplotlib.axes.Axes.annotate

    Notes
    -----
    Deterministic rule (no heuristics):

    - Even indices go below the point.
    - Odd indices go above the point.
    - Every two annotations, increase the magnitude by ``gap``.

    Examples
    --------
    >>> _annotation_xytext(0)
    (-30, -30)
    >>> _annotation_xytext(1)
    (-30, 30)
    """
    if annotation_index < 0:
        raise ValueError("annotation_index must be >= 0.")
    k = annotation_index // 2
    mag = base + gap * k
    y_offset = -mag if (annotation_index % 2) == 0 else mag
    return (x_offset, y_offset)


def _validate_highlight_how(highlight_how: str, *, where: str) -> None:
    """Validate highlight_how.

    Parameters
    ----------
    highlight_how : str
        One of {'plot', 'text', 'plot_text'}.
    where : str
        Caller name.

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
            f"{where}: invalid highlight_how='{highlight_how}'. "
            "Allowed={'plot','text','plot_text'}."
        )


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
        Caller name.

    Returns
    -------
    float
        The value.

    Raises
    ------
    _PlotInputError
        If there is not exactly one row for that ntile.

    See Also
    --------
    pandas.DataFrame.loc

    Notes
    -----
    Dev note: We require a single row at a given ntile to avoid silent
    aggregation.

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


def _annotate_highlight(  # noqa: D417
    ax: plt.Axes,
    *,
    x: int,
    y: float,
    x0: int,
    y0: float,
    color: str,
    text: str,
    xytext: tuple[int, int] = (-30, -30),
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
    ax.plot([x0, x], [y] * 2, linestyle="-.", color=color, lw=1.5)
    ax.plot([x] * 2, [y0, y], linestyle="-.", color=color, lw=1.5)
    ax.plot(x, y, marker="o", ms=6, color=color)

    # Dev note: flip vertical alignment when placing the label below the point.
    va = "bottom" if xytext[1] >= 0 else "top"

    ax.annotate(
        text,
        xy=(x, y),
        xytext=xytext,
        textcoords="offset points",
        ha="center",
        va=va,
        color="black",
        bbox={"boxstyle": "round, pad=0.4", "fc": color, "alpha": 0.8},
        arrowprops={"arrowstyle": "->", "color": "black"},
    )


def _plot_metric_with_reference(  # noqa: PLR0912
    plot_input: pd.DataFrame,
    *,
    metric_col: str,
    ref_col: str | None,
    title: str,
    ylabel: str,
    percent_y: bool,
    xlim_start: int = 1,
    highlight_ntiles: Sequence[int] | None,
    highlight_how: str,
    highlight_value_formatter: Callable[[float], str] | None,
    highlight_text_builder: Callable[[str, int, float], str] | None,
    legend_loc: str,
    x0_for_highlight: int,
    y0_for_highlight: float,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (10, 5),
    plot_kwargs: dict[str, Any] | None = None,
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
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    percent_y : bool
        Format y-axis as percent in [0, 1].
    xlim_start : int, default=1
        Left x-limit for the plot.
    highlight_ntiles : Sequence[int] or None
        Ntile indices to highlight.
    highlight_how : str
        One of {'plot','text','plot_text'}.
    highlight_value_formatter : Callable[[float], str] or None
        Formats the highlighted value for the callout.
    highlight_text_builder : Callable[[str, int, float], str] or None
        Builds the per-group highlight text.
    legend_loc : str
        Legend location.
    x0_for_highlight : int
        X origin for highlight guide lines.
    y0_for_highlight : float
        Y origin for highlight guide lines.
    ax : matplotlib.axes.Axes or None, default=None
        If provided, draw into this axes.
    figsize : tuple[int, int], default=(10, 5)
        Figure size when `ax` is None.
    plot_kwargs : dict[str, Any] or None
        Keyword args passed to the main metric line plot.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.

    Raises
    ------
    _PlotInputError
        If required columns are missing.
    ValueError
        If scope is invalid.

    See Also
    --------
    plot_response, plot_cumresponse, plot_cumlift, plot_cumgains

    Notes
    -----
    - Grouping is controlled by plot_input['scope'].
    - All groups are sorted by ntile before plotting.

    Examples
    --------
    >>> # internal
    """
    where = "plot_metric"
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
    xlabel = _ntile_label(ntiles)

    # Create figure/axes.
    # Dev note: keep a stable flag so we can tell whether we're in multi-panel
    # mode (axes provided) vs single-plot mode (axes created here).
    multi_panel = ax is not None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(title, fontsize=16)
    else:
        fig = ax.figure

    # Build per-scope title details.
    models = [str(v) for v in pd.unique(plot_input["model_label"])]
    datasets = [str(v) for v in pd.unique(plot_input["dataset_label"])]
    classes = [str(v) for v in pd.unique(plot_input["target_class"].astype(str))]

    if scope == "no_comparison":
        details_title = (
            f"model: {models[0]} & dataset: {datasets[0]} & target class: {classes[0]}"
        )
    elif scope == "compare_datasets":
        details_title = f"scope: comparing datasets & model: {models[0]} & target class: {classes[0]}"
    elif scope == "compare_models":
        details_title = f"scope: comparing models & dataset: {datasets[0]} & target class: {classes[0]}"
    else:  # compare_targetclasses
        details_title = f"scope: comparing target classes & dataset: {datasets[0]} & model: {models[0]}"

    # User note: in multi-panel mode (`ax` is provided), prepend the metric title
    # so each subplot is self-describing. In single-plot mode, we keep the legacy
    # layout where the metric name is shown as a figure suptitle.
    ax_title = f"{title}\n{details_title}" if multi_panel else details_title

    _setup_axis(
        ax,
        title=ax_title,
        xlabel=xlabel,
        ylabel=ylabel,
        ntiles=ntiles,
        xlim=(int(xlim_start), int(ntiles)),
        percent_y=percent_y,
    )
    # Plot lines.
    plot_kwargs = dict(plot_kwargs or {})
    text_lines: list[str] = []

    # Dev note: capture line colors from the actual Line2D objects rather than
    # reconstructing Matplotlib's prop_cycle. This stays correct even if callers
    # pass explicit colors via `plot_kwargs`.
    label_to_color: dict[str, str] = {}

    for lab, gdf in _iter_groups(plot_input, scope=scope):
        gdf = gdf.sort_values("ntile")  # noqa: PLW2901

        # Main metric.
        line_label = lab if scope != "no_comparison" else classes[0]
        line = ax.plot(gdf["ntile"], gdf[metric_col], label=line_label, **plot_kwargs)
        color = str(line[0].get_color())
        label_to_color[str(lab)] = color

        # Reference curve, if present.
        if ref_col is not None:
            # User note: dashed line uses the same color as its metric line.
            ax.plot(
                gdf["ntile"],
                gdf[ref_col],
                linestyle="dashed",
                color=color,
                label=(
                    f"overall response ({lab})"
                    if metric_col in ("pct", "cumpct")
                    else f"reference ({lab})"
                ),
            )

    ax.legend(loc=legend_loc, shadow=False, frameon=False)

    # Highlight (supports multiple ntile values).
    hl = list(highlight_ntiles or [])
    if hl:
        _validate_highlight_how(highlight_how, where=where)
        if highlight_value_formatter is None or highlight_text_builder is None:
            raise ValueError(
                f"{where}: highlight formatters must be provided when highlighting is enabled."
            )

        # Apply highlights in a deterministic order: groups-first (as plotted), then by highlight ntile.
        groups = list(_iter_groups(plot_input, scope=scope))
        n_groups = len(groups)

        for j, ntile in enumerate(hl):
            for i, (lab, gdf) in enumerate(groups):
                gdf = gdf.sort_values("ntile")  # noqa: PLW2901
                y = _value_at_ntile(gdf, ntile=int(ntile), col=metric_col, where=where)
                color = label_to_color.get(str(lab), "C0")

                ann_idx = i + j * n_groups
                _annotate_highlight(
                    ax,
                    x=int(ntile),
                    y=float(y),
                    x0=int(x0_for_highlight),
                    y0=float(y0_for_highlight),
                    color=color,
                    text=highlight_value_formatter(float(y)),
                    xytext=_annotation_xytext(ann_idx),
                )
                text_lines.append(
                    highlight_text_builder(str(lab), int(ntile), float(y))
                )

        text = "\n".join(text_lines)
        if highlight_how in ("text", "plot_text"):
            print(text)  # noqa: T201
        if highlight_how in ("plot", "plot_text"):
            # Dev note: Keep stable position for backward compatibility.
            fig.text(0.53, 0.37, text, ha="left")

    return ax


@save_plot_decorator
@_docstring.interpd
def plot_response(
    plot_input: pd.DataFrame,
    *,
    highlight_ntile: int | Sequence[int] | None = None,
    highlight_how: Literal["plot", "text", "plot_text"] = "plot_text",
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
    save_fig : bool, default=True
        Used by :func:`~scikitplot.utils.utils_plot_mpl.save_plot_decorator`.
    save_fig_filename : str, default=''
        Used by :func:`~scikitplot.utils.utils_plot_mpl.save_plot_decorator`.
    **kwargs : Any
        Additional keyword arguments forwarded to :meth:`matplotlib.axes.Axes.plot`
        for the main metric line.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.

    Raises
    ------
    TypeError
        If ``highlight_ntile`` is neither an int nor a sequence of ints.
    ValueError
        If any highlighted ntile is outside the valid range.
    _PlotInputError
        If ``plot_input`` lacks required columns.

    See Also
    --------
    plot_cumresponse, plot_cumlift, plot_cumgains

    Notes
    -----
    Response is the per-ntile positive rate (``pct``).

    Examples
    --------
    >>> # ax = plot_response(plot_input)
    """
    ntiles = _get_ntiles_from_plot_input(plot_input, where="plot_response")
    hl = _normalize_highlight_ntiles(
        highlight_ntile, ntiles=ntiles, where="plot_response"
    )

    def _fmt(v: float) -> str:
        return f"{round(v * 100.0)}%"

    def _text(lab: str, ntile: int, v: float) -> str:
        # User note: For no_comparison, lab is 'single'. We still use it for stable text.
        scope = str(
            _unique_one(plot_input["scope"], name="scope", where="plot_response")
        )
        models = [str(x) for x in pd.unique(plot_input["model_label"])]
        datasets = [str(x) for x in pd.unique(plot_input["dataset_label"])]
        classes = [str(x) for x in pd.unique(plot_input["target_class"].astype(str))]
        desc = _ntile_label(ntiles)

        if scope == "compare_models":
            model_part = f"model {lab}"
            ds_part = f"dataset {datasets[0]}"
            cls_part = f"{classes[0]}"
        elif scope == "compare_datasets":
            model_part = f"model {models[0]}"
            ds_part = f"dataset {lab}"
            cls_part = f"{classes[0]}"
        elif scope == "compare_targetclasses":
            model_part = f"model {models[0]}"
            ds_part = f"dataset {datasets[0]}"
            cls_part = f"{lab}"
        else:
            model_part = f"model {models[0]}"
            ds_part = f"dataset {datasets[0]}"
            cls_part = f"{classes[0]}"

        return (
            f"When we select {desc} {ntile} from {model_part} in {ds_part}, "  # noqa: S608
            f"the response rate for {cls_part} is {round(v * 100.0)}%."
        )

    return _plot_metric_with_reference(
        plot_input,
        metric_col="pct",
        ref_col="pct_ref",
        title="Response",
        ylabel="response",
        percent_y=True,
        highlight_ntiles=hl,
        highlight_how=highlight_how,
        highlight_value_formatter=_fmt,
        highlight_text_builder=_text,
        legend_loc="upper right",
        x0_for_highlight=1,
        y0_for_highlight=0.0,
        plot_kwargs=kwargs,
    )


@save_plot_decorator
@_docstring.interpd
def plot_cumresponse(
    plot_input: pd.DataFrame,
    *,
    highlight_ntile: int | Sequence[int] | None = None,
    highlight_how: Literal["plot", "text", "plot_text"] = "plot_text",
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
    save_fig : bool, default=True
        Used by :func:`~scikitplot.utils.utils_plot_mpl.save_plot_decorator`.
    save_fig_filename : str, default=''
        Used by :func:`~scikitplot.utils.utils_plot_mpl.save_plot_decorator`.
    **kwargs : Any
        Forwarded to :meth:`matplotlib.axes.Axes.plot` for the main metric line.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.

    Raises
    ------
    TypeError
        If ``highlight_ntile`` is neither an int nor a sequence of ints.
    ValueError
        If any highlighted ntile is outside the valid range.
    _PlotInputError
        If ``plot_input`` lacks required columns.

    See Also
    --------
    plot_response, plot_cumlift, plot_cumgains

    Notes
    -----
    Cumulative response is ``cumpct``.

    Examples
    --------
    >>> # ax = plot_cumresponse(plot_input)
    """
    ntiles = _get_ntiles_from_plot_input(plot_input, where="plot_cumresponse")
    hl = _normalize_highlight_ntiles(
        highlight_ntile, ntiles=ntiles, where="plot_cumresponse"
    )

    def _fmt(v: float) -> str:
        return f"{round(v * 100.0)}%"

    def _text(lab: str, ntile: int, v: float) -> str:
        scope = str(
            _unique_one(plot_input["scope"], name="scope", where="plot_cumresponse")
        )
        models = [str(x) for x in pd.unique(plot_input["model_label"])]
        datasets = [str(x) for x in pd.unique(plot_input["dataset_label"])]
        classes = [str(x) for x in pd.unique(plot_input["target_class"].astype(str))]
        desc = _ntile_label(ntiles)

        if scope == "compare_models":
            model_part = f"model {lab}"
            ds_part = f"dataset {datasets[0]}"
            cls_part = f"{classes[0]}"
        elif scope == "compare_datasets":
            model_part = f"model {models[0]}"
            ds_part = f"dataset {lab}"
            cls_part = f"{classes[0]}"
        elif scope == "compare_targetclasses":
            model_part = f"model {models[0]}"
            ds_part = f"dataset {datasets[0]}"
            cls_part = f"{lab}"
        else:
            model_part = f"model {models[0]}"
            ds_part = f"dataset {datasets[0]}"
            cls_part = f"{classes[0]}"

        return (
            f"When we select {desc} 1 until {ntile} from {model_part} in {ds_part}, "  # noqa: S608
            f"the cumulative response rate for {cls_part} is {round(v * 100.0)}%."
        )

    return _plot_metric_with_reference(
        plot_input,
        metric_col="cumpct",
        ref_col="pct_ref",
        title="Cumulative Response",
        ylabel="cumulative response",
        percent_y=True,
        highlight_ntiles=hl,
        highlight_how=highlight_how,
        highlight_value_formatter=_fmt,
        highlight_text_builder=_text,
        legend_loc="upper right",
        x0_for_highlight=1,
        y0_for_highlight=0.0,
        plot_kwargs=kwargs,
    )


@save_plot_decorator
@_docstring.interpd
def plot_cumlift(
    plot_input: pd.DataFrame,
    *,
    highlight_ntile: int | Sequence[int] | None = None,
    highlight_how: Literal["plot", "text", "plot_text"] = "plot_text",
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
    save_fig : bool, default=True
        Used by :func:`~scikitplot.utils.utils_plot_mpl.save_plot_decorator`.
    save_fig_filename : str, default=''
        Used by :func:`~scikitplot.utils.utils_plot_mpl.save_plot_decorator`.
    **kwargs : Any
        Forwarded to :meth:`matplotlib.axes.Axes.plot` for the main metric line.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.

    Raises
    ------
    TypeError
        If ``highlight_ntile`` is neither an int nor a sequence of ints.
    ValueError
        If any highlighted ntile is outside the valid range.
    _PlotInputError
        If ``plot_input`` lacks required columns.

    See Also
    --------
    plot_response, plot_cumresponse, plot_cumgains

    Notes
    -----
    Lift is a ratio (baseline = 1.0). We do **not** format it as percent.

    Examples
    --------
    >>> # ax = plot_cumlift(plot_input)
    """
    ntiles = _get_ntiles_from_plot_input(plot_input, where="plot_cumlift")
    hl = _normalize_highlight_ntiles(
        highlight_ntile, ntiles=ntiles, where="plot_cumlift"
    )

    def _fmt(v: float) -> str:
        return f"{v:.2f}x"

    def _text(lab: str, ntile: int, v: float) -> str:
        scope = str(
            _unique_one(plot_input["scope"], name="scope", where="plot_cumlift")
        )
        models = [str(x) for x in pd.unique(plot_input["model_label"])]
        datasets = [str(x) for x in pd.unique(plot_input["dataset_label"])]
        classes = [str(x) for x in pd.unique(plot_input["target_class"].astype(str))]
        desc = _ntile_label(ntiles)

        if scope == "compare_models":
            model_part = f"model {lab}"
            ds_part = f"dataset {datasets[0]}"
            cls_part = f"{classes[0]}"
        elif scope == "compare_datasets":
            model_part = f"model {models[0]}"
            ds_part = f"dataset {lab}"
            cls_part = f"{classes[0]}"
        elif scope == "compare_targetclasses":
            model_part = f"model {models[0]}"
            ds_part = f"dataset {datasets[0]}"
            cls_part = f"{lab}"
        else:
            model_part = f"model {models[0]}"
            ds_part = f"dataset {datasets[0]}"
            cls_part = f"{classes[0]}"

        return (
            f"When we select {desc} 1 until {ntile} from {model_part} in {ds_part}, "  # noqa: S608
            f"the cumulative lift for {cls_part} is {v:.2f}x."
        )

    ax = _plot_metric_with_reference(
        plot_input,
        metric_col="cumlift",
        ref_col=None,
        title="Cumulative Lift",
        ylabel="cumulative lift (x)",
        percent_y=False,
        highlight_ntiles=hl,
        highlight_how=highlight_how,
        highlight_value_formatter=_fmt,
        highlight_text_builder=_text,
        legend_loc="upper right",
        x0_for_highlight=1,
        y0_for_highlight=0.0,
        plot_kwargs=kwargs,
    )

    # Ensure lift baseline is visible.
    ax.axhline(1.0, linestyle="dashed", color="grey", lw=1.0, label="baseline (1.0x)")
    ax.legend(loc="upper right", shadow=False, frameon=False)
    return ax


@save_plot_decorator
@_docstring.interpd
def plot_cumgains(
    plot_input: pd.DataFrame,
    *,
    highlight_ntile: int | Sequence[int] | None = None,
    highlight_how: Literal["plot", "text", "plot_text"] = "plot_text",
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
    save_fig : bool, default=True
        Used by :func:`~scikitplot.utils.utils_plot_mpl.save_plot_decorator`.
    save_fig_filename : str, default=''
        Used by :func:`~scikitplot.utils.utils_plot_mpl.save_plot_decorator`.
    **kwargs : Any
        Forwarded to :meth:`matplotlib.axes.Axes.plot` for the main metric line.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.

    Raises
    ------
    TypeError
        If ``highlight_ntile`` is neither an int nor a sequence of ints.
    ValueError
        If any highlighted ntile is outside the valid range.
    _PlotInputError
        If ``plot_input`` lacks required columns.

    See Also
    --------
    plot_response, plot_cumresponse, plot_cumlift

    Notes
    -----
    Cumulative gains is ``cumgain``. The reference is the optimal gains curve
    ``gain_opt``.

    Examples
    --------
    >>> # ax = plot_cumgains(plot_input)
    """
    ntiles = _get_ntiles_from_plot_input(plot_input, where="plot_cumgains")
    hl = _normalize_highlight_ntiles(
        highlight_ntile, ntiles=ntiles, where="plot_cumgains"
    )

    def _fmt(v: float) -> str:
        return f"{round(v * 100.0)}%"

    def _text(lab: str, ntile: int, v: float) -> str:
        scope = str(
            _unique_one(plot_input["scope"], name="scope", where="plot_cumgains")
        )
        models = [str(x) for x in pd.unique(plot_input["model_label"])]
        datasets = [str(x) for x in pd.unique(plot_input["dataset_label"])]
        classes = [str(x) for x in pd.unique(plot_input["target_class"].astype(str))]
        desc = _ntile_label(ntiles)

        if scope == "compare_models":
            model_part = f"model {lab}"
            ds_part = f"dataset {datasets[0]}"
            cls_part = f"{classes[0]}"
        elif scope == "compare_datasets":
            model_part = f"model {models[0]}"
            ds_part = f"dataset {lab}"
            cls_part = f"{classes[0]}"
        elif scope == "compare_targetclasses":
            model_part = f"model {models[0]}"
            ds_part = f"dataset {datasets[0]}"
            cls_part = f"{lab}"
        else:
            model_part = f"model {models[0]}"
            ds_part = f"dataset {datasets[0]}"
            cls_part = f"{classes[0]}"

        return (
            f"When we select {desc} 1 until {ntile} from {model_part} in {ds_part}, "  # noqa: S608
            f"the cumulative gain for {cls_part} is {round(v * 100.0)}%."
        )

    ax = _plot_metric_with_reference(
        plot_input,
        metric_col="cumgain",
        ref_col="gain_opt",
        title="Cumulative Gains",
        ylabel="cumulative gain",
        percent_y=True,
        xlim_start=0,
        highlight_ntiles=hl,
        highlight_how=highlight_how,
        highlight_value_formatter=_fmt,
        highlight_text_builder=_text,
        legend_loc="lower right",
        x0_for_highlight=0,
        y0_for_highlight=0.0,
        plot_kwargs=kwargs,
    )

    # Dev note: Add deterministic 'minimal gains' baseline (random selection).
    ax.plot(
        list(range(ntiles + 1)),
        np.linspace(0.0, 1.0, num=ntiles + 1).tolist(),
        linestyle="dashed",
        color="grey",
        label="minimal gains",
    )
    ax.set_xlim(0, ntiles)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", shadow=False, frameon=False)
    return ax


@save_plot_decorator
@_docstring.interpd
def plot_all(
    plot_input: pd.DataFrame,
    *,
    save_fig: bool = True,
    save_fig_filename: str = "",
    **kwargs: Any,
) -> plt.Axes:
    """Plot response, cumulative response, cumulative lift, and cumulative gains as a 2x2 panel.

    Parameters
    ----------
    plot_input : pandas.DataFrame
        Output of :meth:`ModelPlotPy.plotting_scope`.
    save_fig : bool, default=True
        Used by :func:`~scikitplot.utils.utils_plot_mpl.save_plot_decorator`.
    save_fig_filename : str, default=''
        Used by :func:`~scikitplot.utils.utils_plot_mpl.save_plot_decorator`.
    **kwargs : Any
        Additional keyword arguments forwarded to :meth:`matplotlib.axes.Axes.plot`.

        The following optional keys are also supported:

        - ``figsize``: tuple[float, float] for the created figure.

    Returns
    -------
    matplotlib.axes.Axes
        The first subplot axis (cumulative gains), matching legacy behavior.

    Raises
    ------
    _PlotInputError
        If ``plot_input`` lacks required columns.

    See Also
    --------
    plot_response, plot_cumresponse, plot_cumlift, plot_cumgains

    Notes
    -----
    User note: This function draws four subplots on a single figure, but returns
    only the first Axes for backward compatibility.

    Examples
    --------
    >>> # ax = plot_all(plot_input)
    """
    # Dev note: plot_all must not call other decorated plot_* functions.
    # Nesting save_plot_decorator calls can clear/close figures unexpectedly.

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
        ],
        where="plot_all",
    )

    ntiles = _get_ntiles_from_plot_input(plot_input, where="plot_all")
    scope = str(_unique_one(plot_input["scope"], name="scope", where="plot_all"))

    # Extract metadata for the global title.
    models = [str(x) for x in pd.unique(plot_input["model_label"])]
    datasets = [str(x) for x in pd.unique(plot_input["dataset_label"])]
    classes = [str(x) for x in pd.unique(plot_input["target_class"].astype(str))]

    if scope == "no_comparison":
        global_title = (
            f"model: {models[0]} & dataset: {datasets[0]} & target class: {classes[0]}"
        )
    elif scope == "compare_models":
        global_title = f"scope: comparing models & dataset: {datasets[0]} & target class: {classes[0]}"
    elif scope == "compare_datasets":
        global_title = f"scope: comparing datasets & model: {models[0]} & target class: {classes[0]}"
    else:
        global_title = f"scope: comparing target classes & dataset: {datasets[0]} & model: {models[0]}"

    figsize = kwargs.pop("figsize", (15, 10))
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    # --- Cumulative gains (top-left) ---
    # Add the minimal gains diagonal first so it appears in the legend.
    ax1.plot(
        list(range(0, ntiles + 1, 1)),
        np.linspace(0.0, 1.0, num=ntiles + 1).tolist(),
        linestyle="dashed",
        label="minimal gains",
        color="grey",
    )
    _plot_metric_with_reference(
        plot_input,
        metric_col="cumgain",
        ref_col="gain_opt",
        title="Cumulative gains",
        ylabel="cumulative gains",
        percent_y=True,
        highlight_ntiles=[],
        highlight_how="plot_text",
        highlight_value_formatter=None,
        highlight_text_builder=None,
        legend_loc="lower right",
        xlim_start=0,
        x0_for_highlight=0,
        y0_for_highlight=0.0,
        ax=ax1,
        plot_kwargs=kwargs,
    )
    ax1.set_ylim(0.0, 1.0)

    # --- Cumulative lift (top-right) ---
    _plot_metric_with_reference(
        plot_input,
        metric_col="cumlift",
        ref_col=None,
        title="Cumulative lift",
        ylabel="cumulative lift",
        percent_y=False,
        highlight_ntiles=[],
        highlight_how="plot_text",
        highlight_value_formatter=None,
        highlight_text_builder=None,
        legend_loc="upper right",
        xlim_start=1,
        x0_for_highlight=1,
        y0_for_highlight=0.0,
        ax=ax2,
        plot_kwargs=kwargs,
    )
    ax2.set_xlim(1, ntiles)
    ax2.axhline(1.0, linestyle="dashed", color="grey", lw=1.0, label="no lift")
    ax2.legend(loc="upper right", shadow=False, frameon=False)

    # --- Response (bottom-left) ---
    _plot_metric_with_reference(
        plot_input,
        metric_col="pct",
        ref_col="pct_ref",
        title="Response",
        ylabel="response",
        percent_y=True,
        highlight_ntiles=[],
        highlight_how="plot_text",
        highlight_value_formatter=None,
        highlight_text_builder=None,
        legend_loc="upper right",
        xlim_start=1,
        x0_for_highlight=1,
        y0_for_highlight=0.0,
        ax=ax3,
        plot_kwargs=kwargs,
    )

    # --- Cumulative response (bottom-right) ---
    _plot_metric_with_reference(
        plot_input,
        metric_col="cumpct",
        ref_col="pct_ref",
        title="Cumulative response",
        ylabel="cumulative response",
        percent_y=True,
        highlight_ntiles=[],
        highlight_how="plot_text",
        highlight_value_formatter=None,
        highlight_text_builder=None,
        legend_loc="upper right",
        xlim_start=1,
        x0_for_highlight=1,
        y0_for_highlight=0.0,
        ax=ax4,
        plot_kwargs=kwargs,
    )

    fig.suptitle(global_title, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

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


def _currency_fmt(v: float) -> str:
    """Format a value as EUR currency string.

    Parameters
    ----------
    v : float
        Amount.

    Returns
    -------
    str
        EUR string.

    Raises
    ------
    None

    See Also
    --------
    str.format

    Notes
    -----
    Dev note: We keep the legacy EUR sign.

    Examples
    --------
    >>> _currency_fmt(12.3)
    '€12'
    """
    return f"€{round(v)}"


@save_plot_decorator
@_docstring.interpd
def plot_costsrevs(  # noqa: PLR0912
    plot_input: pd.DataFrame,
    *,
    fixed_costs: float,
    variable_costs_per_unit: float,
    profit_per_unit: float,
    highlight_ntile: int | Sequence[int] | None = None,
    highlight_how: Literal["plot", "text", "plot_text"] = "plot_text",
    save_fig: bool = True,
    save_fig_filename: str = "",
    **kwargs: Any,
) -> plt.Axes:
    """Plot costs / revenues curve.

    Parameters
    ----------
    plot_input : pandas.DataFrame
        Output of :meth:`ModelPlotPy.plotting_scope`.
    fixed_costs : float
        Fixed costs related to making a selection. These costs are constant and
        do not vary with selection size.
    variable_costs_per_unit : float
        Variable costs per selected unit. These costs vary with selection size.
    profit_per_unit : float
        Profit per unit in case the selected unit converts / responds positively.
    highlight_ntile : int or Sequence[int] or None, default=None
        Ntile index/indices to highlight (each in 1..ntiles). You may pass a
        single int or a sequence of ints.
    highlight_how : {'plot', 'text', 'plot_text'}, default='plot_text'
        Where to render highlight information.
    save_fig : bool, default=True
        Used by :func:`~scikitplot.utils.utils_plot_mpl.save_plot_decorator`.
    save_fig_filename : str, default=''
        Used by :func:`~scikitplot.utils.utils_plot_mpl.save_plot_decorator`.
    **kwargs : Any
        Forwarded to :meth:`matplotlib.axes.Axes.plot` for the revenue curves.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.

    Raises
    ------
    TypeError
        If any financial parameter is not numeric.
    ValueError
        If highlight parameters are invalid.
    _PlotInputError
        If plot_input lacks required columns, or if in ``compare_models`` scope
        the cumulative selection size (``cumtot``) is not identical across
        models for the same ntile.

    See Also
    --------
    plot_profit, plot_roi

    Notes
    -----
    - This function does not mutate the caller's dataframe.
    - Revenue is computed as ``profit_per_unit * cumpos``.
    - Total costs (investments) are computed as
      ``fixed_costs + variable_costs_per_unit * cumtot``.

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
    hl = _normalize_highlight_ntiles(highlight_ntile, ntiles=ntiles, where=where)
    if hl:
        _validate_highlight_how(highlight_how, where=where)

    df = plot_input.copy()

    df["variable_costs"] = variable_costs_per_unit * df["cumtot"]
    df["investments"] = fixed_costs + df["variable_costs"]
    df["revenues"] = profit_per_unit * df["cumpos"]

    scope = str(_unique_one(df["scope"], name="scope", where=where))
    nlabel = _ntile_label(ntiles)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Costs / Revenues", fontsize=16)
    ax.set_xlabel(nlabel)
    ax.set_ylabel("costs / revenue")

    ax.set_xticks(np.arange(0, ntiles + 1, 1))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.grid(True)
    ax.set_xlim([1, ntiles])

    models = [str(v) for v in pd.unique(df["model_label"])]
    datasets = [str(v) for v in pd.unique(df["dataset_label"])]
    classes = [str(v) for v in pd.unique(df["target_class"].astype(str))]

    if scope == "no_comparison":
        ax.set_title(
            f"model: {models[0]} & dataset: {datasets[0]} & target class: {classes[0]}",
            fontweight="bold",
        )
    elif scope == "compare_datasets":
        ax.set_title(
            f"scope: comparing datasets & model: {models[0]} & target class: {classes[0]}",
            fontweight="bold",
        )
    elif scope == "compare_models":
        ax.set_title(
            f"scope: comparing models & dataset: {datasets[0]} & target class: {classes[0]}",
            fontweight="bold",
        )
    else:
        ax.set_title(
            f"scope: comparing target classes & dataset: {datasets[0]} & model: {models[0]}",
            fontweight="bold",
        )

    text_lines: list[str] = []

    def _append_text(
        *, model: str, dataset: str, target: str, nt: int, rev: float
    ) -> None:
        text_lines.append(
            f"When we select {nlabel} 1 until {nt} from model {model} in dataset {dataset} "  # noqa: S608
            f"the revenue for {target} is {round(rev)}."
        )

    if scope == "no_comparison":
        gdf = df.sort_values("ntile")
        rev_line = ax.plot(gdf["ntile"], gdf["revenues"], label=classes[0], **kwargs)
        color = str(rev_line[0].get_color())
        ax.plot(
            gdf["ntile"],
            gdf["investments"],
            linestyle="dashed",
            label="total costs",
            color=color,
        )
        legend_loc = "lower right"

        if hl:
            for h_idx, nt in enumerate(hl):
                rev = _value_at_ntile(gdf, ntile=int(nt), col="revenues", where=where)
                _annotate_highlight(
                    ax,
                    x=int(nt),
                    y=float(rev),
                    x0=1,
                    y0=0.0,
                    color=color,
                    text=_currency_fmt(float(rev)),
                    xytext=_annotation_xytext(h_idx),
                )
                _append_text(
                    model=models[0],
                    dataset=datasets[0],
                    target=classes[0],
                    nt=int(nt),
                    rev=float(rev),
                )

    elif scope == "compare_datasets":
        legend_loc = "upper right"
        for d_idx, ds in enumerate([str(v) for v in pd.unique(df["dataset_label"])]):
            gdf = df.loc[df["dataset_label"] == ds].sort_values("ntile")
            rev_line = ax.plot(gdf["ntile"], gdf["revenues"], label=ds, **kwargs)
            color = str(rev_line[0].get_color())
            ax.plot(
                gdf["ntile"],
                gdf["investments"],
                linestyle="dashed",
                label=f"total costs ({ds})",
                color=color,
            )

            if hl:
                for h_idx, nt in enumerate(hl):
                    rev = _value_at_ntile(
                        gdf, ntile=int(nt), col="revenues", where=where
                    )
                    anno_idx = d_idx * len(hl) + h_idx
                    _annotate_highlight(
                        ax,
                        x=int(nt),
                        y=float(rev),
                        x0=1,
                        y0=0.0,
                        color=color,
                        text=_currency_fmt(float(rev)),
                        xytext=_annotation_xytext(anno_idx),
                    )
                    _append_text(
                        model=models[0],
                        dataset=ds,
                        target=classes[0],
                        nt=int(nt),
                        rev=float(rev),
                    )

    elif scope == "compare_models":
        legend_loc = "lower right"

        chk = df.groupby("ntile", sort=False)["cumtot"].nunique(dropna=False)
        if (chk > 1).any():
            bad = chk[chk > 1].index.to_list()
            raise _PlotInputError(
                f"{where}: compare_models expects cumtot to be identical across models for each ntile. "
                f"Non-identical cumtot found at ntile(s): {bad}."
            )

        base = df.loc[df["model_label"] == models[0]].sort_values("ntile")
        investments = fixed_costs + variable_costs_per_unit * base["cumtot"]
        ax.plot(
            base["ntile"],
            investments,
            linestyle="dashed",
            label="total costs",
            color="grey",
        )

        for m_idx, m in enumerate([str(v) for v in pd.unique(df["model_label"])]):
            gdf = df.loc[df["model_label"] == m].sort_values("ntile")
            rev_line = ax.plot(
                gdf["ntile"], gdf["revenues"], label=f"revenues ({m})", **kwargs
            )
            color = str(rev_line[0].get_color())

            if hl:
                for h_idx, nt in enumerate(hl):
                    rev = _value_at_ntile(
                        gdf, ntile=int(nt), col="revenues", where=where
                    )
                    anno_idx = m_idx * len(hl) + h_idx
                    _annotate_highlight(
                        ax,
                        x=int(nt),
                        y=float(rev),
                        x0=1,
                        y0=0.0,
                        color=color,
                        text=_currency_fmt(float(rev)),
                        xytext=_annotation_xytext(anno_idx),
                    )
                    _append_text(
                        model=m,
                        dataset=datasets[0],
                        target=classes[0],
                        nt=int(nt),
                        rev=float(rev),
                    )

    else:  # compare_targetclasses
        legend_loc = "lower right"
        for c_idx, cls in enumerate(
            [str(v) for v in pd.unique(df["target_class"].astype(str))]
        ):
            mask = df["target_class"].astype(str) == cls
            gdf = df.loc[mask].sort_values("ntile")
            rev_line = ax.plot(gdf["ntile"], gdf["revenues"], label=cls, **kwargs)
            color = str(rev_line[0].get_color())
            ax.plot(
                gdf["ntile"],
                gdf["investments"],
                linestyle="dashed",
                label=f"total costs ({cls})",
                color=color,
            )

            if hl:
                for h_idx, nt in enumerate(hl):
                    rev = _value_at_ntile(
                        gdf, ntile=int(nt), col="revenues", where=where
                    )
                    anno_idx = c_idx * len(hl) + h_idx
                    _annotate_highlight(
                        ax,
                        x=int(nt),
                        y=float(rev),
                        x0=1,
                        y0=0.0,
                        color=color,
                        text=_currency_fmt(float(rev)),
                        xytext=_annotation_xytext(anno_idx),
                    )
                    _append_text(
                        model=models[0],
                        dataset=datasets[0],
                        target=cls,
                        nt=int(nt),
                        rev=float(rev),
                    )

    ax.legend(loc=legend_loc, shadow=False, frameon=False)

    if hl:
        text = "\n".join(text_lines)
        if highlight_how in ("text", "plot_text"):
            print(text)  # noqa: T201
        if highlight_how in ("plot", "plot_text"):
            fig.text(0.53, 0.37, text, ha="left")

    return ax


@save_plot_decorator
@_docstring.interpd
def plot_profit(
    plot_input: pd.DataFrame,
    *,
    fixed_costs: float,
    variable_costs_per_unit: float,
    profit_per_unit: float,
    highlight_ntile: int | Sequence[int] | None = None,
    highlight_how: Literal["plot", "text", "plot_text"] = "plot_text",
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
    variable_costs_per_unit : float
        Variable cost per selected unit.
    profit_per_unit : float
        Revenue per positive unit.
    highlight_ntile : int or Sequence[int] or None, default=None
        Ntile index/indices to highlight (each in 1..ntiles).
    highlight_how : {'plot', 'text', 'plot_text'}, default='plot_text'
        Where to render highlight information.
    save_fig : bool, default=True
        Used by :func:`~scikitplot.utils.utils_plot_mpl.save_plot_decorator`.
    save_fig_filename : str, default=''
        Used by :func:`~scikitplot.utils.utils_plot_mpl.save_plot_decorator`.
    **kwargs : Any
        Forwarded to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.

    Raises
    ------
    TypeError
        If any financial parameter is not numeric.
    ValueError
        If highlight parameters are invalid.
    _PlotInputError
        If plot_input lacks required columns.

    See Also
    --------
    plot_costsrevs, plot_roi

    Notes
    -----
    Profit is revenues minus investments (fixed + variable costs).

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
    if hl:
        _validate_highlight_how(highlight_how, where=where)

    df = plot_input.copy()

    df["variable_costs"] = variable_costs_per_unit * df["cumtot"]
    df["investments"] = fixed_costs + df["variable_costs"]
    df["revenues"] = profit_per_unit * df["cumpos"]
    df["profit"] = df["revenues"] - df["investments"]

    def _fmt(v: float) -> str:
        return _currency_fmt(v)

    def _text(lab: str, ntile: int, v: float) -> str:
        desc = _ntile_label(ntiles)
        return f"When we select {desc} 1 until {ntile}, expected profit for '{lab}' is {round(v)}."

    return _plot_metric_with_reference(
        df,
        metric_col="profit",
        ref_col=None,
        title="Profit",
        ylabel="€",
        percent_y=False,
        highlight_ntiles=hl,
        highlight_how=highlight_how,
        highlight_value_formatter=_fmt,
        highlight_text_builder=_text,
        legend_loc="upper right",
        x0_for_highlight=1,
        y0_for_highlight=0.0,
        plot_kwargs=kwargs,
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
    save_fig : bool, default=True
        Used by :func:`~scikitplot.utils.utils_plot_mpl.save_plot_decorator`.
    save_fig_filename : str, default=''
        Used by :func:`~scikitplot.utils.utils_plot_mpl.save_plot_decorator`.
    **kwargs : Any
        Forwarded to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.

    Raises
    ------
    TypeError
        If any financial parameter is not numeric.
    ValueError
        If investments are zero (division by zero) or highlight parameters invalid.
    _PlotInputError
        If plot_input lacks required columns.

    See Also
    --------
    plot_profit, plot_costsrevs

    Notes
    -----
    ROI is defined as ``profit / investments``. A dashed break-even line at 0 is
    included.

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
    if hl:
        _validate_highlight_how(highlight_how, where=where)

    df = plot_input.copy()

    df["variable_costs"] = variable_costs_per_unit * df["cumtot"]
    df["investments"] = fixed_costs + df["variable_costs"]
    df["revenues"] = profit_per_unit * df["cumpos"]
    df["profit"] = df["revenues"] - df["investments"]

    inv = df["investments"].to_numpy(dtype=float)
    if np.any(inv == 0.0):
        raise ValueError("ROI undefined when investments are zero for any ntile.")

    df["roi"] = df["profit"] / df["investments"]

    def _fmt(v: float) -> str:
        return f"{round(v * 100.0)}%"

    def _text(lab: str, ntile: int, v: float) -> str:
        desc = _ntile_label(ntiles)
        return f"When we select {desc} 1 until {ntile}, expected ROI for '{lab}' is {round(v * 100.0)}%."

    ax = _plot_metric_with_reference(
        df,
        metric_col="roi",
        ref_col=None,
        title="Return on Investment (ROI)",
        ylabel="ROI",
        percent_y=True,
        highlight_ntiles=hl,
        highlight_how=highlight_how,
        highlight_value_formatter=_fmt,
        highlight_text_builder=_text,
        legend_loc="lower right",
        x0_for_highlight=1,
        y0_for_highlight=0.0,
        plot_kwargs=kwargs,
    )

    # Break-even line at ROI=0.
    ax.axhline(0.0, linestyle="dashed", color="grey", lw=1.0, label="break even")
    ax.legend(loc="lower right", shadow=False, frameon=False)
    return ax
