# scikitplot/feature_selection/_analysis.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Routing tables: which plots, tests and treatments fit each feature.

Two advisory functions turn a fitted schema into plans; neither runs a test,
draws a plot or transforms data:

:func:`analysis_plan`
    Per feature: univariate plots and summaries for its kind, and, given a
    target, the bivariate tests (named after their :mod:`scipy.stats`
    function, so ``getattr(scipy.stats, name)`` finds them) and plots for the
    feature-kind / target-kind pair. Assessments add notes (``skewed`` ->
    ``log_scale_axis``).
:func:`recommend_treatments`
    Per feature: an ordered list of preprocessing steps, named after the
    scikit-learn class that performs them, with the reasons; and
    :func:`treatment_groups` groups features that share a list, which is the
    branch structure of a ``ColumnTransformer``.

Notes
-----
**User note.** The tables are defaults, not rules of nature. Pass your own
``rules`` mapping (same shape as :data:`DEFAULT_ANALYSIS_RULES`) or your own
``rule`` callable to :func:`recommend_treatments` to encode a team's
conventions; the output format stays the same.

**Developer note.** Every entry is a pure function of the fitted kind and the
fitted assessment states, so the plan is deterministic and can be reviewed and
versioned like the policy.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from ._schema import FeatureSchema, FeatureSpec
from ._types import FeatureKind

__all__ = [
    "DEFAULT_ANALYSIS_RULES",
    "analysis_plan",
    "recommend_treatments",
    "treatment_groups",
]

_K = FeatureKind

#: Univariate plots and summaries per feature kind, and bivariate tests/plots
#: per (feature kind, target group). Target groups: ``numeric``, ``binary``,
#: ``categorical``.
DEFAULT_ANALYSIS_RULES: Mapping[str, Any] = {
    "univariate": {
        _K.NUMERIC_CONTINUOUS.value: {
            "plots": ("histogram", "box", "ecdf"),
            "summaries": ("describe", "skewness", "kurtosis", "normaltest"),
        },
        _K.NUMERIC_DISCRETE.value: {
            "plots": ("bar", "histogram"),
            "summaries": ("describe", "value_counts"),
        },
        _K.BINARY.value: {"plots": ("bar",), "summaries": ("value_counts",)},
        _K.CATEGORICAL_NOMINAL.value: {
            "plots": ("bar",),
            "summaries": ("value_counts",),
        },
        _K.CATEGORICAL_ORDINAL.value: {
            "plots": ("bar_ordered",),
            "summaries": ("value_counts", "median_category"),
        },
        _K.DATETIME.value: {"plots": ("line_over_time",), "summaries": ("min_max",)},
        _K.TIMEDELTA.value: {"plots": ("histogram",), "summaries": ("describe",)},
        _K.TEXT.value: {
            "plots": ("length_histogram",),
            "summaries": ("length_describe",),
        },
        _K.OTHER.value: {"plots": (), "summaries": ()},
        _K.UNKNOWN.value: {"plots": (), "summaries": ()},
    },
    "bivariate": {
        (_K.NUMERIC_CONTINUOUS.value, "numeric"): {
            "tests": ("pearsonr", "spearmanr"),
            "plots": ("scatter", "hexbin"),
        },
        (_K.NUMERIC_CONTINUOUS.value, "binary"): {
            "tests": ("mannwhitneyu", "ttest_ind"),
            "plots": ("box_by_target", "kde_by_target"),
        },
        (_K.NUMERIC_CONTINUOUS.value, "categorical"): {
            "tests": ("kruskal", "f_oneway"),
            "plots": ("box_by_target", "violin_by_target"),
        },
        (_K.NUMERIC_DISCRETE.value, "numeric"): {
            "tests": ("spearmanr", "kendalltau"),
            "plots": ("box_by_feature",),
        },
        (_K.NUMERIC_DISCRETE.value, "binary"): {
            "tests": ("mannwhitneyu", "chi2_contingency"),
            "plots": ("stacked_bar",),
        },
        (_K.NUMERIC_DISCRETE.value, "categorical"): {
            "tests": ("kruskal", "chi2_contingency"),
            "plots": ("stacked_bar",),
        },
        (_K.BINARY.value, "numeric"): {
            "tests": ("pointbiserialr", "mannwhitneyu"),
            "plots": ("box_by_feature",),
        },
        (_K.BINARY.value, "binary"): {
            "tests": ("fisher_exact", "chi2_contingency"),
            "plots": ("stacked_bar",),
        },
        (_K.BINARY.value, "categorical"): {
            "tests": ("chi2_contingency",),
            "plots": ("stacked_bar",),
        },
        (_K.CATEGORICAL_NOMINAL.value, "numeric"): {
            "tests": ("kruskal", "f_oneway"),
            "plots": ("box_by_feature",),
        },
        (_K.CATEGORICAL_NOMINAL.value, "binary"): {
            "tests": ("chi2_contingency",),
            "plots": ("stacked_bar", "target_rate_bar"),
        },
        (_K.CATEGORICAL_NOMINAL.value, "categorical"): {
            "tests": ("chi2_contingency",),
            "plots": ("heatmap_crosstab",),
        },
        (_K.CATEGORICAL_ORDINAL.value, "numeric"): {
            "tests": ("spearmanr", "kruskal"),
            "plots": ("box_by_feature",),
        },
        (_K.CATEGORICAL_ORDINAL.value, "binary"): {
            "tests": ("mannwhitneyu", "chi2_contingency"),
            "plots": ("target_rate_bar",),
        },
        (_K.CATEGORICAL_ORDINAL.value, "categorical"): {
            "tests": ("chi2_contingency",),
            "plots": ("heatmap_crosstab",),
        },
    },
    "notes": {
        "skewed": "log_scale_axis",
        "has_outliers": "robust_summaries",
        "high_missing": "plot_missingness_against_target",
        "high_cardinality": "top_k_categories_only",
        "constant": "skip_no_variation",
        "identifier_like": "skip_identifier",
    },
}

_TARGET_GROUPS = {
    _K.NUMERIC_CONTINUOUS.value: "numeric",
    _K.NUMERIC_DISCRETE.value: "numeric",
    _K.BINARY.value: "binary",
    _K.CATEGORICAL_NOMINAL.value: "categorical",
    _K.CATEGORICAL_ORDINAL.value: "categorical",
}


def _target_group(schema: FeatureSchema, target: Any) -> str | None:
    if target is None:
        return None
    if target in schema:  # ruff: ignore[if-else-block-instead-of-if-exp]
        kind = schema[target].kind
    else:
        kind = target
    try:
        return _TARGET_GROUPS[kind]
    except KeyError:
        raise ValueError(
            f"target {target!r} has kind {kind!r}; a target must be numeric, binary or"
            f" categorical (one of {sorted(_TARGET_GROUPS)}), or name such a column"
        ) from None


def analysis_plan(
    schema: FeatureSchema,
    target: str | None = None,
    *,
    rules: Mapping[str, Any] | None = None,
    role: Any = "feature",
) -> pd.DataFrame:
    """
    Route every feature to the plots, summaries and tests that fit its kind.

    Parameters
    ----------
    schema : FeatureSchema
        Fitted schema.
    target : str or None, default=None
        A target column of the schema, or a target kind
        (``"binary"``, ``"numeric_continuous"`` ...). None omits the
        bivariate columns' content.
    rules : mapping or None, default=None
        Replacement for :data:`DEFAULT_ANALYSIS_RULES` (same keys).
    role : str or sequence of str or None, default="feature"
        Roles to include.

    Returns
    -------
    pandas.DataFrame
        Indexed by feature with ``kind``, ``plots``, ``summaries``,
        ``target_tests``, ``target_plots`` and ``notes`` (tuples of names).

    Examples
    --------
    >>> import pandas as pd
    >>> from scikitplot.feature_selection import analysis_plan, profile_features
    >>> X = pd.DataFrame({"x": [1.5, 2.5, 3.5, 50.5] * 5, "y": [0, 1] * 10})
    >>> plan = analysis_plan(profile_features(X, {"roles": {"target": ["y"]}}), "y")
    >>> plan.loc["x", "target_tests"], plan.loc["x", "notes"]
    (('mannwhitneyu', 'ttest_ind'), ('has_outliers: robust_summaries', 'skewed: log_scale_axis'))
    """
    rules = DEFAULT_ANALYSIS_RULES if rules is None else rules
    group = _target_group(schema, target)
    rows = []
    names = schema.select(role=role)
    for name in names:
        if name == target:
            continue
        spec = schema[name]
        uni = rules["univariate"].get(spec.kind, {"plots": (), "summaries": ()})
        bi = rules["bivariate"].get((spec.kind, group), {}) if group else {}
        notes = tuple(
            f"{assessment}: {note}"
            for assessment, note in sorted(rules["notes"].items())
            if assessment in spec.assessment_names and spec.is_(assessment)
        )
        rows.append(
            {
                "feature": name,
                "kind": spec.kind,
                "plots": tuple(uni.get("plots", ())),
                "summaries": tuple(uni.get("summaries", ())),
                "target_tests": tuple(bi.get("tests", ())),
                "target_plots": tuple(bi.get("plots", ())),
                "notes": notes,
            }
        )
    columns = [
        "feature",
        "kind",
        "plots",
        "summaries",
        "target_tests",
        "target_plots",
        "notes",
    ]
    return pd.DataFrame(rows, columns=columns).set_index("feature")


def _state(spec: FeatureSpec, name: str) -> bool:
    return name in spec.assessment_names and spec.is_(name)


def default_treatment(  # ruff: ignore[too-many-branches]
    spec: FeatureSpec,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Return the default treatment of one feature.

    Parameters
    ----------
    spec : FeatureSpec
        A fitted feature.

    Returns
    -------
    steps : tuple of str
        Ordered scikit-learn class names (with key arguments), or ``("drop",)``.
    reasons : tuple of str
        Why each choice was made.

    Notes
    -----
    Decision table (first match wins for the drop rules):

    - ``constant`` or ``identifier_like`` -> ``drop``;
    - missing values -> an imputer first (``median`` for continuous numbers,
      ``most_frequent`` otherwise; ``add_indicator=True`` when ``high_missing``);
    - ``numeric_continuous``: ``skewed`` -> ``PowerTransformer`` (Yeo-Johnson,
      standardising); else ``has_outliers`` -> ``RobustScaler``; else
      ``StandardScaler``;
    - ``numeric_discrete`` -> ``StandardScaler``;
    - ``binary``: numeric -> passthrough, otherwise ``OrdinalEncoder``;
    - ``categorical_nominal``: ``high_cardinality`` -> ``TargetEncoder``, else
      ``OneHotEncoder(handle_unknown='infrequent_if_exist')``;
    - ``categorical_ordinal`` -> ``OrdinalEncoder(categories=<fitted order>)``;
    - ``text`` -> ``TfidfVectorizer``; datetime/timedelta -> a custom expansion;
      ``other``/``unknown`` -> ``review``.
    """
    if _state(spec, "constant"):
        return ("drop",), ("constant",)
    if _state(spec, "identifier_like"):
        return ("drop",), ("identifier_like",)
    steps: list[str] = []
    reasons: list[str] = []
    kind = spec.kind
    if spec.stats.n_missing and kind not in (
        _K.TEXT.value,
        _K.OTHER.value,
        _K.UNKNOWN.value,
    ):
        strategy = "median" if kind == _K.NUMERIC_CONTINUOUS.value else "most_frequent"
        indicator = ", add_indicator=True" if _state(spec, "high_missing") else ""
        steps.append(f"SimpleImputer(strategy='{strategy}'{indicator})")
        reasons.append("has_missing" + ("+high_missing" if indicator else ""))
    if kind == _K.NUMERIC_CONTINUOUS.value:
        if _state(spec, "skewed"):
            steps.append("PowerTransformer(method='yeo-johnson')")
            reasons.append("skewed")
        elif _state(spec, "has_outliers"):
            steps.append("RobustScaler()")
            reasons.append("has_outliers")
        else:
            steps.append("StandardScaler()")
            reasons.append("continuous")
    elif kind == _K.NUMERIC_DISCRETE.value:
        steps.append("StandardScaler()")
        reasons.append("discrete")
    elif kind == _K.BINARY.value:
        if spec.stats.is_numeric or spec.family == "boolean":
            steps.append("passthrough")
            reasons.append("binary_numeric")
        else:
            steps.append("OrdinalEncoder()")
            reasons.append("binary_labels")
    elif kind == _K.CATEGORICAL_NOMINAL.value:
        if _state(spec, "high_cardinality"):
            steps.append("TargetEncoder()")
            reasons.append("high_cardinality")
        else:
            steps.append("OneHotEncoder(handle_unknown='infrequent_if_exist')")
            reasons.append("nominal")
    elif kind == _K.CATEGORICAL_ORDINAL.value:
        steps.append("OrdinalEncoder(categories=<fitted order>)")
        reasons.append("ordinal")
    elif kind == _K.TEXT.value:
        steps.append("TfidfVectorizer()")
        reasons.append("text")
    elif kind in (_K.DATETIME.value, _K.TIMEDELTA.value):
        steps.append("custom: calendar/duration expansion")
        reasons.append(kind)
    else:
        steps.append("review")
        reasons.append(kind)
    return tuple(steps), tuple(reasons)


def recommend_treatments(
    schema: FeatureSchema,
    *,
    rule: Callable[[FeatureSpec], tuple[Sequence[str], Sequence[str]]] | None = None,
    role: Any = "feature",
) -> pd.DataFrame:
    """
    Recommend an ordered preprocessing chain per feature.

    Parameters
    ----------
    schema : FeatureSchema
        Fitted schema.
    rule : callable or None, default=None
        ``rule(spec) -> (steps, reasons)``; :func:`default_treatment` when None.
    role : str or sequence of str or None, default="feature"
        Roles to include.

    Returns
    -------
    pandas.DataFrame
        Indexed by feature with ``kind``, ``steps`` and ``reasons`` (tuples).

    Examples
    --------
    >>> import pandas as pd
    >>> from scikitplot.feature_selection import profile_features, recommend_treatments
    >>> X = pd.DataFrame({"x": [1.5, 2.5, 3.5, 50.5] * 5, "c": list("abcd") * 5})
    >>> recommend_treatments(profile_features(X))["steps"].to_dict()
    {'x': ("PowerTransformer(method='yeo-johnson')",), 'c': ("OneHotEncoder(handle_unknown='infrequent_if_exist')",)}
    """
    rule = default_treatment if rule is None else rule
    rows = []
    for name in schema.select(role=role):
        spec = schema[name]
        steps, reasons = rule(spec)
        rows.append(
            {
                "feature": name,
                "kind": spec.kind,
                "steps": tuple(steps),
                "reasons": tuple(reasons),
            }
        )
    return pd.DataFrame(
        rows, columns=["feature", "kind", "steps", "reasons"]
    ).set_index("feature")


def treatment_groups(treatments: pd.DataFrame) -> dict[tuple[str, ...], list[str]]:
    """
    Group features that share the same recommended chain.

    Parameters
    ----------
    treatments : pandas.DataFrame
        Output of :func:`recommend_treatments`.

    Returns
    -------
    dict of tuple to list of str
        Chain to features, in first-seen order: one ``ColumnTransformer``
        branch per key.
    """
    groups: dict[tuple[str, ...], list[str]] = {}
    for name, steps in treatments["steps"].items():
        groups.setdefault(tuple(steps), []).append(name)
    return groups
