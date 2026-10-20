# scikitplot/feature_selection/__init__.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.feature_selection
============================

Understand, group, track and monitor the features of wide pandas tables.

Three questions, three tools:

*What is each feature?*
    :func:`profile_features` (or :class:`FeatureProfiler` in a pipeline) fits a
    :class:`FeatureSchema` on training data. Each :class:`FeatureSpec` knows
    its role, dtype family, semantic kind, user tags, a bin recommendation,
    and policy-backed assessments with evidence ("am I skewed / discrete /
    constant / binary / identifier-like / high-cardinality ...?"). Thresholds
    live in a :class:`FeaturePolicy` (TOML, YAML or JSON), and custom
    questions are registered with :func:`register_assessment`.

*What happened to each feature in a pipeline?*
    :func:`trace_pipeline` reads a fitted ``Pipeline`` /
    ``ColumnTransformer`` / ``FeatureUnion`` and returns a
    :class:`FeatureLineage`: which steps touched a raw column, which outputs it
    became, where it was dropped, and :meth:`FeatureLineage.attribute` to sum
    one-hot (or any exactly traced) importances back to raw columns.

*Has it drifted?*
    :class:`ReferenceProfile` freezes training buckets per feature and
    reports PSI, Jensen-Shannon distance, missingness change and unseen
    categories for later data.

The scikit-learn pieces (:class:`FeatureColumnSelector` for
``ColumnTransformer``, :class:`FeatureFilter`, :class:`FeatureRenamer`) fit
on the training fold only, so they are safe under cross-validation.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from scikitplot.feature_selection import profile_features
>>> rng = np.random.default_rng(0)
>>> X = pd.DataFrame(
...     {
...         "income": rng.lognormal(10, 1, 300),
...         "age": rng.integers(18, 80, 300),
...         "kids": rng.integers(0, 4, 300),
...         "city": rng.choice(["ankara", "izmir", "bursa"], 300),
...         "customer_id": [f"c{i}" for i in range(300)],
...     }
... )
>>> schema = profile_features(X)
>>> schema.groups(by="kind")["numeric_continuous"]
['income', 'age']
>>> schema["income"].is_("skewed"), schema["kids"].kind
(True, 'numeric_discrete')
>>> schema.select(is_=("identifier_like",))
['customer_id']

Notes
-----
**User note.** Profile the training split only and keep the schema
(:meth:`FeatureSchema.save`); audit later splits with
:meth:`FeatureSchema.audit` and :class:`ReferenceProfile`. Recommendations
(:func:`recommend_treatments`, :func:`analysis_plan`, bin plans) are advice and
never modify data.

**Developer note.** The package is pure Python on top of pandas, NumPy,
SciPy and scikit-learn; PyYAML and ``tomli-w`` are needed only to read YAML or
write TOML. Every persisted artefact carries a ``schema_version`` and is
checked field by field on load.
"""  # ruff: ignore[missing-blank-line-after-summary, missing-trailing-period]

from __future__ import annotations

from . import (
    _analysis,
    _assessments,
    _binning,
    _drift,
    _errors,
    _estimators,
    _lineage,
    _policy,
    _rename,
    _schema,
    _stats,
    _types,
)
from ._analysis import *  # noqa: F403
from ._assessments import *  # noqa: F403
from ._binning import *  # noqa: F403
from ._drift import *  # noqa: F403
from ._errors import *  # noqa: F403
from ._estimators import *  # noqa: F403
from ._lineage import *  # noqa: F403
from ._policy import *  # noqa: F403
from ._rename import *  # noqa: F403
from ._schema import *  # noqa: F403
from ._stats import *  # noqa: F403
from ._types import *  # noqa: F403

__all__: list[str] = []
__all__ += _analysis.__all__
__all__ += _assessments.__all__
__all__ += _binning.__all__
__all__ += _drift.__all__
__all__ += _errors.__all__
__all__ += _estimators.__all__
__all__ += _lineage.__all__
__all__ += _policy.__all__
__all__ += _rename.__all__
__all__ += _schema.__all__
__all__ += _stats.__all__
__all__ += _types.__all__
