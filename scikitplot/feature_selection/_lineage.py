# scikitplot/feature_selection/_lineage.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Feature lineage of a fitted scikit-learn pipeline: what happened to every feature.

:func:`trace_pipeline` walks a *fitted* estimator (``Pipeline``,
``ColumnTransformer``, ``FeatureUnion`` and the transformers inside them) and
records a directed graph:

- **nodes** are the features present after each stage (``"input"`` holds the
  raw columns; ``"ct/num/scaler"`` the outputs of that nested step);
- **edges** link every output feature to the input features it was computed
  from, with a :class:`~scikitplot.feature_selection.Relation` and a
  :class:`~scikitplot.feature_selection.LineageQuality`;
- **stages** record, per step, the estimator, input and output counts and the
  observed matrix width when a sample was given.

Questions it answers without any manual bookkeeping::

    lineage.history("age")  # which steps touched "age", in order
    lineage.treatments()  # raw feature -> chain of transformers
    lineage.origins("cat__city_Paris")  # raw columns behind a model input
    lineage.descendants("city")  # model inputs derived from "city"
    lineage.dropped()  # raw features that never reach the model
    lineage.attribute(model.feature_importances_)  # importances per raw column

Notes
-----
**User note.** Pass ``X=`` (a few training rows) to also check every stage's
actual output width against the traced count; a disagreement raises
:class:`~scikitplot.feature_selection.LineageError` because importance and
drift attribution would otherwise be wrong. Works with any model library:
:meth:`FeatureLineage.attribute` only needs an array aligned with
:attr:`FeatureLineage.output_names`.

**Developer note.** Edges come, in order of preference, from an estimator's
own ``get_feature_lineage(input_features)`` (the protocol of this package,
returning :class:`FeatureMapping`), from public fitted attributes of known
scikit-learn classes (``get_support``, ``categories_``/``drop_idx_``/
``infrequent_categories_``, ``n_bins_``, ``powers_``, ``indicator_.features_``,
``labels_``, ``classes_``), from class contracts (column-wise scalers and
encoders; ``FunctionTransformer(feature_names_out="one-to-one")``, which is
the user's declaration), and otherwise are *mixed* or *unknown*. Mixed: the
values of an output combine several inputs although names stay one-to-one
(``Normalizer``, ``TfidfTransformer``, any other ``OneToOneFeatureMixin``,
KNN and iterative imputers, ``ClassNamePrefixFeaturesOutMixin`` such as PCA).
Unknown: no contract at all; every output is linked to every input and
nothing is attributed. Names are never parsed to guess parents. Every
adapter's count is checked against ``get_feature_names_out``, and a
container without names whose branches would repeat a name is refused.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import _serialize
from ._errors import IncompatibleSchemaError, LineageError
from ._types import LineageQuality, Relation

__all__ = [
    "Attribution",
    "FeatureLineage",
    "FeatureMapping",
    "trace_pipeline",
]

LINEAGE_VERSION = 1

_QUALITY_RANK = {
    LineageQuality.EXACT.value: 0,
    LineageQuality.MIXED.value: 1,
    LineageQuality.UNKNOWN.value: 2,
}
_INPUT = "input"
_OUTPUT = "output"
_ROOT = "(root)"


# --------------------------------------------------------------------------- #
# The protocol an estimator implements to describe itself
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class FeatureMapping:
    """
    How one transformer's outputs derive from its inputs.

    Return it from ``get_feature_lineage(input_features)`` on your own
    transformer to make it exactly traceable.

    Parameters
    ----------
    parents : tuple of tuple of int
        For every output, the positions of the inputs it is computed from.
    relation : str
        A :class:`~scikitplot.feature_selection.Relation` value.
    quality : str, default="exact"
        A :class:`~scikitplot.feature_selection.LineageQuality` value.
    output_qualities : tuple of str, default=()
        Per-output qualities when they differ (an imputer whose filled
        values use other columns, ``mixed``, beside exact missing indicators);
        empty means every output has ``quality``.

    Examples
    --------
    >>> from scikitplot.feature_selection import FeatureMapping
    >>> FeatureMapping(parents=((0,), (0,), (1,)), relation="expand").n_outputs
    3
    """

    parents: tuple[tuple[int, ...], ...]
    relation: str
    quality: str = LineageQuality.EXACT.value
    output_qualities: tuple[str, ...] = ()

    def __post_init__(self) -> None:  # noqa: D105
        parents = tuple(tuple(int(i) for i in p) for p in self.parents)
        object.__setattr__(self, "parents", parents)
        object.__setattr__(self, "relation", Relation(self.relation).value)
        object.__setattr__(self, "quality", LineageQuality(self.quality).value)
        qualities = tuple(LineageQuality(q).value for q in self.output_qualities)
        if qualities and len(qualities) != len(parents):
            raise ValueError(
                f"output_qualities has {len(qualities)} entries for"
                f" {len(parents)} outputs"
            )
        object.__setattr__(self, "output_qualities", qualities)

    def quality_of(self, output: int) -> str:
        """Return the lineage quality of output ``output``."""
        return self.output_qualities[output] if self.output_qualities else self.quality

    @property
    def n_outputs(self) -> int:
        """int: Number of outputs."""
        return len(self.parents)

    @classmethod
    def identity(cls, n: int) -> FeatureMapping:
        """Return the one-to-one mapping of ``n`` features."""
        return cls(tuple((i,) for i in range(n)), Relation.ONE_TO_ONE.value)

    @classmethod
    def subset(cls, indices: Iterable[int]) -> FeatureMapping:
        """Return a mapping that keeps the inputs at ``indices``, in that order."""
        return cls(tuple((int(i),) for i in indices), Relation.SUBSET.value)

    @classmethod
    def all_to_all(cls, n_in: int, n_out: int, quality: str) -> FeatureMapping:
        """Return a mapping where every output depends on every input."""
        relation = (
            Relation.MIXED
            if quality == LineageQuality.MIXED.value
            else Relation.UNKNOWN
        )
        return cls(
            tuple(tuple(range(n_in)) for _ in range(n_out)), relation.value, quality
        )


# --------------------------------------------------------------------------- #
# Graph records
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class FeatureNode:
    """A feature at one stage: ``(stage, position, name)`` with a graph id."""

    id: int
    stage: str
    position: int
    name: str


@dataclasses.dataclass(frozen=True)
class LineageEdge:
    """``child`` (at ``stage``) is computed from ``parent``."""

    parent: int
    child: int
    stage: str
    relation: str
    quality: str


@dataclasses.dataclass(frozen=True)
class StageRecord:
    """
    One traced step.

    Parameters
    ----------
    path : str
        Step path (``"ct/num/scaler"``); ``"(root)"`` for the traced object.
    estimator : str
        Class name of the step (``"passthrough"`` / ``"drop"`` for strings).
    container : str or None
        ``"pipeline"``, ``"column_transformer"`` or ``"feature_union"`` for
        containers; None for leaf transformers.
    relation, quality : str
        Relation and lineage quality of the step's outputs.
    n_in, n_out : int
        Feature counts before and after the step.
    observed_width : int or None
        Width of the step's actual output on the sample ``X``, if given.
    note : str
        Why a quality is not exact, when it is not.
    """

    path: str
    estimator: str
    container: str | None
    relation: str
    quality: str
    n_in: int
    n_out: int
    observed_width: int | None = None
    note: str = ""


# --------------------------------------------------------------------------- #
# Adapters for leaf transformers
# --------------------------------------------------------------------------- #


#: OneToOneFeatureMixin classes whose output ``i`` is computed from input ``i``
#: alone (fitted per column). Any other one-to-one class is ``mixed``.
_COLUMNWISE = frozenset(
    {
        "Binarizer",
        "MaxAbsScaler",
        "MinMaxScaler",
        "OrdinalEncoder",
        "PowerTransformer",
        "QuantileTransformer",
        "RobustScaler",
        "StandardScaler",
        "TargetEncoder",
    }
)


def _names_out(est: Any, input_names: Sequence[str]) -> tuple[list[str] | None, str]:
    method = getattr(est, "get_feature_names_out", None)
    if method is None:
        return None, f"{type(est).__name__} has no get_feature_names_out"
    try:
        names = method(np.asarray(input_names, dtype=object))
    except (AttributeError, ValueError, TypeError) as exc:
        return None, f"get_feature_names_out failed: {type(exc).__name__}: {exc}"
    return [str(n) for n in names], ""


def _class_names(est: Any) -> set[str]:
    return {cls.__name__ for cls in type(est).__mro__}


def _onehot_mapping(est: Any, n_in: int) -> FeatureMapping:
    categories = est.categories_
    drop_idx = getattr(est, "drop_idx_", None)
    infrequent = getattr(est, "infrequent_categories_", None)
    parents = []
    for i in range(n_in):
        n = len(categories[i])
        if infrequent is not None and infrequent[i] is not None and len(infrequent[i]):
            n = n - len(infrequent[i]) + 1
        if drop_idx is not None and drop_idx[i] is not None:
            n -= 1
        parents.extend([(i,)] * n)
    return FeatureMapping(tuple(parents), Relation.EXPAND.value)


def _imputer_mapping(est: Any, input_names: Sequence[str], names: list[str] | None):
    indicator = getattr(est, "indicator_", None)
    features = [] if indicator is None else [int(i) for i in indicator.features_]
    if names is None:
        return None, "imputer without feature names"
    kept = names[: len(names) - len(features)]
    index = {n: i for i, n in enumerate(input_names)}
    try:
        positions = [index[n] for n in kept]
    except KeyError:
        return None, "imputer kept names are not input names"
    if positions != sorted(positions):
        return None, "imputer kept names are out of order"
    indicators = tuple((f,) for f in features)
    if "SimpleImputer" not in _class_names(est):
        # KNN / iterative imputation fills a column from the other columns:
        # every kept output depends on every input; indicators stay exact.
        everything = tuple(range(len(input_names)))
        parents = tuple(everything for _ in positions) + indicators
        qualities = (LineageQuality.MIXED.value,) * len(positions) + (
            LineageQuality.EXACT.value,
        ) * len(indicators)
        return (
            FeatureMapping(
                parents, Relation.MIXED.value, LineageQuality.MIXED.value, qualities
            ),
            "imputed values are computed from other columns",
        )

    def _determine_relation(positions, input_names, features):
        if len(positions) == len(input_names) and not features:
            return Relation.ONE_TO_ONE
        if features:
            return Relation.EXPAND
        return Relation.SUBSET

    relation = _determine_relation(positions, input_names, features)
    parents = tuple((p,) for p in positions) + indicators
    return FeatureMapping(parents, relation.value), ""


def _leaf_mapping(  # ruff: ignore[too-many-branches, too-many-return-statements]
    est: Any,
    input_names: Sequence[str],
    names: list[str] | None,
) -> tuple[FeatureMapping | None, str]:
    """Return the mapping of a fitted leaf transformer, or (None, reason)."""
    n_in = len(input_names)
    classes = _class_names(est)
    own = getattr(est, "get_feature_lineage", None)
    if callable(own):
        mapping = own(list(input_names))
        if not isinstance(mapping, FeatureMapping):
            raise LineageError(
                f"{type(est).__name__}.get_feature_lineage returned"
                f" {type(mapping).__name__}, not FeatureMapping"
            )
        return mapping, ""
    if callable(getattr(est, "get_support", None)):
        return FeatureMapping.subset(np.flatnonzero(est.get_support())), ""
    if "OneHotEncoder" in classes and hasattr(est, "categories_"):
        return _onehot_mapping(est, n_in), ""
    if "KBinsDiscretizer" in classes and hasattr(est, "n_bins_"):
        if est.encode == "ordinal":
            return FeatureMapping.identity(n_in), ""
        parents = [(i,) for i in range(n_in) for _ in range(int(est.n_bins_[i]))]
        return FeatureMapping(tuple(parents), Relation.EXPAND.value), ""
    if "PolynomialFeatures" in classes and hasattr(est, "powers_"):
        parents = tuple(
            tuple(int(i) for i in np.flatnonzero(row)) for row in est.powers_
        )
        return FeatureMapping(parents, Relation.COMBINE.value), ""
    if "SplineTransformer" in classes and hasattr(est, "n_features_out_"):
        per = int(est.n_features_out_) // max(n_in, 1)
        parents = [(i,) for i in range(n_in) for _ in range(per)]
        return FeatureMapping(tuple(parents), Relation.EXPAND.value), ""
    if classes & {"SimpleImputer", "KNNImputer", "IterativeImputer"}:
        return _imputer_mapping(est, input_names, names)
    if "MissingIndicator" in classes and hasattr(est, "features_"):
        return (
            FeatureMapping(
                tuple((int(i),) for i in est.features_), Relation.EXPAND.value
            ),
            "",
        )
    if "FeatureAgglomeration" in classes and hasattr(est, "labels_"):
        labels = np.asarray(est.labels_)
        n_out = int(labels.max()) + 1 if labels.size else 0
        parents = tuple(
            tuple(int(i) for i in np.flatnonzero(labels == j)) for j in range(n_out)
        )
        return FeatureMapping(parents, Relation.COMBINE.value), ""
    if (
        "TargetEncoder" in classes
        and getattr(est, "target_type_", None) == "multiclass"
    ):
        k = len(est.classes_)
        parents = [(i,) for i in range(n_in) for _ in range(k)]
        return FeatureMapping(tuple(parents), Relation.EXPAND.value), ""
    if "FunctionTransformer" in classes:
        if getattr(est, "feature_names_out", None) == "one-to-one":
            return FeatureMapping.identity(n_in), ""
        return None, "FunctionTransformer without feature_names_out='one-to-one'"
    if classes & _COLUMNWISE:
        return FeatureMapping.identity(n_in), ""
    if "OneToOneFeatureMixin" in classes:
        # Same names, but values may combine columns (Normalizer and
        # TfidfTransformer scale each row by a norm over all columns).
        return FeatureMapping.all_to_all(n_in, n_in, LineageQuality.MIXED.value), (
            "one output per input by name; values may depend on every input"
        )
    if "ClassNamePrefixFeaturesOutMixin" in classes and names is not None:
        return FeatureMapping.all_to_all(
            n_in, len(names), LineageQuality.MIXED.value
        ), ("every output combines every input by design")
    return None, f"no lineage contract for {type(est).__name__}"


# --------------------------------------------------------------------------- #
# Tracer
# --------------------------------------------------------------------------- #


def _width(data: Any) -> int:
    shape = getattr(data, "shape", None)
    ndim = 2
    if shape is None or len(shape) != ndim:
        raise LineageError(f"a stage returned {type(data).__name__}, not a 2-D matrix")
    return int(shape[1])


def _take_columns(X: Any, indices: list[int]) -> Any:
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, indices]
    return X[:, indices]


def _column_indices(  # ruff: ignore[too-many-return-statements]
    column: Any,
    names: Sequence[str],
) -> list[int]:
    """Resolve a fitted ColumnTransformer column spec to positions."""
    n = len(names)
    index = {name: i for i, name in enumerate(names)}
    if isinstance(column, str):
        return [index[column]]
    if isinstance(column, (int, np.integer)) and not isinstance(column, bool):
        return [int(column) % n]
    if isinstance(column, slice):
        start, stop = column.start, column.stop
        if isinstance(start, str) or isinstance(
            stop, str
        ):  # label slices are inclusive
            first = 0 if start is None else index[start]
            last = n - 1 if stop is None else index[stop]
            return list(range(first, last + 1, column.step or 1))
        return list(range(n))[column]
    values = list(column)
    if not values:
        return []
    if all(isinstance(v, (bool, np.bool_)) for v in values):
        return [i for i, v in enumerate(values) if v]
    if all(isinstance(v, str) for v in values):
        return [index[v] for v in values]
    return [int(v) % n for v in values]


class _Tracer:
    def __init__(self, strict: bool, trace_final: bool = False) -> None:
        self.strict = strict
        self.trace_final = trace_final
        self.nodes: list[FeatureNode] = []
        self.edges: list[LineageEdge] = []
        self.stages: list[StageRecord] = []
        self.stage_inputs: list[tuple[str, tuple[int, ...]]] = []
        self.final_estimator: str | None = None

    def record(self, record: StageRecord, inputs: Sequence[int]) -> None:
        self.stages.append(record)
        self.stage_inputs.append((record.path, tuple(inputs)))

    def new_nodes(self, stage: str, names: Sequence[str]) -> list[int]:
        start = len(self.nodes)
        for position, name in enumerate(names):
            self.nodes.append(FeatureNode(start + position, stage, position, str(name)))
        return list(range(start, start + len(names)))

    def link(
        self, stage: str, inputs: list[int], outputs: list[int], mapping: FeatureMapping
    ) -> None:
        for j, (child, parents) in enumerate(zip(outputs, mapping.parents)):
            quality = mapping.quality_of(j)
            for p in parents:
                if not 0 <= p < len(inputs):
                    raise LineageError(
                        f"stage {stage!r}: parent position {p} is outside the"
                        f" {len(inputs)} inputs"
                    )
                self.edges.append(
                    LineageEdge(inputs[p], child, stage, mapping.relation, quality)
                )

    def name_of(self, node: int) -> str:
        return self.nodes[node].name

    # -- dispatch -------------------------------------------------------------

    def trace(
        self, est: Any, inputs: list[int], path: str, X: Any
    ) -> tuple[list[int], Any]:
        classes = _class_names(est) if not isinstance(est, str) else set()
        if est is None or (isinstance(est, str) and est == "passthrough"):
            return self.passthrough(inputs, path, X)
        if isinstance(est, str):
            raise LineageError(f"stage {path!r}: unsupported step value {est!r}")
        if "Pipeline" in classes and hasattr(est, "steps"):
            return self.pipeline(est, inputs, path, X)
        if "ColumnTransformer" in classes and hasattr(est, "transformers_"):
            return self.column_transformer(est, inputs, path, X)
        if "FeatureUnion" in classes and hasattr(est, "transformer_list"):
            return self.feature_union(est, inputs, path, X)
        return self.leaf(est, inputs, path, X)

    def passthrough(
        self, inputs: list[int], path: str, X: Any
    ) -> tuple[list[int], Any]:
        names = [self.name_of(i) for i in inputs]
        outputs = self.new_nodes(path, names)
        self.link(path, inputs, outputs, FeatureMapping.identity(len(inputs)))
        self.record(
            StageRecord(
                path,
                "passthrough",
                None,
                Relation.ONE_TO_ONE.value,
                LineageQuality.EXACT.value,
                len(inputs),
                len(inputs),
                None if X is None else _width(X),
            ),
            inputs,
        )
        return outputs, X

    def leaf(
        self,
        est: Any,
        inputs: list[int],
        path: str,
        X: Any,
    ) -> tuple[list[int], Any]:
        input_names = [self.name_of(i) for i in inputs]
        names, why_no_names = _names_out(est, input_names)
        mapping, note = _leaf_mapping(est, input_names, names)
        X_out = None
        observed = None
        if X is not None:
            if not hasattr(est, "transform"):
                raise LineageError(
                    f"stage {path!r}: {type(est).__name__} has no transform"
                )
            X_out = est.transform(X)
            observed = _width(X_out)
        if mapping is None:
            n_out = len(names) if names is not None else observed
            if n_out is None:
                raise LineageError(
                    f"stage {path!r}: cannot tell how many features"
                    f" {type(est).__name__} outputs ({note}; {why_no_names}); pass a"
                    " sample X=..., or give the transformer get_feature_names_out or"
                    " get_feature_lineage"
                )
            mapping = FeatureMapping.all_to_all(
                len(inputs), n_out, LineageQuality.UNKNOWN.value
            )
        if names is not None and len(names) != mapping.n_outputs:
            raise LineageError(
                f"stage {path!r}: {type(est).__name__} lineage has {mapping.n_outputs}"
                f" outputs but get_feature_names_out has {len(names)}"
            )
        if observed is not None and observed != mapping.n_outputs:
            raise LineageError(
                f"stage {path!r}: {type(est).__name__} lineage has {mapping.n_outputs}"
                f" outputs but transform returned {observed} columns"
            )
        if names is None:
            names = (
                input_names
                if mapping.relation == Relation.ONE_TO_ONE.value
                else [f"{path}[{j}]" for j in range(mapping.n_outputs)]
            )
        outputs = self.new_nodes(path, names)
        self.link(path, inputs, outputs, mapping)
        self.record(
            StageRecord(
                path,
                type(est).__name__,
                None,
                mapping.relation,
                mapping.quality,
                len(inputs),
                len(outputs),
                observed,
                note,
            ),
            inputs,
        )
        return outputs, X_out

    def pipeline(
        self,
        est: Any,
        inputs: list[int],
        path: str,
        X: Any,
        *,
        top: bool | None = None,
    ) -> tuple[list[int], Any]:
        # ``top``: this pipeline's last step is the model's position (the root
        # pipeline, or a pipeline that is itself the last step of a top one).
        top = path == _ROOT if top is None else top
        current, data = inputs, X
        steps = list(est.steps)
        for position, (name, step) in enumerate(steps):
            last = position == len(steps) - 1
            child = name if path == _ROOT else f"{path}/{name}"
            if (
                last
                and top
                and "Pipeline" in _class_names(step)
                and hasattr(step, "steps")
            ):
                current, data = self.pipeline(step, current, child, data, top=True)
                break
            is_model = not hasattr(step, "transform") or (
                top and hasattr(step, "predict") and not self.trace_final
            )
            if last and step not in (None, "passthrough") and is_model:
                self.final_estimator = type(step).__name__
                break
            current, data = self.trace(step, current, child, data)
        self.record(
            StageRecord(
                path,
                type(est).__name__,
                "pipeline",
                _summary_relation(len(inputs), len(current)),
                self.worst_quality(current),
                len(inputs),
                len(current),
                None if data is None or data is X else _width(data),
            ),
            inputs,
        )
        return current, data

    def column_transformer(
        self,
        est: Any,
        inputs: list[int],
        path: str,
        X: Any,
    ) -> tuple[list[int], Any]:
        input_names = [self.name_of(i) for i in inputs]
        fitted_names = getattr(est, "feature_names_in_", None)
        if fitted_names is not None and list(fitted_names) != input_names:
            raise LineageError(
                f"stage {path!r}: input names differ from feature_names_in_"
            )
        indices_out = est.output_indices_
        branches: list[tuple[str, list[int]]] = []
        for name, trans, column in est.transformers_:
            if isinstance(trans, str) and trans == "drop":
                continue
            positions = _column_indices(column, input_names)
            span = indices_out.get(name, slice(0, 0))
            if not positions or span.stop - span.start == 0:
                continue
            child = name if path == _ROOT else f"{path}/{name}"
            sub_X = (  # ruff: ignore[non-lowercase-variable-in-function]
                None if X is None else _take_columns(X, positions)
            )
            outs, _ = self.trace(trans, [inputs[p] for p in positions], child, sub_X)
            width = span.stop - span.start
            if width != len(outs):
                raise LineageError(
                    f"stage {child!r}: output_indices_ reserves {width} columns but"
                    f" {len(outs)} features were traced"
                )
            branches.append((name, outs))
        branches.sort(key=lambda b: indices_out[b[0]].start)
        return self.concat(est, inputs, path, X, branches, "column_transformer")

    def feature_union(
        self, est: Any, inputs: list[int], path: str, X: Any
    ) -> tuple[list[int], Any]:
        branches = []
        for name, trans in est.transformer_list:
            if isinstance(trans, str) and trans == "drop":
                continue
            child = name if path == _ROOT else f"{path}/{name}"
            outs, _ = self.trace(trans, list(inputs), child, X)
            branches.append((name, outs))
        return self.concat(est, inputs, path, X, branches, "feature_union")

    def concat(  # ruff: ignore[too-many-positional-arguments]
        self,
        est: Any,
        inputs: list[int],
        path: str,
        X: Any,
        branches: list[tuple[str, list[int]]],
        container: str,
    ) -> tuple[list[int], Any]:
        flat = [node for _, outs in branches for node in outs]
        names, _ = _names_out(est, [self.name_of(i) for i in inputs])
        if names is None or len(names) != len(flat):
            verbose = getattr(est, "verbose_feature_names_out", True)
            names = [
                f"{b}__{self.name_of(n)}" if verbose else self.name_of(n)
                for b, outs in branches
                for n in outs
            ]
            if len(set(names)) != len(names):
                duplicated = sorted({n for n in names if names.count(n) > 1})
                raise LineageError(
                    f"stage {path!r}: {type(est).__name__} has no feature names and"
                    f" its branches produce duplicate names {duplicated[:5]}; set"
                    " verbose_feature_names_out=True"
                )
        outputs = self.new_nodes(path, names)
        for child, parent in zip(outputs, flat):
            self.edges.append(
                LineageEdge(
                    parent,
                    child,
                    path,
                    Relation.CONCAT.value,
                    LineageQuality.EXACT.value,
                )
            )
        X_out, observed = None, None
        if X is not None:
            X_out = est.transform(X)
            observed = _width(X_out)
            if observed != len(outputs):
                raise LineageError(
                    f"stage {path!r}: {len(outputs)} features were traced but transform"
                    f" returned {observed} columns"
                )
        self.record(
            StageRecord(
                path,
                type(est).__name__,
                container,
                _summary_relation(len(inputs), len(outputs)),
                self.worst_quality(outputs),
                len(inputs),
                len(outputs),
                observed,
            ),
            inputs,
        )
        return outputs, X_out

    def worst_quality(self, nodes: Iterable[int]) -> str:
        quality = self.node_quality()
        rank = max((quality.get(n, 0) for n in nodes), default=0)
        return next(q for q, r in _QUALITY_RANK.items() if r == rank)

    def node_quality(self) -> dict[int, int]:
        incoming: dict[int, list[LineageEdge]] = {}
        for edge in self.edges:
            incoming.setdefault(edge.child, []).append(edge)
        quality: dict[int, int] = {}
        for node in self.nodes:  # nodes are created in topological order
            q = 0
            for edge in incoming.get(node.id, ()):
                q = max(q, _QUALITY_RANK[edge.quality], quality.get(edge.parent, 0))
            quality[node.id] = q
        return quality


def _quality_name(rank: int) -> str:
    return next(q for q, r in _QUALITY_RANK.items() if r == rank)


def _summary_relation(n_in: int, n_out: int) -> str:
    if n_in == n_out:
        return Relation.ONE_TO_ONE.value
    return Relation.SUBSET.value if n_out < n_in else Relation.EXPAND.value


def trace_pipeline(
    estimator: Any,
    input_features: Sequence[str] | None = None,
    *,
    X: Any = None,
    trace_final: bool = False,
) -> FeatureLineage:
    """
    Trace the feature lineage of a fitted estimator.

    Parameters
    ----------
    estimator : fitted estimator
        A ``Pipeline``, ``ColumnTransformer``, ``FeatureUnion`` or single
        transformer. A pipeline's final step without ``transform`` (the
        model) is recorded by name and not traced.
    input_features : sequence of str or None, default=None
        Raw input names; defaults to ``estimator.feature_names_in_`` (set when
        it was fitted on a DataFrame), else ``x0, x1, ...``.
    X : DataFrame or array or None, default=None
        Optional sample of training-like rows. Every stage is then run on it
        and its actual output width checked.
    trace_final : bool, default=False
        A top-level pipeline's last step with ``predict`` is the model and is
        not traced, even when it also has ``transform`` (LDA, KMeans). True
        traces it as a transformer.

    Returns
    -------
    FeatureLineage
        The traced graph.

    Raises
    ------
    LineageError
        If a stage's traced count disagrees with its feature names or its
        actual output, or a stage's width cannot be determined.
    sklearn.exceptions.NotFittedError
        If the estimator is not fitted.

    See Also
    --------
    FeatureLineage : Query the result.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.compose import ColumnTransformer
    >>> from sklearn.preprocessing import OneHotEncoder, StandardScaler
    >>> from scikitplot.feature_selection import trace_pipeline
    >>> X = pd.DataFrame({"age": [20.0, 30.0, 40.0, 50.0], "city": list("abab")})
    >>> ct = ColumnTransformer(
    ...     [("num", StandardScaler(), ["age"]), ("cat", OneHotEncoder(), ["city"])]
    ... ).fit(X)
    >>> lineage = trace_pipeline(ct, X=X)
    >>> lineage.output_names
    ['num__age', 'cat__city_a', 'cat__city_b']
    >>> lineage.origins("cat__city_b")
    ('city',)
    >>> lineage.attribute([0.2, 0.3, 0.5]).by_input.to_dict()
    {'age': 0.2, 'city': 0.8}
    """
    from sklearn.utils.validation import check_is_fitted  # noqa: PLC0415

    if not isinstance(estimator, str):
        check_is_fitted(estimator)
    names = input_features
    if names is None:
        names = getattr(estimator, "feature_names_in_", None)
    if names is None:
        n = getattr(estimator, "n_features_in_", None)
        if n is None:
            raise LineageError(
                "pass input_features: the estimator records no input count"
            )
        names = [f"x{i}" for i in range(int(n))]
    names = [str(n) for n in names]
    if len(set(names)) != len(names):
        raise LineageError("input feature names must be unique")
    fitted = getattr(estimator, "feature_names_in_", None)
    if fitted is not None and list(fitted) != names:
        raise LineageError(
            "input_features differ from the estimator's feature_names_in_"
        )
    n_fitted = getattr(estimator, "n_features_in_", None)
    if n_fitted is not None and int(n_fitted) != len(names):
        raise LineageError(
            f"{len(names)} input names given but the estimator was fitted on {n_fitted}"
        )
    if X is not None and _width(X) != len(names):
        raise LineageError(
            f"X has {_width(X)} columns but there are {len(names)} inputs"
        )
    tracer = _Tracer(strict=True, trace_final=bool(trace_final))
    inputs = tracer.new_nodes(_INPUT, names)
    tracer.record(
        StageRecord(
            _INPUT,
            _INPUT,
            None,
            Relation.INPUT.value,
            LineageQuality.EXACT.value,
            len(names),
            len(names),
            None if X is None else _width(X),
        ),
        (),
    )
    outputs, _ = tracer.trace(estimator, inputs, _ROOT, X)
    return FeatureLineage(
        nodes=tuple(tracer.nodes),
        edges=tuple(tracer.edges),
        stages=tuple(tracer.stages),
        stage_inputs=tuple(tracer.stage_inputs),
        input_ids=tuple(inputs),
        output_ids=tuple(outputs),
        estimator=type(estimator).__name__,
        final_estimator=tracer.final_estimator,
    )


# --------------------------------------------------------------------------- #
# Result object
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Record contracts of the persisted lineage (Rule 106)
# --------------------------------------------------------------------------- #

_INT, _STR = "int", "str"
_NODE_FIELDS = {"id": _INT, "stage": _STR, "position": _INT, "name": _STR}
_EDGE_FIELDS = {
    "parent": _INT,
    "child": _INT,
    "stage": _STR,
    "relation": _STR,
    "quality": _STR,
}
_STAGE_FIELDS = {
    "path": _STR,
    "estimator": _STR,
    "container": "str?",
    "relation": _STR,
    "quality": _STR,
    "n_in": _INT,
    "n_out": _INT,
    "observed_width": "int?",
    "note": "str=",
}


def _list(value: Any, field: str) -> list:
    if not isinstance(value, list):
        raise ValueError(  # ruff: ignore[type-check-without-type-error]
            f"lineage {field} must be a list",
        )
    return value


def _record(row: Any, fields: Mapping[str, str], what: str) -> dict[str, Any]:
    """Check one record: exact keys, ``int`` (not bool) or ``str`` values."""
    if not isinstance(row, Mapping):
        raise ValueError(  # ruff: ignore[type-check-without-type-error]
            f"a lineage {what} record must be a mapping, got {row!r}",
        )
    unknown = set(row) - set(fields)
    if unknown:
        raise ValueError(f"lineage {what} record has unknown keys {sorted(unknown)}")
    out: dict[str, Any] = {}
    for name, kind in fields.items():
        optional = kind.endswith(("?", "="))
        if name not in row:
            if not optional:
                raise ValueError(f"lineage {what} record lacks {name!r}")
            out[name] = "" if kind.endswith("=") else None
            continue
        value = row[name]
        if value is None and kind.endswith("?"):
            out[name] = None
            continue
        base = kind.rstrip("?=")
        ok = (
            isinstance(value, str)
            if base == _STR
            else isinstance(value, int) and not isinstance(value, bool) and value >= 0
        )
        if not ok:
            raise ValueError(f"lineage {what}.{name} must be a {base}, got {value!r}")
        out[name] = value
    return out


def _ids(value: Any, n_nodes: int, field: str) -> tuple[int, ...]:
    ids = _list(value, field)
    for i in ids:
        if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n_nodes:
            raise ValueError(f"lineage {field} holds an invalid node id {i!r}")
    return tuple(ids)


class Attribution:
    """
    Values of transformed features summed back onto raw input features.

    Attributes
    ----------
    by_input : pandas.Series or pandas.DataFrame
        For 1-D values, a Series indexed by input name; for 2-D values
        (``n_samples x n_outputs``, such as SHAP values), a DataFrame with one
        column per input. Only exactly attributable outputs contribute.
    unattributed : pandas.DataFrame
        One row per output that could not be attributed to a single input,
        with ``output``, ``origins`` (tuple of input names), ``reason``
        (``multiple_origins``, ``mixed_lineage``, ``unknown_lineage``,
        ``no_origin``) and, for 1-D values, ``value``.
    how : str
        Aggregation used.

    Notes
    -----
    **User note.** Summing is exact for additive quantities: one-hot SHAP
    values, gain-based importances, coefficients' absolute contributions. It
    is a convention, not a theorem, for permutation importances of correlated
    dummies; prefer permuting the raw column for those.
    """

    def __init__(
        self,
        by_input: Any,
        unattributed: pd.DataFrame,
        how: str,
    ) -> None:  # noqa: D107
        self.by_input = by_input
        self.unattributed = unattributed
        self.how = how

    def __repr__(self) -> str:  # noqa: D105
        n = (
            len(self.by_input.columns)
            if isinstance(self.by_input, pd.DataFrame)
            else len(self.by_input)
        )
        return f"Attribution(n_inputs={n}, n_unattributed={len(self.unattributed)}, how={self.how!r})"


_AGGREGATIONS = ("sum", "abs_sum", "mean", "max")


@dataclasses.dataclass(frozen=True)
class FeatureLineage:
    """
    Traced feature graph of a fitted pipeline.

    Parameters
    ----------
    nodes, edges, stages : tuples
        Graph records (see :class:`FeatureNode`, :class:`LineageEdge`,
        :class:`StageRecord`).
    stage_inputs : tuple of (str, tuple of int)
        Input node ids of every stage, in the order of ``stages``.
    input_ids, output_ids : tuple of int
        Node ids of the raw inputs and of the final outputs (model inputs).
    estimator : str
        Class name of the traced object.
    final_estimator : str or None
        Class name of a pipeline's final, non-transformer step.

    See Also
    --------
    trace_pipeline : Build a lineage.
    """

    nodes: tuple[FeatureNode, ...]
    edges: tuple[LineageEdge, ...]
    stages: tuple[StageRecord, ...]
    stage_inputs: tuple[tuple[str, tuple[int, ...]], ...]
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    estimator: str
    final_estimator: str | None = None
    _rank: dict = dataclasses.field(init=False, repr=False, compare=False, hash=False)
    _origins: dict = dataclasses.field(
        init=False, repr=False, compare=False, hash=False
    )
    _parents: dict = dataclasses.field(
        init=False, repr=False, compare=False, hash=False
    )
    _children: dict = dataclasses.field(
        init=False, repr=False, compare=False, hash=False
    )
    _lookup: dict = dataclasses.field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:  # noqa: D105
        parents: dict[int, list[LineageEdge]] = {}
        children: dict[int, list[LineageEdge]] = {}
        for edge in self.edges:
            parents.setdefault(edge.child, []).append(edge)
            children.setdefault(edge.parent, []).append(edge)
        lookup: dict[tuple[str, str], list[int]] = {}
        for node in self.nodes:
            lookup.setdefault((node.stage, node.name), []).append(node.id)
        for name in {self.nodes[i].name for i in self.output_ids}:
            lookup[(_OUTPUT, name)] = [
                i for i in self.output_ids if self.nodes[i].name == name
            ]
        inputs = set(self.input_ids)
        rank: dict[int, int] = {}
        origins: dict[int, frozenset] = {}
        for node in self.nodes:  # every edge points from a lower id to a higher one
            incoming = parents.get(node.id, ())
            if any(e.parent >= node.id for e in incoming):
                raise ValueError("lineage edges must point from earlier to later nodes")
            rank[node.id] = max(
                (max(_QUALITY_RANK[e.quality], rank[e.parent]) for e in incoming),
                default=0,
            )
            own = frozenset((node.id,)) if node.id in inputs else frozenset()
            origins[node.id] = own.union(*(origins[e.parent] for e in incoming))
        object.__setattr__(self, "_rank", rank)
        object.__setattr__(self, "_origins", origins)
        object.__setattr__(self, "_parents", parents)
        object.__setattr__(self, "_children", children)
        object.__setattr__(self, "_lookup", lookup)

    def __getstate__(self) -> dict[str, Any]:  # noqa: D105
        return {
            f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.init
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:  # noqa: D105
        for key, value in state.items():
            object.__setattr__(self, key, value)
        self.__post_init__()

    # -- basic views ----------------------------------------------------------

    @property
    def input_names(self) -> list[str]:
        """List of str: Raw input names."""
        return [self.nodes[i].name for i in self.input_ids]

    @property
    def output_names(self) -> list[str]:
        """List of str: Final output names (what the model sees), in column order."""
        return [self.nodes[i].name for i in self.output_ids]

    @property
    def stage_paths(self) -> list[str]:
        """List of str: Paths of every traced stage, in trace order."""
        return [s.path for s in self.stages]

    def _node(self, name: str, stage: str) -> int:
        stage = _INPUT if stage == _INPUT else stage
        ids = self._lookup.get((stage, name))
        if not ids:
            raise KeyError(f"no feature {name!r} at stage {stage!r}")
        if len(ids) > 1:
            raise KeyError(f"feature name {name!r} is ambiguous at stage {stage!r}")
        return ids[0]

    def _stage_nodes(self, stage: str) -> list[int]:
        if stage == _OUTPUT:
            return list(self.output_ids)
        nodes = [n.id for n in self.nodes if n.stage == stage]
        if not nodes and stage not in self.stage_paths:
            raise KeyError(f"no stage {stage!r}; stages: {self.stage_paths}")
        return nodes

    def _ancestors(self, node: int) -> set[int]:
        seen: set[int] = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for edge in self._parents.get(current, ()):
                if edge.parent not in seen:
                    seen.add(edge.parent)
                    stack.append(edge.parent)
        return seen

    def _descendants(self, node: int) -> set[int]:
        seen: set[int] = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for edge in self._children.get(current, ()):
                if edge.child not in seen:
                    seen.add(edge.child)
                    stack.append(edge.child)
        return seen

    # -- navigation -----------------------------------------------------------

    def parents(self, name: str, stage: str = _OUTPUT) -> list[tuple[str, str]]:
        """
        Return the immediate predecessors of a feature.

        Parameters
        ----------
        name : str
            Feature name at ``stage``.
        stage : str, default="output"
            Stage path, ``"input"`` or ``"output"``.

        Returns
        -------
        list of (stage, name)
            Predecessor features, in input order.
        """
        node = self._node(name, stage)
        parents = sorted({e.parent for e in self._parents.get(node, ())})
        return [(self.nodes[p].stage, self.nodes[p].name) for p in parents]

    def children(self, name: str, stage: str = _INPUT) -> list[tuple[str, str]]:
        """
        Return the immediate successors of a feature.

        Parameters
        ----------
        name : str
            Feature name at ``stage``.
        stage : str, default="input"
            Stage path, ``"input"`` or ``"output"``.

        Returns
        -------
        list of (stage, name)
            Successor features.
        """
        node = self._node(name, stage)
        children = sorted({e.child for e in self._children.get(node, ())})
        return [(self.nodes[c].stage, self.nodes[c].name) for c in children]

    def origins(
        self, name: str, stage: str = _OUTPUT, *, to: str = _INPUT
    ) -> tuple[str, ...]:
        """
        Return the features at stage ``to`` that ``name`` is computed from.

        Parameters
        ----------
        name : str
            Feature name at ``stage``.
        stage : str, default="output"
            Stage of ``name``.
        to : str, default="input"
            Stage to walk back to.

        Returns
        -------
        tuple of str
            Ancestor names at ``to``, in that stage's column order.
        """
        node = self._node(name, stage)
        ancestors = self._origins[node] if to == _INPUT else self._ancestors(node)
        return tuple(
            self.nodes[i].name for i in self._stage_nodes(to) if i in ancestors
        )

    def descendants(
        self, name: str, stage: str = _INPUT, *, to: str = _OUTPUT
    ) -> tuple[str, ...]:
        """
        Return the features at stage ``to`` computed from ``name``.

        Parameters
        ----------
        name : str
            Feature name at ``stage``.
        stage : str, default="input"
            Stage of ``name``.
        to : str, default="output"
            Stage to walk forward to.

        Returns
        -------
        tuple of str
            Descendant names at ``to``, in column order.
        """
        found = self._descendants(self._node(name, stage))
        return tuple(self.nodes[i].name for i in self._stage_nodes(to) if i in found)

    def quality(self, name: str, stage: str = _OUTPUT) -> str:
        """
        Return the weakest lineage quality on any path from the inputs to ``name``.

        Parameters
        ----------
        name : str
            Feature name.
        stage : str, default="output"
            Its stage.

        Returns
        -------
        str
            ``"exact"``, ``"mixed"`` or ``"unknown"``.
        """
        return _quality_name(self._rank[self._node(name, stage)])

    def history(self, name: str) -> pd.DataFrame:
        """
        Return every leaf stage that touched the descendants of input ``name``.

        Parameters
        ----------
        name : str
            Raw input name.

        Returns
        -------
        pandas.DataFrame
            Columns ``stage``, ``estimator``, ``relation``, ``quality`` and
            ``features`` (the names this input became at that stage), in
            pipeline order. Containers (pipelines, column transformers) are
            listed with ``estimator`` set to their class.
        """
        found = self._descendants(self._node(name, _INPUT))
        by_stage: dict[str, list[str]] = {}
        for node in sorted(found):
            by_stage.setdefault(self.nodes[node].stage, []).append(
                self.nodes[node].name
            )
        records = {s.path: s for s in self.stages}
        rows = []
        for path in self.stage_paths:
            if path in by_stage:
                record = records[path]
                rows.append(
                    {
                        "stage": path,
                        "estimator": record.estimator,
                        "relation": record.relation,
                        "quality": record.quality,
                        "features": tuple(by_stage[path]),
                    }
                )
        return pd.DataFrame(
            rows, columns=["stage", "estimator", "relation", "quality", "features"]
        )

    def treatments(self) -> pd.DataFrame:
        """
        Return, per raw input, the chain of leaf transformers applied to it.

        Returns
        -------
        pandas.DataFrame
            Indexed by input name with ``steps`` (tuple of ``"path:Class"``),
            ``n_outputs`` (model inputs derived from it) and ``status``
            (``"kept"`` or ``"dropped"``).
        """
        leaves = {
            s.path: s for s in self.stages if s.container is None and s.path != _INPUT
        }
        outputs = set(self.output_ids)
        rows = []
        for node in self.input_ids:
            found = self._descendants(node)
            stages = [
                f"{path}:{leaves[path].estimator}"
                for path in self.stage_paths
                if path in leaves and any(self.nodes[i].stage == path for i in found)
            ]
            n_out = len(found & outputs)
            rows.append(
                {
                    "feature": self.nodes[node].name,
                    "steps": tuple(stages),
                    "n_outputs": n_out,
                    "status": "kept" if n_out else "dropped",
                }
            )
        return pd.DataFrame(rows).set_index("feature")

    def dropped(self) -> dict[str, str]:
        """
        Return the raw inputs that reach no output, with the stage that removed them.

        Returns
        -------
        dict of str to str
            Input name to the path of the stage that received its last
            surviving feature and did not pass it on (a selector, or a
            ``ColumnTransformer`` whose branches and remainder ignore the
            column).
        """
        outputs = set(self.output_ids)
        order = {p: i for i, (p, _) in enumerate(self.stage_inputs)}
        leaf = {s.path: s.container is None for s in self.stages}
        result = {}
        for node in self.input_ids:
            found = self._descendants(node) | {node}
            if found & outputs:
                continue
            terminals = [n for n in found if not self._children.get(n)]
            stops = []
            for terminal in terminals:
                consumers = [p for p, ins in self.stage_inputs if terminal in ins]
                # a leaf that received it and dropped it; else the innermost
                # container (containers are recorded after their contents)
                consumers.sort(key=lambda p: (not leaf.get(p, False), order[p]))
                if consumers:
                    stops.append(consumers[0])
            result[self.nodes[node].name] = (
                max(stops, key=lambda p: order[p]) if stops else _ROOT
            )
        return result

    def stage_summary(self) -> pd.DataFrame:
        """
        Return one row per traced stage.

        Returns
        -------
        pandas.DataFrame
            Columns ``path``, ``estimator``, ``container``, ``relation``,
            ``quality``, ``n_in``, ``n_out``, ``delta``, ``observed_width``,
            ``note``.
        """
        frame = pd.DataFrame([dataclasses.asdict(s) for s in self.stages])
        frame.insert(
            frame.columns.get_loc("n_out") + 1, "delta", frame["n_out"] - frame["n_in"]
        )
        return frame

    def to_frame(self) -> pd.DataFrame:
        """
        Return every edge as a row.

        Returns
        -------
        pandas.DataFrame
            Columns ``stage``, ``parent_stage``, ``parent``, ``child``,
            ``relation``, ``quality``.
        """
        rows = [
            {
                "stage": e.stage,
                "parent_stage": self.nodes[e.parent].stage,
                "parent": self.nodes[e.parent].name,
                "child": self.nodes[e.child].name,
                "relation": e.relation,
                "quality": e.quality,
            }
            for e in self.edges
        ]
        return pd.DataFrame(
            rows,
            columns=["stage", "parent_stage", "parent", "child", "relation", "quality"],
        )

    # -- attribution ----------------------------------------------------------

    def attribute(self, values: Any, *, how: str = "sum") -> Attribution:
        """
        Aggregate per-output values (importances, SHAP values) onto raw inputs.

        Parameters
        ----------
        values : array-like, pandas.Series or mapping
            1-D: one value per output (array aligned with
            :attr:`output_names`, or a Series/mapping keyed by output name).
            2-D array or DataFrame: ``n_samples x n_outputs`` (a DataFrame is
            aligned by column name).
        how : {"sum", "abs_sum", "mean", "max"}, default="sum"
            Aggregation over the outputs of one input.

        Returns
        -------
        Attribution
            Attributed and unattributed parts.

        Raises
        ------
        ValueError
            If the values do not align with the outputs.

        Notes
        -----
        An output is attributed only when it has exactly one raw origin and
        every edge on its paths is ``exact``. Polynomial interactions,
        PCA components and unknown steps are reported in
        :attr:`Attribution.unattributed` instead of being split by a guess.
        """
        if how not in _AGGREGATIONS:
            raise ValueError(f"how={how!r} is not one of {_AGGREGATIONS}")
        outputs = self.output_names
        matrix, two_d = self._align(values, outputs)
        inputs = self.input_names
        groups: dict[str, list[int]] = {}
        rows = []
        for j, node in enumerate(self.output_ids):
            origin_set = self._origins[node]
            origins = tuple(
                self.nodes[i].name for i in self.input_ids if i in origin_set
            )
            quality = _quality_name(self._rank[node])
            if len(origins) == 1 and quality == LineageQuality.EXACT.value:
                groups.setdefault(origins[0], []).append(j)
                continue
            reason = (
                "no_origin"
                if not origins
                else (
                    "unknown_lineage"
                    if quality == LineageQuality.UNKNOWN.value
                    else (
                        "mixed_lineage"
                        if quality == LineageQuality.MIXED.value
                        else "multiple_origins"
                    )
                )
            )
            row = {"output": outputs[j], "origins": origins, "reason": reason}
            if not two_d:
                row["value"] = float(matrix[0, j])
            rows.append(row)
        reduce = {
            "sum": np.sum,
            "abs_sum": lambda a, axis: np.sum(np.abs(a), axis=axis),
            "mean": np.mean,
            "max": np.max,
        }[how]
        ordered = [name for name in inputs if name in groups]
        data = {name: reduce(matrix[:, groups[name]], axis=1) for name in ordered}
        columns = ["output", "origins", "reason"] + ([] if two_d else ["value"])
        unattributed = pd.DataFrame(rows, columns=columns)
        if two_d:
            index = values.index if isinstance(values, pd.DataFrame) else None
            by_input = pd.DataFrame(data, columns=ordered, index=index)
        else:
            by_input = pd.Series(
                {k: float(v[0]) for k, v in data.items()}, dtype="float64"
            )
            by_input = by_input.reindex(ordered)
        return Attribution(by_input, unattributed, how)

    @staticmethod
    def _align(values: Any, outputs: list[str]) -> tuple[np.ndarray, bool]:
        by_name = isinstance(values, (pd.DataFrame, pd.Series, Mapping))
        if by_name and len(set(outputs)) != len(outputs):
            raise ValueError(
                "the lineage outputs have duplicate names; pass values as an array"
                " aligned by position"
            )
        if isinstance(values, pd.Series) and values.index.has_duplicates:
            raise ValueError("the values' index has duplicate names")
        if isinstance(values, pd.DataFrame):
            missing = [o for o in outputs if o not in values.columns]
            if missing or len(values.columns) != len(outputs):
                raise ValueError(
                    f"DataFrame columns do not match the lineage outputs (missing"
                    f" {missing[:5]})"
                )
            return values[outputs].to_numpy(dtype="float64"), True
        if isinstance(values, (pd.Series, Mapping)):
            keys = list(
                values.index if isinstance(values, pd.Series) else values.keys()
            )
            if sorted(map(str, keys)) != sorted(outputs) or len(keys) != len(outputs):
                raise ValueError("keys of the values do not match the lineage outputs")
            lookup = values.to_dict() if isinstance(values, pd.Series) else dict(values)
            return (
                np.asarray([[float(lookup[o]) for o in outputs]], dtype="float64"),
                False,
            )
        array = np.asarray(values, dtype="float64")
        ndim = 2
        if array.ndim == 1:
            array = array[None, :]
            two_d = False
        elif array.ndim == ndim:
            two_d = True
        else:
            raise ValueError(f"values must be 1-D or 2-D, got {array.ndim}-D")
        if array.shape[1] != len(outputs):
            raise ValueError(
                f"values have {array.shape[1]} columns but the lineage has"
                f" {len(outputs)} outputs"
            )
        return array, two_d

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Return the persisted form.

        Returns
        -------
        dict
            A tree of the shared data model with ``schema_version``.
        """
        return {
            "schema_version": LINEAGE_VERSION,
            "artifact": "feature_lineage",
            "estimator": self.estimator,
            "final_estimator": self.final_estimator,
            "input_ids": list(self.input_ids),
            "output_ids": list(self.output_ids),
            "nodes": [dataclasses.asdict(n) for n in self.nodes],
            "edges": [dataclasses.asdict(e) for e in self.edges],
            "stages": [dataclasses.asdict(s) for s in self.stages],
            "stage_inputs": [[p, list(ids)] for p, ids in self.stage_inputs],
        }

    @classmethod
    def from_dict(  # ruff: ignore[too-many-branches]
        cls,
        data: Mapping[str, Any],
    ) -> FeatureLineage:
        """
        Rebuild a lineage from :meth:`to_dict` output.

        Parameters
        ----------
        data : mapping
            Stored form.

        Returns
        -------
        FeatureLineage
            The lineage.

        Raises
        ------
        IncompatibleSchemaError
            For another generation or artefact type.
        ValueError
            For an off-contract record.
        """
        if not isinstance(data, Mapping) or data.get("artifact") != "feature_lineage":
            raise IncompatibleSchemaError("not a feature lineage document")
        version = data.get("schema_version")
        if isinstance(version, bool) or version != LINEAGE_VERSION:
            raise IncompatibleSchemaError(
                f"lineage schema_version {version!r} is not supported"
            )
        required = {
            "schema_version",
            "artifact",
            "estimator",
            "input_ids",
            "output_ids",
            "nodes",
            "edges",
            "stages",
            "stage_inputs",
        }
        allowed = required | {"final_estimator"}  # None is omitted by TOML
        if not required <= set(data) <= allowed:
            raise ValueError(f"off-contract lineage document: keys {sorted(data)}")
        nodes = tuple(
            FeatureNode(**_record(row, _NODE_FIELDS, "node"))
            for row in _list(data["nodes"], "nodes")
        )
        if [n.id for n in nodes] != list(range(len(nodes))):
            raise ValueError("lineage node ids must be 0..n-1 in order")
        n_nodes = len(nodes)
        edges = []
        for row in _list(data["edges"], "edges"):
            edge = LineageEdge(**_record(row, _EDGE_FIELDS, "edge"))
            if not (0 <= edge.parent < n_nodes and 0 <= edge.child < n_nodes):
                raise ValueError("a lineage edge points outside the nodes")
            if edge.parent >= edge.child:
                raise ValueError("a lineage edge must point to a later node")
            Relation(edge.relation)
            LineageQuality(edge.quality)
            edges.append(edge)
        stages = tuple(
            StageRecord(**_record(row, _STAGE_FIELDS, "stage"))
            for row in _list(data["stages"], "stages")
        )
        for stage in stages:
            Relation(stage.relation)
            LineageQuality(stage.quality)
        stage_inputs = []
        len_item = 2
        for item in _list(data["stage_inputs"], "stage_inputs"):
            if (
                not isinstance(item, list)
                or len(item) != len_item
                or not isinstance(item[0], str)
            ):
                raise ValueError("each stage_inputs entry is [path, [node ids]]")
            stage_inputs.append((item[0], _ids(item[1], n_nodes, "stage_inputs")))
        if [p for p, _ in stage_inputs] != [st.path for st in stages]:
            raise ValueError("stage_inputs must follow the order of stages")
        input_ids = _ids(data["input_ids"], n_nodes, "input_ids")
        output_ids = _ids(data["output_ids"], n_nodes, "output_ids")
        if any(nodes[i].stage != _INPUT for i in input_ids):
            raise ValueError("input_ids must name nodes of the 'input' stage")
        if len(set(output_ids)) != len(output_ids):
            raise ValueError("output_ids must be distinct")
        estimator = data["estimator"]
        final = data.get("final_estimator")
        if not isinstance(estimator, str) or not (
            final is None or isinstance(final, str)
        ):
            raise ValueError("estimator and final_estimator must be strings")
        return cls(
            nodes=nodes,
            edges=tuple(edges),
            stages=stages,
            stage_inputs=tuple(stage_inputs),
            input_ids=input_ids,
            output_ids=output_ids,
            estimator=estimator,
            final_estimator=final,
        )

    def dumps(self, format: str = "json") -> str:  # noqa: A002
        """
        Serialise the lineage; ``"jsonl"`` writes one edge record per line.

        Parameters
        ----------
        format : {"json", "yaml", "toml", "jsonl"}, default="json"
            ``json``/``yaml``/``toml`` hold the complete, reloadable document
            (verified by reading it back); ``jsonl``
            holds the edge table (:meth:`to_frame`) for log pipelines.

        Returns
        -------
        str
            The document.
        """
        if format == "jsonl":
            return _serialize.dumps(self.to_frame().to_dict(orient="records"), "jsonl")
        text = _serialize.dumps(self.to_dict(), format)
        if FeatureLineage.loads(text, format) != self:
            raise ValueError(f"lineage does not round-trip through {format}")
        return text

    @classmethod
    def loads(cls, text: str, format: str = "json") -> FeatureLineage:  # noqa: A002
        """
        Parse a document written by :meth:`dumps` with ``json`` or ``yaml``.

        Parameters
        ----------
        text : str
            The document.
        format : {"json", "yaml"}, default="json"
            Format.

        Returns
        -------
        FeatureLineage
            The lineage.
        """
        if format == "jsonl":
            raise ValueError("a jsonl edge table cannot be loaded back as a lineage")
        return cls.from_dict(_serialize.loads(text, format))
