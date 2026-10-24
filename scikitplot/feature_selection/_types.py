# scikitplot/feature_selection/_types.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Closed vocabularies of :mod:`scikitplot.feature_selection`.

Every enumeration here is a :class:`str` subclass, so a member compares equal
to its value (``AssessmentState.TRUE == "true"``), serialises as plain text and
can be given as a string anywhere the API accepts the enumeration.

Notes
-----
**User note.** You rarely need to import these: pass the string values
(``kind="numeric_continuous"``) and they are validated against the members
listed here.

**Developer note.** :func:`coerce_enum` is the single place where a string
becomes a member, so every public entry point refuses the same misspellings
with the same message (Rule 84).
"""

from __future__ import annotations

import enum
from typing import Any, TypeVar

__all__ = [
    "AssessmentState",
    "DtypeFamily",
    "FeatureKind",
    "FeatureRole",
    "LineageQuality",
    "Relation",
]

_E = TypeVar("_E", bound=enum.Enum)


class _StrEnum(str, enum.Enum):
    """A string-valued enumeration whose ``str()`` is its value."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class AssessmentState(_StrEnum):
    """
    Outcome of one assessment of one feature.

    ``UNKNOWN`` means the evidence was insufficient (too few finite values,
    unhashable cells); ``NOT_APPLICABLE`` means the question has no meaning for
    the feature (skewness of a string column). Neither is ``FALSE``.
    """

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class DtypeFamily(_StrEnum):
    """Physical storage family of a column, read from its pandas dtype and values."""

    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    DATETIME = "datetime"
    TIMEDELTA = "timedelta"
    CATEGORICAL = "categorical"
    STRING = "string"
    OBJECT = "object"


class FeatureKind(_StrEnum):
    """
    Semantic kind of a feature: what analyses and treatments make sense.

    Inferred from the dtype family plus the ``binary`` and ``discrete``
    assessments, or set explicitly in :class:`FeaturePolicy.kinds`. ``TEXT``
    is never inferred: free text cannot be told from a nominal string column
    by cardinality alone, so it is an explicit declaration only.
    """

    BINARY = "binary"
    NUMERIC_DISCRETE = "numeric_discrete"
    NUMERIC_CONTINUOUS = "numeric_continuous"
    CATEGORICAL_NOMINAL = "categorical_nominal"
    CATEGORICAL_ORDINAL = "categorical_ordinal"
    DATETIME = "datetime"
    TIMEDELTA = "timedelta"
    TEXT = "text"
    OTHER = "other"
    UNKNOWN = "unknown"


class FeatureRole(_StrEnum):
    """Role of a column in a modelling task; only ``FEATURE`` columns are model inputs."""

    FEATURE = "feature"
    TARGET = "target"
    DROP = "drop"
    IDENTIFIER = "identifier"
    WEIGHT = "weight"
    GROUP = "group"
    TIME = "time"


class Relation(_StrEnum):
    """How the outputs of one pipeline stage relate to its inputs."""

    INPUT = "input"
    ONE_TO_ONE = "one_to_one"
    SUBSET = "subset"
    EXPAND = "expand"
    COMBINE = "combine"
    MIXED = "mixed"
    CONCAT = "concat"
    UNKNOWN = "unknown"


class LineageQuality(_StrEnum):
    """
    How strongly a lineage edge is supported.

    ``EXACT``: derived from fitted attributes or a class contract of the
    estimator (``get_support``, ``categories_``, ``OneToOneFeatureMixin``).
    ``MIXED``: every output depends on every input by design (PCA, SVD); the
    ancestry is exact but no single raw feature owns an output.
    ``UNKNOWN``: the estimator exposes no mechanical mapping; parents are
    listed conservatively as all inputs and nothing is attributed.
    """

    EXACT = "exact"
    MIXED = "mixed"
    UNKNOWN = "unknown"


def coerce_enum(value: Any, enum_type: type[_E], *, field: str) -> _E:
    """
    Return the member of ``enum_type`` named by ``value``.

    Parameters
    ----------
    value : str or enum member
        The value to convert. Matching is exact and case-sensitive.
    enum_type : type
        The enumeration class.
    field : str
        Name used in the error message.

    Returns
    -------
    enum member
        The matching member.

    Raises
    ------
    ValueError
        If ``value`` is not a member value; the message lists the valid values.
    TypeError
        If ``value`` is neither a string nor a member of ``enum_type``.
    """
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise TypeError(
            f"{field} must be a string, got {type(value).__name__}: {value!r}"
        )
    try:
        return enum_type(value)
    except ValueError:
        valid = ", ".join(repr(m.value) for m in enum_type)
        raise ValueError(f"{field}={value!r} is not one of {valid}") from None
