.. _preprocessing-get-dummies:

GetDummies
==========

:class:`GetDummies` wraps dummy/indicator expansion in an estimator-style
object. Important choices include selected columns, category separator,
prefix/column naming, dropped level, sparse output, dtype and unknown-category
policy.

Naming is part of downstream schema. If encoded feature names are persisted or
consumed by a model/service, treat separator/prefix/drop policy as versioned
configuration rather than presentation.

The maintenance review records a parameter-validation mismatch in this path.
Validate non-default parameter combinations in your own pipeline until that
finding is closed.
