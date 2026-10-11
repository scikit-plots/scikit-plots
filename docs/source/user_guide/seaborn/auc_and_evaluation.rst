.. _seaborn-auc-and-evaluation:

AUC and evaluation
==================

:func:`aucplot` visualizes AUC-oriented model results. :func:`evalplot` renders
classification/evaluation views such as confusion-matrix/report-style output.

Before plotting, decide which values are truth labels, prediction labels and
scores/probabilities. A plotting wrapper cannot infer that a probability column
belongs to the wrong class.

When ``hue`` is used, verify that grouping does not accidentally mix data from
different evaluation splits or models with incompatible score semantics.
