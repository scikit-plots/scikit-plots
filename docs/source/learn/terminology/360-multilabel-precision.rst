:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-multilabel-precision:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Multilabel Precision</b></div>`

======================
Multilabel Precision
======================

*Precision when each instance may carry several labels at once.*

What it is
----------

**Multilabel precision** is precision when each sample can carry **several labels at once** — the labels are
**not** mutually exclusive. Targets are an **indicator matrix** (cell [i, j] = 1 if sample i has label j),
and precision is computed **per label**, each label a binary problem.

How it's aggregated
-------------------

As with multiclass, per-label precisions are combined by **micro**, **macro** or **weighted** averaging —
but multilabel adds a distinctive **'samples'** average, which computes precision **per instance** (across
that sample's labels) and averages over samples.

Why it differs
--------------

Unlike multiclass, where exactly one class is correct, multilabel labels **co-occur**, so a prediction can
be **partly** right (some labels correct, others missed). The averaging choice — especially **samples** vs
**micro** — decides whether you're scoring per-label or per-example accuracy.

----

**Mind map — connected ideas**

   :doc:`Multiclass Precision <359-multiclass-precision>` · :doc:`Per-class Precision (sometimes called class-wise precision) <358-per-class-precision-sometimes-called-class-wise>` · :doc:`One-vs-Rest (OvR) <310-one-vs-rest-ovr>` · :doc:`Micro Averaging <369-micro-averaging>` · :doc:`Precision (a.k.a. Positive Predictive Value, PPV) <429-precision-a-k-a-positive-predictive-value-ppv>` · :doc:`F1-score <363-f1-score>`

----

**More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Multilabel Precision <https://insightful-data-lab.com/2025/08/20/multilabel-precision/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
