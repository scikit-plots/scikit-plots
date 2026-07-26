:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-macro-f1:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Macro F1</b></div>`

==========
Macro F1
==========

*The unweighted mean of per-class F1 scores; exposes weak performance on rare classes.*

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

F1 is the harmonic mean of precision and recall, high only when both are high:

.. math::

   F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}
              {\text{Precision} + \text{Recall}}.

For :math:`K` classes, compute an F1 *per class* (one-vs-rest), then **average them
with equal weight** to get macro F1:

.. math::

   F_{1,\text{macro}} = \frac{1}{K} \sum_{i=1}^{K} F_{1,i}.

Each class contributes the same, whatever its size — so weak performance on a rare
class is fully visible.

Macro vs micro vs weighted
--------------------------

- **Macro** — equal weight per class; surfaces minority-class weakness.
- **Micro** — pool global TP / FP / FN first, then compute F1; dominated by majority
  classes (and equals accuracy in single-label problems).
- **Weighted** — average per-class F1 weighted by class frequency; a compromise that
  respects class sizes while staying per-class.

Worked example
--------------

Three classes:

- F1(A) = 0.80 (P=0.80, R=0.80)
- F1(B) = 0.48 (P=0.60, R=0.40)
- F1(C) = 0.33 (P=0.50, R=0.25)

.. math::

   F_{1,\text{macro}} = \frac{0.80 + 0.48 + 0.33}{3} = 0.54.

Class A is strong, but B and C pull the average down — exactly what you want when
every class matters.

Pitfalls and edge cases
-----------------------

- **Zero-division** — if a class is never predicted, its precision is undefined; by
  convention its F1 is taken as 0, which (correctly) penalises ignoring that class.
- **Rare-class leverage** — one small, hard class can dominate the macro average;
  read the per-class F1s alongside it.

In code
-------

.. code-block:: python

   from sklearn.metrics import f1_score

   macro = f1_score(y_true, y_pred, average="macro")
   weighted = f1_score(y_true, y_pred, average="weighted")

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Micro F1 <013-micro-f1>` · :doc:`Macro Precision <021-macro-precision>` · :doc:`Macro Recall <020-macro-recall>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro Precision <021-macro-precision>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Macro F1 <https://insightful-data-lab.com/2025/08/30/macro-f1/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
