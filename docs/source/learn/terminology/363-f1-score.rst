:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-f1-score:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>F1-score</b></div>`

==========
F1-score
==========

*The harmonic mean of precision and recall.*

What it is
----------

The **F1-score** combines **precision** and **recall** into one number by taking their **harmonic mean**:

.. math::

   F_1 = 2\cdot\frac{P \cdot R}{P + R} = \frac{2\,TP}{2\,TP + FP + FN}.

It ranges from **0 to 1**, and is high only when **both** precision and recall are high.

Why harmonic
------------

Using the harmonic mean makes F1 **penalize lopsided** models — a classifier with 0.95 precision but 0.20
recall scores a low F1, unlike accuracy or a plain average. This makes F1 far more informative than
**accuracy** on **imbalanced** data, and it deliberately **ignores true negatives**.

Its variants
------------

For **multiclass** problems, F1 is aggregated with **micro**, **macro** or **weighted** averaging; the
general **Fβ** score tilts the balance toward recall (β > 1) or precision (β < 1) when the two errors carry
different costs.

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Precision (a.k.a. Positive Predictive Value, PPV) <429-precision-a-k-a-positive-predictive-value-ppv>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Accuracy <323-accuracy>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Micro Averaging <369-micro-averaging>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>` · :doc:`Macro Precision <021-macro-precision>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `F1-score <https://insightful-data-lab.com/2025/08/20/f1-score/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
