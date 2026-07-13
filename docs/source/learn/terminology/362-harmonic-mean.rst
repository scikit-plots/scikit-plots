:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-harmonic-mean:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Harmonic Mean</b></div>`

===============
Harmonic Mean
===============

*The reciprocal-based mean underlying the F1 score.*

What it is
----------

The **harmonic mean** is an average that **leans toward the smaller** of the values — the reciprocal of the
average of reciprocals. For two numbers it is:

.. math::

   \text{HM} = \frac{2ab}{a + b}.

It is always **≤ the arithmetic mean**, and equal only when the values match.

Its key property
----------------

It **penalizes imbalance**. Averaging precision 0.95 and recall 0.20, the arithmetic mean gives a rosy
**0.575**, but the harmonic mean gives **~0.33** — correctly flagging that one component is poor. A high
harmonic mean requires **all** inputs to be high.

Where it's used
---------------

That property is exactly why the **F1-score** uses it to combine precision and recall, and why harmonic
means are the right average for **rates and ratios** (speeds, P/E ratios) rather than additive quantities.

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`F1-score <363-f1-score>` · :doc:`Mean <316-mean>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Precision (a.k.a. Positive Predictive Value, PPV) <429-precision-a-k-a-positive-predictive-value-ppv>` · :doc:`Weighted Averaging <361-weighted-averaging>` · :doc:`Macro Averaging <370-macro-averaging>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>` · :doc:`Macro Precision <021-macro-precision>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Harmonic Mean <https://insightful-data-lab.com/2025/08/20/harmonic-mean/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
