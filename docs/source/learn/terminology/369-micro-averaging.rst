:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-micro-averaging:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Micro Averaging</b></div>`

=================
Micro Averaging
=================

*Aggregating counts across classes before computing a metric.*

What it is
----------

**Micro averaging** aggregates a metric by **pooling the counts** — it sums the **true positives, false
positives and false negatives** across all classes first, then computes precision, recall or F1 from those
totals. Every **instance** counts equally, so **frequent classes dominate**.

Its behavior
------------

Because it is driven by raw counts, micro averaging gives an **overall**, accuracy-flavored number — in
fact, for **single-label multiclass**, micro-F1 **equals accuracy**. Its weakness is that strong performance
on a big class can **mask** poor performance on a small one.

When to use it
--------------

Reach for micro averaging on **balanced** problems, in **multilabel** settings, or whenever you want a
**single global** score reflecting overall correctness. Pair it with **macro** to expose whether minority
classes are being hidden.

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Weighted Averaging <361-weighted-averaging>` · :doc:`Micro AUC <313-micro-auc>` · :doc:`One-vs-Rest (OvR) <310-one-vs-rest-ovr>` · :doc:`Precision (a.k.a. Positive Predictive Value, PPV) <429-precision-a-k-a-positive-predictive-value-ppv>` · :doc:`Multiclass Classification <311-multiclass-classification>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Micro Averaging <https://insightful-data-lab.com/2025/08/20/micro-averaging/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
