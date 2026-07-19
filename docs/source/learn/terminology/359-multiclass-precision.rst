:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-multiclass-precision:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Multiclass Precision</b></div>`

======================
Multiclass Precision
======================

*Precision aggregated across the classes of a multiclass problem.*

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

**Multiclass precision** is precision for a problem with **more than two mutually exclusive** classes. Since
precision is defined on a **binary** positive/negative split, it is computed by treating the task as **K
one-vs-rest** binary problems — the **per-class** precisions — then reduced to a single number.

How it's reduced
----------------

The per-class values are combined by an **averaging** scheme — **micro** (pool counts,
majority-dominated), **macro** (equal weight per class), or **weighted** (by support). The choice
determines whether rare classes are surfaced or hidden, so it must be **stated** with the score.

The caveat
----------

Because each class is scored against "the rest," each binary split is **imbalanced**; reporting the
**per-class** precisions guards against an average that looks good only because the majority class does.

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Per-class Precision (sometimes called class-wise precision) <358-per-class-precision-sometimes-called-class-wise>` · :doc:`Multilabel Precision <360-multilabel-precision>` · :doc:`Multiclass Classification <311-multiclass-classification>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Weighted Averaging <361-weighted-averaging>` · :doc:`Precision (a.k.a. Positive Predictive Value, PPV) <429-precision-a-k-a-positive-predictive-value-ppv>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Multiclass Precision <https://insightful-data-lab.com/2025/08/20/multiclass-precision/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
