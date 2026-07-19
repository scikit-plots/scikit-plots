:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-per-class-precision-sometimes-called-class-wise-precision:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Per-class Precision (sometimes called class-wise precision)</b></div>`

=============================================================
Per-class Precision (sometimes called class-wise precision)
=============================================================

*Precision computed separately for each class.*

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

**Per-class precision** is **precision computed separately for each class**, treating that class as the
**positive** one and everything else as negative (**one-vs-rest**). For class :math:`c` it is

.. math::

   \text{precision}_c = \frac{TP_c}{TP_c + FP_c},

answering *of everything predicted as class c, how much really was c?*

Why report it
-------------

A single averaged number can **hide** a class the model handles badly; per-class precision exposes exactly
**which** classes suffer false positives. In scikit-learn, ``precision_score(average=None)`` returns the
whole **array** of per-class values.

Its role
--------

Per-class precision is the **building block** that **micro**, **macro** and **weighted** averaging then
collapse into one score. Best practice is to report the **per-class** values **alongside** any aggregate.

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Multiclass Precision <359-multiclass-precision>` · :doc:`Multilabel Precision <360-multilabel-precision>` · :doc:`Precision (a.k.a. Positive Predictive Value, PPV) <429-precision-a-k-a-positive-predictive-value-ppv>` · :doc:`One-vs-Rest (OvR) <310-one-vs-rest-ovr>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Micro Averaging <369-micro-averaging>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Per-class Precision (sometimes called class-wise precision) <https://insightful-data-lab.com/2025/08/20/per-class-precision-sometimes-called-class-wise-precision/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
