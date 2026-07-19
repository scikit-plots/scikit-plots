:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-micro-auc:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Micro AUC</b></div>`

===========
Micro AUC
===========

*AUC pooled across all classes by aggregating decisions before averaging.*

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

**Micro AUC** is the multiclass / multilabel AUROC that **pools** every class's predictions into **one
global** ROC curve — aggregating all true-positive and false-positive counts across classes, then computing
a single AUC. Because it counts **every prediction** equally, frequent classes contribute more.

Micro vs macro
--------------

Where **macro AUC** averages per-class AUCs with equal weight, micro AUC is effectively **weighted by
prevalence** — a rare class with few samples barely moves it. Micro answers *how well does the model do on
the average prediction?*, macro *on the average class?*

When to use it
--------------

Micro AUC suits **imbalanced** multiclass problems when you care about **overall** performance dominated by
common classes, and it matches how a **multilabel** system is scored (over the flattened label matrix).
Report it beside **macro** to expose class-size effects.

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Macro AUC <314-macro-auc>` · :doc:`One-vs-Rest (OvR) <310-one-vs-rest-ovr>` · :doc:`Multiclass Classification <311-multiclass-classification>` · :doc:`ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC) <427-roc-auc-receiver-operating-characteristic-area-u>` · :doc:`Multiclass AUROC <022-multiclass-auroc>` · :doc:`Partial AUC (pAUC) <312-partial-auc-pauc>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Micro AUC <https://insightful-data-lab.com/2025/08/21/micro-auc/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
