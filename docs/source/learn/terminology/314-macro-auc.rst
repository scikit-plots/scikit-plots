:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-macro-auc:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Macro AUC</b></div>`

===========
Macro AUC
===========

*AUC averaged equally across per-class scores.*

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

**Macro AUC** averages the per-class AUCs (from **One-vs-Rest**) with **equal weight** — every class counts
the **same**, no matter how rare or common. It answers *how well does the model do on the average class?*

Macro vs micro
--------------

The contrast is **micro AUC**, which pools every class's true/false-positive contributions into **one global**
curve, effectively **weighting by prevalence** so frequent classes dominate. Macro treats a class with 10
samples exactly like one with 10,000.

When to use which
-----------------

**Macro** is the choice when the **rare** classes matter as much as the common ones (you want minority
performance to show), while **micro** (or a **weighted** macro) better reflects **overall** accuracy on
**imbalanced** data. Reporting both reveals whether a good score is carried by the majority classes.

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Micro AUC <313-micro-auc>` · :doc:`One-vs-Rest (OvR) <310-one-vs-rest-ovr>` · :doc:`Multiclass Classification <311-multiclass-classification>` · :doc:`ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC) <427-roc-auc-receiver-operating-characteristic-area-u>` · :doc:`Multiclass AUROC <022-multiclass-auroc>` · :doc:`Partial AUC (pAUC) <312-partial-auc-pauc>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>` · :doc:`Macro Precision <021-macro-precision>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Macro AUC <https://insightful-data-lab.com/2025/08/21/macro-auc/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
