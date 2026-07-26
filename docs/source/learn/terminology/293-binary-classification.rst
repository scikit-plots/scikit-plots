:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-binary-classification:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Binary Classification</b></div>`

=======================
Binary Classification
=======================

*Predicting one of two classes for each instance.*

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

**Binary classification** predicts one of **two classes** — positive/negative, 1/0, spam/not-spam. The
model doesn't output a bare label directly; it estimates the **probability** that an instance belongs to
the **positive** class, then a **decision threshold** turns that probability into a hard label.

The threshold
-------------

By default the cutoff is **0.5** — probability :math:`\ge 0.5` → class 1, else class 0 — but 0.5 is **not**
always right. On **imbalanced** data (e.g. fraud at 1%), 0.5 may label everything negative; the threshold
is tuned against **precision / recall** or an **ROC** curve to match the cost of each error.

How it's judged
---------------

Predictions map to the **confusion matrix** — true and false positives and negatives — from which
precision, recall, F1 and AUROC follow. The threshold choice moves directly along that trade-off.

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Classification Probability <231-classification-probability>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Sigmoid Function <297-sigmoid-function>` · :doc:`Logistic Regression <292-logistic-regression>` · :doc:`Classification Models <294-classification-models>` · :doc:`Neural Networks <287-neural-networks>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>` · :doc:`Macro Precision <021-macro-precision>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Binary Classification <https://insightful-data-lab.com/2025/08/21/binary-classification/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
