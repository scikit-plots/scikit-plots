:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-roc-auc-receiver-operating-characteristic-area-under-curve-auroc:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC)</b></div>`

=========================================================================
ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC)
=========================================================================

*The probability a random positive outranks a random negative.*

What it is
----------

**ROC-AUC** is the **area under the ROC curve**, which plots the **true positive rate** (recall) against the
**false positive rate** as the decision **threshold** sweeps from 0 to 1. It condenses that whole curve into
**one number** in :math:`[0, 1]`.

How to read it
--------------

**1.0** is a perfect classifier, **0.5** is random guessing. It has a clean probabilistic meaning — the
chance that a randomly chosen **positive** is scored **higher** than a randomly chosen **negative** — so it
measures **ranking** quality, independent of any single threshold and invariant to the score **scale**.

The caveat
----------

Because the **false positive rate** has all the **true negatives** in its denominator, ROC-AUC can look
**optimistically high** on **imbalanced** data where negatives dominate — a model can score well while still
flooding a rare positive class with false alarms. There, **PR-AUC** is more honest.

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Precision–Recall AUC (PR-AUC) <430-precisionrecall-auc-pr-auc>` · :doc:`Precision (a.k.a. Positive Predictive Value, PPV) <429-precision-a-k-a-positive-predictive-value-ppv>` · :doc:`Multiclass AUROC <022-multiclass-auroc>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Multiclass Classification <311-multiclass-classification>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC) <https://insightful-data-lab.com/2025/08/17/roc-auc-receiver-operating-characteristic-area-under-curve/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
