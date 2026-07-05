:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-accuracy:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Accuracy</b></div>`

==========
Accuracy
==========

*The fraction of predictions that are correct.*

What it is
----------

**Accuracy** is the simplest classification metric — the **fraction of predictions that are correct**:

.. math::

   \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}.

It answers *what share did the model get right?*

The imbalance trap
------------------

Accuracy is **misleading on imbalanced** data — if 99% of cases are negative, a model that predicts
"negative" for **everything** scores **99%** while catching **zero** positives. That false sense of security
is why fraud, disease and anomaly tasks report **precision / recall**, **AUC** or **F1** instead.

When it's fine
--------------

Accuracy is a reasonable headline when classes are **roughly balanced** and every error costs about the
**same**. Otherwise it hides **which** errors happen — a **confusion matrix** and threshold-aware metrics
tell the real story.

----

**Mind map — connected ideas**

   :doc:`Precision (a.k.a. Positive Predictive Value, PPV) <429-precision-a-k-a-positive-predictive-value-ppv>` · :doc:`ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC) <427-roc-auc-receiver-operating-characteristic-area-u>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Multiclass Classification <311-multiclass-classification>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Precision–Recall AUC (PR-AUC) <430-precisionrecall-auc-pr-auc>`

----

**More in Classification & Averaging Metrics**

   :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>` · :doc:`Macro Precision <021-macro-precision>`

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Accuracy <https://insightful-data-lab.com/2025/08/20/accuracy/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
