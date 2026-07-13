:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-auc-area-under-the-curve:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>AUC (Area Under the Curve)</b></div>`

============================
AUC (Area Under the Curve)
============================

*The area under a ROC or PR curve summarising performance across thresholds.*

What it is
----------

**AUC** — the **area under the curve** — condenses an entire **ROC curve** into one number by measuring the
area beneath it. It ranges from **0 to 1**: **1** is perfect, **0.5** is random guessing, and below 0.5 means
the scores are **inverted**.

Its meaning
-----------

AUC has a clean interpretation — it is the **probability that a random positive is scored higher than a
random negative** (the Wilcoxon–Mann–Whitney statistic). So it measures how well the model **ranks**
positives above negatives, independent of any single threshold.

Why it beats accuracy
---------------------

AUC is **threshold-invariant** and always calibrated so **0.5 = useless**, unlike **accuracy**, which is
misleading under **imbalance** (90% accuracy is trivial when 90% of data is negative). But on **heavily
imbalanced** data the **PR-AUC** often tells a more honest story, since ROC-AUC can look optimistic.

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`ROC Curve (Receiver Operating Characteristic) <277-roc-curve-receiver-operating-characteristic>` · :doc:`ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC) <427-roc-auc-receiver-operating-characteristic-area-u>` · :doc:`Precision–Recall AUC (PR-AUC) <430-precisionrecall-auc-pr-auc>` · :doc:`Model Score <364-model-score>` · :doc:`Partial AUC (pAUC) <312-partial-auc-pauc>` · :doc:`Accuracy <323-accuracy>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>` · :doc:`Macro Precision <021-macro-precision>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `AUC (Area Under the Curve) <https://insightful-data-lab.com/2025/08/20/auc-area-under-the-curve/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
