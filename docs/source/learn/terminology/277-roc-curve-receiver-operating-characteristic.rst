:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-roc-curve-receiver-operating-characteristic:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>ROC Curve (Receiver Operating Characteristic)</b></div>`

===============================================
ROC Curve (Receiver Operating Characteristic)
===============================================

*A plot of true- versus false-positive rate across thresholds.*

What it is
----------

A **ROC curve** (Receiver Operating Characteristic) plots a binary classifier's **true positive rate**
(sensitivity / recall) against its **false positive rate** (1 − specificity) as the **decision threshold**
sweeps from strict to lenient. Each point is one threshold's (FPR, TPR) trade-off.

Reading it
----------

Lowering the threshold labels **more** examples positive, so **both** TPR and FPR rise — the curve runs from
(0, 0) to (1, 1). A curve hugging the **upper-left** corner (high TPR, low FPR) is excellent; the
**diagonal** line is **random guessing**; the closer to the top-left, the better the separation.

Why it's useful
---------------

Because it shows performance at **every** threshold, the ROC curve reveals the full **trade-off** between
catching positives and raising false alarms — letting you pick an operating point for your costs, rather than
being locked to one cutoff. It dates to **radar** signal detection in the 1940s.

----

**Mind map — connected ideas**

   :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Model Score <364-model-score>` · :doc:`ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC) <427-roc-auc-receiver-operating-characteristic-area-u>` · :doc:`Precision–Recall AUC (PR-AUC) <430-precisionrecall-auc-pr-auc>` · :doc:`Accuracy <323-accuracy>` · :doc:`Partial AUC (pAUC) <312-partial-auc-pauc>`

----

**More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `ROC Curve (Receiver Operating Characteristic) <https://insightful-data-lab.com/2025/08/22/roc-curve-receiver-operating-characteristic/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
