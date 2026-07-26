:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-precisionrecall-auc-pr-auc:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Precision–Recall AUC (PR-AUC)</b></div>`

===============================
Precision–Recall AUC (PR-AUC)
===============================

*Area under the precision-recall curve, informative under class imbalance.*

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

**PR-AUC** is the **area under the precision–recall curve**, which plots **precision** against **recall**
across thresholds — also called **Average Precision (AP)**, the mean precision over all recall levels. It
ranges :math:`[0, 1]`, higher is better.

Why it's imbalance-friendly
---------------------------

Unlike ROC-AUC, PR-AUC ignores **true negatives** entirely and focuses on the **positive** class, so it
stays informative when positives are **rare**. Its **baseline** also shifts with prevalence — random
guessing scores the **positive-class ratio** (0.5 when balanced, 0.01 at 1% positive), not a fixed 0.5.

When to use it
--------------

Reach for PR-AUC on **highly imbalanced** problems where finding the **minority** positive is the goal —
fraud, rare-disease, anomaly detection — where a high ROC-AUC can be misleading. Report it **alongside**
ROC-AUC for the full picture.

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC) <427-roc-auc-receiver-operating-characteristic-area-u>` · :doc:`Precision (a.k.a. Positive Predictive Value, PPV) <429-precision-a-k-a-positive-predictive-value-ppv>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Multiclass AUROC <022-multiclass-auroc>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Macro AUC <314-macro-auc>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Precision–Recall AUC (PR-AUC) <https://insightful-data-lab.com/2025/08/17/precision-recall-auc-pr-auc/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
