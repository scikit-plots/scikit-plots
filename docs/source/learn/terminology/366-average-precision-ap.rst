:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-average-precision-ap:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Average Precision (AP)</b></div>`

========================
Average Precision (AP)
========================

*The area under the precision-recall curve summarising ranked retrieval.*

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

**Average precision** summarizes the entire **precision–recall curve** in one number — the mean of precision
across recall levels, computed as precision weighted by the **gain in recall** at each threshold:

.. math::

   \text{AP} = \sum_{n} (R_n - R_{n-1})\,P_n.

It equals the **area under the PR curve** (PR-AUC / AUPRC).

Why it's useful
---------------

Because it sweeps **all thresholds**, AP needs no single cutoff, and because it is built from **precision and
recall** it **ignores true negatives** — making it far more informative than ROC-AUC on **imbalanced** data
where the positive class is rare.

Where it's used
---------------

AP is the standard score for **ranking** and **detection**; averaging it over classes or queries gives
**mean average precision (mAP)**, the headline metric in information retrieval and object detection.

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Precision–Recall AUC (PR-AUC) <430-precisionrecall-auc-pr-auc>` · :doc:`F1-score <363-f1-score>` · :doc:`Precision (a.k.a. Positive Predictive Value, PPV) <429-precision-a-k-a-positive-predictive-value-ppv>` · :doc:`ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC) <427-roc-auc-receiver-operating-characteristic-area-u>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Macro Averaging <370-macro-averaging>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>` · :doc:`Macro Precision <021-macro-precision>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Average Precision (AP) <https://insightful-data-lab.com/2025/08/20/average-precision-ap/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
