:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-precision-a-k-a-positive-predictive-value-ppv:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Precision (a.k.a. Positive Predictive Value, PPV)</b></div>`

===================================================
Precision (a.k.a. Positive Predictive Value, PPV)
===================================================

*The share of predicted positives that are truly positive.*

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

**Precision** (also **positive predictive value**, **PPV**) is the fraction of **correct** positive
predictions among **all** positive predictions:

.. math::

   \text{Precision} = \frac{TP}{TP + FP}.

It answers: *of everything the model flagged positive, how much really was?*

The trade-off
-------------

Precision counts **false positives** against you but ignores **false negatives** — so a model can reach high
precision by only flagging the cases it is surest about. It is therefore read **together with recall**
(:math:`TP/(TP+FN)`), which counts the positives **missed**; the **F1** score is their harmonic mean.

When it matters
---------------

Precision is the priority when a **false positive is costly** — a spam filter deleting real mail, a system
flagging innocent transactions as fraud — where you would rather miss some positives than raise false
alarms.

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`ROC-AUC (Receiver Operating Characteristic – Area Under Curve, = AUROC) <427-roc-auc-receiver-operating-characteristic-area-u>` · :doc:`Precision–Recall AUC (PR-AUC) <430-precisionrecall-auc-pr-auc>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Multiclass AUROC <022-multiclass-auroc>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Multiclass Classification <311-multiclass-classification>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Precision (a.k.a. Positive Predictive Value, PPV) <https://insightful-data-lab.com/2025/08/17/precision-a-k-a-positive-predictive-value-ppv/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
