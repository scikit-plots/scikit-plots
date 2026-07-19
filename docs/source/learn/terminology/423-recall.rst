:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-recall:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Recall</b></div>`

========
Recall
========

*The share of actual positives the model correctly identifies.*

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

**Recall** — also the **true positive rate** or **sensitivity** — is the share of **actual positives** the
model correctly **catches**:

.. math::

   \text{Recall} = \frac{TP}{TP + FN}.

It answers "**of everything that was truly positive, how much did we find?**" and falls when **false
negatives** pile up.

The trade-off
-------------

Recall trades off against **precision** — loosening the threshold catches more positives (higher recall) but
admits more false alarms (lower precision), which is why they're read together via the **F1-score**. High
recall matters most when a **miss** is costly: disease screening, fraud, safety.

Its fairness role
-----------------

Comparing recall **across groups** is exactly the **equal-opportunity** fairness test — it asks whether
**qualified** members of every group have the **same** chance of being correctly selected. Unequal recall
means the model **misses** true positives more often for one group, a common and consequential bias.

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Fairness parity <372-fairness-parity>` · :doc:`Selection Rate <390-selection-rate>` · :doc:`Precision (a.k.a. Positive Predictive Value, PPV) <429-precision-a-k-a-positive-predictive-value-ppv>` · :doc:`F1-score <363-f1-score>` · :doc:`Macro Recall <020-macro-recall>` · :doc:`Micro Recall <015-micro-recall>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Recall <https://insightful-data-lab.com/2025/08/17/recall/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
