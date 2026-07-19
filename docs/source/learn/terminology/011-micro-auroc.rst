:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-micro-auroc:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Micro AUROC</b></div>`

=============
Micro AUROC
=============

*AUROC pooled across classes by aggregating individual decisions first; weights every sample equally.*

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

**AUROC** (Area Under the ROC Curve) is, for a binary task, the probability that
the model scores a randomly chosen positive higher than a randomly chosen negative
— a threshold-free measure of ranking ability. **Micro AUROC** is one of the two
standard ways to extend that single number to :math:`K` classes.

The multiclass problem
----------------------

With :math:`K` classes there is no single ROC curve. Two aggregation strategies
dominate:

- **Macro AUROC** — compute a one-vs-rest AUROC per class, then average them with
  equal weight.
- **Micro AUROC** — *pool* every one-vs-rest binary decision across all classes
  into one big set, then compute a single AUROC.

How micro works
---------------

Flatten all per-class one-vs-rest scores and labels into a single pool, count true
and false positives **globally**, and evaluate AUROC as if it were one binary
problem:

.. math::

   \text{AUROC}_{\text{micro}}
   = \text{AUROC}\!\left(\textstyle\bigcup_{i=1}^{K}\text{positives}_i,\;
     \bigcup_{i=1}^{K}\text{negatives}_i\right).

Because counts are pooled, **frequent classes dominate** the result.

Worked example
--------------

Three classes with one-vs-rest scores AUROC(A)=0.90, AUROC(B)=0.70, AUROC(C)=0.60:

- **Macro AUROC** :math:`= (0.90 + 0.70 + 0.60)/3 = 0.733`.
- **Micro AUROC** pools predictions, so if class A has far more samples the micro
  value is pulled toward 0.90 — the majority class's score.

When to use it (and the trap)
-----------------------------

Micro AUROC answers "how well does the model discriminate **overall, per sample**".
Its trap is imbalance: it can look excellent while rare classes are handled badly,
because their few samples barely move the pooled total. Macro AUROC weights every
class equally and exposes weak minorities. **Report both.**

In code
-------

.. code-block:: python

   from sklearn.metrics import roc_auc_score

   # y_true: one-hot (n_samples, K); y_score: predicted probabilities (n_samples, K)
   micro = roc_auc_score(y_true, y_score, average="micro", multi_class="ovr")
   macro = roc_auc_score(y_true, y_score, average="macro", multi_class="ovr")

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`One-vs-Rest (OvR) AUROC <017-one-vs-rest-ovr-auroc>` · :doc:`Micro F1 <013-micro-f1>` · :doc:`Micro Recall <015-micro-recall>` · :doc:`Micro Precision <016-micro-precision>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Micro AUROC <https://insightful-data-lab.com/2025/08/30/micro-auroc/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
