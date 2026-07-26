:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-macro-recall:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Macro Recall</b></div>`

==============
Macro Recall
==============

*The unweighted mean of per-class recall values.*

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

Recall is the share of actual positives the model catches,
:math:`\text{Recall} = TP/(TP+FN)`. For :math:`K` classes, compute recall *per class*
(one-vs-rest) and take the **unweighted mean**:

.. math::

   \text{Recall}_{\text{macro}} = \frac{1}{K} \sum_{i=1}^{K} \text{Recall}_i.

Every class counts equally regardless of how many samples it has.

A useful equivalence
--------------------

In single-label classification, **macro recall is exactly balanced accuracy** — the
average per-class hit rate. That makes it a go-to headline metric for imbalanced
problems, because it refuses to let a dominant class inflate the score.

Macro vs micro vs weighted
--------------------------

- **Macro** — equal weight per class; good when every class is equally important.
- **Micro** — global TP and FN pooled first; dominated by large classes.
- **Weighted** — per-class recall averaged by the number of true samples in each
  class.

Worked example
--------------

Three classes with Recall(A)=0.90, Recall(B)=0.60, Recall(C)=0.30:

.. math::

   \text{Recall}_{\text{macro}} = \frac{0.90 + 0.60 + 0.30}{3} = 0.60.

Even though C is rare, it carries the same weight as A.

Pitfalls and edge cases
-----------------------

- **Ignores false positives** — recall says nothing about precision, so a model that
  over-predicts a class can still score well; pair it with macro precision or
  macro F1.
- **Empty classes** — a class with no true samples has undefined recall and must be
  handled before averaging.

In code
-------

.. code-block:: python

   from sklearn.metrics import recall_score, balanced_accuracy_score

   macro = recall_score(y_true, y_pred, average="macro")
   # in single-label problems this equals:
   bal_acc = balanced_accuracy_score(y_true, y_pred)

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Macro Precision <021-macro-precision>` · :doc:`Macro F1 <019-macro-f1>` · :doc:`Micro Recall <015-micro-recall>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Macro Recall <https://insightful-data-lab.com/2025/08/30/macro-recall/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
