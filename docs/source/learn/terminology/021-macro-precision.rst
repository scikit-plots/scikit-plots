:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-macro-precision:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Macro Precision</b></div>`

=================
Macro Precision
=================

*The unweighted mean of per-class precision values.*

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

**Precision** is the share of the model's positive predictions that are correct,
:math:`\text{Precision} = TP/(TP+FP)`, so it is sensitive to **false positives**.
For :math:`K` classes, compute precision *per class* (one-vs-rest) and take the
**arithmetic mean** to get macro precision:

.. math::

   \text{Precision}_{\text{macro}} = \frac{1}{K} \sum_{i=1}^{K} \text{Precision}_i.

Every class counts equally, whatever its size, so a small class the model
over-flags pulls the score down as much as a large one.

Macro vs micro vs weighted
--------------------------

- **Macro** — equal weight per class; fair across classes.
- **Micro** — pool global TP and FP first; dominated by large classes (and equals
  accuracy in single-label problems).
- **Weighted** — per-class precision averaged by class frequency.

Worked example
--------------

Three classes with Precision(A)=0.80, Precision(B)=0.60, Precision(C)=0.40:

.. math::

   \text{Precision}_{\text{macro}} = \frac{0.80 + 0.60 + 0.40}{3} = 0.60.

If C is tiny, macro precision still penalises weak performance on it.

Pitfalls and edge cases
-----------------------

- **Zero-division** — a class the model never predicts has 0 in the denominator;
  its precision is undefined and conventionally set to 0, which penalises ignoring
  the class. Set ``zero_division`` explicitly to control this.
- **Pair it with recall** — precision rewards being conservative; a model that
  rarely predicts a class can post high precision while missing most of it.

In code
-------

.. code-block:: python

   from sklearn.metrics import precision_score

   macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
   weighted = precision_score(y_true, y_pred, average="weighted")

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Micro Precision <016-micro-precision>` · :doc:`Macro Recall <020-macro-recall>` · :doc:`Macro F1 <019-macro-f1>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Macro Precision <https://insightful-data-lab.com/2025/08/30/macro-precision/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
