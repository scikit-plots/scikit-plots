:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-micro-f1:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Micro F1</b></div>`

==========
Micro F1
==========

*F1 computed from globally pooled TP/FP/FN; dominated by the more frequent classes.*

What it is
----------

The **F1 score** is the harmonic mean of precision and recall, rewarding a model
only when *both* are high:

.. math::

   F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}
              {\text{Precision} + \text{Recall}}.

**Micro F1** extends F1 to :math:`K` classes by pooling counts before computing the
score, rather than averaging per-class F1 values (which is macro F1).

How it's computed
-----------------

First form **global** precision and recall by summing true/false positives and
false negatives over all classes:

.. math::

   \text{Precision}_{\text{micro}} = \frac{\sum_i TP_i}{\sum_i (TP_i + FP_i)},
   \qquad
   \text{Recall}_{\text{micro}} = \frac{\sum_i TP_i}{\sum_i (TP_i + FN_i)},

then combine them with the F1 formula:

.. math::

   F_{1,\text{micro}} = \frac{2 \cdot \text{Precision}_{\text{micro}}
   \cdot \text{Recall}_{\text{micro}}}
   {\text{Precision}_{\text{micro}} + \text{Recall}_{\text{micro}}}.

A key identity
--------------

In **single-label** (multi-class) problems, micro precision, micro recall and micro
F1 are all equal — and equal to plain **accuracy**. With one label per sample, a
false positive for one class is the same event as a false negative for another, so
the pooled denominators coincide. (In *multi-label* problems they can differ.)

Worked example
--------------

Three classes with TP = (40, 30, 10), FP = (10, 20, 20), FN = (10, 20, 30):

- :math:`\text{Precision}_{\text{micro}} = 80/130 \approx 0.615`
- :math:`\text{Recall}_{\text{micro}} = 80/140 \approx 0.571`
- :math:`F_{1,\text{micro}} \approx 0.592`

When to use it
--------------

Micro F1 measures overall, sample-weighted performance and lets majority classes
dominate — handy on imbalanced data when overall throughput matters. Use **macro
F1** when every class should count equally, including rare ones.

In code
-------

.. code-block:: python

   from sklearn.metrics import f1_score

   micro = f1_score(y_true, y_pred, average="micro")
   macro = f1_score(y_true, y_pred, average="macro")

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Macro F1 <019-macro-f1>` · :doc:`Micro Precision <016-micro-precision>` · :doc:`Micro Recall <015-micro-recall>` · :doc:`Micro AUROC <011-micro-auroc>` · :doc:`Multi-label Classification <012-multi-label-classification>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Micro F1 <https://insightful-data-lab.com/2025/08/30/micro-f1/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
