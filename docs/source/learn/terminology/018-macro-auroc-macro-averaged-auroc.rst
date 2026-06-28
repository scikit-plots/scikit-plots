:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-macro-auroc-macro-averaged-auroc:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Macro AUROC (Macro-Averaged AUROC)</b></div>`

====================================
Macro AUROC (Macro-Averaged AUROC)
====================================

*The mean of per-class AUROC values, weighting every class equally regardless of size.*

What it is
----------

AUROC measures ranking ability for a binary task — the probability a random positive
outscores a random negative, ranging from 0.5 (chance) to 1.0 (perfect). Since ROC
is inherently binary, a :math:`K`-class model is scored by reducing it with
**one-vs-rest (OvR)** (or one-vs-one). **Macro AUROC** averages the per-class OvR
AUROCs with **equal weight**:

.. math::

   \text{AUROC}_{\text{macro}}
   = \frac{1}{K} \sum_{i=1}^{K} \text{AUROC}(\text{class}_i \;\text{vs}\; \text{rest}).

Because every class counts the same, a rare class the model handles badly drags the
score down just as much as a common one.

Macro vs micro
--------------

**Macro** gives equal weight per class, so it is sensitive to **minority-class**
performance. **Micro** pools all OvR decisions into one global AUROC and is therefore
dominated by **majority classes**. They answer different questions — fairness across
classes vs overall sample-level discrimination — so reporting both is good practice.
The scale matches binary AUROC in every case.

Worked example
--------------

Three classes with AUROC(A vs rest)=0.85, AUROC(B vs rest)=0.72,
AUROC(C vs rest)=0.65:

.. math::

   \text{AUROC}_{\text{macro}} = \frac{0.85 + 0.72 + 0.65}{3} = 0.74.

If C is rare but poorly separated, macro AUROC reflects it; micro might not.

Pitfalls and edge cases
-----------------------

- **Undefined per-class AUROC** — if a class has no positive (or no negative)
  samples in the evaluation set, its OvR AUROC is undefined and breaks the average;
  guard against empty classes.
- **Equal weighting cuts both ways** — a single tiny, hard class can dominate the
  headline number; inspect the per-class AUROCs, not just the mean.
- **Weighted variant** — averaging the per-class AUROCs by class frequency gives a
  middle ground between macro and micro.

In code
-------

.. code-block:: python

   from sklearn.metrics import roc_auc_score

   macro = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
   per_class = roc_auc_score(y_true, y_score, multi_class="ovr", average=None)

----

**Mind map — connected ideas**

   :doc:`Micro AUROC <011-micro-auroc>` · :doc:`One-vs-Rest (OvR) AUROC <017-one-vs-rest-ovr-auroc>` · :doc:`Macro F1 <019-macro-f1>` · :doc:`Macro Recall <020-macro-recall>`

----

**More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>` · :doc:`Macro Precision <021-macro-precision>`

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Macro AUROC (Macro-Averaged AUROC) <https://insightful-data-lab.com/2025/08/30/macro-auroc/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
