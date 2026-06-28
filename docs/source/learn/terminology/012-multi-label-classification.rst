:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-multi-label-classification:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Multi-label Classification</b></div>`

============================
Multi-label Classification
============================

*Tasks where each instance may carry several non-exclusive labels at once.*

What it is
----------

In **multi-label classification** each sample can carry *several* labels at once.
This is different from **multi-class** classification, where every sample gets
exactly one label from a set. Here the labels are **not mutually exclusive** — the
model makes an independent yes/no decision for every class:

.. math::

   f : X \;\rightarrow\; \{0, 1\}^K,

so for :math:`K` classes the output is a length-:math:`K` vector of binary
decisions. A movie can be *Action + Comedy*, a news story *Politics + Economy*, a
patient *diabetic + hypertensive*, an image *dog + car + tree*.

Multi-label vs multi-class
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - Feature
     - Multi-class
     - Multi-label
   * - Labels per sample
     - Exactly one
     - One or more
   * - Class exclusivity
     - Mutually exclusive
     - Independent
   * - Output layer
     - Softmax (probabilities sum to 1)
     - Sigmoid (independent probability per class)
   * - Loss
     - Categorical cross-entropy
     - Binary cross-entropy per class

How it's modelled
-----------------

The network ends in a **sigmoid** per class rather than a single softmax, and each
output is thresholded independently. The loss is **binary cross-entropy** summed or
averaged over the :math:`K` labels — effectively :math:`K` coupled binary problems
sharing one backbone.

How it's scored
---------------

Because a prediction is a *set* of labels, the metrics differ from single-label
ones:

- **Per-label precision / recall / F1**, then aggregated with **micro**, **macro**
  or **weighted** averaging.
- **Hamming loss** — the fraction of individual label slots that are wrong.
- **Subset accuracy** — strict: 1 only if *every* label of the sample is correct,
  else 0.
- **Jaccard similarity** — intersection over union of predicted and true label
  sets.

Worked example
--------------

True labels for an image: :math:`\{\text{Cat}, \text{Dog}\}`; the model predicts
:math:`\{\text{Cat}, \text{Horse}\}`.

- Precision :math:`= 1/(1+1) = 0.5` (Cat right, Horse is a false positive).
- Recall :math:`= 1/(1+1) = 0.5` (Dog was missed — a false negative).
- Jaccard :math:`= |\{\text{Cat}\}| / |\{\text{Cat},\text{Dog},\text{Horse}\}| = 1/3 \approx 0.33`.

Pitfalls and edge cases
-----------------------

- **Subset accuracy is harsh** — one wrong label out of many zeroes the whole
  sample; report it alongside Hamming loss, not alone.
- **Per-label thresholds** — a single 0.5 cutoff is rarely optimal for every label;
  tune thresholds per class, especially under imbalance.
- **Label imbalance and correlation** — rare labels and co-occurring labels (Cat
  with Dog) are easy to under-predict; micro averaging will hide that, macro will
  expose it.

----

**Mind map — connected ideas**

   :doc:`Single-label Classification <014-single-label-classification>` · :doc:`Micro F1 <013-micro-f1>` · :doc:`Macro F1 <019-macro-f1>` · :doc:`Micro AUROC <011-micro-auroc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>`

----

**More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Multi-label Classification <https://insightful-data-lab.com/2025/08/30/multi-label-classification/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
