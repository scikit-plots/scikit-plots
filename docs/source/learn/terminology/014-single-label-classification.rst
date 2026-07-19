:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-single-label-classification:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Single-label Classification</b></div>`

=============================
Single-label Classification
=============================

*Tasks where each instance is assigned exactly one label.*

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

In **single-label classification** every sample is assigned **exactly one** label
from a set of :math:`K` mutually exclusive classes. Formally the classifier maps an
input to one label,

.. math::

   f : X \rightarrow Y, \qquad Y = \{1, 2, \dots, K\},

and typically picks the highest-probability class,

.. math::

   f(x) = \arg\max_{k}\, P(y = k \mid x).

This is the most common classification setting: spam vs not-spam, an image that is a
cat *or* a dog *or* a horse (never two at once), a single diagnosis from mutually
exclusive outcomes.

How it's modelled
-----------------

The output layer is a **softmax**, so the predicted class probabilities sum to one
and the classes compete — raising one lowers the others. Training uses **categorical
cross-entropy**.

How it's scored
---------------

Because there is one prediction per sample, the natural metrics are **accuracy**
plus **precision, recall and F1** (with micro / macro / weighted averaging for the
multi-class case) and **AUROC / AUPRC** via one-vs-rest. A useful identity: under
single-label evaluation, *micro* precision, recall and F1 all equal accuracy.

vs multi-label
--------------

The contrast is exclusivity. Single-label gives one label per sample (an image is
cat *or* dog); **multi-label** allows any subset (a news story tagged *politics* and
*economy*), uses independent sigmoids instead of a softmax, and needs set-based
metrics like Hamming loss and Jaccard.

Pitfalls and edge cases
-----------------------

- **The exclusivity assumption** — if samples can truly belong to several classes,
  forcing one label loses information; use multi-label instead.
- **Calibrated probabilities** — ``argmax`` discards confidence; keep the softmax
  scores if you need ranking, thresholds or AUROC.
- **Imbalance** — plain accuracy flatters a majority-heavy dataset; prefer macro
  metrics when minority classes matter.

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Multi-label Classification <012-multi-label-classification>` · :doc:`Micro F1 <013-micro-f1>` · :doc:`Macro F1 <019-macro-f1>` · :doc:`One-vs-Rest (OvR) AUROC <017-one-vs-rest-ovr-auroc>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Single-label Classification <https://insightful-data-lab.com/2025/08/30/single-label-classification/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
