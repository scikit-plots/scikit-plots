:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-discriminatory-power:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Discriminatory Power</b></div>`

======================
Discriminatory Power
======================

*A model's ability to separate positive from negative cases.*

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

**Discriminatory power** is a model's (or test's) ability to **distinguish correctly between two
groups** — usually positives versus negatives: good versus bad credit, disease versus healthy,
responder versus not. The plain question it answers: *how well can the model separate those who will
do X from those who won't?*

Where it applies
----------------

A classifier with high discriminatory power assigns **higher scores to positives** than negatives. A
**credit scorecard** is judged on how cleanly it separates defaulters from non-defaulters; a
**diagnostic test** on how well it separates the sick from the healthy.

How it's measured
-----------------

Several metrics. **AUC** is the probability a random positive outscores a random negative — 0.5 is
random, 1.0 is perfect. The **KS statistic** is the maximum gap between the cumulative score
distributions of positives and negatives (0.4-0.6 is strong in credit risk). The **Gini coefficient**
rescales AUC, :math:`\text{Gini} = 2 \times \text{AUC} - 1`. Lift and CAP curves give a visual read.

An example, and why it matters
--------------------------------

If good borrowers average a score of 0.8 and bad ones 0.3, with **AUC 0.85 and KS 0.45**, the model
has strong discriminatory power; **AUC 0.55, KS 0.08** is near-random. It drives better **targeting**
and **lending** decisions, is a **regulatory** reporting requirement in finance, and reduces false
positives and negatives for fairer outcomes.

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Multiclass AUROC <022-multiclass-auroc>` · :doc:`KS Statistic (Kolmogorov–Smirnov Statistic) <186-ks-statistic-kolmogorovsmirnov-statistic>` · :doc:`Demographic Parity (Statistical Parity) <030-demographic-parity-statistical-parity>` · :doc:`Recalibration <159-recalibration>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>` · :doc:`Macro Precision <021-macro-precision>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Discriminatory Power <https://insightful-data-lab.com/2025/08/23/discriminatory-power/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
