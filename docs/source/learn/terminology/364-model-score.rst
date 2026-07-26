:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-model-score:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Model Score</b></div>`

=============
Model Score
=============

*The raw numeric output a model assigns before thresholding.*

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

A **model score** is the **continuous output** a classifier assigns each example — a **probability** or
real-valued score of belonging to the **positive** class — *before* it becomes a hard label. Logistic
regression, random forests, and neural nets all emit scores.

Score vs label
--------------

Turning a score into a **decision** requires a **threshold** — above it, positive; below, negative. The score
carries **more** information than the label: its **ranking** (are positives scored above negatives?) is what
threshold-free metrics like **AUC** measure, and its **magnitude** matters for ranking and prioritization.

Score vs probability
--------------------

A score need not be a **calibrated** probability — a score of 0.9 doesn't guarantee a 90% chance of being
positive unless the model is calibrated (e.g., via **temperature** or **Platt scaling**). Use the raw score
for **ranking**, the calibrated one for **decisions** that need real probabilities.

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`ROC Curve (Receiver Operating Characteristic) <277-roc-curve-receiver-operating-characteristic>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Sigmoid Function <297-sigmoid-function>` · :doc:`Temperature Scaling <279-temperature-scaling>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Model Score <https://insightful-data-lab.com/2025/08/20/model-score/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
