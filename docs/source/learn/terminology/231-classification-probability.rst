:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-classification-probability:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📏&nbsp;&nbsp;<b>Classification Probability</b></div>`

============================
Classification Probability
============================

*The probability a model assigns to a class for an instance.*

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

The **classification probability** is the **score** a classifier assigns that an instance belongs to a
class — an estimate of :math:`P(\text{class} \mid \text{features})` between **0 and 1**. It is the model's
**confidence** *before* any hard decision is made.

From probability to label
-------------------------

A **threshold** converts it to a class (in scikit-learn, ``predict_proba`` gives the probability,
``predict`` applies the cutoff). Two instances scored 0.51 and 0.99 both become "positive," but they are
**not** equally certain — which is why the probability carries more information than the label.

Why calibration matters
-----------------------

The probability is only trustworthy if it is **calibrated** — if events predicted at 0.7 actually happen
about 70% of the time. **Over-** or **under-confident** scores mislead any downstream **risk-based
decision**, so probabilities are validated with calibration curves, not just accuracy.

----

*Theme:* :ref:`Classification & Averaging Metrics <term-theme-metrics>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Binary Classification <293-binary-classification>` · :doc:`Log-Odds <295-log-odds>` · :doc:`Sigmoid Function <297-sigmoid-function>` · :doc:`Logistic Regression <292-logistic-regression>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Multiclass AUROC <022-multiclass-auroc>`

----

.. hint::
   **More in Classification & Averaging Metrics**

   :doc:`Accuracy <323-accuracy>` · :doc:`AUC (Area Under the Curve) <371-auc-area-under-the-curve>` · :doc:`Average Precision (AP) <366-average-precision-ap>` · :doc:`Binary Classification <293-binary-classification>` · :doc:`Discriminatory Power <185-discriminatory-power>` · :doc:`F1-score <363-f1-score>` · :doc:`Gini Coefficient <023-gini-coefficient>` · :doc:`Harmonic Mean <362-harmonic-mean>` · :doc:`Log Loss (also called Logarithmic Loss or Cross-Entropy Loss) <417-log-loss-also-called-logarithmic-loss-or-cross-e>` · :doc:`Macro AUC <314-macro-auc>` · :doc:`Macro AUROC (Macro-Averaged AUROC) <018-macro-auroc-macro-averaged-auroc>` · :doc:`Macro Averaging <370-macro-averaging>` · :doc:`Macro F1 <019-macro-f1>` · :doc:`Macro Precision <021-macro-precision>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Classification Probability <https://insightful-data-lab.com/2025/08/23/classification-probability/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: intermediate
