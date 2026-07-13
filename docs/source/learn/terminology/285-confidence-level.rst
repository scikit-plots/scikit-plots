:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-confidence-level:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Confidence Level</b></div>`

==================
Confidence Level
==================

*The long-run proportion of intervals expected to contain the true parameter.*

What it is
----------

A model's **confidence level** is the **probability it attaches** to its prediction — how sure it is that an
instance belongs to the predicted class. It is only meaningful if it is **calibrated**: a model is
**well-calibrated** when predictions made with confidence :math:`p` are correct about **100p%** of the time
(predict 0.9 → right 90% of the time).

How it's checked
----------------

A **reliability diagram** bins predictions by confidence level and plots confidence against actual
**accuracy**; perfect calibration lies on the **diagonal**. The gap is summarized by the **Expected
Calibration Error (ECE)** — the average distance between confidence and accuracy across bins.

Why it matters
--------------

Raw **accuracy** says nothing about whether the confidence is honest, yet downstream **risk-based
decisions** (which cases to escalate, when to defer) depend on trusting the number. This is the *model*
sense of confidence, distinct from a statistical **confidence interval**.

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Overconfident <284-overconfident>` · :doc:`Underconfident <283-underconfident>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Binary Cross-Entropy (BCE) <288-binary-cross-entropy-bce>` · :doc:`Risk-Based Decisions <286-risk-based-decisions>` · :doc:`Multiclass AUROC <022-multiclass-auroc>`

----

.. hint::
   **More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>` · :doc:`Probability <025-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Confidence Level <https://insightful-data-lab.com/2025/08/21/confidence-level/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
