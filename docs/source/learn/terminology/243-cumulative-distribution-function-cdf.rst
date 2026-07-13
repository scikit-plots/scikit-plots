:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-cumulative-distribution-function-cdf:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Cumulative Distribution Function (CDF)</b></div>`

========================================
Cumulative Distribution Function (CDF)
========================================

*The probability that a variable is at most a given value.*

What it is
----------

The **cumulative distribution function (CDF)** gives the probability that a random variable is **at
most** :math:`x`:

.. math::

   F(x) = \Pr[X \le x].

Unlike the PMF or PDF, it works for **both** discrete and continuous variables.

Properties
----------

The CDF is **non-decreasing**, runs from **0 to 1**, and is right-continuous. For a **discrete** variable
it is a **step function**; for a **continuous** one it is smooth:

.. math::

   F(x) = \sum_{k \le x} p(k) \quad\text{(discrete)}, \qquad F(x) = \int_{-\infty}^{x} f(t)\,dt \quad\text{(continuous)}.

Why it's useful
---------------

It directly answers "**at most**" and interval questions, and it is the **bridge** between
representations: **quantiles** are read off its inverse and the **density** is its derivative. For a fair
die, :math:`F(2) = 1/3`.

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Probability Density <237-probability-density>` · :doc:`Probability Mass <239-probability-mass>` · :doc:`Probability Distribution <240-probability-distribution>` · :doc:`Quantile Level <255-quantile-level>` · :doc:`Quantile Regression <254-quantile-regression>` · :doc:`Normal Distribution <238-normal-distribution>`

----

.. hint::
   **More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>` · :doc:`Probability <025-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Cumulative Distribution Function (CDF) <https://insightful-data-lab.com/2025/08/22/cumulative-distribution-function-cdf/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
