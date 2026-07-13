:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-probability-mass:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Probability Mass</b></div>`

==================
Probability Mass
==================

*The probability assigned to each value of a discrete variable.*

What it is
----------

The **probability mass function (PMF)** of a **discrete** random variable gives the probability that it
takes **exactly** a given value:

.. math::

   p(x) = \Pr[X = x].

It maps each possible value to a probability — for a fair die, :math:`p(k) = 1/6` for :math:`k = 1,\dots,6`.

Properties
----------

Every mass lies in :math:`[0, 1]`, and all masses **sum to one**:

.. math::

   \sum_x p(x) = 1.

Discrete only
-------------

Mass applies to **discrete** outcomes, where a single value can carry **positive** probability — unlike a
continuous variable, where any *exact* point has probability **zero** (there, density takes its place).

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Probability Density <237-probability-density>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Probability Distribution <240-probability-distribution>` · :doc:`Classification Probability <231-classification-probability>` · :doc:`Probability <025-probability>` · :doc:`Normal Distribution <238-normal-distribution>`

----

.. hint::
   **More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Probability Mass <https://insightful-data-lab.com/2025/08/22/probability-mass/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
