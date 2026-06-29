:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-proportion:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Proportion</b></div>`

============
Proportion
============

*The fraction of a sample or population that has a given attribute.*

What it is
----------

A **proportion** is a **part-to-whole ratio** — the fraction of a sample or population
with a given characteristic, usually the fraction of **successes** ("yes" outcomes). The
**sample proportion** estimates the true population proportion :math:`p`:

.. math::

   \hat{p} = \frac{x}{n},

with :math:`x` the number of successes and :math:`n` the sample size.

Population vs sample
--------------------

:math:`p` is the **true** proportion in the whole population (fixed, usually unknown);
:math:`\hat{p}` is computed from a sample and **estimates** :math:`p`.

Examples
--------

In a poll, 540 of 1,000 voters back candidate A → :math:`\hat{p} = 0.54` (true :math:`p`
might be 0.55). In quality control, 10 defective bulbs out of 200 →
:math:`\hat{p} = 0.05`, a 5% defect rate.

In inference
------------

Proportions drive categorical inference: a **confidence interval**

.. math::

   \hat{p} \pm z\, \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}},

and hypothesis tests — a **one-sample** proportion test compares :math:`\hat{p}` to a
hypothesised :math:`p_0`, and a **two-proportion z-test** compares :math:`\hat{p}_1` and
:math:`\hat{p}_2` (the workhorse of A/B testing).

Proportion ≈ probability
------------------------

A sample proportion **estimates a population probability**: if 30% of surveyed users
clicked an ad, the probability a random user clicks is :math:`\approx 0.30`. This is why
proportions sit at the centre of surveys, A/B tests, medical studies and quality control —
they turn yes/no data into estimable probabilities.

----

**Mind map — connected ideas**

   :doc:`True Conversion Rate <083-true-conversion-rate>` · :doc:`Standard Error (SE) <084-standard-error-se>` · :doc:`Probability <025-probability>` · :doc:`True Population Parameter <092-true-population-parameter>` · :doc:`Frequentist <059-frequentist>` · :doc:`A/B Testing <380-a-b-testing>`

----

**More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>`

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Proportion <https://insightful-data-lab.com/2025/08/25/proportion/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: beginner
