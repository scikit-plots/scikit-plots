:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-population-proportion:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Population Proportion</b></div>`

=======================
Population Proportion
=======================

*The fraction of an entire population with a given attribute.*

What it is
----------

The **population proportion** is the **true fraction** of an entire population that has some characteristic —
the real conversion rate, defect rate, or approval share — usually written **p**. It is a **parameter**: a
fixed but usually **unknown** number you want to learn.

Estimating it
-------------

Since you rarely measure everyone, you estimate p with the **sample proportion** :math:`\hat{p} = x/n`
(successes over sample size). :math:`\hat{p}` is a **statistic** that varies from sample to sample, with
**standard error** :math:`\sqrt{p(1-p)/n}` shrinking as **n** grows — the basis for **confidence intervals**
and **hypothesis tests** about p.

Where it shows up
-----------------

Proportions are everywhere in ML and analytics — **conversion rates**, click rates, and accuracy are all
population proportions estimated from samples. That is why **A/B tests** and polls rest on proportion
inference, and why bigger samples give **tighter** estimates.

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Statistical Tests <328-statistical-tests>` · :doc:`Standard Error (SE) <084-standard-error-se>` · :doc:`Statistical Power <348-statistical-power>` · :doc:`Conversion Rate (CR) <299-conversion-rate-cr>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>`

----

.. hint::
   **More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Probability <025-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Population Proportion <https://insightful-data-lab.com/2025/08/23/population-proportion/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
