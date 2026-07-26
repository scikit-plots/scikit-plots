:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-normal-distribution:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Normal Distribution</b></div>`

=====================
Normal Distribution
=====================

*The bell-shaped Gaussian distribution defined by its mean and variance.*

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

The **normal (Gaussian) distribution** is the continuous, **bell-shaped**, symmetric distribution defined
by two parameters — the **mean** :math:`\mu` (its center) and the **standard deviation** :math:`\sigma`
(its spread; variance :math:`\sigma^2`):

.. math::

   X \sim \mathcal{N}(\mu, \sigma^2).

Its **mean, median and mode coincide**, and it extends from :math:`-\infty` to :math:`+\infty`.

The density
-----------

.. math::

   f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right).

About **68%** of values lie within :math:`1\sigma` of the mean, **95%** within :math:`2\sigma`, and
**99.7%** within :math:`3\sigma` (the *68–95–99.7 rule*). Standardizing with :math:`z = (x-\mu)/\sigma`
maps any normal onto the **standard normal** :math:`\mathcal{N}(0, 1)`, so a single table serves all.

Why it's everywhere
-------------------

The **Central Limit Theorem** — averages of many independent, finite-variance quantities tend toward a
normal — makes it the default model for **measurement errors** and **aggregates**. But it has **light
tails**: with heavy-tailed data or frequent **outliers** (e.g. Cauchy, Pareto) it fits poorly and
least-squares methods grow unreliable.

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Probability Distribution <240-probability-distribution>` · :doc:`Probability Density <237-probability-density>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Z-Score <097-z-score>` · :doc:`Standard Error (SE) <084-standard-error-se>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>`

----

.. hint::
   **More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>` · :doc:`Probability <025-probability>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Normal Distribution <https://insightful-data-lab.com/2025/08/22/normal-distribution/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
