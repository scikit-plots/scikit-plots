:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-probability-distribution:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Probability Distribution</b></div>`

==========================
Probability Distribution
==========================

*A description of how probability is spread over a variable's possible values.*

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

A **probability distribution** describes how probability is **spread over the possible values** of a
random variable — the full accounting of what can happen and how likely each outcome is.

How it's described
------------------

For a **discrete** variable it is given by a **probability mass function**, for a **continuous** one by a
**probability density function**, and for either by a **cumulative distribution function**. Whatever the
form, the total probability is **one**:

.. math::

   \sum_x p(x) = 1 \quad\text{(discrete)}, \qquad \int_{-\infty}^{\infty} f(x)\,dx = 1 \quad\text{(continuous)}.

Characterizing it
-----------------

Distributions are summarized by **parameters** (a **mean** for location, a **variance** for spread) and by
**moments**; you can **fit** a distribution to data or **sample** synthetic data from one. Common families
include the **normal**, **Bernoulli / binomial**, **Poisson** and **exponential**.

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Probability Density <237-probability-density>` · :doc:`Probability Mass <239-probability-mass>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Probability <025-probability>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>`

----

.. hint::
   **More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Probability Distribution <https://insightful-data-lab.com/2025/08/22/probability-distributions/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
