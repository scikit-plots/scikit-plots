:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-probability-density:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Probability Density</b></div>`

=====================
Probability Density
=====================

*The relative likelihood of a continuous variable at a value, given by its PDF.*

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

The **probability density function (PDF)** :math:`f(x)` describes a **continuous** random variable. It is
**not** a probability itself — its **area** is: the probability of landing in an interval is the integral

.. math::

   P(a \le X \le b) = \int_a^b f(x)\,dx.

Properties
----------

The density satisfies :math:`f(x) \ge 0`, may **exceed 1** (it is a density, not a probability),
integrates to **one** over the whole line, and assigns probability **zero** to any *exact* value. The bell
curve of the **normal distribution** is the classic PDF.

Link to the CDF
---------------

The density is the **derivative** of the cumulative distribution function,

.. math::

   f(x) = F'(x),

so equivalently :math:`F` is the running integral of :math:`f`.

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Probability Mass <239-probability-mass>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Probability Distribution <240-probability-distribution>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Probabilistic Forecasts <241-probabilistic-forecasts>` · :doc:`Quantile Regression <254-quantile-regression>`

----

.. hint::
   **More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Probability Density <https://insightful-data-lab.com/2025/08/22/probability-density/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
