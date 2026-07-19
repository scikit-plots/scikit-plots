:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-true-mean-population-mean:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>True Mean (Population Mean)</b></div>`

=============================
True Mean (Population Mean)
=============================

*The actual mean of the whole population that a sample mean estimates.*

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

The **true mean**, or **population mean** :math:`\mu`, is the **actual arithmetic average
of an entire population**. It is a **fixed** value but usually **unknown** — we can rarely
measure everyone — so we estimate it with the **sample mean** :math:`\bar{x}`.

Parameter vs estimate
---------------------

.. math::

   \mu = \frac{1}{N}\sum_{i=1}^{N} x_i, \qquad \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i.

Here :math:`\mu` is a **parameter** (fixed, unknown) and :math:`\bar{x}` a **statistic**
(random, changing from sample to sample). By the **Law of Large Numbers**, as :math:`n`
grows, :math:`\bar{x} \to \mu`.

Example
-------

For the population :math:`\{2, 4, 6, 8, 10\}`, the true mean is
:math:`\mu = (2+4+6+8+10)/5 = 6`. A sample :math:`\{4, 10\}` gives :math:`\bar{x} = 7` —
an *estimate* of the true 6, off by sampling luck.

Inference about μ
-----------------

Everything in classical inference targets :math:`\mu`: a **hypothesis test** checks a
claim like :math:`H_0 : \mu = 100`, and a **confidence interval** says "we're 95%
confident the true mean :math:`\mu` lies between :math:`X` and :math:`Y`." The sample mean
is the **best unbiased estimator** of :math:`\mu`, and tests and intervals quantify how
far it might be from the truth.

The proportion analogue
-----------------------

For yes/no outcomes the same parameter-vs-estimate story holds with the **true conversion
rate** :math:`p` estimated by :math:`\hat{p}` — :math:`\mu \leftrightarrow p`,
:math:`\bar{x} \leftrightarrow \hat{p}` — the mean and proportion versions of one idea.

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`True Conversion Rate <083-true-conversion-rate>` · :doc:`Standard Error (SE) <084-standard-error-se>` · :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Frequentist <059-frequentist>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>`

----

.. hint::
   **More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `True Mean (Population Mean) <https://insightful-data-lab.com/2025/08/25/true-mean-population-mean/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
