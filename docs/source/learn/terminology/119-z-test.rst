:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-z-test:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Z-Test</b></div>`

========
Z-Test
========

*A hypothesis test using the normal distribution when variance is known or n is large.*

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

A **z-test** is a hypothesis test for whether a **sample mean (or proportion) differs from a
known population value**, built on the **standard normal** distribution. It applies when the
**sample is large** (:math:`n > 30`, so the CLT holds) and the **population variance is
known** — or well approximated by a large sample.

The statistic
-------------

For a one-sample mean,

.. math::

   z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}},

the gap between the sample mean :math:`\bar{X}` and the hypothesised :math:`\mu`, measured in
**standard errors**. There are three common forms: **one-sample** (mean vs population),
**two-sample** (two independent means), and the **proportion** z-test.

The procedure
-------------

State :math:`H_0` (no difference) and :math:`H_1`; pick :math:`\alpha`; compute :math:`z`;
and compare to the **critical value** (:math:`\pm 1.96` at :math:`\alpha = 0.05`, two-tailed)
or read off a p-value — reject when :math:`|z|` exceeds it.

Examples
--------

With :math:`\mu = 100, \sigma = 15, n = 50` and a sample mean of 105,
:math:`z = 5 / 2.12 \approx 2.36 > 1.96`, so reject :math:`H_0`. For a proportion, 320 of 500
(0.64) against a hypothesised 0.60 gives :math:`z = 0.04 / 0.022 \approx 1.82 < 1.96` —
**fail** to reject.

Z-test vs t-test
----------------

The choice turns on what you know about the variance. Use a **z-test** when :math:`\sigma` is
**known** and the sample is **large**, working from the normal distribution; use a
**t-test** when :math:`\sigma` is **unknown** and estimated from the sample :math:`s`,
working from the heavier-tailed **Student's t** — the gap between them vanishes as :math:`n`
grows.

----

*Theme:* :ref:`Statistical Inference & Power <term-theme-inference>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Z-Score <097-z-score>` · :doc:`Two-Proportion Z-Test <098-two-proportion-z-test>` · :doc:`Critical Value <087-critical-value>` · :doc:`Standard Error (SE) <084-standard-error-se>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`T-Test <120-t-test>`

----

.. hint::
   **More in Statistical Inference & Power**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Sample size <103-sample-size>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Z-Test <https://insightful-data-lab.com/2025/08/24/z-test/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
