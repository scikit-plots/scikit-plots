:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-t-test:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>T-Test</b></div>`

========
T-Test
========

*A hypothesis test comparing means using the t-distribution for smaller samples.*

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

A **t-test** asks whether the **means of two groups differ significantly**, using the
**Student's t-distribution**. It is the tool of choice when the **sample is small**
(:math:`n < 30`) and the **population standard deviation is unknown** — estimated instead
from the data, which the heavier-tailed t accounts for.

The three forms
---------------

A **one-sample** t-test compares a sample mean to a known value (is the class average
different from 70?); an **independent two-sample** test compares two separate groups (male
vs female scores); and a **paired** test compares the *same* units measured twice (weight
before vs after a diet).

The statistic
-------------

For the independent test,

.. math::

   t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\dfrac{s_1^2}{n_1} + \dfrac{s_2^2}{n_2}}},

the difference in means over its standard error. Compare :math:`t` to a **critical value**
from the t-distribution at the appropriate **degrees of freedom** (for the independent test,
:math:`df = n_1 + n_2 - 2`), or read a p-value.

Example
-------

A sample of 25 with mean 72, against a hypothesised population mean of 70 with
:math:`s = 5`, gives :math:`t = (72 - 70)/(5/\sqrt{25}) = 2`. The critical value at
:math:`df = 24, \alpha = 0.05` is about **2.064**, and since :math:`2 < 2.064` we **fail to
reject** :math:`H_0` — no significant difference.

Assumptions
-----------

The data should be roughly **normal**; the independent test also assumes **independent
groups** and **equal variances** — if the variances differ, use **Welch's t-test**. As
:math:`n` grows the t-distribution approaches the normal, and the t-test converges to the
z-test.

----

*Theme:* :ref:`Statistical Inference & Power <term-theme-inference>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Z-Test <119-z-test>` · :doc:`Sample Mean <089-sample-mean>` · :doc:`Sample Standard Deviation <088-sample-standard-deviation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Effect Size (δ) <106-effect-size>`

----

.. hint::
   **More in Statistical Inference & Power**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Sample size <103-sample-size>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `T-Test <https://insightful-data-lab.com/2025/08/24/t-test/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
