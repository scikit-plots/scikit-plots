:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-a-priori-power-analysis:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>A Priori Power Analysis</b></div>`

=========================
A Priori Power Analysis
=========================

*Computing the sample size required before a study for a target power and effect size.*

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

**A-priori power analysis** computes the **sample size** :math:`n` **needed before
collecting data**, from four inputs: the significance level :math:`\alpha` (Type I risk),
the desired **power** :math:`1 - \beta` (chance of detecting a true effect), the expected
**effect size** :math:`\delta`, and the data variance. The aim: a study **large enough to
detect a meaningful effect** but no larger.

Why it matters
--------------

It guards against **underpowered** studies (false negatives — missing real effects) and
**overpowered** ones (wasted resources chasing trivial effects), and forces you to commit
to a **minimum meaningful effect size** in advance rather than rationalising after the
fact.

The sample-size formula
-----------------------

For a two-sample mean test,

.. math::

   n = \left(\frac{Z_{1-\alpha/2} + Z_{1-\beta}}{\delta}\right)^2,

where :math:`Z_{1-\alpha/2}` is the critical value for :math:`\alpha` (1.96 at
:math:`\alpha = 0.05`, two-tailed), :math:`Z_{1-\beta}` the value for the target power
(0.84 for 80%), and :math:`\delta = (\mu_1 - \mu_2)/\sigma` the standardised effect size
(Cohen's d). In practice tools like **G\*Power**, R or ``statsmodels`` do the arithmetic.

Example
-------

To detect a conversion lift from 10% to 11% at :math:`\alpha = 0.05` and **80% power**
with an effect of 0.01, the analysis returns about **7,850 users per group** — testing
only 1,000 per arm would be badly underpowered.

The power-analysis family
-------------------------

**A-priori** (before) sets the sample size; **post-hoc** (after) estimates achieved power
and is controversial; **sensitivity** asks, for a given :math:`n`, :math:`\alpha` and
power, the *smallest detectable* effect. The everyday convention is :math:`\alpha = 0.05`,
power = 0.80.

----

*Theme:* :ref:`Statistical Inference & Power <term-theme-inference>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Statistical Power <348-statistical-power>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Type I Error <080-type-i-error>` · :doc:`Fixed-Horizon Testing <082-fixed-horizon-testing>`

----

.. hint::
   **More in Statistical Inference & Power**

   :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Sample size <103-sample-size>` · :doc:`Significance Level (α) <105-significance-level>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `A Priori Power Analysis <https://insightful-data-lab.com/2025/08/24/a-priori-power-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
