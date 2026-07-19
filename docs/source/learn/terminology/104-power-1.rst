:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-power-1:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Power (1 – β)</b></div>`

===============
Power (1 – β)
===============

*The probability a test correctly detects a real effect (rejects a false null).*

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

**Power** is the probability of **correctly rejecting** the null hypothesis :math:`H_0`
when the alternative is true — of **detecting a real effect**. Formally,

.. math::

   \text{Power} = 1 - \beta,

where :math:`\beta` is the probability of a **Type II error** (missing a true effect). The
usual target is **power** :math:`\ge 0.80`: an 80% chance of catching an effect that is
really there.

The error triad
---------------

Three quantities partition the possibilities when :math:`H_0` is actually false or true:
:math:`\alpha` is the **Type I** error (a false positive — rejecting a true :math:`H_0`),
:math:`\beta` the **Type II** error (a false negative), and :math:`1 - \beta` the power (a
true positive).

What raises power
-----------------

Four levers. A larger **effect size** :math:`\delta` is easier to detect; a larger
**sample size** :math:`n` shrinks the standard error and lifts power; a more lenient
**significance level** :math:`\alpha` (say 0.10 rather than 0.05) raises power but admits
more false positives; and **lower variance** :math:`\sigma^2` sharpens detection.

Example
-------

Testing whether a drug lowers blood pressure, with a medium effect (:math:`\delta = 0.5`),
:math:`n = 30` and :math:`\alpha = 0.05`, power might be only **0.60** — a 40% chance of
missing the effect. Raising :math:`n` to **100** lifts power to about **0.90**.

Where it's used
---------------

Power is the target of **a-priori power analysis**: fixing :math:`\alpha`, a desired power
(commonly 0.80) and an expected :math:`\delta`, one solves for the **minimum sample size**
needed — so that a true effect is very likely to register rather than slip away as a false
negative.

----

*Theme:* :ref:`Statistical Inference & Power <term-theme-inference>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Type I Error <080-type-i-error>` · :doc:`Statistical Power <348-statistical-power>` · :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Sample size <103-sample-size>` · :doc:`Statistical Significance <096-statistical-significance>`

----

.. hint::
   **More in Statistical Inference & Power**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Sample size <103-sample-size>` · :doc:`Significance Level (α) <105-significance-level>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Power (1 – β) <https://insightful-data-lab.com/2025/08/24/power-1-%ce%b2/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
