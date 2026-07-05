:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-statistical-power:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Statistical Power</b></div>`

===================
Statistical Power
===================

*The probability of detecting a true effect when one exists.*

What it is
----------

**Statistical power** is the probability that a test **correctly detects a real effect** — that it **rejects**
the null hypothesis when the null is genuinely **false**. Formally it is :math:`1 - \beta`, where
:math:`\beta` is the **Type II error** (false-negative) rate.

What it depends on
------------------

Power rises with **larger effect sizes**, **bigger samples**, a **looser** significance level :math:`\alpha`,
and **lower** variance. Researchers conventionally target **0.80** — an 80% chance of catching a true effect —
and solve for the **sample size** that achieves it via **power analysis**.

Why it matters
--------------

An **underpowered** study is likely to **miss** true effects and produces findings that **don't replicate**;
power is the guard against **false negatives**, the complement of the :math:`\alpha` that guards against false
positives. Too much power on a huge sample flips the risk — flagging **trivial** effects as significant, which
is why **effect size** is reported alongside significance.

----

**Mind map — connected ideas**

   :doc:`Power Analysis <378-power-analysis>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Statistical Tests <328-statistical-tests>` · :doc:`Population Proportion <199-population-proportion>` · :doc:`Correlation <305-correlation>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>`

----

**More in Statistical Inference & Power**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Sample size <103-sample-size>`

----

*Theme:* :ref:`Statistical Inference & Power <term-theme-inference>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Statistical Power <https://insightful-data-lab.com/2025/08/20/statistical-power/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: beginner
