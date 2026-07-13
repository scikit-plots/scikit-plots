:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-power-analysis:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Power Analysis</b></div>`

================
Power Analysis
================

*Planning sample size so a test can detect an effect of interest.*

What it is
----------

**Power analysis** plans a study around its **statistical power** — the probability of **detecting a real
effect** (correctly rejecting H₀ when it's false), which equals :math:`1 - \beta`. Most often it answers
*how many samples do I need?*

The four levers
---------------

Power, **effect size**, **significance level α**, and **sample size** are locked in a relationship — fix any
**three** and the fourth is determined. Bigger effects, larger samples, or a looser α all **raise** power;
the usual target is **≥ 0.80**.

Why do it first
---------------

Run **before** collecting data, power analysis prevents **underpowered** experiments that waste resources
and are likely to **miss** true effects (a high Type II risk). Run **after**, it tells you how much you could
realistically have detected — and warns against over-reading a **non-significant** result.

----

*Theme:* :ref:`Statistical Inference & Power <term-theme-inference>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Statistical Tests <328-statistical-tests>` · :doc:`Statistical Power <348-statistical-power>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`A/B Testing <380-a-b-testing>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>`

----

.. hint::
   **More in Statistical Inference & Power**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Sample size <103-sample-size>` · :doc:`Significance Level (α) <105-significance-level>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Power Analysis <https://insightful-data-lab.com/2025/08/19/power-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
