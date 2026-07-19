:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-statistical-tests:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Statistical Tests</b></div>`

===================
Statistical Tests
===================

*Procedures for deciding whether data support a hypothesis.*

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

A **statistical test** is a formal procedure for deciding whether data provide enough evidence to **reject**
a default assumption. Every test follows the same **five steps**: state a **null (H₀)** and **alternative
(Hₐ)** hypothesis, pick a **significance level α**, compute a **test statistic**, find its **p-value**, and
**interpret**.

The decision rule
-----------------

**Reject H₀ when p < α** (the data would be surprising if H₀ were true), otherwise **fail to reject** it.
Crucially, failing to reject is **not** proof that H₀ is true — absence of evidence is not evidence of
absence. Two errors are possible: **Type I** (rejecting a true H₀, rate α) and **Type II** (missing a real
effect, rate β).

The families
------------

Tests split into **parametric** (assuming a distribution — t-test, ANOVA) and **non-parametric**
(assumption-free — **KS**, chi-square), and into one- vs two-sided. The right test depends on the **data
type**, the **question**, and the assumptions you can defend.

----

*Theme:* :ref:`Statistical Inference & Power <term-theme-inference>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Statistical Power <348-statistical-power>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`A/B Testing <380-a-b-testing>`

----

.. hint::
   **More in Statistical Inference & Power**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Sample size <103-sample-size>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Statistical Tests <https://insightful-data-lab.com/2025/08/20/statistical-tests/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
