:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-p-value-probability-value:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>P-Value (probability value)</b></div>`

=============================
P-Value (probability value)
=============================

*The probability of data at least as extreme as observed, assuming the null is true.*

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

The **p-value** is the probability of observing data **at least as extreme** as what you
saw, **assuming the null hypothesis** :math:`H_0` **is true**. It is a measure of **evidence
against** :math:`H_0`: a **low** p-value means the data would be surprising under
:math:`H_0` (stronger evidence against it), a **high** one means the data sit comfortably
with :math:`H_0`.

Reading it
----------

By convention :math:`p < 0.01` is very strong evidence, :math:`p < 0.05` is "significant",
:math:`p < 0.10` is weak, and :math:`p > 0.10` is not significant. The decision rule pairs it
with a threshold: if :math:`p \le \alpha`, reject :math:`H_0`; if :math:`p > \alpha`, fail to
reject.

How it's computed
-----------------

Form the **test statistic** (z, t, :math:`\chi^2`…) and find the tail probability of a value
that extreme under its distribution — one tail for a one-sided test, both for two-sided. For
a two-tailed t-test,

.. math::

   p = 2 \times P\!\left(T \ge |t_{\text{observed}}|\right).

For example, 17 heads in 20 tosses gives a binomial p of about **0.003** against a fair coin
(reject); a two-sample test with :math:`t = 2.1, df = 28` gives :math:`p \approx 0.045`
(reject at 0.05).

What it is not
--------------

Three persistent misconceptions: the p-value is **not** the probability that :math:`H_0` is
true; it is **not** an effect size (a tiny p means *surprising*, not *large*); and it is
**sample-size sensitive** — huge samples make trivial effects significant, tiny samples can
miss real ones. Always read it alongside an effect size and a confidence interval.

----

*Theme:* :ref:`Statistical Inference & Power <term-theme-inference>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Statistical Significance <096-statistical-significance>` · :doc:`Significance Level (α) <105-significance-level>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Type I Error <080-type-i-error>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Z-Test <119-z-test>`

----

.. hint::
   **More in Statistical Inference & Power**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Sample size <103-sample-size>` · :doc:`Significance Level (α) <105-significance-level>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `P-Value (probability value) <https://insightful-data-lab.com/2025/08/24/p-value-probability-value/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
