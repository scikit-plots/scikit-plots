:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-confidence-intervals-cis:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Confidence Intervals (CIs)</b></div>`

============================
Confidence Intervals (CIs)
============================

*A range that would contain the true parameter a stated fraction of the time.*

What it is
----------

A **confidence interval** is a **range** of plausible values for an unknown parameter — a mean, a
proportion — computed from a sample together with a **confidence level** (typically **95%**). It expresses
the **uncertainty** in a point estimate: a wider interval means less precision.

What the level means
--------------------

The confidence level is a statement about the **procedure**, not any one interval. If you repeated the study
many times, about **95% of the intervals** you built would contain the true value — it is **not** a 95%
probability that the parameter lies in *this* interval (in the frequentist view the parameter is fixed).
Intervals **narrow** as the sample size **grows**.

How they're built
-----------------

A CI is typically an estimate **± a margin of error** (a critical value times a **standard error**), but for
tricky quantities like a **binomial proportion** there are several methods — **Wald**, **Wilson**,
**Clopper–Pearson**, **bootstrap** — that trade **coverage** against **width**.

----

*Theme:* :ref:`Statistical Inference & Power <term-theme-inference>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Wilson Score Interval <357-wilson-score-interval>` · :doc:`Bootstrap Confidence Intervals (CIs) <024-bootstrap-confidence-intervals-cis>` · :doc:`Standard Error (SE) <084-standard-error-se>` · :doc:`Statistical Tests <328-statistical-tests>` · :doc:`Normal Distribution <238-normal-distribution>`

----

.. hint::
   **More in Statistical Inference & Power**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Sample size <103-sample-size>` · :doc:`Significance Level (α) <105-significance-level>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Confidence Intervals (CIs) <https://insightful-data-lab.com/2025/08/19/confidence-intervals-cis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
