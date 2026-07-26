:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-type-i-error:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Type I Error</b></div>`

==============
Type I Error
==============

*Rejecting a true null hypothesis — a false positive, controlled at level alpha.*

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

A **Type I error** is a **false positive**: you **reject the null hypothesis**
:math:`H_0` **when it is actually true** — concluding there is an effect or difference
when in reality there is none.

Its probability is α
--------------------

The probability of a Type I error is exactly the **significance level** :math:`\alpha`,
fixed *before* the test: :math:`\alpha = 0.05` accepts a 5% chance of wrongly rejecting a
true :math:`H_0`; :math:`\alpha = 0.01` a 1% chance. Choosing :math:`\alpha` *is* choosing
how often you are willing to cry wolf.

Examples
--------

- **Medicine** — :math:`H_0`: the drug has no effect. If it truly doesn't, but the data
  happen to give :math:`p < 0.05`, you reject :math:`H_0` and declare it works — a Type I
  error.
- **A/B testing** — :math:`H_0`: conversion rates are equal. If they really are, but
  random variation produces a "significant" gap, you've made a Type I error.

Geometrically, with overlapping :math:`H_0` and :math:`H_1` distributions, :math:`\alpha`
is the **rejection region** in the tail; a statistic landing there *while* :math:`H_0`
holds is the error.

Type I vs Type II vs power
--------------------------

There are two ways to be wrong and one way the test "works":

- **Type I (false positive)** — reject a true :math:`H_0`; probability :math:`\alpha`.
- **Type II (false negative)** — fail to reject a false :math:`H_0`; probability
  :math:`\beta`.
- **Power** — correctly reject a false :math:`H_0`; equals :math:`1 - \beta` (and grows
  with sample size, effect size and :math:`\alpha`).

Lowering :math:`\alpha` reduces Type I errors but, all else equal, raises :math:`\beta` —
the two trade off.

Controlling it
--------------

Use a **stricter** :math:`\alpha`; apply **Bonferroni** or other multiple-testing
corrections when running many tests; stick to **fixed-horizon** testing (no peeking), or
an α-spending design if you must look early; and **replicate** to confirm.

----

*Theme:* :ref:`Statistical Inference & Power <term-theme-inference>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Frequentist <059-frequentist>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Group Sequential Testing <079-group-sequential-testing>` · :doc:`A/B Testing <380-a-b-testing>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>`

----

.. hint::
   **More in Statistical Inference & Power**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Sample size <103-sample-size>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Type I Error <https://insightful-data-lab.com/2025/08/25/type-i-error/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
