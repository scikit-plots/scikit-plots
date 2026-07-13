:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-effect-size:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Effect Size (δ)</b></div>`

=================
Effect Size (δ)
=================

*The magnitude of a difference or relationship, independent of sample size.*

What it is
----------

**Effect size** :math:`\delta` measures the **magnitude** of a difference or relationship
— *how big* an effect is, not merely *whether* it exists. Where a p-value answers the
yes/no question of detectability, effect size carries the **practical importance**, and it
is the key input to **power analysis**.

Why it matters
--------------

Statistical and practical significance diverge: with a huge sample a 0.1% difference can be
"significant" yet trivial. Effect size restores the real-world magnitude, which is why it
anchors **sample-size planning**, results reporting in medicine and psychology, and A/B
testing in business.

The common forms
----------------

- **Cohen's d** (standardised mean difference): :math:`d = (\bar{X}_1 - \bar{X}_2)/s_p`
  with pooled SD :math:`s_p`; by rule of thumb **0.2 small, 0.5 medium, 0.8 large**.
- **Noncentrality** :math:`\delta` (power analysis):
  :math:`\delta = (\mu - \mu_0)/(\sigma/\sqrt{n})`, essentially the **expected
  t-statistic** under :math:`H_1` — larger :math:`\delta`, higher power.
- **Association**: Pearson :math:`r`, variance-explained :math:`R^2`, and ANOVA's
  :math:`\eta^2`.
- **Proportions** (A/B): the raw gap :math:`\delta = p_1 - p_2`, or the standardised
  :math:`h = 2\arcsin\!\sqrt{p_1} - 2\arcsin\!\sqrt{p_2}`.

Example
-------

A one-sample t-test with :math:`H_0` mean 100, sample mean 104, :math:`\sigma = 10`,
:math:`n = 25` gives :math:`\delta = (104 - 100)/(10/\sqrt{25}) = 4/2 = 2.0` — the mean is
**2 standard errors** from :math:`H_0`, a very large effect.

Effect size vs p-value
----------------------

The crucial contrast: a **p-value** says whether an effect exists and **depends on sample
size** (large :math:`n` makes tiny effects significant), while **effect size** says how big
it is and is **independent of sample size**. Report them together — significance for
detectability, effect size for meaning.

----

*Theme:* :ref:`Statistical Inference & Power <term-theme-inference>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Statistical Significance <096-statistical-significance>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Statistical Power <348-statistical-power>` · :doc:`Sample size <103-sample-size>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`Hypothesis Testing <107-hypothesis-testing>`

----

.. hint::
   **More in Statistical Inference & Power**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Sample size <103-sample-size>` · :doc:`Significance Level (α) <105-significance-level>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Effect Size (δ) <https://insightful-data-lab.com/2025/08/24/effect-size-%ce%b4/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
