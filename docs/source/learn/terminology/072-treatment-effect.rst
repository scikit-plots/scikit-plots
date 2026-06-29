:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-treatment-effect:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🧫&nbsp;&nbsp;<b>Treatment Effect</b></div>`

==================
Treatment Effect
==================

*The causal difference in outcome between treated and untreated units.*

What it is
----------

The **treatment effect** is the **causal impact of an intervention** versus a control —
*how much did the treatment change the outcome compared with what would have happened
without it?* In the **potential-outcomes** framework,

.. math::

   \text{Treatment Effect} = Y(1) - Y(0),

where :math:`Y(1)` is the outcome if the unit *is* treated and :math:`Y(0)` the outcome
if it is *not*.

The fundamental problem
-----------------------

For any single unit you only ever observe **one** of :math:`Y(1)` or :math:`Y(0)` — never
both — so the individual effect is unobservable. This *fundamental problem of causal
inference* is why we estimate **averages** instead of individual effects.

The hierarchy of effects
------------------------

- **ITE (individual)** — :math:`Y_i(1) - Y_i(0)` for one unit; unobservable.
- **ATE (average)** — :math:`\text{ATE} = \mathbb{E}[Y(1) - Y(0)]`, the
  population-average effect, the usual target of RCTs and A/B tests.
- **CATE (conditional average)** —
  :math:`\text{CATE}(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]`, the effect within a
  subgroup defined by covariates :math:`x` (age, segment) — the basis of personalised
  interventions.
- **LATE (local average)** — the effect for a specific compliant subgroup, typically via
  instrumental variables.

Examples
--------

A drug with 60% recovery vs 50% on placebo has :math:`\text{ATE} = +10` points. A website
variant at 5.5% vs 5% conversion has :math:`\text{ATE} = +0.5` points, a
:math:`(0.055 - 0.05)/0.05 = +10\%` relative lift.

How it's estimated
------------------

In a **randomised controlled trial** (or A/B test), randomisation makes the groups
comparable, so the **difference in group means** is an unbiased estimate of the ATE. In
**observational** data, confounding must be removed with causal-inference tools —
matching, regression adjustment, instrumental variables, difference-in-differences or
**propensity scores**. ML methods increasingly estimate **heterogeneous (CATE)** effects
for targeting.

----

**Mind map — connected ideas**

   :doc:`A/B Testing <380-a-b-testing>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Parameter(s) of Interest <065-parameter-s-of-interest>` · :doc:`Causal Inference <117-causal-inference>` · :doc:`Posterior probability of uplift <053-posterior-probability-of-uplift>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>`

----

**More in A/B Testing & Experimentation**

   :doc:`A/B Testing <380-a-b-testing>` · :doc:`A/B/n Test <114-a-b-n-test>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Bayesian Stopping Rules <068-bayesian-stopping-rules>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Fixed-Horizon Testing <082-fixed-horizon-testing>` · :doc:`Group Sequential Testing <079-group-sequential-testing>` · :doc:`Multivariate Test (MVT) <115-multivariate-test-mvt>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`Optimizely <069-optimizely>` · :doc:`Risk of Peeking <116-risk-of-peeking>` · :doc:`Sequential Testing (also called sequential analysis) <376-sequential-testing-also-called-sequential-analys>` · :doc:`Stopping Rules <071-stopping-rules>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>`

----

*Theme:* :ref:`A/B Testing & Experimentation <term-theme-abtest>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Treatment Effect <https://insightful-data-lab.com/2025/08/25/treatment-effect/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
