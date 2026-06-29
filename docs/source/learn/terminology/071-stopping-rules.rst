:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-stopping-rules:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🧫&nbsp;&nbsp;<b>Stopping Rules</b></div>`

================
Stopping Rules
================

*Predefined conditions that determine when to stop collecting data in a test.*

What it is
----------

A **stopping rule** is the **pre-specified condition for ending an experiment** — it says
*when* to stop collecting data and decide. Without one, you can **"peek" until the result
looks significant**, which inflates the **Type I error** (false-positive) rate. The
choice of stopping rule is what makes continuous monitoring valid or invalid.

The families
------------

- **Fixed-horizon (the traditional A/B test)** — fix the sample size up front (say
  10,000 per arm), analyse **once** at the end. Simple and protects :math:`\alpha`, but
  inflexible.
- **Group-sequential** — pre-plan **interim looks** (e.g. every 25%) and spend the error
  budget with an **α-spending** rule (O'Brien–Fleming, Pocock), allowing early stops for
  efficacy or futility. Efficient; standard in clinical trials.
- **Sequential probability ratio test (SPRT)** — check a **likelihood ratio** after each
  observation and stop once it crosses a bound for :math:`H_1` or :math:`H_0`; often uses
  far fewer samples.
- **Bayesian** — stop when a **posterior probability** crosses a threshold (e.g.
  :math:`P(H_1 \mid \text{data}) > 0.95`); intuitive and needs no α-spending.
- **Ethical / practical** — in trials, stop for **harm**, for **clear benefit** (it's
  unethical to withhold), or for **futility**.

Examples
--------

An A/B test might run to a fixed **50,000 visitors per arm**; or check every **5,000**
and stop early if :math:`p < 0.001` under an O'Brien–Fleming bound; or stop once the
posterior probability that B beats A exceeds **0.95**. A clinical trial planned for 1,000
patients might look at 250, 500 and 750 — stopping early for strong benefit, immediately
for harm, or for futility.

Why they matter
---------------

Stopping rules prevent **p-hacking** (stopping the moment things look good), keep **Type
I error** controlled, **save time and cost** by ending early when results are clear, and
**protect participants** in clinical settings. The dividing line: **fixed horizon** (one
look) versus **sequential / adaptive** (group-sequential, SPRT, Bayesian) that permit
interim looks without inflating false positives.

----

**Mind map — connected ideas**

   :doc:`Bayesian Stopping Rules <068-bayesian-stopping-rules>` · :doc:`Sequential Settings <058-sequential-settings>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Frequentist <059-frequentist>` · :doc:`A/B Testing <380-a-b-testing>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>`

----

**More in A/B Testing & Experimentation**

   :doc:`A/B Testing <380-a-b-testing>` · :doc:`A/B/n Test <114-a-b-n-test>` · :doc:`Bayesian Sequential Testing <074-bayesian-sequential-testing>` · :doc:`Bayesian Stopping Rules <068-bayesian-stopping-rules>` · :doc:`Conversion Rate Uplift <067-conversion-rate-uplift>` · :doc:`Fixed-Horizon Testing <082-fixed-horizon-testing>` · :doc:`Group Sequential Testing <079-group-sequential-testing>` · :doc:`Multivariate Test (MVT) <115-multivariate-test-mvt>` · :doc:`Online Experimentation Platforms <070-online-experimentation-platforms>` · :doc:`Optimizely <069-optimizely>` · :doc:`Risk of Peeking <116-risk-of-peeking>` · :doc:`Sequential Testing (also called sequential analysis) <376-sequential-testing-also-called-sequential-analys>` · :doc:`Traditional A/B Test (Fixed-Horizon A/B Test) <081-traditional-a-b-test-fixed-horizon-a-b-test>` · :doc:`Treatment Effect <072-treatment-effect>`

----

*Theme:* :ref:`A/B Testing & Experimentation <term-theme-abtest>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Stopping Rules <https://insightful-data-lab.com/2025/08/25/stopping-rules/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: intermediate
