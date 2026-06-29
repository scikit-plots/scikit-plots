:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-statistical-significance:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🎲&nbsp;&nbsp;<b>Statistical Significance</b></div>`

==========================
Statistical Significance
==========================

*Evidence that an observed effect is unlikely under the null hypothesis, judged against a chosen threshold.*

What it is
----------

A result is **statistically significant** when it is **unlikely to have arisen by random
chance alone, assuming the null hypothesis** :math:`H_0` **is true** — operationally, when
the **p-value** :math:`\le \alpha`, the predefined significance level. It answers one
narrow question: *is this result sufficiently inconsistent with* :math:`H_0`?

What it does not tell you
-------------------------

Significance says nothing about **how large** the effect is, **whether it matters**, or
**whether it will replicate**. And the **p-value** is widely misread: it is the
probability, *under* :math:`H_0`, of data as extreme or more extreme than observed — *not*
the probability that :math:`H_0` is true, nor the probability the result is "due to
chance."

Statistical vs practical significance
-------------------------------------

These come apart. **Statistical** significance is about *detectability* and depends
heavily on sample size; **practical** significance is about *real-world importance* and
depends on effect size and context. With a large enough :math:`n`, a **trivial** effect
becomes significant; with a small :math:`n`, a **meaningful** one may not — so a result can
be significant yet practically meaningless.

Significance vs power
---------------------

Significance is a **binary** outcome (yes/no); **power** is the *probability* of achieving
it when a real effect exists. High power makes a true effect likely to register; under low
power, a non-significant result is **ambiguous** (it may just reflect too little data).

A decision rule, not a verdict
------------------------------

Treat significance as a **decision rule for controlling false positives under repeated
use** — part of a risk-management system, not a proof of truth. The :math:`\alpha = 0.05`
line is a **convention**, not a law: "significant" is not "important," and "not
significant" is not "no effect." Good practice reports **effect sizes, confidence
intervals, and power** alongside it, never significance alone.

----

**Mind map — connected ideas**

   :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Type I Error <080-type-i-error>` · :doc:`Statistical Power <348-statistical-power>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Frequentist <059-frequentist>` · :doc:`A/B Testing <380-a-b-testing>`

----

**More in Probability & Statistics Foundations**

   :doc:`Beta Distribution <099-beta-distribution>` · :doc:`Confidence Level <285-confidence-level>` · :doc:`Correlation <305-correlation>` · :doc:`Critical Value <087-critical-value>` · :doc:`Cumulative Distribution Function (CDF) <243-cumulative-distribution-function-cdf>` · :doc:`Frequentist <059-frequentist>` · :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Likelihood <304-likelihood>` · :doc:`Margin of Error (MoE) <086-margin-of-error-moe>` · :doc:`Mean <316-mean>` · :doc:`Median <315-median>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Outlier <307-outlier>` · :doc:`Population Proportion <199-population-proportion>`

----

*Theme:* :ref:`Probability & Statistics Foundations <term-theme-probstats>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Statistical Significance <https://insightful-data-lab.com/2025/08/24/significance/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: beginner
