:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-sample-size:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Sample size</b></div>`

=============
Sample size
=============

*The number of observations collected; larger samples shrink estimation error.*

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

**Sample size** :math:`n` is the **number of observations** used to estimate a parameter
or test a hypothesis. It is the master dial of inference, setting the **precision** of
estimates, the **power** of tests, and the **stability** of conclusions — in a word, the
*resolution* of what the data can tell you.

The square-root law
-------------------

The central relationship is

.. math::

   \text{SE} \propto \frac{1}{\sqrt{n}}.

The square root has a sharp consequence: **doubling** :math:`n` does *not* halve the
standard error — to **halve** it you must **quadruple** :math:`n`. Precision is bought at
an accelerating price.

In estimation and testing
-------------------------

For **estimation**, larger :math:`n` means **narrower confidence intervals**. For
**testing**, the statistic is roughly :math:`\text{effect}/\text{SE}`, so as the SE shrinks
the statistic grows — the same effect becomes **easier to push past the critical value**,
and **power rises**. Small effects only become detectable once :math:`n` is large enough.

The trade-off with effect size
------------------------------

Sample size and effect size **substitute** for one another: a **large** effect shows up in
a **small** sample, while a **small** effect needs a **large** one. In short, *sample size
compensates for a weak signal* — which is exactly why it is chosen up front via **power
analysis** from :math:`\alpha`, target power, the expected effect and the variance.

What n cannot do
----------------

More data **cannot fix bias, poor measurement or a wrong model**, and it can make
**trivial effects statistically significant** — precision is not correctness or importance.
So interpretation, not just design, depends on :math:`n`: a non-significant result with
small :math:`n` is *inconclusive*, and a significant one with enormous :math:`n` should be
checked for **practical relevance**. The emphasis shifts by context — precision in
estimation, power in testing, generalisation in ML, confounding in observational work — but
the concept is one.

----

*Theme:* :ref:`Statistical Inference & Power <term-theme-inference>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Standard Error (SE) <084-standard-error-se>` · :doc:`Statistical Power <348-statistical-power>` · :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Trivial Effects <102-trivial-effects>`

----

.. hint::
   **More in Statistical Inference & Power**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Compromise Power Analysis <093-compromise-power-analysis>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Significance Level (α) <105-significance-level>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Sample size <https://insightful-data-lab.com/2025/08/24/sample-size-n/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
