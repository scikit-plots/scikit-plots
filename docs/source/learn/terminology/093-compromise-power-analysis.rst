:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-compromise-power-analysis:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">🧮&nbsp;&nbsp;<b>Compromise Power Analysis</b></div>`

===========================
Compromise Power Analysis
===========================

*Sizing a study by trading Type I against Type II error at a fixed ratio rather than fixing one.*

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

**Compromise power analysis** finds a sensible **balance between the Type I error**
:math:`\alpha` (false positives) **and the Type II error** :math:`\beta` (false
negatives) **when the sample size** :math:`n` **is fixed**. Unlike *a-priori* analysis —
which fixes :math:`\alpha` and power and solves for :math:`n` — here you already know
:math:`n` and ask what :math:`\alpha`/:math:`\beta` trade-off is reasonable.

Why it's useful
---------------

Sometimes :math:`n` simply **cannot change** — a limited participant pool, a budget cap,
or a historical dataset. A-priori analysis might say "you need 500 subjects" when you have
200; compromise analysis answers the real question: *with 200, what :math:`\alpha` and
:math:`\beta` give a balanced test?*

How it works
------------

Specify the **effect size** :math:`\delta`, the **available** :math:`n`, and a desired
**ratio of Type I to Type II error** (often :math:`\alpha = \beta`, i.e. a 1:1 ratio). The
procedure then solves for the :math:`\alpha` and :math:`\beta` that satisfy the
constraint.

Example
-------

With a medium effect (Cohen's :math:`d = 0.5`), :math:`n = 40` (20 per group), and a
requirement that :math:`\alpha = \beta`, the analysis might return
:math:`\alpha = \beta = 0.12` (power :math:`= 0.88`). You **accept a higher false-positive
rate (12%)** to keep false negatives equally low, given the small sample.

The three power analyses
------------------------

- **A-priori** — input effect size, :math:`\alpha`, power → output **required** :math:`n`
  ("how many subjects do I need?").
- **Post-hoc** — input observed :math:`n` and effect size → output **achieved power**
  ("given what I saw, what was the power?").
- **Compromise** — input effect size, available :math:`n`, error ratio → output
  **appropriate** :math:`\alpha` **and** :math:`\beta` ("with this :math:`n`, how do I
  balance the two errors?").

It is the pragmatic choice when data is scarce, though it is less conventional than the
fixed :math:`\alpha = 0.05` and depends on an assumed effect size.

----

*Theme:* :ref:`Statistical Inference & Power <term-theme-inference>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Type I Error <080-type-i-error>` · :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Statistical Power <348-statistical-power>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Frequentist <059-frequentist>`

----

.. hint::
   **More in Statistical Inference & Power**

   :doc:`A Priori Power Analysis <095-a-priori-power-analysis>` · :doc:`Chi-square (χ²) Test <324-chi-square-2-test>` · :doc:`Clopper–Pearson Interval <356-clopperpearson-interval>` · :doc:`Confidence Intervals (CIs) <377-confidence-intervals-cis>` · :doc:`Effect Size (δ) <106-effect-size>` · :doc:`Hypothesis Testing <107-hypothesis-testing>` · :doc:`Kolmogorov–Smirnov (KS) Test <325-kolmogorovsmirnov-ks-test>` · :doc:`Minimum Detectable Lift (MDL) <101-minimum-detectable-lift-mdl>` · :doc:`P-Value (probability value) <118-p-value-probability-value>` · :doc:`Post Hoc Power Analysis <094-post-hoc-power-analysis>` · :doc:`Power (1 – β) <104-power-1>` · :doc:`Power Analysis <378-power-analysis>` · :doc:`Sample size <103-sample-size>` · :doc:`Significance Level (α) <105-significance-level>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Compromise Power Analysis <https://insightful-data-lab.com/2025/08/24/compromise-power-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: beginner
